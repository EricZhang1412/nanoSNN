from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401 - registers torch.npu
from torch.profiler import ProfilerActivity as TorchProfilerActivity
from torch.profiler import profile as torch_profile
from spikingjelly.activation_based import functional

from models.build_model import init_weights  # triggers model registration imports
from models.common.registry import get_model_cls
from models.common.triton_lif import patch_lif_nodes_for_inference
from models.spikformer.streaming_attention import patch_gated_attention_streaming_inference
from utils.load_config import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile Spikformer top ops on Ascend NPU")
    p.add_argument("--model_config", default="configs/model_configs/pilot/spikformer_shd_c0_sdla.yaml")
    p.add_argument("--T", type=int, default=None, help="override model T")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--mode", choices=["eval", "train"], default="eval")
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--record_shapes", action="store_true")
    p.add_argument("--profile_memory", action="store_true")
    p.add_argument("--row_limit", type=int, default=40)
    p.add_argument("--sort_by", default="self_cpu_time_total")
    p.add_argument("--npu_trace", action="store_true", help="also export torch_npu profiler trace")
    p.add_argument("--trace_dir", default="kernel_profiles/spikformer_npu")
    p.add_argument("--triton_lif", action="store_true", help="patch LIF nodes with fused Triton-Ascend LIF")
    p.add_argument("--triton_lif_train", action="store_true", help="enable fused LIF ATan backward in --mode train")
    p.add_argument("--lif_block", type=int, default=256)
    p.add_argument("--streaming_attention", action="store_true", help="patch C0/C1/C2/C3 eval attention to avoid S_seq allocation")
    p.add_argument("--triton_attention", action="store_true", help="use Triton-Ascend kernel inside the streaming attention patch")
    return p.parse_args()


def _make_input(model_config, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    name = str(model_config.name).lower()
    T = int(getattr(model_config, "T", 4))
    num_classes = int(getattr(model_config, "num_classes", 10))
    if name == "spikformer_audio":
        in_channels = int(getattr(model_config, "in_channels", 1))
        n_in = int(getattr(model_config, "n_in", 700))
        x = (torch.rand(T, batch_size, in_channels, n_in, device=device) > 0.95).float()
    elif name == "spikformer_sequence":
        input_dim = int(getattr(model_config, "input_dim", 1))
        x = torch.rand(T, batch_size, input_dim, device=device)
    elif name == "spikformer":
        in_channels = int(getattr(model_config, "in_channels", 3))
        image_size = int(getattr(model_config, "image_size", 32))
        x = torch.rand(T, batch_size, in_channels, image_size, image_size, device=device)
    else:
        raise ValueError(f"Unsupported model name for synthetic input: {name}")
    y = torch.randint(0, num_classes, (batch_size,), device=device)
    return x, y


def _build_model(model_config, device: torch.device) -> torch.nn.Module:
    model_cls = get_model_cls(str(model_config.name).lower())
    model = model_cls(model_config)
    init_weights(model)
    return model.to(device)


def _run_step(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "eval":
        with torch.no_grad():
            logits = model(x)
        functional.reset_net(model)
        return logits

    logits = model(x)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    functional.reset_net(model)
    return loss.detach()


def _sync(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize()


def _print_npu_operator_summary(trace_dir: Path, row_limit: int) -> None:
    import csv
    from collections import defaultdict

    csv_files = sorted(trace_dir.glob("**/ASCEND_PROFILER_OUTPUT/operator_details.csv"))
    if not csv_files:
        print("No operator_details.csv found under torch_npu trace dir")
        return
    path = csv_files[-1]
    totals = defaultdict(lambda: [0.0, 0.0, 0])
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Name", "<unknown>")
            device_us = float(row.get("Device Total Duration(us)") or 0.0)
            host_us = float(row.get("Host Total Duration(us)") or 0.0)
            totals[name][0] += device_us
            totals[name][1] += host_us
            totals[name][2] += 1

    print(f"\nTop torch_npu operators from {path}:")
    print("rank,name,device_total_ms,host_total_ms,calls")
    for rank, (name, (device_us, host_us, calls)) in enumerate(
        sorted(totals.items(), key=lambda item: item[1][0], reverse=True)[:row_limit],
        start=1,
    ):
        print(f"{rank},{name},{device_us / 1000.0:.3f},{host_us / 1000.0:.3f},{calls}")


def main() -> None:
    args = parse_args()
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("torch.npu is not available; source Ascend set_env.sh and check NPU visibility")

    device = torch.device("npu:0")
    model_config = load_config(args.model_config)
    if args.T is not None:
        setattr(model_config, "T", int(args.T))

    model = _build_model(model_config, device)
    model.train(args.mode == "train")
    if args.mode == "eval":
        model.eval()
    if args.triton_lif:
        if args.mode == "train" and not args.triton_lif_train:
            print("--triton_lif in train mode will patch eval path only; pass --triton_lif_train to enable fused backward")
        patched = patch_lif_nodes_for_inference(
            model, block=args.lif_block, enable_training=(args.mode == "train" and args.triton_lif_train)
        )
        print(f"patched_lif_nodes={patched} triton_lif_train={args.mode == 'train' and args.triton_lif_train}")
    if args.streaming_attention:
        if args.mode != "eval":
            raise ValueError("--streaming_attention is inference-only; use --mode eval")
        patched_attn = patch_gated_attention_streaming_inference(model, use_triton=args.triton_attention)
        print(f"patched_streaming_attention={patched_attn}")

    x, y = _make_input(model_config, args.batch_size, device)

    for _ in range(args.warmup):
        _run_step(model, x, y, args.mode)
    _sync(device)

    print(
        f"model={model_config.name} attention={getattr(model_config, 'attention_type', 'n/a')} "
        f"T={getattr(model_config, 'T', None)} batch={args.batch_size} mode={args.mode}"
    )

    with torch_profile(
        activities=[TorchProfilerActivity.CPU],
        record_shapes=args.record_shapes,
        profile_memory=args.profile_memory,
    ) as prof:
        for _ in range(args.steps):
            _run_step(model, x, y, args.mode)
        _sync(device)

    print(prof.key_averages().table(sort_by=args.sort_by, row_limit=args.row_limit))

    if args.npu_trace:
        from torch_npu.profiler import ProfilerActivity, profile, tensorboard_trace_handler

        trace_dir = Path(args.trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)
        handler = tensorboard_trace_handler(str(trace_dir))
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
            on_trace_ready=handler,
            record_shapes=args.record_shapes,
            profile_memory=args.profile_memory,
        ) as npu_prof:
            for _ in range(args.steps):
                _run_step(model, x, y, args.mode)
                npu_prof.step()
            _sync(device)
        print(f"torch_npu trace exported to: {trace_dir.resolve()}")
        _print_npu_operator_summary(trace_dir, args.row_limit)
        print("Open it with MindStudio/msprof tooling, or TensorBoard profile plugin if available.")


if __name__ == "__main__":
    main()
