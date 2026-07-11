from __future__ import annotations

import argparse
import copy
import time

import torch
import torch_npu  # noqa: F401
from spikingjelly.activation_based import functional

from models.build_model import init_weights
from models.common.registry import get_model_cls
from models.common.triton_lif import patch_lif_nodes_for_inference
from models.spikformer.streaming_attention import patch_gated_attention_streaming_inference
from utils.load_config import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="End-to-end Spikformer eval benchmark with optional fused Triton LIF")
    p.add_argument("--model_config", default="configs/model_configs/pilot/spikformer_shd_c0_sdla.yaml")
    p.add_argument("--T", type=int, nargs="+", default=[100, 200])
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--lif_block", type=int, default=256)
    p.add_argument("--streaming_attention", action="store_true", help="patch C0/C1/C2/C3 eval attention to avoid S_seq allocation")
    p.add_argument("--triton_attention", action="store_true", help="use Triton-Ascend kernel inside the streaming attention patch")
    return p.parse_args()


def _make_input(model_config, batch_size: int, device: torch.device) -> torch.Tensor:
    name = str(model_config.name).lower()
    T = int(getattr(model_config, "T", 4))
    if name == "spikformer_audio":
        return (torch.rand(T, batch_size, int(getattr(model_config, "in_channels", 1)), int(getattr(model_config, "n_in", 700)), device=device) > 0.95).float()
    if name == "spikformer_sequence":
        return torch.rand(T, batch_size, int(getattr(model_config, "input_dim", 1)), device=device)
    if name == "spikformer":
        return torch.rand(
            T,
            batch_size,
            int(getattr(model_config, "in_channels", 3)),
            int(getattr(model_config, "image_size", 32)),
            int(getattr(model_config, "image_size", 32)),
            device=device,
        )
    raise ValueError(f"Unsupported model name: {name}")


def _build(model_config, device: torch.device):
    model_cls = get_model_cls(str(model_config.name).lower())
    model = model_cls(model_config)
    init_weights(model)
    model.eval().to(device)
    return model


def _sync() -> None:
    torch.npu.synchronize()


def _forward(model, x):
    with torch.no_grad():
        y = model(x)
    functional.reset_net(model)
    return y


def _measure(fn, repeats: int) -> tuple[float, float]:
    _sync()
    torch.npu.reset_peak_memory_stats()
    base_mem = torch.npu.memory_allocated()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    _sync()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / repeats
    peak_mb = max(0, torch.npu.max_memory_allocated() - base_mem) / (1024 ** 2)
    return elapsed_ms, peak_mb


def main() -> None:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("torch.npu is not available")
    args = parse_args()
    device = torch.device("npu:0")
    base_cfg = load_config(args.model_config)

    print("T,batch,attention,patched_lif,patched_attn,torch_ms,opt_ms,speedup,torch_peak_delta_mb,opt_peak_delta_mb,max_abs_diff")
    for T in args.T:
        cfg = copy.deepcopy(base_cfg)
        setattr(cfg, "T", int(T))
        x = _make_input(cfg, args.batch_size, device)

        torch.manual_seed(123)
        model_ref = _build(cfg, device)
        torch.manual_seed(123)
        model_tri = _build(cfg, device)
        patched = patch_lif_nodes_for_inference(model_tri, block=args.lif_block)
        patched_attn = patch_gated_attention_streaming_inference(model_tri, use_triton=args.triton_attention) if args.streaming_attention else 0

        with torch.no_grad():
            y_ref = _forward(model_ref, x)
            y_tri = _forward(model_tri, x)
            diff = (y_ref - y_tri).abs().max().item()
            for _ in range(args.warmup):
                _forward(model_ref, x)
                _forward(model_tri, x)

        torch_ms, torch_peak = _measure(lambda: _forward(model_ref, x), args.repeats)
        triton_ms, triton_peak = _measure(lambda: _forward(model_tri, x), args.repeats)
        speedup = torch_ms / triton_ms if triton_ms > 0 else float("inf")
        print(
            f"{T},{args.batch_size},{getattr(cfg, 'attention_type', 'n/a')},{patched},{patched_attn},"
            f"{torch_ms:.3f},{triton_ms:.3f},{speedup:.3f},{torch_peak:.1f},{triton_peak:.1f},{diff:.6g}"
        )


if __name__ == "__main__":
    main()
