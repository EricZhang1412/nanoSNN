from __future__ import annotations

import argparse
import copy
import time

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
from spikingjelly.activation_based import functional

from models.build_model import init_weights
from models.common.registry import get_model_cls
from models.common.triton_lif import patch_lif_nodes_for_inference
from utils.load_config import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="End-to-end Spikformer training-step benchmark with fused Triton LIF FWD/BWD")
    p.add_argument("--model_config", default="configs/model_configs/pilot/spikformer_shd_c0_sdla.yaml")
    p.add_argument("--T", type=int, nargs="+", default=[100, 200])
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--lif_block", type=int, default=256)
    return p.parse_args()


def _make_input(model_config, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    name = str(model_config.name).lower()
    T = int(getattr(model_config, "T", 4))
    if name == "spikformer_audio":
        x = (torch.rand(T, batch_size, int(getattr(model_config, "in_channels", 1)), int(getattr(model_config, "n_in", 700)), device=device) > 0.95).float()
    elif name == "spikformer_sequence":
        x = torch.rand(T, batch_size, int(getattr(model_config, "input_dim", 1)), device=device)
    elif name == "spikformer":
        image_size = int(getattr(model_config, "image_size", 32))
        x = torch.rand(T, batch_size, int(getattr(model_config, "in_channels", 3)), image_size, image_size, device=device)
    else:
        raise ValueError(f"Unsupported model name: {name}")
    y = torch.randint(0, int(getattr(model_config, "num_classes", 10)), (batch_size,), device=device)
    return x, y


def _build(model_config, device: torch.device):
    model_cls = get_model_cls(str(model_config.name).lower())
    model = model_cls(model_config)
    init_weights(model)
    model.train().to(device)
    return model


def _sync() -> None:
    torch.npu.synchronize()


def _zero_grad(model) -> None:
    for p in model.parameters():
        p.grad = None


def _step(model, x, y):
    _zero_grad(model)
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    functional.reset_net(model)
    return loss.detach(), logits.detach()


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


def _max_grad_diff(a, b) -> float:
    max_diff = 0.0
    for p_a, p_b in zip(a.parameters(), b.parameters()):
        if p_a.grad is None and p_b.grad is None:
            continue
        if p_a.grad is None or p_b.grad is None:
            return float("inf")
        diff = (p_a.grad.detach() - p_b.grad.detach()).abs().max().item()
        max_diff = max(max_diff, diff)
    return max_diff


def main() -> None:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("torch.npu is not available")
    args = parse_args()
    device = torch.device("npu:0")
    base_cfg = load_config(args.model_config)

    print("T,batch,attention,patched_lif,torch_ms,triton_train_lif_ms,speedup,torch_peak_delta_mb,triton_peak_delta_mb,loss_abs_diff,logit_max_abs_diff,grad_max_abs_diff")
    for T in args.T:
        cfg = copy.deepcopy(base_cfg)
        setattr(cfg, "T", int(T))
        x, y = _make_input(cfg, args.batch_size, device)

        torch.manual_seed(123)
        model_ref = _build(cfg, device)
        torch.manual_seed(123)
        model_tri = _build(cfg, device)
        patched = patch_lif_nodes_for_inference(model_tri, block=args.lif_block, enable_training=True)

        loss_ref, logits_ref = _step(model_ref, x, y)
        loss_tri, logits_tri = _step(model_tri, x, y)
        _sync()
        loss_diff = abs(loss_ref.item() - loss_tri.item())
        logit_diff = (logits_ref - logits_tri).abs().max().item()
        grad_diff = _max_grad_diff(model_ref, model_tri)

        for _ in range(args.warmup):
            _step(model_ref, x, y)
            _step(model_tri, x, y)

        torch_ms, torch_peak = _measure(lambda: _step(model_ref, x, y), args.repeats)
        triton_ms, triton_peak = _measure(lambda: _step(model_tri, x, y), args.repeats)
        speedup = torch_ms / triton_ms if triton_ms > 0 else float("inf")
        print(
            f"{T},{args.batch_size},{getattr(cfg, 'attention_type', 'n/a')},{patched},"
            f"{torch_ms:.3f},{triton_ms:.3f},{speedup:.3f},{torch_peak:.1f},{triton_peak:.1f},"
            f"{loss_diff:.6g},{logit_diff:.6g},{grad_diff:.6g}"
        )


if __name__ == "__main__":
    main()
