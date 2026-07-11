from __future__ import annotations

import argparse
import time

import torch
import torch_npu  # noqa: F401
from spikingjelly.activation_based import neuron, surrogate

from models.common.triton_lif import patch_lif_nodes_for_inference


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark SpikingJelly eval LIF vs fused Triton-Ascend LIF")
    p.add_argument("--T", type=int, nargs="+", default=[100, 200])
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--tokens", type=int, default=70)
    p.add_argument("--channels", type=int, default=256)
    p.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--lif_block", type=int, default=256)
    p.add_argument("--mode", choices=["eval", "train"], default="eval")
    return p.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


def _sync() -> None:
    torch.npu.synchronize()


def _measure(fn, repeats: int) -> tuple[float, float]:
    _sync()
    torch.npu.reset_peak_memory_stats()
    base_mem = torch.npu.memory_allocated()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    _sync()
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / repeats
    peak_mb = max(0, torch.npu.max_memory_allocated() - base_mem) / (1024 ** 2)
    return elapsed_ms, peak_mb


def _make_node(device: torch.device, mode: str):
    node = neuron.LIFNode(
        tau=2.0,
        v_threshold=1.0,
        v_reset=0.0,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
        step_mode="m",
        backend="torch",
    ).to(device)
    node.train(mode == "train")
    return node


def main() -> None:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("torch.npu is not available")
    args = parse_args()
    device = torch.device("npu:0")
    dtype = _dtype(args.dtype)

    print("mode,T,B,N,C,dtype,torch_ms,triton_ms,speedup,torch_peak_delta_mb,triton_peak_delta_mb,max_abs_diff,grad_max_abs_diff")
    for T in args.T:
        ref_node = _make_node(device, args.mode)
        tri_node = _make_node(device, args.mode)
        patched = patch_lif_nodes_for_inference(
            tri_node, block=args.lif_block, enable_training=(args.mode == "train")
        )
        if patched != 1:
            raise RuntimeError(f"expected to patch 1 LIF node, patched {patched}")

        if args.mode == "eval":
            x = torch.randn(T, args.batch_size, args.tokens, args.channels, device=device, dtype=dtype)
            with torch.no_grad():
                y_ref = ref_node(x); ref_node.reset()
                y_tri = tri_node(x); tri_node.reset()
                _sync()
                diff = (y_ref - y_tri).abs().max().item()
                grad_diff = 0.0
                for _ in range(args.warmup):
                    ref_node(x); ref_node.reset()
                    tri_node(x); tri_node.reset()
                _sync()
                torch_ms, torch_peak = _measure(lambda: (ref_node(x), ref_node.reset()), args.repeats)
                triton_ms, triton_peak = _measure(lambda: (tri_node(x), tri_node.reset()), args.repeats)
        else:
            x_ref = torch.randn(T, args.batch_size, args.tokens, args.channels, device=device, dtype=dtype, requires_grad=True)
            x_tri = x_ref.detach().clone().requires_grad_(True)
            weight = torch.randn_like(x_ref)

            def ref_step():
                x_ref.grad = None
                y = ref_node(x_ref)
                loss = (y * weight).sum()
                loss.backward()
                ref_node.reset()
                return y

            def tri_step():
                x_tri.grad = None
                y = tri_node(x_tri)
                loss = (y * weight).sum()
                loss.backward()
                tri_node.reset()
                return y

            y_ref = ref_step(); y_tri = tri_step(); _sync()
            diff = (y_ref.detach() - y_tri.detach()).abs().max().item()
            grad_diff = (x_ref.grad.detach() - x_tri.grad.detach()).abs().max().item()
            for _ in range(args.warmup):
                ref_step(); tri_step()
            _sync()
            torch_ms, torch_peak = _measure(ref_step, args.repeats)
            triton_ms, triton_peak = _measure(tri_step, args.repeats)

        speedup = torch_ms / triton_ms if triton_ms > 0 else float("inf")
        print(f"{args.mode},{T},{args.batch_size},{args.tokens},{args.channels},{args.dtype},{torch_ms:.3f},{triton_ms:.3f},{speedup:.3f},{torch_peak:.1f},{triton_peak:.1f},{diff:.6g},{grad_diff:.6g}")


if __name__ == "__main__":
    main()
