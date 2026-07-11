from __future__ import annotations

import argparse
import time

import torch
import torch_npu  # noqa: F401

from models.spikformer.triton_attention import streaming_linear_attention_fwd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark original materialized S_seq vs Triton streaming linear attention")
    p.add_argument("--mode", choices=["c0", "diag", "mga"], default="diag")
    p.add_argument("--T", type=int, nargs="+", default=[100, 200])
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--tokens", type=int, default=70)
    p.add_argument("--head_dim", type=int, default=64)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    return p.parse_args()


def _sync() -> None:
    torch.npu.synchronize()


def _measure(fn, repeats: int) -> tuple[float, float]:
    _sync()
    torch.npu.reset_peak_memory_stats()
    base = torch.npu.memory_allocated()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    _sync()
    ms = (time.perf_counter() - t0) * 1000.0 / repeats
    peak = max(0, torch.npu.max_memory_allocated() - base) / (1024 ** 2)
    return ms, peak


def _ref(q, k, v, mode, gate_d, gate_scalar, write_scale, scale, shift_scale):
    T, B, H, N, D = q.shape
    outs = []
    S = q.new_zeros(B, H, D, D)
    for t in range(T):
        KV = k[t].transpose(-2, -1) @ v[t]
        if mode == "c0":
            S_cur = KV
        elif mode == "diag":
            S = gate_d[t].unsqueeze(-1) * S + KV
            S_cur = S
        else:
            S = (1.0 - gate_d[t] * shift_scale).unsqueeze(-1) * S + write_scale.view(1, H, 1, 1) * gate_scalar[t].unsqueeze(-1).unsqueeze(-1) * KV
            S_cur = S
        outs.append((q[t] @ S_cur) * scale)
    return torch.stack(outs, dim=0)


def main() -> None:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("torch.npu is not available")
    args = parse_args()
    device = torch.device("npu:0")
    print("mode,T,B,H,N,D,torch_ms,triton_ms,speedup,torch_peak_delta_mb,triton_peak_delta_mb,max_abs_diff")
    for T in args.T:
        shape = (T, args.batch_size, args.heads, args.tokens, args.head_dim)
        q = (torch.rand(shape, device=device) > 0.75).float()
        k = (torch.rand(shape, device=device) > 0.75).float()
        v = (torch.rand(shape, device=device) > 0.75).float()
        gate_d = torch.rand(T, args.batch_size, args.heads, args.head_dim, device=device)
        gate_scalar = (torch.rand(T, args.batch_size, args.heads, device=device) > 0.5).float()
        write_scale = torch.full((args.heads,), 0.125, device=device)
        scale = args.head_dim ** -0.5
        shift_scale = 0.125

        y_ref = _ref(q, k, v, args.mode, gate_d, gate_scalar, write_scale, scale, shift_scale)
        y_tri = streaming_linear_attention_fwd(
            q,
            k,
            v,
            mode=args.mode,
            gate_d=None if args.mode == "c0" else gate_d,
            gate_scalar=gate_scalar if args.mode == "mga" else None,
            write_scale=write_scale if args.mode == "mga" else None,
            scale=scale,
            shift_scale=shift_scale,
        )
        _sync()
        diff = (y_ref - y_tri).abs().max().item()

        for _ in range(args.warmup):
            _ref(q, k, v, args.mode, gate_d, gate_scalar, write_scale, scale, shift_scale)
            streaming_linear_attention_fwd(
                q,
                k,
                v,
                mode=args.mode,
                gate_d=None if args.mode == "c0" else gate_d,
                gate_scalar=gate_scalar if args.mode == "mga" else None,
                write_scale=write_scale if args.mode == "mga" else None,
                scale=scale,
                shift_scale=shift_scale,
            )
        torch_ms, torch_peak = _measure(lambda: _ref(q, k, v, args.mode, gate_d, gate_scalar, write_scale, scale, shift_scale), args.repeats)
        tri_ms, tri_peak = _measure(lambda: streaming_linear_attention_fwd(
            q,
            k,
            v,
            mode=args.mode,
            gate_d=None if args.mode == "c0" else gate_d,
            gate_scalar=gate_scalar if args.mode == "mga" else None,
            write_scale=write_scale if args.mode == "mga" else None,
            scale=scale,
            shift_scale=shift_scale,
        ), args.repeats)
        speedup = torch_ms / tri_ms if tri_ms > 0 else float("inf")
        print(f"{args.mode},{T},{args.batch_size},{args.heads},{args.tokens},{args.head_dim},{torch_ms:.3f},{tri_ms:.3f},{speedup:.3f},{torch_peak:.1f},{tri_peak:.1f},{diff:.6g}")


if __name__ == "__main__":
    main()
