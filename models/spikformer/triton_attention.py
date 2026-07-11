from __future__ import annotations

from typing import Literal

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _streaming_linear_attention_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        gate_d_ptr,
        gate_scalar_ptr,
        write_scale_ptr,
        out_ptr,
        BH: tl.constexpr,
        H: tl.constexpr,
        N: tl.constexpr,
        D: tl.constexpr,
        T: tl.constexpr,
        scale: tl.constexpr,
        shift_scale: tl.constexpr,
        MODE: tl.constexpr,
        BLOCK_N_OUT: tl.constexpr,
        BLOCK_N_KV: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_bh = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_no = pid_n * BLOCK_N_OUT + tl.arange(0, BLOCK_N_OUT)
        offs_nk = tl.arange(0, BLOCK_N_KV)
        offs_d = tl.arange(0, BLOCK_D)

        mask_no = offs_no < N
        mask_nk = offs_nk < N
        mask_d = offs_d < D
        S = tl.zeros((BLOCK_D, BLOCK_D), tl.float32)
        h = pid_bh % H

        for t in range(T):
            base = (t * BH + pid_bh) * N * D
            k_tile = tl.load(
                k_ptr + base + offs_nk[:, None] * D + offs_d[None, :],
                mask=mask_nk[:, None] & mask_d[None, :],
                other=0.0,
            )
            v_tile = tl.load(
                v_ptr + base + offs_nk[:, None] * D + offs_d[None, :],
                mask=mask_nk[:, None] & mask_d[None, :],
                other=0.0,
            )
            KV = tl.dot(tl.trans(k_tile), v_tile, input_precision="hf32")

            if MODE == 0:  # C0 memoryless SDLA
                S_cur = KV
            elif MODE == 1:  # C1/C2 diagonal recurrent gate over D rows
                gate_d = tl.load(
                    gate_d_ptr + (t * BH + pid_bh) * D + offs_d,
                    mask=mask_d,
                    other=0.0,
                ).to(tl.float32)
                S = gate_d[:, None] * S + KV
                S_cur = S
            else:  # C3 MGA: decay gate + scalar write gate
                s_gamma = tl.load(
                    gate_d_ptr + (t * BH + pid_bh) * D + offs_d,
                    mask=mask_d,
                    other=0.0,
                ).to(tl.float32)
                s_beta = tl.load(gate_scalar_ptr + t * BH + pid_bh).to(tl.float32)
                write_scale = tl.load(write_scale_ptr + h).to(tl.float32)
                alpha_eff = 1.0 - s_gamma * shift_scale
                S = alpha_eff[:, None] * S + write_scale * s_beta * KV
                S_cur = S

            q_tile = tl.load(
                q_ptr + base + offs_no[:, None] * D + offs_d[None, :],
                mask=mask_no[:, None] & mask_d[None, :],
                other=0.0,
            )
            O = tl.dot(q_tile, S_cur, input_precision="hf32") * scale
            tl.store(
                out_ptr + base + offs_no[:, None] * D + offs_d[None, :],
                O,
                mask=mask_no[:, None] & mask_d[None, :],
            )


def _next_power_of_2(x: int) -> int:
    return 1 << (int(x) - 1).bit_length()


def _empty_gate(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.empty((1,), device=device, dtype=dtype)


def streaming_linear_attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    mode: Literal["c0", "diag", "mga"],
    gate_d: torch.Tensor | None = None,
    gate_scalar: torch.Tensor | None = None,
    write_scale: torch.Tensor | None = None,
    scale: float = 1.0,
    shift_scale: float = 0.125,
    block_n_out: int = 32,
    block_n_kv: int | None = None,
) -> torch.Tensor:
    """Triton-Ascend streaming linear attention forward.

    Inputs are contiguous `[T, B, H, N, D]` tensors. The recurrence state `S`
    lives inside the Triton program and is never materialized as `[T,B,H,D,D]`.
    This is forward-only; use the PyTorch path for autograd/training.
    """
    if triton is None or q.device.type != "npu":
        raise RuntimeError("streaming_linear_attention_fwd requires Triton-Ascend on NPU")
    if q.ndim != 5 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError(f"q/k/v must share shape [T,B,H,N,D], got {q.shape}, {k.shape}, {v.shape}")

    T, B, H, N, D = map(int, q.shape)
    if D > 64:
        raise ValueError(f"Triton streaming attention currently supports D <= 64, got D={D}")
    if block_n_kv is None:
        block_n_kv = _next_power_of_2(N)
    block_d = _next_power_of_2(D)
    if block_d < 16:
        block_d = 16

    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    out = torch.empty_like(q_c)
    dummy = _empty_gate(q_c.device, q_c.dtype)

    if mode == "c0":
        mode_id = 0
        gate_d_c = dummy
        gate_scalar_c = dummy
        write_scale_c = dummy
    elif mode == "diag":
        if gate_d is None:
            raise ValueError("mode='diag' requires gate_d [T,B,H,D]")
        mode_id = 1
        gate_d_c = gate_d.contiguous()
        gate_scalar_c = dummy
        write_scale_c = dummy
    elif mode == "mga":
        if gate_d is None or gate_scalar is None or write_scale is None:
            raise ValueError("mode='mga' requires gate_d [T,B,H,D], gate_scalar [T,B,H], write_scale [H]")
        mode_id = 2
        gate_d_c = gate_d.contiguous()
        gate_scalar_c = gate_scalar.contiguous()
        write_scale_c = write_scale.contiguous()
    else:
        raise ValueError(f"unknown mode: {mode}")

    BH = B * H
    grid = (BH, triton.cdiv(N, int(block_n_out)))
    _streaming_linear_attention_kernel[grid](
        q_c,
        k_c,
        v_c,
        gate_d_c,
        gate_scalar_c,
        write_scale_c,
        out,
        BH=BH,
        H=H,
        N=N,
        D=D,
        T=T,
        scale=float(scale),
        shift_scale=float(shift_scale),
        MODE=mode_id,
        BLOCK_N_OUT=int(block_n_out),
        BLOCK_N_KV=int(block_n_kv),
        BLOCK_D=int(block_d),
    )
    return out
