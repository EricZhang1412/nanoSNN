from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional dependency
    triton = None
    tl = None


_TRITON_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


def _gate_accum_dtype(spike: torch.Tensor, psp: torch.Tensor) -> torch.dtype:
    dtype = torch.promote_types(spike.dtype, psp.dtype)
    if dtype in {torch.float16, torch.bfloat16}:
        return torch.float32
    return dtype


def _check_inputs(spike: torch.Tensor, psp: torch.Tensor) -> None:
    if spike.shape != psp.shape:
        raise ValueError(f"spike shape {tuple(spike.shape)} does not match psp shape {tuple(psp.shape)}")
    if spike.device != psp.device:
        raise ValueError(f"spike device {spike.device} does not match psp device {psp.device}")


def _can_use_triton(spike: torch.Tensor, psp: torch.Tensor) -> bool:
    return (
        triton is not None
        and spike.is_cuda
        and psp.is_cuda
        and spike.dtype == psp.dtype
        and spike.dtype in _TRITON_DTYPES
    )


def _resolve_backend(spike: torch.Tensor, psp: torch.Tensor, backend: str) -> str:
    backend = backend.lower()
    if backend not in {"auto", "torch", "triton"}:
        raise ValueError(f"Unsupported pspgate backend: {backend}")
    if backend == "auto":
        return "triton" if _can_use_triton(spike, psp) else "torch"
    if backend == "triton" and not _can_use_triton(spike, psp):
        raise RuntimeError("backend='triton' requires CUDA tensors with matching dtype and Triton installed")
    return backend


if triton is not None:
    @triton.jit
    def _pspgate_fwd_kernel(spike_ptr, psp_ptr, out_ptr, n_elements, scale, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        spike = tl.load(spike_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        psp = tl.load(psp_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        gate = 1.0 / (1.0 + tl.exp(-(scale * psp)))
        out = spike * gate
        tl.store(out_ptr + offsets, out, mask=mask)


def pspgate_torch(spike: torch.Tensor, psp: torch.Tensor, scale: float) -> torch.Tensor:
    _check_inputs(spike, psp)
    accum_dtype = _gate_accum_dtype(spike, psp)
    gate = torch.sigmoid(float(scale) * psp.to(accum_dtype))
    return (spike.to(accum_dtype) * gate).to(spike.dtype)


def _pspgate_triton_forward(spike: torch.Tensor, psp: torch.Tensor, scale: float) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is not installed")
    _check_inputs(spike, psp)
    if not _can_use_triton(spike, psp):
        raise RuntimeError("Triton pspgate requires CUDA tensors with matching dtype")
    if spike.numel() == 0:
        return spike.clone()

    spike_flat = spike.contiguous().reshape(-1)
    psp_flat = psp.contiguous().reshape(-1)
    out_flat = torch.empty_like(spike_flat)
    n_elements = out_flat.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _pspgate_fwd_kernel[grid](
        spike_flat,
        psp_flat,
        out_flat,
        n_elements,
        float(scale),
        BLOCK_SIZE=1024,
    )
    return out_flat.view_as(spike)


class _PSPGateTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, spike: torch.Tensor, psp: torch.Tensor, scale: float):
        _check_inputs(spike, psp)
        accum_dtype = _gate_accum_dtype(spike, psp)
        gate = torch.sigmoid(float(scale) * psp.to(accum_dtype))
        out = _pspgate_triton_forward(spike, psp, float(scale))

        ctx.scale = float(scale)
        ctx.spike_dtype = spike.dtype
        ctx.psp_dtype = psp.dtype
        ctx.save_for_backward(spike.to(accum_dtype), gate)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        spike, gate = ctx.saved_tensors
        grad = grad_out.to(gate.dtype)
        grad_spike = grad * gate
        grad_psp = grad * spike * ctx.scale * gate * (1.0 - gate)
        return grad_spike.to(ctx.spike_dtype), grad_psp.to(ctx.psp_dtype), None


def pspgate_apply(
    spike: torch.Tensor,
    psp: torch.Tensor,
    scale: float,
    backend: str = "auto",
) -> torch.Tensor:
    _check_inputs(spike, psp)
    resolved_backend = _resolve_backend(spike, psp, backend)
    if resolved_backend == "torch":
        return pspgate_torch(spike, psp, scale=float(scale))
    return _PSPGateTritonFunction.apply(spike, psp, float(scale))
