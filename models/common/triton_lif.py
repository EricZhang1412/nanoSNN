from __future__ import annotations

import types
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - import guard for non-Triton environments
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _lif_hard_reset_decay_fwd_kernel(
        x_ptr,
        v0_ptr,
        y_ptr,
        v_last_ptr,
        m_size: tl.constexpr,
        tau: tl.constexpr,
        v_threshold: tl.constexpr,
        v_reset: tl.constexpr,
        T: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < m_size

        v = tl.load(v0_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        inv_tau = 1.0 / tau
        for t in range(T):
            x_t = tl.load(x_ptr + t * m_size + offs, mask=mask, other=0.0).to(tl.float32)
            v = v + (x_t - (v - v_reset)) * inv_tau
            spike = v >= v_threshold
            spike_f = spike.to(tl.float32)
            tl.store(y_ptr + t * m_size + offs, spike_f, mask=mask)
            v = v_reset * spike_f + (1.0 - spike_f) * v

        tl.store(v_last_ptr + offs, v, mask=mask)


    @triton.jit
    def _lif_hard_reset_decay_train_fwd_kernel(
        x_ptr,
        v0_ptr,
        spike_ptr,
        v_pre_ptr,
        v_last_ptr,
        m_size: tl.constexpr,
        tau: tl.constexpr,
        v_threshold: tl.constexpr,
        v_reset: tl.constexpr,
        T: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < m_size

        v = tl.load(v0_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        inv_tau = 1.0 / tau
        for t in range(T):
            ptrs = t * m_size + offs
            x_t = tl.load(x_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
            v = v + (x_t - (v - v_reset)) * inv_tau
            tl.store(v_pre_ptr + ptrs, v, mask=mask)
            spike = v >= v_threshold
            spike_f = spike.to(tl.float32)
            tl.store(spike_ptr + ptrs, spike_f, mask=mask)
            v = v_reset * spike_f + (1.0 - spike_f) * v

        tl.store(v_last_ptr + offs, v, mask=mask)


    @triton.jit
    def _lif_hard_reset_decay_train_bwd_kernel(
        grad_spike_ptr,
        spike_ptr,
        v_pre_ptr,
        grad_x_ptr,
        m_size: tl.constexpr,
        tau: tl.constexpr,
        v_threshold: tl.constexpr,
        surrogate_alpha: tl.constexpr,
        T: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < m_size

        grad_v_post = tl.zeros((BLOCK,), tl.float32)
        inv_tau = 1.0 / tau
        alpha_v = 1.0 - inv_tau
        half_pi_alpha = 1.5707963267948966 * surrogate_alpha

        for t_rev in range(T, 0, -1):
            t = t_rev - 1
            ptrs = t * m_size + offs
            grad_s = tl.load(grad_spike_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
            spike = tl.load(spike_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
            v_pre = tl.load(v_pre_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
            z = v_pre - v_threshold
            tmp = half_pi_alpha * z
            surrogate_grad = surrogate_alpha * 0.5 / (1.0 + tmp * tmp)
            grad_v_pre = grad_s * surrogate_grad + grad_v_post * (1.0 - spike)
            tl.store(grad_x_ptr + ptrs, grad_v_pre * inv_tau, mask=mask)
            grad_v_post = grad_v_pre * alpha_v


def _torch_lif_hard_reset_decay_fwd(
    x_seq: torch.Tensor,
    v0: torch.Tensor | None = None,
    *,
    tau: float = 2.0,
    v_threshold: float = 1.0,
    v_reset: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference eval LIF: hard reset + decay_input=True, matching SpikingJelly eval path."""
    if v0 is None:
        v = x_seq.new_full(x_seq.shape[1:], float(v_reset))
    else:
        v = v0.to(device=x_seq.device, dtype=x_seq.dtype).reshape(x_seq.shape[1:]).clone()
    y = torch.empty_like(x_seq)
    inv_tau = 1.0 / float(tau)
    for t in range(x_seq.shape[0]):
        v = v + (x_seq[t] - (v - float(v_reset))) * inv_tau
        spike = (v >= float(v_threshold)).to(x_seq.dtype)
        y[t] = spike
        v = float(v_reset) * spike + (1.0 - spike) * v
    return y, v


def lif_hard_reset_decay_fwd(
    x_seq: torch.Tensor,
    v0: torch.Tensor | None = None,
    *,
    tau: float = 2.0,
    v_threshold: float = 1.0,
    v_reset: float = 0.0,
    block: int = 256,
    force_torch: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inference-only fused multi-step LIF forward.

    This matches SpikingJelly's eval equation for LIFNode with
    `decay_input=True` and hard reset (`v_reset` not None):

        v_t = v_{t-1} + (x_t - (v_{t-1} - v_reset)) / tau
        s_t = 1[v_t >= v_threshold]
        v_t = v_reset * s_t + (1 - s_t) * v_t

    On Ascend NPU this dispatches a Triton-Ascend kernel that scans `T` inside
    one device kernel, avoiding T separate launches and repeated global-memory
    round trips for membrane state. On CPU / unsupported devices it falls back to
    the reference PyTorch scan, so import-time smoke tests remain portable.
    """
    if x_seq.ndim < 2:
        raise ValueError(f"x_seq must be [T, ...], got {tuple(x_seq.shape)}")
    if x_seq.requires_grad:
        raise RuntimeError("lif_hard_reset_decay_fwd is inference-only; use it under torch.no_grad().")

    if force_torch or triton is None or x_seq.device.type != "npu":
        return _torch_lif_hard_reset_decay_fwd(
            x_seq, v0, tau=tau, v_threshold=v_threshold, v_reset=v_reset
        )

    x_contig = x_seq.contiguous()
    T = int(x_contig.shape[0])
    flat = x_contig.reshape(T, -1)
    m_size = int(flat.shape[1])

    if v0 is None:
        v0_flat = x_contig.new_full((m_size,), float(v_reset))
    else:
        v0_flat = v0.to(device=x_contig.device, dtype=x_contig.dtype).reshape(-1).contiguous()
        if v0_flat.numel() != m_size:
            raise ValueError(f"v0 has {v0_flat.numel()} elements but expected {m_size}")

    y_flat = torch.empty_like(flat)
    v_last_flat = torch.empty_like(v0_flat)
    grid = (triton.cdiv(m_size, int(block)),)
    _lif_hard_reset_decay_fwd_kernel[grid](
        flat,
        v0_flat,
        y_flat,
        v_last_flat,
        m_size,
        tau=float(tau),
        v_threshold=float(v_threshold),
        v_reset=float(v_reset),
        T=T,
        BLOCK=int(block),
    )
    return y_flat.reshape_as(x_contig), v_last_flat.reshape(x_contig.shape[1:])


class _TritonLIFHardResetDecayTrainFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x_seq: torch.Tensor,
        v0: torch.Tensor,
        tau: float,
        v_threshold: float,
        v_reset: float,
        surrogate_alpha: float,
        block: int,
    ):
        if triton is None or x_seq.device.type != "npu":
            raise RuntimeError("training fused LIF requires Triton-Ascend on NPU")

        x_contig = x_seq.contiguous()
        T = int(x_contig.shape[0])
        flat = x_contig.reshape(T, -1)
        m_size = int(flat.shape[1])
        v0_flat = v0.to(device=x_contig.device, dtype=x_contig.dtype).reshape(-1).contiguous()
        spike_flat = torch.empty_like(flat)
        v_pre_flat = torch.empty_like(flat)
        v_last_flat = torch.empty_like(v0_flat)
        grid = (triton.cdiv(m_size, int(block)),)
        _lif_hard_reset_decay_train_fwd_kernel[grid](
            flat,
            v0_flat,
            spike_flat,
            v_pre_flat,
            v_last_flat,
            m_size,
            tau=float(tau),
            v_threshold=float(v_threshold),
            v_reset=float(v_reset),
            T=T,
            BLOCK=int(block),
        )
        ctx.save_for_backward(spike_flat, v_pre_flat)
        ctx.shape = tuple(x_contig.shape)
        ctx.m_size = m_size
        ctx.tau = float(tau)
        ctx.v_threshold = float(v_threshold)
        ctx.surrogate_alpha = float(surrogate_alpha)
        ctx.block = int(block)
        return spike_flat.reshape_as(x_contig), v_last_flat.reshape(x_contig.shape[1:])

    @staticmethod
    def backward(ctx, grad_spike: torch.Tensor, grad_v_last: torch.Tensor | None = None):
        spike_flat, v_pre_flat = ctx.saved_tensors
        grad_flat = grad_spike.contiguous().reshape(int(ctx.shape[0]), ctx.m_size)
        grad_x_flat = torch.empty_like(grad_flat)
        grid = (triton.cdiv(ctx.m_size, ctx.block),)
        _lif_hard_reset_decay_train_bwd_kernel[grid](
            grad_flat,
            spike_flat,
            v_pre_flat,
            grad_x_flat,
            ctx.m_size,
            tau=ctx.tau,
            v_threshold=ctx.v_threshold,
            surrogate_alpha=ctx.surrogate_alpha,
            T=int(ctx.shape[0]),
            BLOCK=ctx.block,
        )
        return grad_x_flat.reshape(ctx.shape), None, None, None, None, None, None


def lif_hard_reset_decay_train(
    x_seq: torch.Tensor,
    v0: torch.Tensor,
    *,
    tau: float = 2.0,
    v_threshold: float = 1.0,
    v_reset: float = 0.0,
    surrogate_alpha: float = 2.0,
    block: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Training fused LIF with ATan surrogate backward for detach_reset=True.

    This covers the nanoSNN default LIF configuration used by the Spikformer
    configs: hard reset, decay_input=True, detach_reset=True, ATan surrogate.
    It returns `(spike_seq, v_last)`.
    """
    return _TritonLIFHardResetDecayTrainFn.apply(
        x_seq, v0, float(tau), float(v_threshold), float(v_reset), float(surrogate_alpha), int(block)
    )


def _is_supported_lif_node(module: Any) -> bool:
    return (
        module.__class__.__name__ == "LIFNode"
        and getattr(module, "step_mode", None) == "m"
        and bool(getattr(module, "decay_input", True))
        and getattr(module, "v_reset", 0.0) is not None
        and not bool(getattr(module, "store_v_seq", False))
    )


def _triton_multi_step_forward(self: Any, x_seq: torch.Tensor) -> torch.Tensor:
    if not _is_supported_lif_node(self) or x_seq.device.type != "npu":
        return self._nanosnn_orig_multi_step_forward(x_seq)

    self.v_float_to_tensor(x_seq[0])
    v0 = self.v
    block = int(getattr(self, "_nanosnn_triton_lif_block", 256))

    if self.training and torch.is_grad_enabled() and x_seq.requires_grad:
        if not getattr(self, "_nanosnn_triton_lif_train", False) or not bool(getattr(self, "detach_reset", True)):
            return self._nanosnn_orig_multi_step_forward(x_seq)
        surrogate_alpha = float(getattr(getattr(self, "surrogate_function", None), "alpha", 2.0))
        spike_seq, v_last = lif_hard_reset_decay_train(
            x_seq,
            v0,
            tau=float(self.tau),
            v_threshold=float(self.v_threshold),
            v_reset=float(self.v_reset),
            surrogate_alpha=surrogate_alpha,
            block=block,
        )
    else:
        spike_seq, v_last = lif_hard_reset_decay_fwd(
            x_seq,
            v0,
            tau=float(self.tau),
            v_threshold=float(self.v_threshold),
            v_reset=float(self.v_reset),
            block=block,
        )
    self.v = v_last
    return spike_seq


def patch_lif_nodes_for_inference(
    module: torch.nn.Module, *, block: int = 256, enable_training: bool = False
) -> int:
    """Monkey-patch SpikingJelly LIFNode multi-step with fused Triton LIF.

    Eval/inference is always enabled by this patch. Training uses the fused
    ATan-backward path only when `enable_training=True`; otherwise it falls
    back to the original SpikingJelly implementation. The model topology and
    state_dict are unchanged. Unsupported LIF variants silently fall back.
    """
    patched = 0
    for submodule in module.modules():
        if not _is_supported_lif_node(submodule):
            continue
        if getattr(submodule, "_nanosnn_triton_lif_patched", False):
            continue
        submodule._nanosnn_orig_multi_step_forward = submodule.multi_step_forward
        submodule._nanosnn_triton_lif_block = int(block)
        submodule._nanosnn_triton_lif_train = bool(enable_training)
        submodule.multi_step_forward = types.MethodType(_triton_multi_step_forward, submodule)
        submodule._nanosnn_triton_lif_patched = True
        patched += 1
    return patched


def unpatch_lif_nodes(module: torch.nn.Module) -> int:
    restored = 0
    for submodule in module.modules():
        if not getattr(submodule, "_nanosnn_triton_lif_patched", False):
            continue
        submodule.multi_step_forward = submodule._nanosnn_orig_multi_step_forward
        delattr(submodule, "_nanosnn_orig_multi_step_forward")
        delattr(submodule, "_nanosnn_triton_lif_block")
        delattr(submodule, "_nanosnn_triton_lif_train")
        delattr(submodule, "_nanosnn_triton_lif_patched")
        restored += 1
    return restored
