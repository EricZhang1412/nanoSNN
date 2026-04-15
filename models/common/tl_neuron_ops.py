from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn as nn
from spikingjelly.activation_based import base

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


_TRITON_DTYPES = {torch.float16, torch.bfloat16, torch.float32}
_PI = 3.141592653589793
_NEURON_KIND = {"if": 0, "lif": 1, "plif": 2}
_SURROGATE_KIND = {"atan": 0, "sigmoid": 1}


class DualOutput(NamedTuple):
    spike: torch.Tensor
    v_seq: torch.Tensor


def _default_alpha(surrogate_type: str) -> float:
    if surrogate_type == "sigmoid":
        return 4.0
    return 2.0


def _can_use_triton(x: torch.Tensor) -> bool:
    return triton is not None and x.is_cuda and x.dtype in _TRITON_DTYPES


def _flatten_time(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    shape = tuple(x.shape)
    x_flat = x.reshape(shape[0], -1).contiguous()
    return x_flat, shape


def _restore_time(x: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    return x.reshape(shape)


def _state_base(v_reset: float | None) -> float:
    return 0.0 if v_reset is None else float(v_reset)


def _init_state(x0: torch.Tensor, state: torch.Tensor | float, v_reset: float | None) -> torch.Tensor:
    flat = x0.reshape(-1)
    if isinstance(state, torch.Tensor):
        state = state.reshape(-1).to(device=flat.device, dtype=flat.dtype)
        if state.numel() == flat.numel():
            return state.detach().contiguous()
    return torch.full_like(flat, _state_base(v_reset))


def _surrogate_grad_torch(u: torch.Tensor, surrogate_kind: int, alpha: float) -> torch.Tensor:
    alpha = float(alpha)
    if surrogate_kind == _SURROGATE_KIND["atan"]:
        z = 0.5 * _PI * alpha * u
        return alpha / (2.0 * (1.0 + z * z))
    sig = torch.sigmoid(alpha * u)
    return alpha * sig * (1.0 - sig)


def _charge_torch(
    x_t: torch.Tensor,
    v_prev: torch.Tensor,
    param: torch.Tensor,
    neuron_kind: int,
    reset_mode: int,
    v_reset_value: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if neuron_kind == _NEURON_KIND["if"]:
        h = v_prev + x_t
        dh_dx = torch.ones_like(x_t)
        dh_dv = torch.ones_like(x_t)
        dh_dp = torch.zeros_like(x_t)
        return h, dh_dx, dh_dv, dh_dp

    if neuron_kind == _NEURON_KIND["lif"]:
        tau = param.reshape(()).to(device=x_t.device, dtype=x_t.dtype)
        decay = tau.reciprocal()
        base = v_prev if reset_mode == 0 or v_reset_value == 0.0 else (v_prev - v_reset_value)
        delta = x_t - base
        h = v_prev + delta * decay
        dh_dx = torch.full_like(x_t, decay)
        dh_dv = torch.full_like(x_t, 1.0 - decay)
        dh_dp = -delta / (tau * tau)
        return h, dh_dx, dh_dv, dh_dp

    w = param.reshape(()).to(device=x_t.device, dtype=x_t.dtype)
    decay = torch.sigmoid(w)
    base = v_prev if reset_mode == 0 or v_reset_value == 0.0 else (v_prev - v_reset_value)
    delta = x_t - base
    h = v_prev + delta * decay
    dh_dx = torch.full_like(x_t, decay)
    dh_dv = torch.full_like(x_t, 1.0 - decay)
    dh_dp = delta * decay * (1.0 - decay)
    return h, dh_dx, dh_dv, dh_dp


def _forward_torch_flat(
    x_flat: torch.Tensor,
    v_init: torch.Tensor,
    param: torch.Tensor,
    v_threshold: float,
    v_reset: float | None,
    neuron_kind: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    reset_mode = 0 if v_reset is None else 1
    v_reset_value = 0.0 if v_reset is None else float(v_reset)
    T, N = x_flat.shape
    spike = torch.empty_like(x_flat)
    v_seq = torch.empty_like(x_flat)
    v_prev = v_init

    for t in range(T):
        h, _, _, _ = _charge_torch(x_flat[t], v_prev, param, neuron_kind, reset_mode, v_reset_value)
        s = (h >= float(v_threshold)).to(x_flat.dtype)
        if reset_mode == 0:
            v = h - s * float(v_threshold)
        else:
            v = h * (1.0 - s) + s * v_reset_value
        spike[t] = s
        v_seq[t] = v
        v_prev = v

    return spike, v_seq


def _forward_torch_spike_flat(
    x_flat: torch.Tensor,
    v_init: torch.Tensor,
    param: torch.Tensor,
    v_threshold: float,
    v_reset: float | None,
    neuron_kind: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    spike, v_seq = _forward_torch_flat(x_flat, v_init, param, v_threshold, v_reset, neuron_kind)
    return spike, v_seq[-1]


def _backward_torch_flat(
    x_flat: torch.Tensor,
    v_init: torch.Tensor,
    v_seq: torch.Tensor,
    param: torch.Tensor,
    grad_spike: torch.Tensor,
    grad_v_seq: torch.Tensor,
    v_threshold: float,
    v_reset: float | None,
    detach_reset: bool,
    neuron_kind: int,
    surrogate_kind: int,
    surrogate_alpha: float,
    need_param_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    reset_mode = 0 if v_reset is None else 1
    v_reset_value = 0.0 if v_reset is None else float(v_reset)
    T, N = x_flat.shape
    grad_x = torch.empty_like(x_flat)
    carry = torch.zeros(N, device=x_flat.device, dtype=x_flat.dtype)
    grad_param = torch.zeros((), device=x_flat.device, dtype=x_flat.dtype) if need_param_grad else None

    for t in range(T - 1, -1, -1):
        v_prev = v_init if t == 0 else v_seq[t - 1]
        h, dh_dx, dh_dv, dh_dp = _charge_torch(
            x_flat[t],
            v_prev,
            param,
            neuron_kind,
            reset_mode,
            v_reset_value,
        )
        s = (h >= float(v_threshold)).to(x_flat.dtype)
        g_v = grad_v_seq[t] + carry
        g_s = grad_spike[t]
        if reset_mode == 0:
            g_h = g_v
            if not detach_reset:
                g_s = g_s - g_v * float(v_threshold)
        else:
            g_h = g_v * (1.0 - s)
            if not detach_reset:
                g_s = g_s + g_v * (v_reset_value - h)
        g_h = g_h + g_s * _surrogate_grad_torch(h - float(v_threshold), surrogate_kind, surrogate_alpha)
        grad_x[t] = g_h * dh_dx
        carry = g_h * dh_dv
        if need_param_grad:
            grad_param = grad_param + (g_h * dh_dp).sum()

    return grad_x, grad_param


if triton is not None:
    @triton.jit
    def _forward_kernel(
        x_ptr,
        v_init_ptr,
        param_ptr,
        spike_ptr,
        v_seq_ptr,
        T,
        N,
        v_threshold,
        v_reset_value,
        BLOCK_N: tl.constexpr,
        neuron_kind: tl.constexpr,
        reset_mode: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs < N
        v_prev = tl.load(v_init_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        if neuron_kind == 1:
            tau = tl.load(param_ptr).to(tl.float32)
            decay = 1.0 / tau
        elif neuron_kind == 2:
            w = tl.load(param_ptr).to(tl.float32)
            decay = 1.0 / (1.0 + tl.exp(-w))

        for t in range(T):
            x_t = tl.load(x_ptr + t * N + offs, mask=mask, other=0.0).to(tl.float32)

            if neuron_kind == 0:
                h = v_prev + x_t
            else:
                base = v_prev if reset_mode == 0 or v_reset_value == 0.0 else (v_prev - v_reset_value)
                h = v_prev + (x_t - base) * decay

            s = h >= v_threshold
            s_f = s.to(tl.float32)

            if reset_mode == 0:
                v = h - s_f * v_threshold
            else:
                v = h * (1.0 - s_f) + s_f * v_reset_value

            tl.store(spike_ptr + t * N + offs, s_f.to(tl.float32), mask=mask)
            tl.store(v_seq_ptr + t * N + offs, v.to(tl.float32), mask=mask)
            v_prev = v

    @triton.jit
    def _forward_spike_kernel(
        x_ptr, v_init_ptr, param_ptr, spike_ptr, v_last_ptr, T, N, v_threshold, v_reset_value,
        BLOCK_N: tl.constexpr, neuron_kind: tl.constexpr, reset_mode: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs < N
        v_prev = tl.load(v_init_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        if neuron_kind == 1:
            tau = tl.load(param_ptr).to(tl.float32)
            decay = 1.0 / tau
        elif neuron_kind == 2:
            w = tl.load(param_ptr).to(tl.float32)
            decay = 1.0 / (1.0 + tl.exp(-w))
        for t in range(T):
            x_t = tl.load(x_ptr + t * N + offs, mask=mask, other=0.0).to(tl.float32)
            if neuron_kind == 0:
                h = v_prev + x_t
            else:
                base = v_prev if reset_mode == 0 or v_reset_value == 0.0 else (v_prev - v_reset_value)
                h = v_prev + (x_t - base) * decay
            s_f = (h >= v_threshold).to(tl.float32)
            v_prev = h - s_f * v_threshold if reset_mode == 0 else h * (1.0 - s_f) + s_f * v_reset_value
            tl.store(spike_ptr + t * N + offs, s_f, mask=mask)
        tl.store(v_last_ptr + offs, v_prev, mask=mask)

    @triton.jit
    def _backward_kernel(
        x_ptr,
        v_init_ptr,
        v_seq_ptr,
        param_ptr,
        grad_spike_ptr,
        grad_v_seq_ptr,
        grad_x_ptr,
        grad_param_ptr,
        T,
        N,
        v_threshold,
        v_reset_value,
        surrogate_alpha,
        BLOCK_N: tl.constexpr,
        neuron_kind: tl.constexpr,
        reset_mode: tl.constexpr,
        surrogate_kind: tl.constexpr,
        detach_reset: tl.constexpr,
        accumulate_param: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs < N
        carry = tl.zeros((BLOCK_N,), dtype=tl.float32)

        if neuron_kind == 1:
            tau = tl.load(param_ptr).to(tl.float32)
            decay = 1.0 / tau
            dh_dx_const = decay
            dh_dv_const = 1.0 - decay
        elif neuron_kind == 2:
            w = tl.load(param_ptr).to(tl.float32)
            decay = 1.0 / (1.0 + tl.exp(-w))
            dh_dx_const = decay
            dh_dv_const = 1.0 - decay

        for rev_t in range(T):
            t = T - 1 - rev_t
            x_t = tl.load(x_ptr + t * N + offs, mask=mask, other=0.0).to(tl.float32)
            grad_s = tl.load(grad_spike_ptr + t * N + offs, mask=mask, other=0.0).to(tl.float32)
            grad_v = tl.load(grad_v_seq_ptr + t * N + offs, mask=mask, other=0.0).to(tl.float32) + carry

            if t == 0:
                v_prev = tl.load(v_init_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            else:
                v_prev = tl.load(v_seq_ptr + (t - 1) * N + offs, mask=mask, other=0.0).to(tl.float32)

            if neuron_kind == 0:
                h = v_prev + x_t
                dh_dx = 1.0
                dh_dv = 1.0
                dh_dp = tl.zeros((BLOCK_N,), dtype=tl.float32)
            else:
                base = v_prev if reset_mode == 0 or v_reset_value == 0.0 else (v_prev - v_reset_value)
                delta = x_t - base
                h = v_prev + delta * decay
                dh_dx = dh_dx_const
                dh_dv = dh_dv_const
                if neuron_kind == 1:
                    dh_dp = -(delta / (tau * tau))
                else:
                    dh_dp = delta * decay * (1.0 - decay)

            s = h >= v_threshold
            s_f = s.to(tl.float32)
            u = h - v_threshold

            if surrogate_kind == 0:
                pi = 3.141592653589793
                z = 0.5 * pi * surrogate_alpha * u
                surrogate_grad = surrogate_alpha / (2.0 * (1.0 + z * z))
            else:
                sig = 1.0 / (1.0 + tl.exp(-surrogate_alpha * u))
                surrogate_grad = surrogate_alpha * sig * (1.0 - sig)

            if reset_mode == 0:
                grad_h = grad_v
                if not detach_reset:
                    grad_s = grad_s - grad_v * v_threshold
            else:
                grad_h = grad_v * (1.0 - s_f)
                if not detach_reset:
                    grad_s = grad_s + grad_v * (v_reset_value - h)

            grad_h = grad_h + grad_s * surrogate_grad
            grad_x = grad_h * dh_dx
            tl.store(grad_x_ptr + t * N + offs, grad_x.to(tl.float32), mask=mask)
            carry = grad_h * dh_dv

            if accumulate_param:
                tl.atomic_add(grad_param_ptr, tl.sum(grad_h * dh_dp, axis=0))


def _forward_triton_flat(
    x_flat: torch.Tensor,
    v_init: torch.Tensor,
    param: torch.Tensor,
    v_threshold: float,
    v_reset: float | None,
    neuron_kind: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    T, N = x_flat.shape
    spike = torch.empty_like(x_flat, dtype=torch.float32)
    v_seq = torch.empty_like(x_flat, dtype=torch.float32)
    reset_mode = 0 if v_reset is None else 1
    v_reset_value = 0.0 if v_reset is None else float(v_reset)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)
    _forward_kernel[grid](x_flat, v_init, param, spike, v_seq, T, N, float(v_threshold), float(v_reset_value), BLOCK_N=128, neuron_kind=neuron_kind, reset_mode=reset_mode, num_warps=4)
    return spike.to(x_flat.dtype), v_seq.to(x_flat.dtype)


def _forward_triton_spike_flat(
    x_flat: torch.Tensor,
    v_init: torch.Tensor,
    param: torch.Tensor,
    v_threshold: float,
    v_reset: float | None,
    neuron_kind: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    T, N = x_flat.shape
    spike = torch.empty_like(x_flat, dtype=torch.float32)
    v_last = torch.empty_like(v_init, dtype=torch.float32)
    reset_mode = 0 if v_reset is None else 1
    v_reset_value = 0.0 if v_reset is None else float(v_reset)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)
    _forward_spike_kernel[grid](x_flat, v_init, param, spike, v_last, T, N, float(v_threshold), float(v_reset_value), BLOCK_N=128, neuron_kind=neuron_kind, reset_mode=reset_mode, num_warps=4)
    return spike.to(x_flat.dtype), v_last.to(x_flat.dtype)


def _backward_triton_flat(
    x_flat: torch.Tensor,
    v_init: torch.Tensor,
    v_seq: torch.Tensor,
    param: torch.Tensor,
    grad_spike: torch.Tensor,
    grad_v_seq: torch.Tensor,
    v_threshold: float,
    v_reset: float | None,
    detach_reset: bool,
    neuron_kind: int,
    surrogate_kind: int,
    surrogate_alpha: float,
    need_param_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    T, N = x_flat.shape
    grad_x = torch.empty_like(x_flat, dtype=torch.float32)
    grad_param = torch.zeros(1, device=x_flat.device, dtype=torch.float32) if need_param_grad else torch.empty(1, device=x_flat.device, dtype=torch.float32)
    reset_mode = 0 if v_reset is None else 1
    v_reset_value = 0.0 if v_reset is None else float(v_reset)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)
    _backward_kernel[grid](
        x_flat,
        v_init,
        v_seq,
        param,
        grad_spike,
        grad_v_seq,
        grad_x,
        grad_param,
        T,
        N,
        float(v_threshold),
        float(v_reset_value),
        float(surrogate_alpha),
        BLOCK_N=128,
        neuron_kind=neuron_kind,
        reset_mode=reset_mode,
        surrogate_kind=surrogate_kind,
        detach_reset=detach_reset,
        accumulate_param=need_param_grad,
        num_warps=4,
    )
    grad_param_out = grad_param[0].to(param.dtype) if need_param_grad else None
    return grad_x.to(x_flat.dtype), grad_param_out


class _SpikeOnlyNeuronFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_flat: torch.Tensor, v_init: torch.Tensor, param: torch.Tensor, v_threshold: float, v_reset: float | None, detach_reset: bool, neuron_kind: int, surrogate_kind: int, surrogate_alpha: float):
        if _can_use_triton(x_flat):
            spike, v_last = _forward_triton_spike_flat(x_flat, v_init, param, v_threshold, v_reset, neuron_kind)
            ctx.use_triton = True
        else:
            with torch.no_grad():
                spike, v_last = _forward_torch_spike_flat(x_flat, v_init, param, v_threshold, v_reset, neuron_kind)
            ctx.use_triton = False
        ctx.v_threshold = float(v_threshold)
        ctx.v_reset = v_reset
        ctx.detach_reset = bool(detach_reset)
        ctx.neuron_kind = int(neuron_kind)
        ctx.surrogate_kind = int(surrogate_kind)
        ctx.surrogate_alpha = float(surrogate_alpha)
        ctx.save_for_backward(x_flat, v_init, param)
        return spike, v_last

    @staticmethod
    def backward(ctx, grad_spike: torch.Tensor, grad_v_last: torch.Tensor):
        x_flat, v_init, param = ctx.saved_tensors
        grad_spike = torch.zeros_like(x_flat) if grad_spike is None else grad_spike.contiguous()
        need_param_grad = bool(ctx.needs_input_grad[2] and ctx.neuron_kind == _NEURON_KIND["plif"])
        if ctx.use_triton and _can_use_triton(x_flat):
            _, v_seq = _forward_triton_flat(x_flat, v_init, param, ctx.v_threshold, ctx.v_reset, ctx.neuron_kind)
            grad_x, grad_param = _backward_triton_flat(x_flat, v_init, v_seq, param, grad_spike, torch.zeros_like(v_seq), ctx.v_threshold, ctx.v_reset, ctx.detach_reset, ctx.neuron_kind, ctx.surrogate_kind, ctx.surrogate_alpha, need_param_grad)
        else:
            _, v_seq = _forward_torch_flat(x_flat, v_init, param, ctx.v_threshold, ctx.v_reset, ctx.neuron_kind)
            grad_x, grad_param = _backward_torch_flat(x_flat, v_init, v_seq, param, grad_spike, torch.zeros_like(v_seq), ctx.v_threshold, ctx.v_reset, ctx.detach_reset, ctx.neuron_kind, ctx.surrogate_kind, ctx.surrogate_alpha, need_param_grad)
        return grad_x, None, grad_param, None, None, None, None, None, None


class _FusedNeuronFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x_flat: torch.Tensor,
        v_init: torch.Tensor,
        param: torch.Tensor,
        v_threshold: float,
        v_reset: float | None,
        detach_reset: bool,
        neuron_kind: int,
        surrogate_kind: int,
        surrogate_alpha: float,
    ):
        if _can_use_triton(x_flat):
            spike, v_seq = _forward_triton_flat(
                x_flat=x_flat,
                v_init=v_init,
                param=param,
                v_threshold=v_threshold,
                v_reset=v_reset,
                neuron_kind=neuron_kind,
            )
            ctx.use_triton = True
        else:
            with torch.no_grad():
                spike, v_seq = _forward_torch_flat(
                    x_flat=x_flat,
                    v_init=v_init,
                    param=param,
                    v_threshold=v_threshold,
                    v_reset=v_reset,
                    neuron_kind=neuron_kind,
                )
            ctx.use_triton = False

        ctx.v_threshold = float(v_threshold)
        ctx.v_reset = v_reset
        ctx.detach_reset = bool(detach_reset)
        ctx.neuron_kind = int(neuron_kind)
        ctx.surrogate_kind = int(surrogate_kind)
        ctx.surrogate_alpha = float(surrogate_alpha)
        ctx.save_for_backward(x_flat, v_init, param, v_seq.detach())
        return spike, v_seq

    @staticmethod
    def backward(ctx, grad_spike: torch.Tensor, grad_v_seq: torch.Tensor):
        x_flat, v_init, param, v_seq = ctx.saved_tensors
        grad_spike = torch.zeros_like(x_flat) if grad_spike is None else grad_spike.contiguous()
        grad_v_seq = torch.zeros_like(v_seq) if grad_v_seq is None else grad_v_seq.contiguous()
        need_param_grad = bool(ctx.needs_input_grad[2] and ctx.neuron_kind == _NEURON_KIND["plif"])

        if ctx.use_triton and _can_use_triton(x_flat):
            grad_x, grad_param = _backward_triton_flat(
                x_flat=x_flat,
                v_init=v_init,
                v_seq=v_seq,
                param=param,
                grad_spike=grad_spike,
                grad_v_seq=grad_v_seq,
                v_threshold=ctx.v_threshold,
                v_reset=ctx.v_reset,
                detach_reset=ctx.detach_reset,
                neuron_kind=ctx.neuron_kind,
                surrogate_kind=ctx.surrogate_kind,
                surrogate_alpha=ctx.surrogate_alpha,
                need_param_grad=need_param_grad,
            )
        else:
            grad_x, grad_param = _backward_torch_flat(
                x_flat=x_flat,
                v_init=v_init,
                v_seq=v_seq,
                param=param,
                grad_spike=grad_spike,
                grad_v_seq=grad_v_seq,
                v_threshold=ctx.v_threshold,
                v_reset=ctx.v_reset,
                detach_reset=ctx.detach_reset,
                neuron_kind=ctx.neuron_kind,
                surrogate_kind=ctx.surrogate_kind,
                surrogate_alpha=ctx.surrogate_alpha,
                need_param_grad=need_param_grad,
            )

        return grad_x, None, grad_param, None, None, None, None, None, None


# class TritonBaseNode(nn.Module):
class TritonBaseNode(base.MemoryModule):
    def __init__(
        self,
        neuron_type: str,
        surrogate_type: str = "atan",
        surrogate_alpha: float | None = None,
        v_threshold: float = 1.0,
        v_reset: float | None = 0.0,
        detach_reset: bool = True,
        step_mode: str = "m",
    ):
        super().__init__()
        neuron_type = neuron_type.lower()
        surrogate_type = surrogate_type.lower()

        if neuron_type not in _NEURON_KIND:
            raise ValueError(f"Unsupported neuron_type: {neuron_type}")
        if surrogate_type not in _SURROGATE_KIND:
            raise ValueError(f"Unsupported surrogate_type: {surrogate_type}")

        self.neuron_type = neuron_type
        self.surrogate_type = surrogate_type
        self.surrogate_alpha = float(_default_alpha(surrogate_type) if surrogate_alpha is None else surrogate_alpha)
        self.v_threshold = float(v_threshold)
        self.v_reset = v_reset
        self.detach_reset = bool(detach_reset)
        self.step_mode = step_mode
        self.store_v_seq = False
        self.v: torch.Tensor | float = _state_base(v_reset)
        self.v_seq: torch.Tensor | None = None

    def reset(self):
        self.v = _state_base(self.v_reset)
        self.v_seq = None

    def _param(self, x_flat: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.step_mode not in {"s", "m"}:
            raise ValueError(f"Unsupported step_mode: {self.step_mode}")

        squeeze = self.step_mode == "s"
        x_seq = x.unsqueeze(0) if squeeze else x
        x_flat, original_shape = _flatten_time(x_seq)
        v_init = _init_state(x_seq[0], self.v, self.v_reset)
        param = self._param(x_flat)

        if self.store_v_seq:
            spike_flat, v_flat = _FusedNeuronFunction.apply(x_flat, v_init, param, self.v_threshold, self.v_reset, self.detach_reset, _NEURON_KIND[self.neuron_type], _SURROGATE_KIND[self.surrogate_type], self.surrogate_alpha)
            spike_seq = _restore_time(spike_flat, original_shape)
            v_seq = _restore_time(v_flat, original_shape)
            current_v = v_seq[-1] if not squeeze else v_seq[0]
            self.v_seq = v_seq if not squeeze else v_seq[0]
        else:
            spike_flat, v_last = _SpikeOnlyNeuronFunction.apply(x_flat, v_init, param, self.v_threshold, self.v_reset, self.detach_reset, _NEURON_KIND[self.neuron_type], _SURROGATE_KIND[self.surrogate_type], self.surrogate_alpha)
            spike_seq = _restore_time(spike_flat, original_shape)
            current_v = v_last.reshape(x_seq[0].shape)
            self.v_seq = None
        self.v = current_v.detach()
        return spike_seq if not squeeze else spike_seq[0]


class TritonIFNode(TritonBaseNode):
    def __init__(
        self,
        surrogate_type: str = "atan",
        surrogate_alpha: float | None = None,
        v_threshold: float = 1.0,
        v_reset: float | None = 0.0,
        detach_reset: bool = True,
        step_mode: str = "m",
    ):
        super().__init__(
            neuron_type="if",
            surrogate_type=surrogate_type,
            surrogate_alpha=surrogate_alpha,
            v_threshold=v_threshold,
            v_reset=v_reset,
            detach_reset=detach_reset,
            step_mode=step_mode,
        )
        self.register_buffer("_dummy_param", torch.tensor(0.0, dtype=torch.float32))

    def _param(self, x_flat: torch.Tensor) -> torch.Tensor:
        return self._dummy_param.to(device=x_flat.device, dtype=torch.float32)


class TritonLIFNode(TritonBaseNode):
    def __init__(
        self,
        tau: float = 2.0,
        surrogate_type: str = "atan",
        surrogate_alpha: float | None = None,
        v_threshold: float = 1.0,
        v_reset: float | None = 0.0,
        detach_reset: bool = True,
        step_mode: str = "m",
    ):
        super().__init__(
            neuron_type="lif",
            surrogate_type=surrogate_type,
            surrogate_alpha=surrogate_alpha,
            v_threshold=v_threshold,
            v_reset=v_reset,
            detach_reset=detach_reset,
            step_mode=step_mode,
        )
        self.register_buffer("tau_tensor", torch.tensor(float(tau), dtype=torch.float32))

    def _param(self, x_flat: torch.Tensor) -> torch.Tensor:
        return self.tau_tensor.to(device=x_flat.device, dtype=torch.float32)


class TritonParametricLIFNode(TritonBaseNode):
    def __init__(
        self,
        init_tau: float = 2.0,
        surrogate_type: str = "atan",
        surrogate_alpha: float | None = None,
        v_threshold: float = 1.0,
        v_reset: float | None = 0.0,
        detach_reset: bool = True,
        step_mode: str = "m",
    ):
        super().__init__(
            neuron_type="plif",
            surrogate_type=surrogate_type,
            surrogate_alpha=surrogate_alpha,
            v_threshold=v_threshold,
            v_reset=v_reset,
            detach_reset=detach_reset,
            step_mode=step_mode,
        )
        if init_tau <= 1.0:
            raise ValueError(f"init_tau must be > 1, but got {init_tau}")
        init_w = -math.log(init_tau - 1.0)
        self.w = nn.Parameter(torch.tensor(init_w, dtype=torch.float32))

    def _param(self, x_flat: torch.Tensor) -> torch.Tensor:
        return self.w


class _PSPReadout(nn.Module):
    def __init__(self, node: nn.Module, step_mode: str):
        super().__init__()
        self.node = node
        self.step_mode = step_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ = self.node(x)
        return self.node.v_seq


class _DualReadout(nn.Module):
    def __init__(self, node: nn.Module):
        super().__init__()
        self.node = node

    def forward(self, x: torch.Tensor) -> DualOutput:
        spike = self.node(x)
        return DualOutput(spike=spike, v_seq=self.node.v_seq)


def build_triton_neuron(
    model_config,
    step_mode: str = "m",
    v_threshold: float | None = None,
    output_mode: str | None = None,
) -> nn.Module:
    neuron_type = str(getattr(model_config, "neuron_type", "lif")).lower()
    surrogate_type = str(getattr(model_config, "surrogate", "atan")).lower()
    surrogate_alpha = getattr(model_config, "surrogate_alpha", None)
    detach_reset = bool(getattr(model_config, "detach_reset", True))
    tau = float(getattr(model_config, "tau", 2.0))
    v_reset = getattr(model_config, "v_reset", 0.0)

    if v_threshold is None:
        v_threshold = float(getattr(model_config, "v_threshold", 1.0))

    if output_mode is None:
        output_mode = str(getattr(model_config, "neuron_output", "spike")).lower()
    else:
        output_mode = output_mode.lower()

    if output_mode not in {"spike", "psp", "dual"}:
        raise ValueError(f"Unknown output_mode: {output_mode}")

    if neuron_type == "if":
        node = TritonIFNode(
            surrogate_type=surrogate_type,
            surrogate_alpha=surrogate_alpha,
            v_threshold=v_threshold,
            v_reset=v_reset,
            detach_reset=detach_reset,
            step_mode=step_mode,
        )
    elif neuron_type == "lif":
        node = TritonLIFNode(
            tau=tau,
            surrogate_type=surrogate_type,
            surrogate_alpha=surrogate_alpha,
            v_threshold=v_threshold,
            v_reset=v_reset,
            detach_reset=detach_reset,
            step_mode=step_mode,
        )
    elif neuron_type == "plif":
        node = TritonParametricLIFNode(
            init_tau=tau,
            surrogate_type=surrogate_type,
            surrogate_alpha=surrogate_alpha,
            v_threshold=v_threshold,
            v_reset=v_reset,
            detach_reset=detach_reset,
            step_mode=step_mode,
        )
    else:
        raise ValueError(f"Unsupported neuron_type: {neuron_type}")

    if output_mode == "spike":
        node.store_v_seq = False
        return node
    node.store_v_seq = True
    if output_mode == "psp":
        return _PSPReadout(node, step_mode)
    return _DualReadout(node)