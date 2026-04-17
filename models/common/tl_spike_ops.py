"""
triton_gate_lif.py
==================
Memory-efficient Triton implementation of LIF neurons for SNN Transformers.

Key idea:
  - Forward:  fused T-step scan, does NOT store v_seq in autograd graph.
  - Backward: recomputes v on-the-fly in a scratch buffer → saves T*N floats
              of activation memory compared to spikingjelly's default.

Provides:
  - TritonGateLIF : drop-in for GateLIFNode  (per-channel learnable τ)
  - TritonLIF     : drop-in for standard LIF  (scalar τ, optionally learnable)
  - TritonDualLIF : LIF that also outputs spatially-pooled v_seq (for k_lif)

Compatible with: multi-step (T>1), hard/soft reset, ATan surrogate gradient.
"""

import math
import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch.autograd import Function


# =====================================================================
#  Triton Kernels
# =====================================================================

@triton.jit
def _lif_fwd_kernel(
    x_ptr,       # [T * N]   input current (contiguous)
    s_ptr,       # [T * N]   output spikes
    v0_ptr,      # [N]       initial membrane potential
    vT_ptr,      # [N]       final membrane potential  (written)
    decay_ptr,   # [C]       per-channel decay  (broadcast if C < N)
    v_seq_ptr,   # [T * N]   optional v_seq output
    # ---- constexpr ----
    V_TH: tl.constexpr,
    V_RESET: tl.constexpr,      # only used when HARD_RESET=True
    T: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,            # channel dim size; decay is indexed as off % C
    STORE_V: tl.constexpr,      # whether to write v_seq
    HARD_RESET: tl.constexpr,   # True → hard reset; False → soft reset
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < N

    decay = tl.load(decay_ptr + (off % C), mask=mask, other=1.0)
    v = tl.load(v0_ptr + off, mask=mask, other=0.0).to(tl.float32)

    for t in range(T):
        idx = t * N + off
        # inp = tl.load(x_ptr + idx, mask=mask, other=0.0)
        inp = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)

        # Charge
        v = decay * v + inp

        # Store v before reset (optional)
        if STORE_V:
            tl.store(v_seq_ptr + idx, v, mask=mask)

        # Fire  (Heaviside – surrogate only in backward)
        s = (v >= V_TH).to(tl.float32)
        tl.store(s_ptr + idx, s, mask=mask)

        # Reset
        if HARD_RESET:
            v = (1.0 - s) * v + s * V_RESET
        else:
            v = v - s * V_TH

    tl.store(vT_ptr + off, v, mask=mask)


@triton.jit
def _lif_bwd_kernel(
    # Forward tensors (for recomputation)
    x_ptr,       # [T * N]
    v0_ptr,      # [N]
    decay_ptr,   # [C]
    # Incoming gradients
    gs_ptr,      # [T * N]   grad w.r.t. spikes
    gvT_ptr,     # [N]       grad w.r.t. v_last
    # Output gradients
    gx_ptr,      # [T * N]   grad w.r.t. input
    gd_ptr,      # [N]       per-element grad_decay accumulator
    # Scratch
    vbuf_ptr,    # [T * N]   scratch for recomputed v (before reset)
    # ---- constexpr ----
    V_TH: tl.constexpr,
    V_RESET: tl.constexpr,
    ALPHA: tl.constexpr,        # ATan surrogate α
    DETACH_RESET: tl.constexpr,
    HARD_RESET: tl.constexpr,
    T: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < N

    decay = tl.load(decay_ptr + (off % C), mask=mask, other=1.0)

    # ---- Pass 1: forward recompute  →  store v (before reset) ----
    v = tl.load(v0_ptr + off, mask=mask, other=0.0).to(tl.float32)
    for t in range(T):
        idx = t * N + off
        # inp = tl.load(x_ptr + idx, mask=mask, other=0.0)
        inp = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        v = decay * v + inp
        tl.store(vbuf_ptr + idx, v, mask=mask)          # v before reset
        s = (v >= V_TH).to(tl.float32)
        if HARD_RESET:
            v = (1.0 - s) * v + s * V_RESET
        else:
            v = v - s * V_TH

    # ---- Pass 2: backward through time ----
    g_var = tl.load(gvT_ptr + off, mask=mask, other=0.0).to(tl.float32)
    g_decay_acc = tl.zeros([BLOCK], dtype=tl.float32)
    PI_H: tl.constexpr = 1.5707963267948966     # π / 2

    for t_rev in range(T):
        t = T - 1 - t_rev
        idx = t * N + off

        v = tl.load(vbuf_ptr + idx, mask=mask, other=0.0).to(tl.float32)      # v before reset
        s = (v >= V_TH).to(tl.float32)

        # ATan surrogate:  α / (2 · (1 + (π/2 · α · h)²))
        h = v - V_TH
        pah = PI_H * ALPHA * h
        sg = ALPHA / (2.0 * (1.0 + pah * pah))

        # ---- grad through reset ----
        if HARD_RESET:
            # v_ar = (1 - s) * v + s * V_RESET
            if DETACH_RESET:
                g_v = g_var * (1.0 - s)          # s treated as const
                g_s_reset = 0.0
            else:
                g_v = g_var * (1.0 - s)
                g_s_reset = g_var * (V_RESET - v)
        else:
            # v_ar = v - s * V_TH
            g_v = g_var                           # ∂v_ar/∂v = 1
            if DETACH_RESET:
                g_s_reset = 0.0
            else:
                g_s_reset = -V_TH * g_var

        # ---- grad through firing (surrogate) ----
        g_s_ext = tl.load(gs_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        g_v = g_v + (g_s_ext + g_s_reset) * sg

        # ---- grad through charge: v = decay * v_ar_prev + x ----
        tl.store(gx_ptr + idx, g_v, mask=mask)

        # v_ar[t-1] for grad_decay
        if t > 0:
            vp = tl.load(vbuf_ptr + (t - 1) * N + off, mask=mask, other=0.0).to(tl.float32)
            sp = (vp >= V_TH).to(tl.float32)
            if HARD_RESET:
                var_prev = (1.0 - sp) * vp + sp * V_RESET
            else:
                var_prev = vp - sp * V_TH
        else:
            var_prev = tl.load(v0_ptr + off, mask=mask, other=0.0).to(tl.float32)

        g_decay_acc += g_v * var_prev
        g_var = g_v * decay      # propagate to v_ar[t-1]

    tl.store(gd_ptr + off, g_decay_acc, mask=mask)


# =====================================================================
#  Fused LIF + spatial-pool kernel  (for DualLIF / k_lif)
# =====================================================================

@triton.jit
def _lif_fwd_pool_kernel(
    x_ptr,       # [T * N_total]
    s_ptr,       # [T * N_total]
    v0_ptr,      # [N_total]
    vT_ptr,      # [N_total]
    decay_ptr,   # [C]
    vpool_ptr,   # [T * N_pool]  spatially-pooled v output
    # ---- constexpr ----
    V_TH: tl.constexpr,
    V_RESET: tl.constexpr,
    T: tl.constexpr,
    N_TOTAL: tl.constexpr,   # B * N_spatial * C
    C: tl.constexpr,
    N_SPATIAL: tl.constexpr, # number of spatial tokens
    N_POOL: tl.constexpr,    # B * C  (pooled size per timestep)
    HARD_RESET: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Forward LIF + output spatially-pooled v (mean over N_spatial).

    Layout: x is [T, B, N_spatial, C] flattened to [T, B*N_spatial*C].
    Element off in a timestep:  off = b * N_spatial * C + n * C + c
    Pool target:                pool_off = b * C + c = (off // (N_spatial*C)) * C + (off % C)
    """
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < N_TOTAL

    decay = tl.load(decay_ptr + (off % C), mask=mask, other=1.0)
    v = tl.load(v0_ptr + off, mask=mask, other=0.0).to(tl.float32)

    # Compute pool index for this element
    pool_off = (off // (N_SPATIAL * C)) * C + (off % C)
    inv_n = 1.0 / N_SPATIAL

    for t in range(T):
        idx = t * N_TOTAL + off
        # inp = tl.load(x_ptr + idx, mask=mask, other=0.0)
        inp = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        v = decay * v + inp

        # Atomic add pooled v  (v / N_spatial)
        pool_idx = t * N_POOL + pool_off
        tl.atomic_add(vpool_ptr + pool_idx, v * inv_n, mask=mask)

        s = (v >= V_TH).to(tl.float32)
        tl.store(s_ptr + idx, s, mask=mask)

        if HARD_RESET:
            v = (1.0 - s) * v + s * V_RESET
        else:
            v = v - s * V_TH

    tl.store(vT_ptr + off, v, mask=mask)


# =====================================================================
#  Autograd Functions
# =====================================================================

def _grid(N, BLOCK):
    return ((N + BLOCK - 1) // BLOCK,)


class _LIFFunction(Function):
    """Generic LIF autograd — works for both GateLIF and standard LIF."""

    @staticmethod
    def forward(ctx, x_flat, v_init, decay, v_threshold, v_reset,
                alpha, detach_reset, hard_reset):
        T, N = x_flat.shape
        C = decay.shape[0]
        BLOCK = min(1024, triton.next_power_of_2(N))

        spike = torch.empty_like(x_flat)
        v_last = torch.empty(N, device=x_flat.device, dtype=x_flat.dtype)

        _lif_fwd_kernel[_grid(N, BLOCK)](
            x_flat, spike, v_init, v_last, decay,
            spike,                         # dummy v_seq_ptr
            V_TH=v_threshold, V_RESET=v_reset,
            T=T, N=N, C=C,
            STORE_V=False, HARD_RESET=hard_reset,
            BLOCK=BLOCK,
        )

        ctx.save_for_backward(x_flat, v_init, decay)
        ctx.v_threshold = v_threshold
        ctx.v_reset = v_reset
        ctx.alpha = alpha
        ctx.detach_reset = detach_reset
        ctx.hard_reset = hard_reset
        ctx.T, ctx.N, ctx.C = T, N, C
        return spike, v_last

    @staticmethod
    def backward(ctx, grad_spike, grad_v_last):
        x_flat, v_init, decay = ctx.saved_tensors
        T, N, C = ctx.T, ctx.N, ctx.C
        BLOCK = min(1024, triton.next_power_of_2(N))

        grad_x = torch.empty_like(x_flat)
        grad_decay_buf = torch.empty(N, device=x_flat.device, dtype=x_flat.dtype)
        v_scratch = torch.empty_like(x_flat)  # transient [T, N]

        if grad_v_last is None:
            grad_v_last = torch.zeros(N, device=x_flat.device, dtype=x_flat.dtype)

        _lif_bwd_kernel[_grid(N, BLOCK)](
            x_flat, v_init, decay,
            grad_spike, grad_v_last,
            grad_x, grad_decay_buf,
            v_scratch,
            V_TH=ctx.v_threshold, V_RESET=ctx.v_reset,
            ALPHA=ctx.alpha, DETACH_RESET=ctx.detach_reset,
            HARD_RESET=ctx.hard_reset,
            T=T, N=N, C=C, BLOCK=BLOCK,
        )
        del v_scratch  # free immediately

        # Reduce per-element grad_decay → per-channel [C]
        grad_decay = grad_decay_buf.reshape(-1, C).sum(dim=0)
        return grad_x, None, grad_decay, None, None, None, None, None


class _DualLIFFunction(Function):
    """LIF that returns spike + spatially-pooled v_seq."""

    @staticmethod
    def forward(ctx, x_flat, v_init, decay, v_threshold, v_reset,
                alpha, detach_reset, hard_reset,
                N_spatial, N_pool):
        T, N_total = x_flat.shape
        C = decay.shape[0]
        BLOCK = min(1024, triton.next_power_of_2(N_total))

        spike = torch.empty_like(x_flat)
        v_last = torch.empty(N_total, device=x_flat.device, dtype=x_flat.dtype)
        v_pool = torch.zeros(T * N_pool, device=x_flat.device, dtype=torch.float32)

        _lif_fwd_pool_kernel[_grid(N_total, BLOCK)](
            x_flat, spike, v_init, v_last, decay,
            v_pool,
            V_TH=v_threshold, V_RESET=v_reset,
            T=T, N_TOTAL=N_total, C=C,
            N_SPATIAL=N_spatial, N_POOL=N_pool,
            HARD_RESET=hard_reset, BLOCK=BLOCK,
        )

        ctx.save_for_backward(x_flat, v_init, decay)
        ctx.v_threshold = v_threshold
        ctx.v_reset = v_reset
        ctx.alpha = alpha
        ctx.detach_reset = detach_reset
        ctx.hard_reset = hard_reset
        ctx.T, ctx.N_total, ctx.C = T, N_total, C

        v_pool = v_pool.reshape(T, N_pool).to(x_flat.dtype)
        return spike, v_last, v_pool

    @staticmethod
    def backward(ctx, grad_spike, grad_v_last, grad_v_pool):
        # grad_v_pool is NOT backpropagated through atomic adds easily.
        # We fall back to the standard LIF backward for grad_x and grad_decay,
        # ignoring grad_v_pool (it's used as a detached input to gates anyway).
        x_flat, v_init, decay = ctx.saved_tensors
        T, N, C = ctx.T, ctx.N_total, ctx.C
        BLOCK = min(1024, triton.next_power_of_2(N))

        grad_x = torch.empty_like(x_flat)
        grad_decay_buf = torch.empty(N, device=x_flat.device, dtype=x_flat.dtype)
        v_scratch = torch.empty_like(x_flat)

        if grad_v_last is None:
            grad_v_last = torch.zeros(N, device=x_flat.device, dtype=x_flat.dtype)

        _lif_bwd_kernel[_grid(N, BLOCK)](
            x_flat, v_init, decay,
            grad_spike, grad_v_last,
            grad_x, grad_decay_buf,
            v_scratch,
            V_TH=ctx.v_threshold, V_RESET=ctx.v_reset,
            ALPHA=ctx.alpha, DETACH_RESET=ctx.detach_reset,
            HARD_RESET=ctx.hard_reset,
            T=T, N=N, C=C, BLOCK=BLOCK,
        )
        del v_scratch

        grad_decay = grad_decay_buf.reshape(-1, C).sum(dim=0)
        return grad_x, None, grad_decay, None, None, None, None, None, None, None


# =====================================================================
#  nn.Module Wrappers
# =====================================================================

class TritonGateLIF(nn.Module):
    """Drop-in replacement for GateLIFNode.

    Per-channel learnable τ via parameter ``w``:  decay = 1 − σ(w).
    Saves ~T×N floats of activation memory by not storing v_seq.

    Args:
        channels:      number of channels (= head_dim typically)
        init_tau:      initial time constant  (τ = 1/σ(w))
        v_threshold:   firing threshold
        v_reset:       reset voltage (float → hard reset; None → soft reset)
        alpha:         ATan surrogate gradient parameter
        detach_reset:  if True, detach spike in reset (standard in spikingjelly)
    """

    def __init__(self, channels: int, init_tau: float = 2.0,
                 v_threshold: float = 1.0, v_reset: float = 0.0,
                 alpha: float = 2.0, detach_reset: bool = True):
        super().__init__()
        init_w = -math.log(init_tau - 1.0)        # σ(init_w) = 1/init_tau
        self.w = nn.Parameter(torch.full((channels,), init_w))
        self.channels = channels
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.hard_reset = v_reset is not None
        self.alpha = alpha
        self.detach_reset = detach_reset
        self.last_firing_rate = None
        self._v: torch.Tensor | None = None       # hidden state

    def reset(self):
        self._v = None

    @property
    def decay(self):
        return 1.0 - self.w.sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [T, *spatial_dims, C]   where C == self.channels
        Returns:
            spike: same shape as x
        """
        T = x.shape[0]
        orig_shape = x.shape
        x_flat = x.reshape(T, -1).contiguous()      # [T, N]
        N = x_flat.shape[1]

        v_init = (self._v if self._v is not None
                  else torch.zeros(N, device=x.device, dtype=x.dtype))

        v_reset_val = self.v_reset if self.v_reset is not None else 0.0

        spike, v_last = _LIFFunction.apply(
            x_flat, v_init, self.decay,
            self.v_threshold, v_reset_val,
            self.alpha, self.detach_reset, self.hard_reset,
        )
        self._v = v_last.detach()
        self.last_firing_rate = spike.detach().mean()
        return spike.reshape(orig_shape)


class TritonLIF(nn.Module):
    """Memory-efficient standard LIF neuron (scalar or learnable τ).

    Drop-in for spikingjelly ``LIFNode`` / ``ParametricLIFNode``.
    """

    def __init__(self, tau: float = 2.0, learnable: bool = False,
                 v_threshold: float = 1.0, v_reset: float = 0.0,
                 alpha: float = 2.0, detach_reset: bool = True):
        super().__init__()
        init_w = -math.log(tau - 1.0)
        if learnable:
            self.w = nn.Parameter(torch.tensor([init_w]))
        else:
            self.register_buffer('w', torch.tensor([init_w]))
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.hard_reset = v_reset is not None
        self.alpha = alpha
        self.detach_reset = detach_reset
        self._v: torch.Tensor | None = None

    def reset(self):
        self._v = None

    @property
    def decay(self):
        return 1.0 - self.w.sigmoid()              # [1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[0]
        orig_shape = x.shape
        x_flat = x.reshape(T, -1).contiguous()
        N = x_flat.shape[1]

        v_init = (self._v if self._v is not None
                  else torch.zeros(N, device=x.device, dtype=x.dtype))

        v_reset_val = self.v_reset if self.v_reset is not None else 0.0

        spike, v_last = _LIFFunction.apply(
            x_flat, v_init, self.decay,
            self.v_threshold, v_reset_val,
            self.alpha, self.detach_reset, self.hard_reset,
        )
        self._v = v_last.detach()
        return spike.reshape(orig_shape)


class TritonDualLIF(nn.Module):
    """LIF that outputs spike AND spatially-pooled v_seq.

    Replaces k_lif with ``output_mode="dual"`` — fuses the spatial mean
    into the forward kernel so the full [T, B, N, C] v_seq is never
    materialised.  Saves N × sizeof(float) per timestep.

    Input layout:  [T, B, N_spatial, C]
    Outputs:
        spike:     [T, B, N_spatial, C]
        v_pooled:  [T, B, C]              (mean over N_spatial)
    """

    def __init__(self, tau: float = 2.0, learnable: bool = False,
                 v_threshold: float = 1.0, v_reset: float = 0.0,
                 alpha: float = 2.0, detach_reset: bool = True):
        super().__init__()
        init_w = -math.log(tau - 1.0)
        if learnable:
            self.w = nn.Parameter(torch.tensor([init_w]))
        else:
            self.register_buffer('w', torch.tensor([init_w]))
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.hard_reset = v_reset is not None
        self.alpha = alpha
        self.detach_reset = detach_reset
        self._v: torch.Tensor | None = None

    def reset(self):
        self._v = None

    @property
    def decay(self):
        return 1.0 - self.w.sigmoid()

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [T, B, N_spatial, C]
        Returns:
            spike:    [T, B, N_spatial, C]
            v_pooled: [T, B, C]
        """
        T, B, N_spatial, C = x.shape
        N_total = B * N_spatial * C
        N_pool = B * C

        x_flat = x.reshape(T, N_total).contiguous()

        v_init = (self._v if self._v is not None
                  else torch.zeros(N_total, device=x.device, dtype=x.dtype))

        v_reset_val = self.v_reset if self.v_reset is not None else 0.0

        spike, v_last, v_pool = _DualLIFFunction.apply(
            x_flat, v_init, self.decay,
            self.v_threshold, v_reset_val,
            self.alpha, self.detach_reset, self.hard_reset,
            N_spatial, N_pool,
        )

        self._v = v_last.detach()

        spike = spike.reshape(T, B, N_spatial, C)
        v_pooled = v_pool.reshape(T, B, C)
        return spike, v_pooled


# =====================================================================
#  Quick smoke test
# =====================================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda"
    T, B, N, C = 4, 2, 16, 32

    # --- Test TritonGateLIF ---
    x = torch.randn(T, B, N, C, device=device, requires_grad=True)
    lif = TritonGateLIF(channels=C, init_tau=2.0, v_threshold=1.0, v_reset=0.0).to(device)
    lif.reset()
    s = lif(x)
    loss = s.sum()
    loss.backward()
    print(f"[TritonGateLIF]  spike shape={s.shape}  "
          f"grad_x norm={x.grad.norm():.4f}  "
          f"grad_w norm={lif.w.grad.norm():.4f}  "
          f"firing_rate={lif.last_firing_rate:.4f}")

    # --- Test TritonLIF ---
    x2 = torch.randn(T, B, N, C, device=device, requires_grad=True)
    lif2 = TritonLIF(tau=2.0, learnable=True).to(device)
    lif2.reset()
    s2 = lif2(x2)
    s2.sum().backward()
    print(f"[TritonLIF]      spike shape={s2.shape}  "
          f"grad_x norm={x2.grad.norm():.4f}  "
          f"grad_w norm={lif2.w.grad.norm():.4f}")

    # --- Test TritonDualLIF ---
    x3 = torch.randn(T, B, N, C, device=device, requires_grad=True)
    lif3 = TritonDualLIF(tau=2.0).to(device)
    lif3.reset()
    s3, vp = lif3(x3)
    (s3.sum() + vp.sum()).backward()
    print(f"[TritonDualLIF]  spike={s3.shape}  v_pooled={vp.shape}  "
          f"grad_x norm={x3.grad.norm():.4f}")

    # --- Numerical check vs naive PyTorch ---
    print("\n--- Numerical verification ---")
    lif_t = TritonGateLIF(channels=C, init_tau=2.0, v_reset=0.0).to(device)
    decay = lif_t.decay.detach()

    x_test = torch.randn(T, B, N, C, device=device)
    lif_t.reset()
    s_triton = lif_t(x_test)

    # Naive loop
    v = torch.zeros(B, N, C, device=device)
    s_ref = []
    for t in range(T):
        v = decay * v + x_test[t]
        s = (v >= 1.0).float()
        s_ref.append(s)
        v = (1.0 - s) * v   # hard reset, v_reset=0
    s_ref = torch.stack(s_ref)

    err = (s_triton - s_ref).abs().max().item()
    print(f"Max spike diff = {err}")
    assert err == 0.0, f"Mismatch! {err}"
    print("✓ Triton kernel matches naive PyTorch loop.")
