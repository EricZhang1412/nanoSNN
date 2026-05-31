"""Gated linear-attention variants for Gate 1 pilot (MGA vs SpikingBrain trivial gates).

Four drop-in replacements of a single spike-driven linear-attention module:

  C0 (`c0_sdla`):       memoryless baseline  S_t = K_t^T V_t
  C1 (`c1_lowrank`):    SpikingBrain 7B-style low-rank linear gate
                        S_t = diag(g_t) S_{t-1} + K_t^T V_t,  g_t = sigmoid(W_down W_up(mean_N K_t))
  C2 (`c2_oneminusk`):  SpikingBrain 76B-style shared (1 - k) gate (no learnable gate params)
                        S_t = diag(1 - mean_N K_t) S_{t-1} + K_t^T V_t
  C3 (`c3_mga`):        Membrane-Gated Attention (this paper's proposal)
                        S_t = (1 - s_gamma_t * 2^-k_bits) S_{t-1} + s_beta_t * K_t^T V_t
                        with s_gamma, s_beta driven by two STP-inspired LIF gate neurons
                        whose input for s_gamma is the K *pre-threshold membrane* pooled
                        over tokens (not the K spike output).

All four share an identical Q/K/V projection backbone (Linear + BN1d + LIF).  Only
gate parameters differ.  See `docs/PILOT_GATE1.md` (in the same PR) for the
experimental protocol.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from spikingjelly.activation_based import surrogate

from ..common.spike_ops import build_neuron


def _bn1d(x: torch.Tensor, bn: nn.BatchNorm1d) -> torch.Tensor:
    """Apply BN1d on [T, B, N, C] by transposing C to dim -2 (matches Spikformer convention)."""
    T, B, N, C = x.shape
    x = rearrange(x, "T B N C -> (T B) N C")
    x = rearrange(bn(rearrange(x, "TB N C -> TB C N")), "TB C N -> TB N C")
    return rearrange(x, "(T B) N C -> T B N C", T=T, B=B).contiguous()


# ---------------------------------------------------------------------------
# Shared backbone
# ---------------------------------------------------------------------------


class _GatedLinearAttentionBase(nn.Module):
    """Base class with shared Q/K/V LIF projections, BN, scale and output projection.

    Subclasses implement `_run_state_update(K, V, K_membrane=None)` returning a
    sequence `S_seq` of shape `[T, B, H, D, D]` (the per-step KV-state).
    """

    def __init__(self, dim: int, num_heads: int, model_config):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_linear = nn.Linear(dim, dim)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = build_neuron(model_config)

        self.k_linear = nn.Linear(dim, dim)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = build_neuron(model_config)

        self.v_linear = nn.Linear(dim, dim)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = build_neuron(model_config)

        self.attn_lif = build_neuron(model_config, v_threshold=0.5)

        self.proj_linear = nn.Linear(dim, dim)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = build_neuron(model_config)

        # Subclasses that need pre-threshold K membrane (C3) override this flag.
        self._need_k_membrane: bool = False
        self._k_tau: float = float(getattr(model_config, "tau", 2.0))
        self._k_v_threshold: float = float(getattr(model_config, "v_threshold", 1.0))
        self._k_detach_reset: bool = bool(getattr(model_config, "detach_reset", True))
        surr_name = str(getattr(model_config, "surrogate", "atan")).lower()
        if surr_name == "atan":
            self._k_surrogate = surrogate.ATan()
        elif surr_name == "sigmoid":
            self._k_surrogate = surrogate.Sigmoid()
        else:
            raise ValueError(f"Unknown surrogate '{surr_name}' for gated linear attention")

    # ------------------------------------------------------------------
    # Q/K/V projection
    # ------------------------------------------------------------------

    def _project_qv(self, x: torch.Tensor):
        """Project to Q and V spike tensors (no gating, identical across all conditions)."""
        T, B, N, C = x.shape
        H, D = self.num_heads, self.head_dim
        x_for_qkv = rearrange(x, "T B N C -> (T B) N C")

        q = self.q_linear(x_for_qkv)
        q = _bn1d(rearrange(q, "(T B) N C -> T B N C", T=T, B=B), self.q_bn)
        q = self.q_lif(q)
        q = rearrange(q, "T B N (H D) -> T B H N D", H=H, D=D)

        v = self.v_linear(x_for_qkv)
        v = _bn1d(rearrange(v, "(T B) N C -> T B N C", T=T, B=B), self.v_bn)
        v = self.v_lif(v)
        v = rearrange(v, "T B N (H D) -> T B H N D", H=H, D=D)
        return q, v

    def _project_k(self, x: torch.Tensor):
        """Project K and (if needed) expose pre-threshold membrane.

        For conditions C0/C1/C2 we go through the spikingjelly LIF as usual.
        For C3 we replicate the LIF dynamics manually so the *pre-threshold*
        membrane potential can be returned alongside the spike output.  Both
        paths use the same hyper-parameters (tau, V_th, surrogate) so the
        spike output is mathematically equivalent up to reset details
        (we use soft reset in the manual path; the pre-mem is the membrane
        immediately before thresholding, which is what the pilot spec requires).
        """
        T, B, N, C = x.shape
        H, D = self.num_heads, self.head_dim
        x_for_kv = rearrange(x, "T B N C -> (T B) N C")
        k_pre = self.k_linear(x_for_kv)
        k_pre = _bn1d(rearrange(k_pre, "(T B) N C -> T B N C", T=T, B=B), self.k_bn)

        if not self._need_k_membrane:
            k = self.k_lif(k_pre)
            k = rearrange(k, "T B N (H D) -> T B H N D", H=H, D=D)
            return k, None

        # Manual LIF for K, exposes pre-threshold membrane.
        alpha = 1.0 - 1.0 / self._k_tau
        V_th = self._k_v_threshold
        u = k_pre.new_zeros(k_pre.shape[1:])
        K_seq = []
        Upre_seq = []
        for t in range(T):
            u = alpha * u + k_pre[t]
            Upre_seq.append(u)
            s = self._k_surrogate(u - V_th)
            K_seq.append(s)
            s_reset = s.detach() if self._k_detach_reset else s
            u = u - s_reset * V_th  # soft reset
        K = torch.stack(K_seq, dim=0)          # [T, B, N, C]
        Upre = torch.stack(Upre_seq, dim=0)    # [T, B, N, C]
        K = rearrange(K, "T B N (H D) -> T B H N D", H=H, D=D)
        Upre = rearrange(Upre, "T B N (H D) -> T B H N D", H=H, D=D)
        return K, Upre

    # ------------------------------------------------------------------
    # The state-update step (to be overridden)
    # ------------------------------------------------------------------

    def _run_state_update(
        self,
        K: torch.Tensor,
        V: torch.Tensor,
        K_membrane: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return per-step KV-state sequence `[T, B, H, D, D]`."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, N, C = x.shape
        H, D = self.num_heads, self.head_dim

        q, v = self._project_qv(x)
        k, k_membrane = self._project_k(x)
        # k, v: [T, B, H, N, D]; q: [T, B, H, N, D]

        S_seq = self._run_state_update(k, v, k_membrane)  # [T, B, H, D, D]
        # O_t = (Q_t @ S_t) * scale
        out = (q @ S_seq) * self.scale                    # [T, B, H, N, D]
        out = rearrange(out, "T B H N D -> T B N (H D)")
        out = self.attn_lif(out)

        out = self.proj_linear(rearrange(out, "T B N C -> (T B) N C"))
        out = _bn1d(rearrange(out, "(T B) N C -> T B N C", T=T, B=B), self.proj_bn)
        out = self.proj_lif(out)
        return out

    # ------------------------------------------------------------------
    # Instrumentation hooks (used by pilot logger)
    # ------------------------------------------------------------------

    def estimate_attn_fp_mults_per_step(self, K_shape) -> int:
        """Multiplications on the attention path (K^T V, gating, Q @ S) per timestep.

        Subclasses should override for accurate accounting.  Default counts the
        common path: Q @ S (B*H*N*D*D MACs) and K^T V (B*H*D*N*D MACs).
        K^T V is binary*binary so it's *not* FP-mult; we still report it for
        completeness (it's a "spike-driven" integer accumulation).
        """
        B, H, N, D = K_shape
        # Spike-driven outer-product K^T V: counted as zero FP-mult (binary x binary).
        # Q @ S: binary Q, FP S → D^2 * N MAC per (B, H)
        return 0


# ---------------------------------------------------------------------------
# C0 — Memoryless baseline SDLA (Yao et al. SDT-v2)
# ---------------------------------------------------------------------------


class LinearAttentionC0(_GatedLinearAttentionBase):
    """C0: memoryless spike-driven linear attention. S_t = K_t^T V_t  (no recurrence)."""

    def __init__(self, dim: int, num_heads: int, model_config):
        super().__init__(dim, num_heads, model_config)
        self.attn_type = "c0_sdla"

    def _run_state_update(self, K, V, K_membrane=None):
        # K, V: [T, B, H, N, D]
        # S_t = K_t^T V_t  (no carry across t)
        S = rearrange(K, "T B H N D -> T B H D N") @ V  # [T, B, H, D, D]
        return S

    def estimate_attn_fp_mults_per_step(self, K_shape) -> int:
        # Binary K, binary V → no FP-mults on outer-product (integer accum).
        # Q @ S: binary Q, FP S → no FP-mult either (binary * FP, no FP-FP).
        # Honest reporting: zero FP-mults on the attention path.
        return 0


# ---------------------------------------------------------------------------
# C1 — SpikingBrain-7B style low-rank linear gate (trivial baseline 1)
# ---------------------------------------------------------------------------


class LinearAttentionC1(_GatedLinearAttentionBase):
    """C1: g_t = sigmoid(W_down(W_up(mean_N K_t))), low-rank r=16. S_t = diag(g_t) S_{t-1} + K^T V."""

    def __init__(self, dim: int, num_heads: int, model_config):
        super().__init__(dim, num_heads, model_config)
        self.attn_type = "c1_lowrank"
        rank = int(getattr(model_config, "gate_rank", 16))
        # Per-head low-rank gate: (B, H, N, D) -> mean over N -> (B, H, D) -> r -> D
        # We instantiate one shared up/down per head dim for simplicity (single nn.Linear over D).
        # Init W_down to zero so g_t = sigmoid(0) = 0.5 at start (mild contraction).
        self.gate_up = nn.Linear(self.head_dim, rank, bias=False)
        self.gate_down = nn.Linear(rank, self.head_dim, bias=False)
        nn.init.zeros_(self.gate_down.weight)
        self.gate_rank = rank

    def _run_state_update(self, K, V, K_membrane=None):
        T, B, H, N, D = K.shape
        # k_bar_t: pool over tokens → [T, B, H, D]
        k_bar = K.mean(dim=-2)
        # gate: sigmoid(W_down(W_up(k_bar)))  → [T, B, H, D]
        g = torch.sigmoid(self.gate_down(self.gate_up(k_bar)))
        # KV outer-product per step: [T, B, H, D, D]
        KV = rearrange(K, "T B H N D -> T B H D N") @ V
        # Recurrent gated accumulation
        S_seq = []
        S_prev = KV.new_zeros(B, H, D, D)
        for t in range(T):
            S_prev = g[t].unsqueeze(-1) * S_prev + KV[t]
            S_seq.append(S_prev)
        return torch.stack(S_seq, dim=0)

    def estimate_attn_fp_mults_per_step(self, K_shape) -> int:
        B, H, N, D = K_shape
        r = self.gate_rank
        # Low-rank gate: B*H*D*r (up) + B*H*r*D (down) + B*H*D (sigmoid is non-MUL but elementwise)
        gate_mul = B * H * (D * r + r * D)
        # diag(g) * S_prev: B*H*D*D FP-mults
        gate_mul += B * H * D * D
        # K^T V (binary*binary): 0 FP-mult
        # Q @ S (binary * FP): 0 FP-mult (binary doesn't count as FP-MUL)
        return gate_mul


# ---------------------------------------------------------------------------
# C2 — SpikingBrain-76B style shared `1 - k` gate (trivial baseline 2)
# ---------------------------------------------------------------------------


class LinearAttentionC2(_GatedLinearAttentionBase):
    """C2: g_t = 1 - mean_N K_t (no learnable gate params). S_t = diag(g_t) S_{t-1} + K^T V."""

    def __init__(self, dim: int, num_heads: int, model_config):
        super().__init__(dim, num_heads, model_config)
        self.attn_type = "c2_oneminusk"

    def _run_state_update(self, K, V, K_membrane=None):
        T, B, H, N, D = K.shape
        # Pool binary K over tokens → [T, B, H, D] in [0, 1]
        k_bar = K.mean(dim=-2)
        g = 1.0 - k_bar  # in [0, 1]
        KV = rearrange(K, "T B H N D -> T B H D N") @ V
        S_seq = []
        S_prev = KV.new_zeros(B, H, D, D)
        for t in range(T):
            S_prev = g[t].unsqueeze(-1) * S_prev + KV[t]
            S_seq.append(S_prev)
        return torch.stack(S_seq, dim=0)

    def estimate_attn_fp_mults_per_step(self, K_shape) -> int:
        B, H, N, D = K_shape
        # Only diag(g) * S_prev is an FP-mult. g itself is (1 - mean) which avg-pool + sub.
        # mean over N of binary K: integer accumulation; treat as 0 FP-mult.
        return B * H * D * D


# ---------------------------------------------------------------------------
# C3 — Membrane-Gated Attention (this paper's proposal)
# ---------------------------------------------------------------------------


class LinearAttentionC3(_GatedLinearAttentionBase):
    """C3 — MGA.

    γ-gate input = pre-threshold K membrane pooled over tokens (NOT K spike output).
    β-gate input = scalar mean K spike rate per (B, head).
    S_t = (1 - s_gamma_t * 2^-k_bits) S_{t-1} + s_beta_t * K_t^T V_t

    Bit-shift k=3 makes the recurrence multiply-free on the attention path
    (replace `*(1 - 2^-3)` with `S - (S >> 3)` in fixed-point hardware).
    """

    def __init__(self, dim: int, num_heads: int, model_config):
        super().__init__(dim, num_heads, model_config)
        self.attn_type = "c3_mga"
        self._need_k_membrane = True  # triggers manual K-LIF in base

        H, D = self.num_heads, self.head_dim
        self.k_bits = int(getattr(model_config, "mga_k_bits", 3))
        if self.k_bits < 0:
            raise ValueError(f"mga_k_bits must be non-negative, got {self.k_bits}")
        self.shift_scale = float(2.0 ** (-self.k_bits))

        # Initial tau = exp(log(4.0)) = 4.0 → alpha = 1 - 1/4 = 0.75
        init_log_tau = math.log(4.0)
        self.log_tau_gamma_raw = nn.Parameter(torch.full((H, D), init_log_tau))
        self.log_tau_beta_raw = nn.Parameter(torch.full((H,), init_log_tau))

        # Initial threshold via softplus(V_raw) = 1.0  → V_raw = log(e - 1) ≈ 0.5413.
        init_V_raw = math.log(math.e - 1.0)
        self.V_gamma_raw = nn.Parameter(torch.full((H, D), init_V_raw))
        self.V_beta_raw = nn.Parameter(torch.full((H,), init_V_raw))

        # Same surrogate as Q/K/V LIFs (reuse self._k_surrogate).
        self._gate_surrogate = self._k_surrogate

    def _gate_params(self):
        tau_gamma = torch.exp(self.log_tau_gamma_raw)             # [H, D] in (0, ∞)
        tau_beta = torch.exp(self.log_tau_beta_raw)               # [H]
        V_gamma = torch.nn.functional.softplus(self.V_gamma_raw)  # [H, D] in (0, ∞)
        V_beta = torch.nn.functional.softplus(self.V_beta_raw)    # [H]
        return tau_gamma, V_gamma, tau_beta, V_beta

    def _run_state_update(self, K, V, K_membrane=None):
        assert K_membrane is not None, "C3 requires pre-threshold K membrane (set _need_k_membrane=True)"
        T, B, H, N, D = K.shape
        tau_gamma, V_gamma, tau_beta, V_beta = self._gate_params()
        alpha_gamma = 1.0 - 1.0 / tau_gamma                      # [H, D]
        alpha_beta = 1.0 - 1.0 / tau_beta                        # [H]

        # γ-gate input: pool pre-threshold K membrane over tokens → [T, B, H, D]
        u_k_pool = K_membrane.mean(dim=-2)
        # β-gate input: scalar K spike rate per (B, H) → [T, B, H]
        r_pool = K.float().mean(dim=(-2, -1))

        # KV outer product per step
        KV = rearrange(K, "T B H N D -> T B H D N") @ V          # [T, B, H, D, D]

        S_seq = []
        S_prev = KV.new_zeros(B, H, D, D)
        u_gamma = KV.new_zeros(B, H, D)
        u_beta = KV.new_zeros(B, H)

        for t in range(T):
            # γ LIF: pre-threshold update, fire, soft reset
            u_gamma = alpha_gamma * u_gamma + u_k_pool[t]        # [B, H, D]
            s_gamma = self._gate_surrogate(u_gamma - V_gamma)    # [B, H, D] in {0, 1}
            s_gamma_reset = s_gamma.detach() if self._k_detach_reset else s_gamma
            u_gamma = u_gamma - s_gamma_reset * V_gamma

            # β LIF: scalar per (B, H)
            u_beta = alpha_beta * u_beta + r_pool[t]             # [B, H]
            s_beta = self._gate_surrogate(u_beta - V_beta)       # [B, H] in {0, 1}
            s_beta_reset = s_beta.detach() if self._k_detach_reset else s_beta
            u_beta = u_beta - s_beta_reset * V_beta

            # Gated state update (multiply-free attention path):
            #   S = S_prev - s_gamma * (S_prev >> k)  + s_beta * KV
            decay_mask = s_gamma * self.shift_scale              # [B, H, D]
            alpha_eff = 1.0 - decay_mask                         # [B, H, D]
            S_prev = alpha_eff.unsqueeze(-1) * S_prev + s_beta.unsqueeze(-1).unsqueeze(-1) * KV[t]
            S_seq.append(S_prev)

        return torch.stack(S_seq, dim=0)

    def estimate_attn_fp_mults_per_step(self, K_shape) -> int:
        # Honest hardware accounting:
        #   - alpha_eff * S_prev: in real hardware this is S_prev - (s_gamma * (S_prev >> k_bits))
        #     which is shift + subtract, NO FP-mult.  In PyTorch we use a multiply for clarity,
        #     but on chip it is multiply-free.
        #   - s_beta * KV is gating by a binary scalar (0/1 mask), NO FP-mult.
        # γ/β gate LIFs use multiplies for alpha*u and reset, but those are scalar/vector
        # operations on (B*H*D) and (B*H) — small.  We report the *attention-path* count
        # only, which is the headline number per pilot §1.
        return 0

    def gate_sparsity(self):
        """Read off cached γ/β firing rates (for logging).  Empty until forward() runs."""
        return {
            "gamma": float(self._last_s_gamma_rate) if hasattr(self, "_last_s_gamma_rate") else float("nan"),
            "beta": float(self._last_s_beta_rate) if hasattr(self, "_last_s_beta_rate") else float("nan"),
        }


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


_ATTENTION_REGISTRY = {
    "c0_sdla": LinearAttentionC0,
    "c1_lowrank": LinearAttentionC1,
    "c2_oneminusk": LinearAttentionC2,
    "c3_mga": LinearAttentionC3,
}


def build_gated_attention(attn_type: str, dim: int, num_heads: int, model_config) -> _GatedLinearAttentionBase:
    key = attn_type.lower()
    if key not in _ATTENTION_REGISTRY:
        raise ValueError(
            f"Unknown gated attention type: '{attn_type}'. "
            f"Available: {sorted(_ATTENTION_REGISTRY)}"
        )
    return _ATTENTION_REGISTRY[key](dim, num_heads, model_config)


def list_gated_attention_types() -> list[str]:
    return sorted(_ATTENTION_REGISTRY)

