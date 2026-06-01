"""Quantify γ/β gate firing rates and S magnitude at INIT for C3.

If both gates are dead (s_γ, s_β ≈ 0) at init, the recurrence degenerates:
  α_eff ≈ 1 (no decay)  AND  s_β ≈ 0 (no writes)  →  S stuck at 0
or
  α_eff ≈ 1 (no decay)  AND  s_β ≈ 1 (writes)    →  S accumulates → attn_lif saturates.

Either way C3 cannot learn.  This script reports the per-timestep gate stats
and S-norm trajectory so we know whether init is the culprit.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from types import SimpleNamespace

import torch
from spikingjelly.activation_based import functional


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _REPO_ROOT)


def _load_ga():
    if "models.spikformer.gate_attention" in sys.modules:
        return sys.modules["models.spikformer.gate_attention"]
    for pkg in ("models", "models.common", "models.spikformer"):
        if pkg not in sys.modules:
            sys.modules[pkg] = types.ModuleType(pkg)

    def _load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    _load("models.common.ilif_ops", os.path.join(_REPO_ROOT, "models/common/ilif_ops.py"))
    _load("models.common.spike_ops", os.path.join(_REPO_ROOT, "models/common/spike_ops.py"))
    return _load("models.spikformer.gate_attention", os.path.join(_REPO_ROOT, "models/spikformer/gate_attention.py"))


def main():
    ga = _load_ga()
    cfg = SimpleNamespace(
        neuron_type="lif", tau=2.0, v_threshold=1.0,
        detach_reset=True, surrogate="atan", mga_k_bits=3,
    )
    T, B, N, C, H = 16, 4, 64, 256, 4
    D = C // H
    torch.manual_seed(0)
    x = (torch.rand(T, B, N, C) > 0.7).float()

    c3 = ga.LinearAttentionC3(C, H, cfg)
    functional.reset_net(c3)

    # Instrument _run_state_update to record gate rates and S magnitudes.
    rates_g, rates_b, snorms = [], [], []
    u_k_pool_mags = []
    orig = c3._run_state_update

    def wrapped(K, V, K_membrane):
        T_, B_, H_, N_, D_ = K.shape
        tau_g, V_g, tau_b, V_b = c3._gate_params()
        alpha_g = 1.0 - 1.0 / tau_g
        alpha_b = 1.0 - 1.0 / tau_b
        u_k_pool = K_membrane.mean(dim=-2)        # [T, B, H, D]
        if c3.gate_input_norm is not None:
            u_k_pool = c3.gate_input_norm(u_k_pool)
        r_pool = K.float().mean(dim=(-2, -1))     # [T, B, H]
        KV = K.transpose(-1, -2) @ V              # [T, B, H, D, D]
        S = KV.new_zeros(B_, H_, D_, D_)
        u_g = KV.new_zeros(B_, H_, D_)
        u_b = KV.new_zeros(B_, H_)
        for t in range(T_):
            u_g = alpha_g * u_g + u_k_pool[t]
            s_g = c3._gate_surrogate(u_g - V_g)
            u_g = u_g - s_g.detach() * V_g
            u_b = alpha_b * u_b + r_pool[t]
            s_b = c3._gate_surrogate(u_b - V_b)
            u_b = u_b - s_b.detach() * V_b
            alpha_eff = 1.0 - s_g * c3.shift_scale
            S = alpha_eff.unsqueeze(-1) * S + s_b.unsqueeze(-1).unsqueeze(-1) * KV[t]
            rates_g.append(s_g.float().mean().item())
            rates_b.append(s_b.float().mean().item())
            u_k_pool_mags.append(u_k_pool[t].abs().mean().item())
            snorms.append(S.float().pow(2).sum().sqrt().item())

        # Diagnostic: signed stats of γ-gate input + state of u_gamma
        print(f"\n  signed u_k_pool: mean={u_k_pool.mean().item():+.4f}  "
              f"std={u_k_pool.std().item():.4f}  "
              f"P(u_k_pool>V_γ)≈{(u_k_pool > V_g.unsqueeze(0).unsqueeze(0)).float().mean().item():.4f}")
        print(f"  u_gamma steady-state estimate: mean(u_k_pool) / (1 - α_γ) "
              f"= {(u_k_pool.mean() / (1 - alpha_g)).mean().item():+.4f}")
        return orig(K, V, K_membrane)

    c3._run_state_update = wrapped
    with torch.no_grad():
        _ = c3(x)
    c3._run_state_update = orig

    tau_g, V_g, tau_b, V_b = c3._gate_params()
    print("=== C3 gate init diagnostic ===")
    print(f"  config: T={T} B={B} N={N} C={C} H={H} D={D}")
    print(f"  tau_gamma mean = {tau_g.mean().item():.3f}   V_gamma mean = {V_g.mean().item():.3f}")
    print(f"  tau_beta  mean = {tau_b.mean().item():.3f}   V_beta  mean = {V_b.mean().item():.3f}")
    print(f"  k_bits = {c3.k_bits}  → shift_scale = {c3.shift_scale}")
    print()
    print(f"  γ-rate per t: {[f'{r:.3f}' for r in rates_g]}")
    print(f"  β-rate per t: {[f'{r:.3f}' for r in rates_b]}")
    print(f"  |u_k_pool| per t: {[f'{m:.3f}' for m in u_k_pool_mags]}")
    print(f"  ||S||_F per t: {[f'{s:.2f}' for s in snorms]}")
    print()
    print(f"  γ-rate mean = {sum(rates_g)/T:.4f}")
    print(f"  β-rate mean = {sum(rates_b)/T:.4f}")
    print(f"  ||S|| at last step = {snorms[-1]:.2f}")
    # Compute approximate decay over T given mean γ-rate:
    g_rate_mean = sum(rates_g) / T
    retention = (1 - g_rate_mean * c3.shift_scale) ** T
    print(f"  approximate T-step retention factor = {retention:.4f}")
    if g_rate_mean < 0.05:
        print("  ⚠️  γ-gate effectively dead — S has near-zero decay.")
    if sum(rates_b) / T < 0.05:
        print("  ⚠️  β-gate effectively dead — no writes to S → S stuck at ≈ 0.")
    if snorms[-1] / max(snorms[0], 1e-6) > 5:
        print("  ⚠️  ||S|| grew >5× — likely saturates attn_lif (v_th=0.5) downstream.")


if __name__ == "__main__":
    main()
