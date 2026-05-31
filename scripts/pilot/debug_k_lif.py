"""Compare K-LIF spike statistics between C0/C2 (spikingjelly LIF) and C3
(custom manual LIF).  If isolation holds, K spike rate should be identical
(modulo tiny numerical noise).  If C3's K fires much more, we have a bug.
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
        detach_reset=True, surrogate="atan",
    )
    T, B, N, C, H = 16, 4, 64, 256, 4
    torch.manual_seed(0)
    x = (torch.rand(T, B, N, C) > 0.7).float()

    # Build C0 and C3 with IDENTICAL Q/K/V weights.
    c0 = ga.LinearAttentionC0(C, H, cfg)
    c3 = ga.LinearAttentionC3(C, H, cfg)
    # Copy backbone weights c0 -> c3 so the only difference is the K-LIF path
    backbone_keys = ["q_linear", "q_bn", "k_linear", "k_bn", "v_linear", "v_bn",
                     "proj_linear", "proj_bn"]
    for k in backbone_keys:
        getattr(c3, k).load_state_dict(getattr(c0, k).state_dict())

    # Hook to capture K spikes inside _project_k
    def capture_k(attn, name, x):
        functional.reset_net(attn)
        with torch.no_grad():
            k, u_pre = attn._project_k(x)
        rate = k.float().mean().item()
        print(f"  [{name}]  K spike rate = {rate:.4f}   "
              f"K shape = {tuple(k.shape)}   "
              f"u_pre None? {u_pre is None}")
        return rate, k

    print("=== K-LIF isolation check (C0 vs C3 with same weights) ===")
    r0, k0 = capture_k(c0, "C0 (spikingjelly LIF)", x)
    r3, k3 = capture_k(c3, "C3 (manual LIF)     ", x)
    ratio = r3 / max(r0, 1e-8)
    print(f"\n  C3/C0 ratio = {ratio:.3f}   (expected ≈ 1.0 if isolated)")
    # Pairwise diff
    diff = (k0 != k3).float().mean().item()
    print(f"  fraction of K bits that disagree: {diff:.4f}")

    # Also report K^T V magnitude
    KtV_c0 = (k0.transpose(-1, -2) @ k0).float().sum().item()
    KtV_c3 = (k3.transpose(-1, -2) @ k3).float().sum().item()
    print(f"  ||K^T K||_sum  C0={KtV_c0:.1f}  C3={KtV_c3:.1f}   ratio={KtV_c3 / max(KtV_c0,1e-8):.2f}")


if __name__ == "__main__":
    main()
