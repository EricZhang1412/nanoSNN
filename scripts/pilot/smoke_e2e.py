"""End-to-end smoke test for Spikformer + SpikformerAudio with all 4 attention types.

Loads each model_config YAML, instantiates the model, runs a forward pass on a
random binary input of the expected shape, and reports total params + output
shape + finiteness.  Intended to catch wiring errors before launching the full
pilot on a cluster.

Run:  uv run python -m scripts.pilot.smoke_e2e
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _REPO_ROOT)


def _load_spikformer_models():
    """Direct-import models/spikformer/model.py without going through
    `models/__init__.py` (which forces a `lightning` import we don't need here).
    """
    if "models.spikformer.model" in sys.modules:
        return sys.modules["models.spikformer.model"]

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
    _load("models.common.registry", os.path.join(_REPO_ROOT, "models/common/registry.py"))
    _load("models.spikformer.gate_attention", os.path.join(_REPO_ROOT, "models/spikformer/gate_attention.py"))
    return _load("models.spikformer.model", os.path.join(_REPO_ROOT, "models/spikformer/model.py"))


def _load_config(path: str) -> SimpleNamespace:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)

    def to_ns(v):
        if isinstance(v, dict):
            return SimpleNamespace(**{k: to_ns(x) for k, x in v.items()})
        if isinstance(v, list):
            return [to_ns(x) for x in v]
        return v

    return to_ns(d)


def smoke_dvs():
    print("\n=== Spikformer (DVS128 backbone) ===")
    models_mod = _load_spikformer_models()
    Spikformer = models_mod.Spikformer

    from spikingjelly.activation_based import functional

    # Small toy input so it's fast: T=4, B=2, C=2, H=W=32 (Spikformer's SPS does 4x downsample → 8x8 = 64 tokens)
    T, B = 4, 2
    in_channels, img_size = 2, 32

    for cfg_name in [
        "configs/model_configs/pilot/spikformer_dvs_c0_sdla.yaml",
        "configs/model_configs/pilot/spikformer_dvs_c1_lowrank.yaml",
        "configs/model_configs/pilot/spikformer_dvs_c2_oneminusk.yaml",
        "configs/model_configs/pilot/spikformer_dvs_c3_mga.yaml",
    ]:
        cfg = _load_config(os.path.join(_REPO_ROOT, cfg_name))
        # Override to a tiny variant for the smoke test (don't run T=16 @ 128x128 on CPU).
        cfg.T = T
        cfg.image_size = img_size
        cfg.in_channels = in_channels
        cfg.embed_dim = 64
        cfg.num_heads = 4
        cfg.depth = 2
        model = Spikformer(cfg)
        x = (torch.rand(T, B, in_channels, img_size, img_size) > 0.7).float()
        functional.reset_net(model)
        with torch.no_grad():
            out = model(x)
        n_params = sum(p.numel() for p in model.parameters())
        finite = bool(torch.isfinite(out).all().item())
        print(f"  {cfg.attention_type:18s}  out.shape={tuple(out.shape)}  "
              f"finite={finite}  params={n_params:,}")


def smoke_shd():
    print("\n=== SpikformerAudio (SHD backbone) ===")
    models_mod = _load_spikformer_models()
    SpikformerAudio = models_mod.SpikformerAudio

    from spikingjelly.activation_based import functional

    T, B = 4, 2

    for cfg_name in [
        "configs/model_configs/pilot/spikformer_shd_c0_sdla.yaml",
        "configs/model_configs/pilot/spikformer_shd_c1_lowrank.yaml",
        "configs/model_configs/pilot/spikformer_shd_c2_oneminusk.yaml",
        "configs/model_configs/pilot/spikformer_shd_c3_mga.yaml",
    ]:
        cfg = _load_config(os.path.join(_REPO_ROOT, cfg_name))
        # Shrink to keep CPU smoke test cheap.
        cfg.T = T
        cfg.embed_dim = 64
        cfg.num_heads = 4
        cfg.depth = 2
        cfg.n_in = 700
        cfg.token_count = 70
        model = SpikformerAudio(cfg)
        x = (torch.rand(T, B, 1, 700) > 0.95).float()  # sparse spikes
        functional.reset_net(model)
        with torch.no_grad():
            out = model(x)
        n_params = sum(p.numel() for p in model.parameters())
        finite = bool(torch.isfinite(out).all().item())
        print(f"  {cfg.attention_type:18s}  out.shape={tuple(out.shape)}  "
              f"finite={finite}  params={n_params:,}")


def main():
    smoke_dvs()
    smoke_shd()
    print("\nE2E smoke test passed.")


if __name__ == "__main__":
    main()
