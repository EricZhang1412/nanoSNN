from __future__ import annotations

import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate


_NEURON_REGISTRY = {
    "lif": neuron.LIFNode,
    "if": neuron.IFNode,
    "plif": neuron.ParametricLIFNode,
}

_SURROGATE_REGISTRY = {
    "atan": surrogate.ATan,
    "sigmoid": surrogate.Sigmoid,
}


def build_neuron(model_config, step_mode: str = "m") -> nn.Module:
    neuron_type = str(getattr(model_config, "neuron_type", "lif")).lower()
    surrogate_type = str(getattr(model_config, "surrogate", "atan")).lower()
    tau = float(getattr(model_config, "tau", 2.0))
    v_threshold = float(getattr(model_config, "v_threshold", 1.0))
    detach_reset = bool(getattr(model_config, "detach_reset", True))

    if surrogate_type not in _SURROGATE_REGISTRY:
        raise ValueError(f"Unknown surrogate: {surrogate_type}")
    surr_fn = _SURROGATE_REGISTRY[surrogate_type]()

    if neuron_type not in _NEURON_REGISTRY:
        raise ValueError(f"Unknown neuron type: {neuron_type}")
    cls = _NEURON_REGISTRY[neuron_type]

    kwargs = dict(
        v_threshold=v_threshold,
        surrogate_function=surr_fn,
        detach_reset=detach_reset,
        step_mode=step_mode,
    )
    if neuron_type in {"lif", "plif"}:
        kwargs["tau"] = tau

    return cls(**kwargs)


def temporal_mean(x: torch.Tensor) -> torch.Tensor:
    """Aggregate [T, B, C] or [T, B, C, H, W] -> [B, C, ...]."""
    return x.mean(dim=0)


def expand_static_to_temporal(x: torch.Tensor, T: int) -> torch.Tensor:
    """Repeat [B, C, H, W] -> [T, B, C, H, W]."""
    return x.unsqueeze(0).repeat(T, 1, 1, 1, 1)
