from __future__ import annotations

import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate
from .ilif_ops import MultiSpike

_NEURON_REGISTRY = {
    "lif": neuron.LIFNode,
    "if": neuron.IFNode,
    "plif": neuron.ParametricLIFNode,
    "ilif": MultiSpike,
}

_SURROGATE_REGISTRY = {
    "atan": surrogate.ATan,
    "sigmoid": surrogate.Sigmoid,
}


class _PSPReadout(nn.Module):
    def __init__(self, node: nn.Module, step_mode: str):
        super().__init__()
        self.node = node
        self.step_mode = step_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ = self.node(x)
        if self.step_mode == "m":
            v_seq = getattr(self.node, "v_seq", None)
            if v_seq is None:
                raise RuntimeError("PSP readout in multi-step mode requires store_v_seq=True")
            return v_seq
        return self.node.v


def build_neuron(model_config, step_mode: str = "m", v_threshold: float | None = None, output_mode: str | None = None) -> nn.Module:
    neuron_type = str(getattr(model_config, "neuron_type", "lif")).lower()
    surrogate_type = str(getattr(model_config, "surrogate", "atan")).lower()
    tau = float(getattr(model_config, "tau", 2.0))
    if v_threshold is None:
        v_threshold = float(getattr(model_config, "v_threshold", 1.0))
    detach_reset = bool(getattr(model_config, "detach_reset", True))
    if output_mode is None:
        output_mode = str(getattr(model_config, "neuron_output", "spike")).lower()
    else:
        output_mode = output_mode.lower()
    if output_mode not in {"spike", "psp"}:
        raise ValueError(f"Unknown output mode: {output_mode}")

    if neuron_type == "ilif":
        if output_mode == "psp":
            raise ValueError("neuron_type=ilif does not support output_mode='psp'")
        return _NEURON_REGISTRY[neuron_type]()

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

    if output_mode == "psp":
        if step_mode == "m":
            kwargs["store_v_seq"] = True
        return _PSPReadout(cls(**kwargs), step_mode=step_mode)

    return cls(**kwargs)


def temporal_mean(x: torch.Tensor) -> torch.Tensor:
    """Aggregate [T, B, C] or [T, B, C, H, W] -> [B, C, ...]."""
    return x.mean(dim=0)


def expand_static_to_temporal(x: torch.Tensor, T: int) -> torch.Tensor:
    """Repeat [B, C, H, W] -> [T, B, C, H, W]."""
    return x.unsqueeze(0).repeat(T, 1, 1, 1, 1)
