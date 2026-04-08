from __future__ import annotations

import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate
from .ilif_ops import MultiSpike
from typing import NamedTuple

_NEURON_REGISTRY = {
    "lif":  neuron.LIFNode,
    "if":   neuron.IFNode,
    "plif": neuron.ParametricLIFNode,
    "ilif": MultiSpike,
}

_SURROGATE_REGISTRY = {
    "atan":    surrogate.ATan,
    "sigmoid": surrogate.Sigmoid,
}


# ---------------------------------------------------------------------------
# Output containers
# ---------------------------------------------------------------------------

class DualOutput(NamedTuple):
    spike: torch.Tensor   # binary spikes,    same shape as input
    v_seq: torch.Tensor   # membrane potential, same shape as input


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------

class _PSPReadout(nn.Module):
    """Legacy wrapper: returns membrane potential only."""
    def __init__(self, node: nn.Module, step_mode: str):
        super().__init__()
        self.node = node
        self.step_mode = step_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ = self.node(x)
        if self.step_mode == "m":
            v_seq = getattr(self.node, "v_seq", None)
            if v_seq is None:
                raise RuntimeError(
                    "PSP readout in multi-step mode requires store_v_seq=True"
                )
            return v_seq
        return self.node.v


class _DualReadout(nn.Module):
    """Returns both spike and membrane potential as a DualOutput named tuple.

    Requires the underlying node to be built with store_v_seq=True
    and step_mode='m'.
    """
    def __init__(self, node: nn.Module):
        super().__init__()
        self.node = node

    def forward(self, x: torch.Tensor) -> DualOutput:
        spike = self.node(x)
        v_seq = getattr(self.node, "v_seq", None)
        if v_seq is None:
            raise RuntimeError(
                "_DualReadout requires the node to be built with store_v_seq=True"
            )
        return DualOutput(spike=spike, v_seq=v_seq)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_neuron(
    model_config,
    step_mode: str = "m",
    v_threshold: float | None = None,
    output_mode: str | None = None,
) -> nn.Module:
    """Build a spiking neuron node.

    output_mode
    -----------
    "spike"  — (default) returns binary spike tensor  [T, ...]
    "psp"    — returns membrane potential tensor       [T, ...]
    "dual"   — returns DualOutput(spike, v_seq),       both [T, ...]
               use this when you need both outputs from the same node.
    """
    neuron_type    = str(getattr(model_config, "neuron_type",    "lif")).lower()
    surrogate_type = str(getattr(model_config, "surrogate",      "atan")).lower()
    tau            = float(getattr(model_config, "tau",          2.0))
    detach_reset   = bool(getattr(model_config, "detach_reset",  True))

    if v_threshold is None:
        v_threshold = float(getattr(model_config, "v_threshold", 1.0))

    if output_mode is None:
        output_mode = str(getattr(model_config, "neuron_output", "spike")).lower()
    else:
        output_mode = output_mode.lower()

    if output_mode not in {"spike", "psp", "dual"}:
        raise ValueError(f"Unknown output_mode: {output_mode!r}")

    # ilif has no surrogate / tau kwargs and doesn't support psp/dual
    if neuron_type == "ilif":
        if output_mode != "spike":
            raise ValueError("neuron_type=ilif only supports output_mode='spike'")
        return _NEURON_REGISTRY["ilif"]()

    if surrogate_type not in _SURROGATE_REGISTRY:
        raise ValueError(f"Unknown surrogate: {surrogate_type!r}")
    if neuron_type not in _NEURON_REGISTRY:
        raise ValueError(f"Unknown neuron type: {neuron_type!r}")

    surr_fn = _SURROGATE_REGISTRY[surrogate_type]()
    cls     = _NEURON_REGISTRY[neuron_type]

    kwargs = dict(
        v_threshold=v_threshold,
        surrogate_function=surr_fn,
        detach_reset=detach_reset,
        step_mode=step_mode,
    )
    if neuron_type in {"lif", "plif"}:
        kwargs["tau"] = tau

    if output_mode == "spike":
        return cls(**kwargs)

    # psp and dual both need store_v_seq
    kwargs["store_v_seq"] = True
    node = cls(**kwargs)

    if output_mode == "psp":
        return _PSPReadout(node, step_mode)

    # dual
    return _DualReadout(node)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def temporal_mean(x: torch.Tensor) -> torch.Tensor:
    """Aggregate [T, B, C] or [T, B, C, H, W] -> [B, C, ...]."""
    return x.mean(dim=0)


def expand_static_to_temporal(x: torch.Tensor, T: int) -> torch.Tensor:
    """Repeat [B, C, H, W] -> [T, B, C, H, W]."""
    return x.unsqueeze(0).repeat(T, 1, 1, 1, 1)