from __future__ import annotations

import torch
import torch.nn as nn
from spikingjelly.activation_based import layer

from .spike_ops import build_neuron


class ConvBNLIF(nn.Module):
    """Conv2d -> BN -> LIF (multi-step mode)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, model_config=None):
        super().__init__()
        self.conv = layer.Conv2d(in_channels, out_channels, kernel_size,
                                 stride=stride, padding=padding, bias=False, step_mode="m")
        self.bn = layer.BatchNorm2d(out_channels, step_mode="m")
        self.lif = build_neuron(model_config) if model_config is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        if self.lif is not None:
            x = self.lif(x)
        return x


class SpikeLinear(nn.Module):
    """Linear -> BN1d -> LIF (multi-step mode)."""

    def __init__(self, in_features: int, out_features: int, model_config=None):
        super().__init__()
        self.linear = layer.Linear(in_features, out_features, bias=False, step_mode="m")
        self.bn = nn.BatchNorm1d(out_features)
        self.lif = build_neuron(model_config) if model_config is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, B, C]
        T, B, C = x.shape
        x = self.linear(x)
        x_flat = x.reshape(T * B, -1)
        x_flat = self.bn(x_flat)
        x = x_flat.reshape(T, B, -1)
        if self.lif is not None:
            x = self.lif(x)
        return x
