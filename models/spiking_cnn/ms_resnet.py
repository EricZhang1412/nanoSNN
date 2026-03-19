from __future__ import annotations

import torch
import torch.nn as nn
from spikingjelly.activation_based import layer

from ..common.layers import ConvBNLIF
from ..common.spike_ops import build_neuron, temporal_mean
from ..common.registry import register_model


class MSResBlock(nn.Module):
    """Multi-Scale Residual Block from MS-ResNet (Hu et al., 2024).

    Adds a 1x1 shortcut branch alongside the standard 3x3 residual path,
    enabling multi-scale feature reuse without extra spike operations.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, model_config=None):
        super().__init__()
        self.conv1 = ConvBNLIF(in_ch, out_ch, 3, stride, 1, model_config)
        self.conv2 = ConvBNLIF(out_ch, out_ch, 3, 1, 1, model_config)

        self.shortcut_3x3 = None
        if stride != 1 or in_ch != out_ch:
            self.shortcut_3x3 = nn.Sequential(
                layer.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False, step_mode="m"),
                layer.BatchNorm2d(out_ch, step_mode="m"),
            )

        self.shortcut_1x1 = nn.Sequential(
            layer.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False, step_mode="m"),
            layer.BatchNorm2d(out_ch, step_mode="m"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity_3x3 = self.shortcut_3x3(x) if self.shortcut_3x3 is not None else x
        identity_1x1 = self.shortcut_1x1(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return out + identity_3x3 + identity_1x1


_MSRESNET_CONFIGS = {
    "ms_resnet18": [2, 2, 2, 2],
    "ms_resnet34": [3, 4, 6, 3],
}


@register_model("ms_resnet")
class MSResNet(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        variant = str(getattr(model_config, "variant", "ms_resnet18")).lower()
        num_classes = int(getattr(model_config, "num_classes", 10))
        self.T = int(getattr(model_config, "T", 4))
        in_channels = int(getattr(model_config, "in_channels", 3))

        layers_cfg = _MSRESNET_CONFIGS[variant]

        self.stem = ConvBNLIF(in_channels, 64, 3, 1, 1, model_config)
        self.layer1 = self._make_layer(64, 64, layers_cfg[0], 1, model_config)
        self.layer2 = self._make_layer(64, 128, layers_cfg[1], 2, model_config)
        self.layer3 = self._make_layer(128, 256, layers_cfg[2], 2, model_config)
        self.layer4 = self._make_layer(256, 512, layers_cfg[3], 2, model_config)

        self.pool = layer.AdaptiveAvgPool2d((1, 1), step_mode="m")
        self.head = layer.Linear(512, num_classes, step_mode="m")

    @staticmethod
    def _make_layer(in_ch: int, out_ch: int, num_blocks: int, stride: int, model_config):
        blocks = [MSResBlock(in_ch, out_ch, stride, model_config)]
        for _ in range(1, num_blocks):
            blocks.append(MSResBlock(out_ch, out_ch, 1, model_config))
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = x.flatten(2)
        x = self.head(x)
        return temporal_mean(x)
