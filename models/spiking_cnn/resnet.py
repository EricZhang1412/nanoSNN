from __future__ import annotations

import torch
import torch.nn as nn
from spikingjelly.activation_based import layer

from ..common.layers import ConvBNLIF
from ..common.spike_ops import build_neuron, temporal_mean
from ..common.registry import register_model


class SpikingBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, model_config=None):
        super().__init__()
        self.conv1 = ConvBNLIF(in_ch, out_ch, 3, stride, 1, model_config)
        self.conv2 = ConvBNLIF(out_ch, out_ch, 3, 1, 1, model_config)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                layer.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False, step_mode="m"),
                layer.BatchNorm2d(out_ch, step_mode="m"),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return out + identity


_RESNET_CONFIGS = {
    "resnet18": [2, 2, 2, 2],
    "resnet34": [3, 4, 6, 3],
}


@register_model("spiking_resnet")
class SpikingResNet(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        variant = str(getattr(model_config, "variant", "resnet18")).lower()
        num_classes = int(getattr(model_config, "num_classes", 10))
        self.T = int(getattr(model_config, "T", 4))
        in_channels = int(getattr(model_config, "in_channels", 3))

        layers_cfg = _RESNET_CONFIGS[variant]
        channels = [64, 128, 256, 512]

        self.stem = ConvBNLIF(in_channels, 64, 3, 1, 1, model_config)

        self.layer1 = self._make_layer(64, 64, layers_cfg[0], 1, model_config)
        self.layer2 = self._make_layer(64, 128, layers_cfg[1], 2, model_config)
        self.layer3 = self._make_layer(128, 256, layers_cfg[2], 2, model_config)
        self.layer4 = self._make_layer(256, 512, layers_cfg[3], 2, model_config)

        self.pool = layer.AdaptiveAvgPool2d((1, 1), step_mode="m")
        self.head = layer.Linear(512, num_classes, step_mode="m")

    @staticmethod
    def _make_layer(in_ch: int, out_ch: int, num_blocks: int, stride: int, model_config):
        blocks = [SpikingBasicBlock(in_ch, out_ch, stride, model_config)]
        for _ in range(1, num_blocks):
            blocks.append(SpikingBasicBlock(out_ch, out_ch, 1, model_config))
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
