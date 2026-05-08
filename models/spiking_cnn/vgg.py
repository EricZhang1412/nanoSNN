from __future__ import annotations

import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, functional

from ..common.layers import ConvBNLIF
from ..common.spike_ops import build_neuron, temporal_mean
from ..common.registry import register_model


class SpikingVGGBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_convs: int, model_config):
        super().__init__()
        convs = []
        for i in range(num_convs):
            convs.append(ConvBNLIF(in_ch if i == 0 else out_ch, out_ch, 3, 1, 1, model_config))
        self.convs = nn.Sequential(*convs)
        self.pool = layer.MaxPool2d(2, 2, step_mode="m")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.convs(x))


_VGG_CONFIGS = {
    "vgg11": [1, 1, 2, 2, 2],
    "vgg13": [2, 2, 2, 2, 2],
    "vgg16": [2, 2, 3, 3, 3],
    "vgg19": [2, 2, 4, 4, 4],
}

_VGG_CHANNELS = [64, 128, 256, 512, 512]


@register_model("spiking_vgg")
class SpikingVGG(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        variant = str(getattr(model_config, "variant", "vgg11")).lower()
        num_classes = int(getattr(model_config, "num_classes", 10))
        self.T = int(getattr(model_config, "T", 4))
        dropout = float(getattr(model_config, "dropout", 0.0))

        num_convs_per_stage = _VGG_CONFIGS[variant]
        channels = _VGG_CHANNELS

        stages = []
        in_ch = int(getattr(model_config, "in_channels", 3))
        for num_convs, out_ch in zip(num_convs_per_stage, channels):
            stages.append(SpikingVGGBlock(in_ch, out_ch, num_convs, model_config))
            in_ch = out_ch
        self.features = nn.Sequential(*stages)

        self.pool = layer.AdaptiveAvgPool2d((1, 1), step_mode="m")

        classifier = [layer.Linear(512, 4096, step_mode="m"),
                      build_neuron(model_config)]
        if dropout > 0:
            classifier.append(layer.Dropout(dropout, step_mode="m"))
        classifier += [layer.Linear(4096, 4096, step_mode="m"),
                       build_neuron(model_config)]
        if dropout > 0:
            classifier.append(layer.Dropout(dropout, step_mode="m"))
        classifier.append(layer.Linear(4096, num_classes, step_mode="m"))
        self.classifier = nn.Sequential(*classifier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, B, C, H, W]
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(2)
        x = self.classifier(x)
        return temporal_mean(x)
