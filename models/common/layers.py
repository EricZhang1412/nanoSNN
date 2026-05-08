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

class BNAndPadLayer(nn.Module):
    def __init__(
        self,
        pad_pixels,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super(BNAndPadLayer, self).__init__()
        self.bn = nn.BatchNorm2d(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.pad_pixels = pad_pixels

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            if self.bn.affine:
                pad_values = (
                    self.bn.bias.detach()
                    - self.bn.running_mean
                    * self.bn.weight.detach()
                    / torch.sqrt(self.bn.running_var + self.bn.eps)
                )
            else:
                pad_values = -self.bn.running_mean / torch.sqrt(
                    self.bn.running_var + self.bn.eps
                )
            output = F.pad(output, [self.pad_pixels] * 4)
            pad_values = pad_values.view(1, -1, 1, 1)
            output[:, :, 0 : self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels :, :] = pad_values
            output[:, :, :, 0 : self.pad_pixels] = pad_values
            output[:, :, :, -self.pad_pixels :] = pad_values
        return output

    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def running_mean(self):
        return self.bn.running_mean

    @property
    def running_var(self):
        return self.bn.running_var

    @property
    def eps(self):
        return self.bn.eps

class RepConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        bias=False,
    ):
        super().__init__()
        # hidden_channel = in_channel
        conv1x1 = nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=False, groups=1)
        bn = BNAndPadLayer(pad_pixels=1, num_features=in_channel)
        conv3x3 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 0, groups=in_channel, bias=False),
            nn.Conv2d(in_channel, out_channel, 1, 1, 0, groups=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )

        self.body = nn.Sequential(conv1x1, bn, conv3x3)

    def forward(self, x):
        return self.body(x)

class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(
        self,
        dim,
        expansion_ratio=2,
        act2_layer=nn.Identity,
        bias=False,
        kernel_size=7,
        padding=3,
    ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.spike1 = MultiSpike()
        self.pwconv1 = nn.Conv2d(dim, med_channels, kernel_size=1, stride=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(med_channels)
        self.spike2 = MultiSpike()
        self.dwconv = nn.Conv2d(
            med_channels,
            med_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=med_channels,
            bias=bias,
        )  # depthwise conv
        self.pwconv2 = nn.Conv2d(med_channels, dim, kernel_size=1, stride=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        
        x = self.spike1(x)
            
        x = self.bn1(self.pwconv1(x))
        
        x = self.spike2(x)
            
        x = self.dwconv(x)
        x = self.bn2(self.pwconv2(x))
        return x

class SepConv_Spike(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(
        self,
        dim,
        expansion_ratio=2,
        act2_layer=nn.Identity,
        bias=False,
        kernel_size=7,
        padding=3,
        model_config=None,
    ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.spike1 = build_neuron(model_config)
        self.pwconv1 = nn.Sequential(
            nn.Conv2d(dim, med_channels, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(med_channels)
            )
        self.spike2 = build_neuron(model_config)
        self.dwconv = nn.Sequential(
            nn.Conv2d(med_channels, med_channels, kernel_size=kernel_size, padding=padding, groups=med_channels, bias=bias),
            nn.BatchNorm2d(med_channels)
        )
        self.spike3 = build_neuron(model_config)
        self.pwconv2 = nn.Sequential(
            nn.Conv2d(med_channels, dim, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        x = self.spike1(x)
        x = self.pwconv1(x)

        x = self.spike2(x)
        x = self.dwconv(x)

        x = self.spike3(x)
        x = self.pwconv2(x)
        return x

class MS_ConvBlock(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        model_config=None,
    ):
        super().__init__()

        self.Conv = SepConv_Spike(dim=dim, model_config=model_config)

        self.mlp_ratio = mlp_ratio

        self.spike1 = build_neuron(model_config)
        self.conv1 = nn.Conv2d(
            dim, dim * mlp_ratio, kernel_size=3, padding=1, groups=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(dim * mlp_ratio)  # 这里可以进行改进
        self.spike2 = build_neuron(model_config)
        self.conv2 = nn.Conv2d(
            dim * mlp_ratio, dim, kernel_size=3, padding=1, groups=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(dim)  # 这里可以进行改进

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.Conv(x) + x
        x_feat = x
        x = self.spike1(x)
        x = self.bn1(self.conv1(x)).reshape(B, self.mlp_ratio * C, H, W)
        x = self.spike2(x)
        x = self.bn2(self.conv2(x)).reshape(B, C, H, W)
        x = x_feat + x

        return x

class MS_ConvBlock_spike_SepConv(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        model_config=None,
    ):
        super().__init__()

        self.Conv = SepConv_Spike(dim=dim, model_config=model_config)

        self.mlp_ratio = mlp_ratio

        self.spike1 = build_neuron(model_config)
        self.conv1 = nn.Conv2d(
            dim, dim * mlp_ratio, kernel_size=3, padding=1, groups=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(dim * mlp_ratio) 
        self.spike2 = build_neuron(model_config)
        self.conv2 = nn.Conv2d(
            dim * mlp_ratio, dim, kernel_size=3, padding=1, groups=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(dim)  

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.Conv(x) + x
        x_feat = x
        x = self.spike1(x)
        x = self.bn1(self.conv1(x)).reshape(B, self.mlp_ratio * C, H, W)
        x = self.spike2(x)
        x = self.bn2(self.conv2(x)).reshape(B, C, H, W)
        x = x_feat + x

        return x

class MS_DownSampling(nn.Module):
    def __init__(
        self,
        in_channels=2,
        embed_dims=256,
        kernel_size=3,
        stride=2,
        padding=1,
        first_layer=True,
        T=None,
        model_config=None,
    ):
        super().__init__()

        self.encode_conv = nn.Conv2d(
            in_channels,
            embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.encode_bn = nn.BatchNorm2d(embed_dims)
        self.first_layer = first_layer
        if not first_layer:
            self.encode_spike = build_neuron(model_config)

    def forward(self, x):

        if hasattr(self, "encode_spike"):
            x = self.encode_spike(x)
        x = self.encode_conv(x)
        x = self.encode_bn(x)

        return x