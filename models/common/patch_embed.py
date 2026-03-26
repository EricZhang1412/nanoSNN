from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

from .spike_ops import build_neuron


class SDTv1PatchSpliting(nn.Module):
    def __init__(
        self,
        img_size_h: int = 128,
        img_size_w: int = 128,
        patch_size: int | tuple[int, int] = 4,
        in_channels: int = 2,
        embed_dims: int = 256,
        pooling_stat: str = "1111",
        spike_mode: str = "lif",
        model_config=None,
    ):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.pooling_stat = pooling_stat
        self.spike_mode = spike_mode

        self.proj_conv = nn.Conv2d(in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims // 8)
        self.proj_lif = build_neuron(model_config)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv1 = nn.Conv2d(embed_dims // 8, embed_dims // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn1 = nn.BatchNorm2d(embed_dims // 4)
        self.proj_lif1 = build_neuron(model_config)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv2 = nn.Conv2d(embed_dims // 4, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(embed_dims // 2)
        self.proj_lif2 = build_neuron(model_config)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv3 = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        self.proj_lif3 = build_neuron(model_config)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.rpe_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dims)

        self.image_size = [img_size_h, img_size_w]
        self.H = self.image_size[0] // self.patch_size[0]
        self.W = self.image_size[1] // self.patch_size[1]
        self.num_patches = self.H * self.W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, _, H, W = x.shape
        ratio = 1

        x = self.proj_conv(rearrange(x, "T B C H W -> (T B) C H W"))
        x = rearrange(self.proj_bn(x), "(T B) C h w -> T B C h w", T=T, B=B)
        x = self.proj_lif(x)
        x = rearrange(x, "T B C h w -> (T B) C h w")
        if self.pooling_stat[0] == "1":
            x = self.maxpool(x)
            ratio *= 2

        x = self.proj_conv1(x)
        x = rearrange(self.proj_bn1(x), "(T B) C h w -> T B C h w", T=T, B=B)
        x = self.proj_lif1(x)
        x = rearrange(x, "T B C h w -> (T B) C h w")
        if self.pooling_stat[1] == "1":
            x = self.maxpool1(x)
            ratio *= 2

        x = self.proj_conv2(x)
        x = rearrange(self.proj_bn2(x), "(T B) C h w -> T B C h w", T=T, B=B)
        x = self.proj_lif2(x)
        x = rearrange(x, "T B C h w -> (T B) C h w")
        if self.pooling_stat[2] == "1":
            x = self.maxpool2(x)
            ratio *= 2

        x = self.proj_conv3(x)
        x = self.proj_bn3(x)
        if self.pooling_stat[3] == "1":
            x = self.maxpool3(x)
            ratio *= 2

        x_feat = x
        x = rearrange(x, "(T B) C h w -> T B C h w", T=T, B=B)
        x = self.proj_lif3(x)
        x = rearrange(x, "T B C h w -> (T B) C h w")
        x = self.rpe_bn(self.rpe_conv(x))
        x = rearrange(x + x_feat, "(T B) C h w -> T B C h w", T=T, B=B)
        return x