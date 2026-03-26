from __future__ import annotations

import torch
import torch.nn as nn
from spikingjelly.activation_based import layer

from ..common.spike_ops import build_neuron, temporal_mean
from ..common.registry import register_model


def _bn1d(x: torch.Tensor, bn: nn.BatchNorm1d) -> torch.Tensor:
    """Apply BN1d on [T, B, N, C] by transposing C to dim -2 (matches original repo)."""
    T, B, N, C = x.shape
    # flatten T,B -> TB, then transpose to [TB, C, N] for BN, then restore
    x = x.flatten(0, 1)                          # [TB, N, C]
    x = bn(x.transpose(-1, -2)).transpose(-1, -2)  # [TB, N, C]
    return x.reshape(T, B, N, C).contiguous()


class SPS(nn.Module):
    """Spike-driven Patch Splitting from Spikformer (Zhou et al., 2022).

    4-layer Conv stack with 2x MaxPool (total 4x spatial downsampling) + RPE branch.
    Input:  [T, B, C, H, W]
    Output: [T, B, N, embed_dim]  where N = (H//4) * (W//4)
    """

    def __init__(self, in_channels: int, embed_dim: int, img_size: int,
                 patch_size: int, model_config):
        super().__init__()
        # patch_size is kept for API compatibility but SPS always does 4x downsampling
        self.proj_conv = nn.Conv2d(in_channels, embed_dim // 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dim // 8)
        self.proj_lif = build_neuron(model_config)

        self.proj_conv1 = nn.Conv2d(embed_dim // 8, embed_dim // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn1 = nn.BatchNorm2d(embed_dim // 4)
        self.proj_lif1 = build_neuron(model_config)

        self.proj_conv2 = nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(embed_dim // 2)
        self.proj_lif2 = build_neuron(model_config)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv3 = nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn3 = nn.BatchNorm2d(embed_dim)
        self.proj_lif3 = build_neuron(model_config)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.rpe_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dim)
        self.rpe_lif = build_neuron(model_config)

        self.num_patches = (img_size // 4) ** 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, C, H, W = x.shape

        x = self.proj_conv(x.flatten(0, 1))
        x = self.proj_bn(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif(x).flatten(0, 1).contiguous()

        x = self.proj_conv1(x)
        x = self.proj_bn1(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif1(x).flatten(0, 1).contiguous()

        x = self.proj_conv2(x)
        x = self.proj_bn2(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif2(x).flatten(0, 1).contiguous()
        x = self.maxpool2(x)

        x = self.proj_conv3(x)
        x = self.proj_bn3(x).reshape(T, B, -1, H // 2, W // 2).contiguous()
        x = self.proj_lif3(x).flatten(0, 1).contiguous()
        x = self.maxpool3(x)

        x_feat = x.reshape(T, B, -1, H // 4, W // 4).contiguous()
        x = self.rpe_conv(x)
        x = self.rpe_bn(x).reshape(T, B, -1, H // 4, W // 4).contiguous()
        x = self.rpe_lif(x)
        x = x + x_feat

        x = x.flatten(-2).transpose(-1, -2)  # [T, B, N, C]
        return x


class SpikeSelfAttention(nn.Module):
    """Spike-driven self-attention from Spikformer (Zhou et al., 2022)."""

    def __init__(self, dim: int, num_heads: int, model_config):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 0.125  # fixed as in original repo

        self.q_linear = nn.Linear(dim, dim)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = build_neuron(model_config)

        self.k_linear = nn.Linear(dim, dim)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = build_neuron(model_config)

        self.v_linear = nn.Linear(dim, dim)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = build_neuron(model_config)

        # attn_lif uses v_threshold=0.5 as in original repo
        self.attn_lif = build_neuron(model_config, v_threshold=0.5)

        self.proj_linear = nn.Linear(dim, dim)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = build_neuron(model_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, N, C = x.shape
        H, D = self.num_heads, self.head_dim

        x_for_qkv = x.flatten(0, 1)  # [TB, N, C]

        q = self.q_linear(x_for_qkv)
        q = _bn1d(q.reshape(T, B, N, C), self.q_bn)
        q = self.q_lif(q)
        q = q.reshape(T, B, N, H, D).permute(0, 1, 3, 2, 4)  # [T,B,H,N,D]

        k = self.k_linear(x_for_qkv)
        k = _bn1d(k.reshape(T, B, N, C), self.k_bn)
        k = self.k_lif(k)
        k = k.reshape(T, B, N, H, D).permute(0, 1, 3, 2, 4)

        v = self.v_linear(x_for_qkv)
        v = _bn1d(v.reshape(T, B, N, C), self.v_bn)
        v = self.v_lif(v)
        v = v.reshape(T, B, N, H, D).permute(0, 1, 3, 2, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [T,B,H,N,N]
        x = attn @ v                                    # [T,B,H,N,D]
        x = x.transpose(2, 3).reshape(T, B, N, C).contiguous()  # [T,B,N,C]
        x = self.attn_lif(x)

        x = self.proj_linear(x.flatten(0, 1))           # [TB, N, C]
        x = _bn1d(x.reshape(T, B, N, C), self.proj_bn)
        x = self.proj_lif(x)
        return x


class SpikformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, model_config):
        super().__init__()
        self.attn = SpikeSelfAttention(dim, num_heads, model_config)
        mlp_hidden = int(dim * mlp_ratio)

        self.fc1_linear = nn.Linear(dim, mlp_hidden)
        self.fc1_bn = nn.BatchNorm1d(mlp_hidden)
        self.fc1_lif = build_neuron(model_config)

        self.fc2_linear = nn.Linear(mlp_hidden, dim)
        self.fc2_bn = nn.BatchNorm1d(dim)
        self.fc2_lif = build_neuron(model_config)

        self.c_hidden = mlp_hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)

        T, B, N, C = x.shape
        h = self.fc1_linear(x.flatten(0, 1))            # [TB, N, mlp_hidden]
        h = _bn1d(h.reshape(T, B, N, self.c_hidden), self.fc1_bn)
        h = self.fc1_lif(h)

        h = self.fc2_linear(h.flatten(0, 1))            # [TB, N, C]
        h = _bn1d(h.reshape(T, B, N, C), self.fc2_bn)
        h = self.fc2_lif(h)
        return x + h


@register_model("spikformer")
class Spikformer(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.T = int(getattr(model_config, "T", 4))
        embed_dim = int(getattr(model_config, "embed_dim", 256))
        depth = int(getattr(model_config, "depth", 4))
        num_heads = int(getattr(model_config, "num_heads", 8))
        mlp_ratio = float(getattr(model_config, "mlp_ratio", 4.0))
        num_classes = int(getattr(model_config, "num_classes", 10))
        img_size = int(getattr(model_config, "image_size", 32))
        patch_size = int(getattr(model_config, "patch_size", 4))
        in_channels = int(getattr(model_config, "in_channels", 3))

        self.patch_embed = SPS(in_channels, embed_dim, img_size, patch_size, model_config)
        self.blocks = nn.ModuleList([
            SpikformerBlock(embed_dim, num_heads, mlp_ratio, model_config)
            for _ in range(depth)
        ])
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, B, C, H, W]
        x = self.patch_embed(x)   # [T, B, N, C]
        for blk in self.blocks:
            x = blk(x)
        x = x.mean(dim=2)         # [T, B, C]
        x = temporal_mean(x)      # [B, C]
        return self.head(x)
