from __future__ import annotations

import torch
import torch.nn as nn
from spikingjelly.activation_based import layer

from ..common.layers import ConvBNLIF
from ..common.spike_ops import build_neuron, temporal_mean
from ..common.registry import register_model


class SpikeSelfAttention(nn.Module):
    """Spike-driven self-attention from Spikformer (Zhou et al., 2022)."""

    def __init__(self, dim: int, num_heads: int, model_config):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_lif = build_neuron(model_config)
        self.k_lif = build_neuron(model_config)
        self.v_lif = build_neuron(model_config)
        self.attn_lif = build_neuron(model_config)

        self.q_proj = layer.Linear(dim, dim, bias=False, step_mode="m")
        self.k_proj = layer.Linear(dim, dim, bias=False, step_mode="m")
        self.v_proj = layer.Linear(dim, dim, bias=False, step_mode="m")
        self.out_proj = layer.Linear(dim, dim, bias=False, step_mode="m")

        self.q_bn = nn.BatchNorm1d(dim)
        self.k_bn = nn.BatchNorm1d(dim)
        self.v_bn = nn.BatchNorm1d(dim)
        self.out_bn = nn.BatchNorm1d(dim)

    def _bn1d(self, x: torch.Tensor, bn: nn.BatchNorm1d) -> torch.Tensor:
        T, B, N, C = x.shape
        x = x.reshape(T * B * N, C)
        x = bn(x)
        return x.reshape(T, B, N, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, B, N, C]
        T, B, N, C = x.shape
        H, D = self.num_heads, self.head_dim

        q = self._bn1d(self.q_proj(x.reshape(T, B * N, C)).reshape(T, B, N, C), self.q_bn)
        k = self._bn1d(self.k_proj(x.reshape(T, B * N, C)).reshape(T, B, N, C), self.k_bn)
        v = self._bn1d(self.v_proj(x.reshape(T, B * N, C)).reshape(T, B, N, C), self.v_bn)

        q = self.q_lif(q)
        k = self.k_lif(k)
        v = self.v_lif(v)

        q = q.reshape(T, B, N, H, D).permute(0, 1, 3, 2, 4)  # [T,B,H,N,D]
        k = k.reshape(T, B, N, H, D).permute(0, 1, 3, 2, 4)
        v = v.reshape(T, B, N, H, D).permute(0, 1, 3, 2, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [T,B,H,N,N]
        attn = attn.reshape(T, B * H, N, N)
        attn = self.attn_lif(attn)
        attn = attn.reshape(T, B, H, N, N)

        out = (attn @ v).permute(0, 1, 3, 2, 4).reshape(T, B, N, C)
        out = self._bn1d(self.out_proj(out.reshape(T, B * N, C)).reshape(T, B, N, C), self.out_bn)
        return out


class SpikformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, model_config):
        super().__init__()
        self.attn = SpikeSelfAttention(dim, num_heads, model_config)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp_fc1 = layer.Linear(dim, mlp_hidden, bias=False, step_mode="m")
        self.mlp_fc2 = layer.Linear(mlp_hidden, dim, bias=False, step_mode="m")
        self.mlp_lif1 = build_neuron(model_config)
        self.mlp_lif2 = build_neuron(model_config)
        self.mlp_bn1 = nn.BatchNorm1d(mlp_hidden)
        self.mlp_bn2 = nn.BatchNorm1d(dim)

    def _bn1d(self, x: torch.Tensor, bn: nn.BatchNorm1d) -> torch.Tensor:
        T, B, N, C = x.shape
        x = x.reshape(T * B * N, C)
        x = bn(x)
        return x.reshape(T, B, N, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        h = self._bn1d(self.mlp_fc1(x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])).reshape(*x.shape[:3], -1), self.mlp_bn1)
        h = self.mlp_lif1(h)
        h = self._bn1d(self.mlp_fc2(h.reshape(h.shape[0], h.shape[1] * h.shape[2], h.shape[3])).reshape(*x.shape), self.mlp_bn2)
        h = self.mlp_lif2(h)
        return x + h


class SpikeConvPatchEmbed(nn.Module):
    """Convolutional patch embedding for Spikformer."""

    def __init__(self, in_channels: int, embed_dim: int, img_size: int, patch_size: int, model_config):
        super().__init__()
        self.proj = ConvBNLIF(in_channels, embed_dim, kernel_size=patch_size,
                              stride=patch_size, padding=0, model_config=model_config)
        self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, B, C, H, W] -> [T, B, N, embed_dim]
        T, B, C, H, W = x.shape
        x = self.proj(x)  # [T, B, embed_dim, H', W']
        x = x.flatten(3).transpose(2, 3)  # [T, B, N, embed_dim]
        return x


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

        self.patch_embed = SpikeConvPatchEmbed(in_channels, embed_dim, img_size, patch_size, model_config)
        self.blocks = nn.ModuleList([
            SpikformerBlock(embed_dim, num_heads, mlp_ratio, model_config)
            for _ in range(depth)
        ])
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, B, C, H, W]
        x = self.patch_embed(x)  # [T, B, N, C]
        for blk in self.blocks:
            x = blk(x)
        x = x.mean(dim=2)  # [T, B, C]
        x = temporal_mean(x)  # [B, C]
        return self.head(x)
