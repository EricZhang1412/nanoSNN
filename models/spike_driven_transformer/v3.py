from __future__ import annotations

import torch
import torch.nn as nn
from spikingjelly.activation_based import layer

from ..common.layers import ConvBNLIF
from ..common.spike_ops import build_neuron, temporal_mean
from ..common.registry import register_model


class SDTv3Attention(nn.Module):
    """Spike-Driven Self-Attention from SDT v3 (Yao et al., 2024).

    Key change vs v2: replaces the linear KV product with a windowed
    local attention to reduce memory while keeping spike-form ops.
    For simplicity this implementation uses global spike-driven attention
    with the v3 BN-LIF gating scheme; window partitioning can be added
    as a follow-up.
    """

    def __init__(self, dim: int, num_heads: int, model_config):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = layer.Linear(dim, 3 * dim, bias=False, step_mode="m")
        self.out_proj = layer.Linear(dim, dim, bias=False, step_mode="m")

        self.q_lif = build_neuron(model_config)
        self.k_lif = build_neuron(model_config)
        self.v_lif = build_neuron(model_config)
        self.attn_lif = build_neuron(model_config)
        self.out_lif = build_neuron(model_config)

        self.qkv_bn = nn.BatchNorm1d(3 * dim)
        self.attn_bn = nn.BatchNorm1d(dim)
        self.out_bn = nn.BatchNorm1d(dim)

    def _bn1d(self, x: torch.Tensor, bn: nn.BatchNorm1d) -> torch.Tensor:
        T, B, N, C = x.shape
        x = x.reshape(T * B * N, C)
        x = bn(x)
        return x.reshape(T, B, N, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, N, C = x.shape
        H, D = self.num_heads, self.head_dim

        qkv = self._bn1d(self.qkv_proj(x.reshape(T, B * N, C)).reshape(T, B, N, 3 * C), self.qkv_bn)
        q, k, v = qkv.chunk(3, dim=-1)

        q = self.q_lif(q).reshape(T, B, N, H, D).permute(0, 1, 3, 2, 4)
        k = self.k_lif(k).reshape(T, B, N, H, D).permute(0, 1, 3, 2, 4)
        v = self.v_lif(v).reshape(T, B, N, H, D).permute(0, 1, 3, 2, 4)

        kv = k.transpose(-2, -1) @ v
        attn = (q @ kv) * self.scale
        attn = attn.permute(0, 1, 3, 2, 4).reshape(T, B, N, C)
        attn = self._bn1d(attn, self.attn_bn)
        attn = self.attn_lif(attn)

        out = self._bn1d(self.out_proj(attn.reshape(T, B * N, C)).reshape(T, B, N, C), self.out_bn)
        return self.out_lif(out)


class SDTv3Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, model_config):
        super().__init__()
        self.attn = SDTv3Attention(dim, num_heads, model_config)
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
        T, B, N, C = x.shape
        h = self._bn1d(self.mlp_fc1(x.reshape(T, B * N, C)).reshape(T, B, N, -1), self.mlp_bn1)
        h = self.mlp_lif1(h)
        h = self._bn1d(self.mlp_fc2(h.reshape(T, B * N, -1)).reshape(T, B, N, C), self.mlp_bn2)
        h = self.mlp_lif2(h)
        return x + h


@register_model("sdt_v3")
class SpikeDrivenTransformerV3(nn.Module):
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

        from ..spikformer.model import SpikeConvPatchEmbed
        self.patch_embed = SpikeConvPatchEmbed(in_channels, embed_dim, img_size, patch_size, model_config)
        self.blocks = nn.ModuleList([
            SDTv3Block(embed_dim, num_heads, mlp_ratio, model_config)
            for _ in range(depth)
        ])
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = x.mean(dim=2)
        x = temporal_mean(x)
        return self.head(x)
