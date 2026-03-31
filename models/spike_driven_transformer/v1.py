from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from spikingjelly.activation_based import layer
from einops import rearrange, repeat
from ..common.layers import ConvBNLIF
from ..common.spike_ops import build_neuron, temporal_mean
from ..common.registry import register_model


class SDTv1Attention(nn.Module):
    """Spike-Driven Self-Attention from SDT v1 (Yao et al., 2023).

    Uses spike-form Q/K/V with integer-like multiply-accumulate ops.
    """

    def __init__(self, dim: int, num_heads: int, model_config):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        
        # The original implementation uses Conv1x1 to project Q/K/V. see https://github.com/BICLab/Spike-Driven-Transformer/blob/3faa82272eb499e4d56ea77707d4536e7c69a54b/module/ms_conv.py#L105
        self.dvs = bool(getattr(model_config, "dvs", False))
        self.layer = int(getattr(model_config, "layer", 0))

        self.pool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))

        self.shortcut_lif = build_neuron(model_config)

        self.q_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm2d(dim)
        self.q_lif = build_neuron(model_config)

        self.k_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm2d(dim)
        self.k_lif = build_neuron(model_config)

        self.v_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm2d(dim)
        self.v_lif = build_neuron(model_config)

        self.kv_attn_lif = build_neuron(model_config, v_threshold=0.5)

        self.proj_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm2d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, C, H, W = x.shape

        x = self.shortcut_lif(x)
        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, H, W).contiguous()
        q_conv_out = self.q_lif(q_conv_out)
        q = rearrange(q_conv_out, "T B (h d) H W -> T B h (H W) d", h=self.num_heads)

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, H, W).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        k = rearrange(k_conv_out, "T B (h d) H W -> T B h (H W) d", h=self.num_heads)

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, H, W).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        v = rearrange(v_conv_out, "T B (h d) H W -> T B h (H W) d", h=self.num_heads)

        kv = k.mul(v)
        kv = kv.sum(dim=-2, keepdim=True)
        kv = self.kv_attn_lif(kv)
        x = q.mul(kv)
        if self.dvs:
            x = self.pool(x)

        x = rearrange(x, "T B h (H W) d -> T B (h d) H W", H=H, W=W)
        x = self.proj_bn(self.proj_conv(x.flatten(0, 1))).reshape(T, B, C, H, W).contiguous()

        return x

class SDTv1MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, model_config):
        super().__init__()
        self.fc1_conv = nn.Conv2d(dim, int(dim * mlp_ratio), kernel_size=1, stride=1, bias=False)
        self.fc1_bn = nn.BatchNorm2d(int(dim * mlp_ratio))
        self.fc1_lif = build_neuron(model_config)

        self.fc2_conv = nn.Conv2d(int(dim * mlp_ratio), dim, kernel_size=1, stride=1, bias=False)
        self.fc2_bn = nn.BatchNorm2d(dim)
        self.fc2_lif = build_neuron(model_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, _, H, W = x.shape

        x = self.fc1_lif(x)
        x = self.fc1_conv(rearrange(x, "T B C H W -> (T B) C H W"))
        x = rearrange(self.fc1_bn(x), "(T B) C H W -> T B C H W", T=T, B=B)

        x = self.fc2_lif(x)
        x = self.fc2_conv(rearrange(x, "T B C H W -> (T B) C H W"))
        x = rearrange(self.fc2_bn(x), "(T B) C H W -> T B C H W", T=T, B=B)

        return x

class SDTv1Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, model_config):
        super().__init__()
        self.attn = SDTv1Attention(dim, num_heads, model_config)
        self.mlp = SDTv1MLP(dim, mlp_ratio, model_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


@register_model("sdt_v1")
class SpikeDrivenTransformerV1(nn.Module):
    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self):
        self.apply(self._init_weights)

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
        pooling_stat = str(getattr(model_config, "pooling_stat", "0011"))

        from ..common.patch_embed import SDTv1PatchSpliting
        self.patch_embed = SDTv1PatchSpliting(
            img_size_h=img_size,
            img_size_w=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dim,
            pooling_stat=pooling_stat,
            model_config=model_config,
        )
        self.blocks = nn.ModuleList([
            SDTv1Block(embed_dim, num_heads, mlp_ratio, model_config)
            for _ in range(depth)
        ])
        self.head_lif = build_neuron(model_config)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 5:
            x = repeat(x, "B C H W -> T B C H W", T=self.T)
        elif x.shape[0] == self.T:
            pass
        elif x.shape[1] == self.T:
            x = rearrange(x, "B T C H W -> T B C H W")
        else:
            raise ValueError(f"Expected 5D input as [T,B,C,H,W] or [B,T,C,H,W] with T={self.T}, got {tuple(x.shape)}")

        x = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = rearrange(x, "T B C H W -> T B C (H W)").mean(-1)
        x = self.head_lif(x)
        # x = temporal_mean(x)
        x = temporal_mean(self.head(x))

        return x
