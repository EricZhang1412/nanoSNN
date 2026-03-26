from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange


from ..common.layers import MS_ConvBlock_spike_SepConv, MS_DownSampling, RepConv, SepConv_Spike
from ..common.spike_ops import build_neuron
from ..common.registry import register_model


def _cfg(model_config, name, default):
    return getattr(model_config, name, default)


def _cfg_stage_list(model_config, name, default, cast):
    value = getattr(model_config, name, None)
    if value is None:
        out = list(default)
    elif isinstance(value, (list, tuple)):
        out = [cast(v) for v in value]
    else:
        out = [cast(value)]
    while len(out) < 4:
        out.append(out[-1])
    return out[:4]

class SDTv3MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        model_config=None,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_spike = build_neuron(model_config)

        self.fc2_conv = nn.Conv1d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_spike = build_neuron(model_config)


    def forward(self, x):
        _, _, H, W = x.shape
        x = rearrange(x, "B C H W -> B C (H W)")
        x = self.fc1_spike(x)
        x = self.fc1_conv(x)
        x = self.fc1_bn(x).contiguous()
        x = self.fc2_spike(x)
        x = self.fc2_conv(x)
        x = self.fc2_bn(x)
        x = rearrange(x, "B C (H W) -> B C H W", H=H, W=W)

        return x


class SDTv3AttentionRepConv(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        model_config=None,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.num_heads = num_heads
        self.scale = (dim//num_heads) ** -0.5

        self.head_spike = build_neuron(model_config)

        self.q_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))
        self.k_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))
        self.v_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))

        self.q_spike = build_neuron(model_config)
        self.k_spike = build_neuron(model_config)
        self.v_spike = build_neuron(model_config)
        self.attn_spike = build_neuron(model_config)

        self.proj_conv = nn.Sequential(
            RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim)
        )


    def forward(self, x):
        _, C, H, W = x.shape

        x = self.head_spike(x)

        q = self.q_conv(x)
        k = self.k_conv(x)
        v = self.v_conv(x)

        q = rearrange(self.q_spike(q), "B (h d) H W -> B h (H W) d", h=self.num_heads)
        k = rearrange(self.k_spike(k), "B (h d) H W -> B h (H W) d", h=self.num_heads)
        v = rearrange(self.v_spike(v), "B (h d) H W -> B h (H W) d", h=self.num_heads)

        x = k.transpose(-2, -1) @ v
        x = (q @ x) * self.scale

        x = rearrange(x, "B h N d -> B (h d) N")
        x = self.attn_spike(x)
        x = rearrange(x, "B C (H W) -> B C H W", H=H, W=W)
        x = self.proj_conv(x)

        return x


class SDTv3AttentionLinear(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=None,
        lamda_ratio=None,
        model_config=None,
    ):
        super().__init__()
        num_heads = int(num_heads if num_heads is not None else _cfg(model_config, "num_heads", 8))
        lamda_ratio = float(lamda_ratio if lamda_ratio is not None else _cfg(model_config, "attn_lamda_ratio", 1.0))
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.lamda_ratio = lamda_ratio
        v_dim = int(dim * self.lamda_ratio)

        self.head_spike = build_neuron(model_config)

        self.q_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=False), nn.BatchNorm2d(dim))
        self.q_spike = build_neuron(model_config)

        self.k_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=False), nn.BatchNorm2d(dim))
        self.k_spike = build_neuron(model_config)

        self.v_conv = nn.Sequential(nn.Conv2d(dim, v_dim, 1, 1, bias=False), nn.BatchNorm2d(v_dim))
        self.v_spike = build_neuron(model_config)

        self.attn_spike = build_neuron(model_config)
        self.proj_conv = nn.Sequential(
            nn.Conv2d(v_dim, dim, 1, 1, bias=False), nn.BatchNorm2d(dim)
        )


    def forward(self, x):
        _, _, H, W = x.shape

        x = self.head_spike(x)

        q = self.q_conv(x)
        k = self.k_conv(x)
        v = self.v_conv(x)

        q = rearrange(self.q_spike(q), "B (h d) H W -> B h (H W) d", h=self.num_heads)
        k = rearrange(self.k_spike(k), "B (h d) H W -> B h (H W) d", h=self.num_heads)
        v = rearrange(self.v_spike(v), "B (h d) H W -> B h (H W) d", h=self.num_heads)

        x = q @ k.transpose(-2, -1)
        x = (x @ v) * self.scale

        x = rearrange(x, "B h N d -> B (h d) N")
        x = self.attn_spike(x)
        x = rearrange(x, "B C (H W) -> B C H W", H=H, W=W)
        x = self.proj_conv(x)

        return x

class SDTv3AttnMLPBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, lamda_ratio: float = 1.0, model_config=None):
        super().__init__()
        self.attn = SDTv3AttentionLinear(dim=dim, num_heads=num_heads, lamda_ratio=lamda_ratio, model_config=model_config)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = SDTv3MLP(in_features=dim, hidden_features=mlp_hidden, model_config=model_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x

class SDTv3ConvAttnMLPBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, lamda_ratio: float = 1.0, model_config=None):
        super().__init__()
        self.conv = SepConv_Spike(dim=dim, kernel_size=3, padding=1, model_config=model_config)
        self.attn = SDTv3AttentionLinear(dim=dim, num_heads=num_heads, lamda_ratio=lamda_ratio, model_config=model_config)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = SDTv3MLP(in_features=dim, hidden_features=mlp_hidden, model_config=model_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.conv(x)
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


@register_model("sdt_v3")
class SpikeDrivenTransformerV3(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        in_channels = int(_cfg(model_config, "in_channels", 2))
        num_classes = int(_cfg(model_config, "num_classes", 11))
        embed_dim = _cfg_stage_list(model_config, "embed_dim", [64, 128, 256, 360], int)
        num_heads = _cfg_stage_list(model_config, "num_heads", [1, 2, 4, 8], int)
        mlp_ratios = _cfg_stage_list(model_config, "mlp_ratio", [4, 4, 4, 4], int)
        depths = _cfg_stage_list(model_config, "depth", [1, 1, 6, 2], int)
        lamda_ratios = _cfg_stage_list(model_config, "attn_lamda_ratio", [1.0, 1.0, 1.0, 1.0], float)
        ds_k = _cfg_stage_list(model_config, "downsample_kernel_size", [7, 3, 3, 3], int)
        ds_s = _cfg_stage_list(model_config, "downsample_stride", [2, 2, 2, 1], int)
        ds_p = _cfg_stage_list(model_config, "downsample_padding", [3, 1, 1, 1], int)

        self.downsample1_1 = MS_DownSampling(in_channels=in_channels, embed_dims=embed_dim[0] // 2, kernel_size=ds_k[0], stride=ds_s[0], padding=ds_p[0], first_layer=True)
        self.ConvBlock1_1 = nn.ModuleList([MS_ConvBlock_spike_SepConv(dim=embed_dim[0] // 2, mlp_ratio=mlp_ratios[0], model_config=model_config) for _ in range(depths[0])])
        self.downsample1_2 = MS_DownSampling(in_channels=embed_dim[0] // 2, embed_dims=embed_dim[0], kernel_size=ds_k[1], stride=ds_s[1], padding=ds_p[1], first_layer=False)
        self.ConvBlock1_2 = nn.ModuleList([MS_ConvBlock_spike_SepConv(dim=embed_dim[0], mlp_ratio=mlp_ratios[0], model_config=model_config) for _ in range(depths[0])])

        self.downsample2 = MS_DownSampling(in_channels=embed_dim[0], embed_dims=embed_dim[1], kernel_size=ds_k[2], stride=ds_s[2], padding=ds_p[2], first_layer=False)
        self.ConvBlock2_1 = nn.ModuleList([MS_ConvBlock_spike_SepConv(dim=embed_dim[1], mlp_ratio=mlp_ratios[1], model_config=model_config) for _ in range(depths[1])])
        self.ConvBlock2_2 = nn.ModuleList([MS_ConvBlock_spike_SepConv(dim=embed_dim[1], mlp_ratio=mlp_ratios[1], model_config=model_config) for _ in range(depths[1])])

        self.downsample3 = MS_DownSampling(in_channels=embed_dim[1], embed_dims=embed_dim[2], kernel_size=ds_k[2], stride=ds_s[2], padding=ds_p[2], first_layer=False)
        self.block3 = nn.ModuleList([SDTv3ConvAttnMLPBlock(embed_dim[2], num_heads[2], mlp_ratios[2], lamda_ratios[2], model_config) for _ in range(depths[2])])
        self.downsample4 = MS_DownSampling(in_channels=embed_dim[2], embed_dims=embed_dim[3], kernel_size=ds_k[3], stride=ds_s[3], padding=ds_p[3], first_layer=False)
        self.block4 = nn.ModuleList([SDTv3ConvAttnMLPBlock(embed_dim[3], num_heads[3], mlp_ratios[3], lamda_ratios[3], model_config) for _ in range(depths[3])])

        self.head_spike = build_neuron(model_config)
        self.head = nn.Linear(embed_dim[3], num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample1_1(x)
        for blk in self.ConvBlock1_1:
            x = blk(x)
        x = self.downsample1_2(x)
        for blk in self.ConvBlock1_2:
            x = blk(x)
        x = self.downsample2(x)
        for blk in self.ConvBlock2_1:
            x = blk(x)
        for blk in self.ConvBlock2_2:
            x = blk(x)
        x = self.downsample3(x)
        for blk in self.block3:
            x = blk(x)
        x = self.downsample4(x)
        for blk in self.block4:
            x = blk(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = x.flatten(2).mean(2)
        x = self.head_spike(x)
        return self.head(x)
