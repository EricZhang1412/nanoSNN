from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import rearrange
from spikingjelly.activation_based import layer

from ..common.spike_ops import build_neuron, temporal_mean
from ..common.registry import register_model


def _bn1d(x: torch.Tensor, bn: nn.BatchNorm1d) -> torch.Tensor:
    """Apply BN1d on [T, B, N, C] by transposing C to dim -2 (matches original repo)."""
    T, B, N, C = x.shape
    # flatten T,B -> TB, then transpose to [TB, C, N] for BN, then restore
    x = rearrange(x, "T B N C -> (T B) N C")
    x = rearrange(bn(rearrange(x, "TB N C -> TB C N")), "TB C N -> TB N C")
    return rearrange(x, "(T B) N C -> T B N C", T=T, B=B).contiguous()


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


class TemporalPSPGate(nn.Module):
    def __init__(self, tau: float = 2.0, gate_fn: str = "sigmoid", scale: float = 1.0):
        super().__init__()
        if tau <= 0:
            raise ValueError(f"psp_tau must be positive, got {tau}")
        self.alpha = math.exp(-1.0 / tau)
        self.gate_fn = gate_fn.lower()
        self.scale = scale
        self.state: torch.Tensor | None = None

    def reset(self):
        self.state = None

    def _make_gate(self, state: torch.Tensor) -> torch.Tensor:
        if self.gate_fn == "sigmoid":
            return torch.sigmoid(self.scale * state)
        raise ValueError(f"Unsupported psp_gate_fn: {self.gate_fn}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"TemporalPSPGate expects [T, B, H, N, D], got shape {tuple(x.shape)}")

        state = self.state
        if state is None or state.shape != x.shape[1:] or state.device != x.device or state.dtype != x.dtype:
            state = x.new_zeros(x.shape[1:])

        outputs = []
        for t in range(x.shape[0]):
            state = self.alpha * state + x[t]
            outputs.append(x[t] * self._make_gate(state))

        self.state = state.detach()
        return torch.stack(outputs, dim=0)


class PSPGatedLinearSpikeSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, model_config):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_linear = nn.Linear(dim, dim)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = build_neuron(model_config)

        self.k_linear = nn.Linear(dim, dim)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = build_neuron(model_config)

        self.v_linear = nn.Linear(dim, dim)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = build_neuron(model_config)

        self.attn_lif = build_neuron(model_config, v_threshold=0.5)

        self.proj_linear = nn.Linear(dim, dim)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = build_neuron(model_config)

        gate_on = str(getattr(model_config, "psp_gate_on", "qk")).lower()
        if gate_on == "off":
            gate_on = "none"
        if gate_on not in {"qk", "q", "k", "none"}:
            raise ValueError(f"Unsupported psp_gate_on: {gate_on}")

        gate_tau = float(getattr(model_config, "psp_tau", 2.0))
        gate_fn = str(getattr(model_config, "psp_gate_fn", "sigmoid")).lower()
        gate_scale = float(getattr(model_config, "psp_gate_scale", 1.0))

        self.q_gate = TemporalPSPGate(gate_tau, gate_fn, gate_scale) if gate_on in {"qk", "q"} else None
        self.k_gate = TemporalPSPGate(gate_tau, gate_fn, gate_scale) if gate_on in {"qk", "k"} else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, N, C = x.shape
        H, D = self.num_heads, self.head_dim

        x_for_qkv = rearrange(x, "T B N C -> (T B) N C")

        q = self.q_linear(x_for_qkv)
        q = _bn1d(rearrange(q, "(T B) N C -> T B N C", T=T, B=B), self.q_bn)
        q = self.q_lif(q)
        q = rearrange(q, "T B N (H D) -> T B H N D", H=H, D=D)

        k = self.k_linear(x_for_qkv)
        k = _bn1d(rearrange(k, "(T B) N C -> T B N C", T=T, B=B), self.k_bn)
        k = self.k_lif(k)
        k = rearrange(k, "T B N (H D) -> T B H N D", H=H, D=D)

        v = self.v_linear(x_for_qkv)
        v = _bn1d(rearrange(v, "(T B) N C -> T B N C", T=T, B=B), self.v_bn)
        v = self.v_lif(v)
        v = rearrange(v, "T B N (H D) -> T B H N D", H=H, D=D)

        if self.q_gate is not None:
            q = self.q_gate(q)
        if self.k_gate is not None:
            k = self.k_gate(k)

        kv = rearrange(k, "T B H N D -> T B H D N") @ v
        x = (q @ kv) * self.scale
        x = rearrange(x, "T B H N D -> T B N (H D)")
        x = self.attn_lif(x)

        x = self.proj_linear(rearrange(x, "T B N C -> (T B) N C"))
        x = _bn1d(rearrange(x, "(T B) N C -> T B N C", T=T, B=B), self.proj_bn)
        x = self.proj_lif(x)
        return x


class SpikformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, model_config):
        super().__init__()
        attention_type = str(getattr(model_config, "attention_type", "quadratic")).lower()
        if attention_type == "quadratic":
            self.attn = SpikeSelfAttention(dim, num_heads, model_config)
        elif attention_type == "psp_linear":
            self.attn = PSPGatedLinearSpikeSelfAttention(dim, num_heads, model_config)
        else:
            raise ValueError(f"Unsupported attention_type: {attention_type}")
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
