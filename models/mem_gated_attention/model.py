"""
SD-BGLA: Spike-Driven Binary Gated Linear Attention
=====================================================

Purely spike-driven gated linear attention for SNN transformers.

Core idea: add temporal recurrence to the KV state with two spike gates
(decay + write), using bit-shift decay and binary gate neurons.
No floating-point multiplication in the attention mechanism.

Training uses the parallel (causal weighted sum) form for GPU efficiency.
Inference uses the sequential (spike-driven recurrent) form for neuromorphic hw.
"""

from __future__ import annotations

from typing import Optional
import math

import torch
import torch.nn as nn
from einops import rearrange
from spikingjelly.activation_based import neuron, surrogate
from spikingjelly.activation_based.neuron import BaseNode

from torch.utils.checkpoint import checkpoint

from ..common.spike_ops import build_neuron, temporal_mean
from ..common.tl_neuron_ops import build_triton_neuron
from ..common.registry import register_model
from ..common.tl_spike_ops import TritonLIF, TritonGateLIF, TritonDualLIF

from .mga_triton_ops import sdbgla_attention


def _bn1d(x: torch.Tensor, bn: nn.BatchNorm1d) -> torch.Tensor:
    """Apply nn.BatchNorm1d(C) on the last dim of x. Shape-agnostic.
    
    Semantically equivalent to transpose-based version but avoids ~3 HBM
    copies per call by using a single reshape→BN→reshape.
    
    Args:
        x: tensor with C in last dim, e.g. [T, B, N, C] or [TB, N, C]
        bn: nn.BatchNorm1d(C)
    
    Returns:
        same shape as x, BN applied per-C over all (*) positions.
    """
    orig_shape = x.shape
    C = orig_shape[-1]
    if not x.is_contiguous():
        x = x.contiguous()
    x_flat = x.view(-1, C)
    x_flat = bn(x_flat)
    return x_flat.view(orig_shape)

# ---------------------------------------------------------------------------
# Gate LIF Neuron — produces binary gate spikes
# ---------------------------------------------------------------------------
class GateLIFNode(BaseNode):
    """Per-channel learnable tau 的 Parametric LIF。
    
    基于 SpikingJelly 的 BaseNode，只需实现 neuronal_charge()。
    """

    def __init__(self, channels: int, init_tau: float = 2.0,
                 v_threshold: float = 1.0, v_reset: float = 0.0,
                 surrogate_function=None, detach_reset=True,
                 step_mode='m', backend='torch'):
        super().__init__(
            v_threshold=v_threshold,
            v_reset=v_reset,
            surrogate_function=surrogate_function or surrogate.ATan(),
            detach_reset=detach_reset,
            step_mode=step_mode,
            backend=backend,
        )
        # per-channel 可学习参数，w → tau = 1 / sigmoid(w)
        init_w = -math.log(init_tau - 1.0)  # sigmoid(w) = 1/tau
        self.w = nn.Parameter(torch.full((channels,), init_w))
        self.last_firing_rate = None

    @property
    def supported_backends(self):
        return ('torch',)

    def neuronal_charge(self, x: torch.Tensor):
        # decay = 1 - 1/tau = 1 - sigmoid(w)
        # self.v_float_to_tensor(x)
        decay = 1.0 - self.w.sigmoid()
        self.v = decay * self.v + x
    
    def forward(self, x: torch.Tensor):
        out = super().forward(x)
        self.last_firing_rate = out.detach().mean()
        return out


# ---------------------------------------------------------------------------
# SD-BGLA Attention
# ---------------------------------------------------------------------------

class SDBGLAttention(nn.Module):
    """Spike-Driven Binary Gated Linear Attention.

    Training: parallel form (causal temporal attention matrix L).
    Inference: sequential spike-driven form (bit-shift + conditional add/sub).

    Parameters
    ----------
    dim : int
        Total embedding dimension.
    num_heads : int
        Number of attention heads.
    shift_k : int
        Bit-shift amount for decay (δ = 2^{-k}).  Default 1 → δ = 0.5.
    model_config : object
        Config for building LIF neurons.
    """

    def __init__(self, dim: int, num_heads: int, shift_k: int = 1,
                 model_config=None):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # d_k = d_v
        self.scale = self.head_dim ** -0.5
        self.shift_k = shift_k
        self.delta = 2.0 ** (-shift_k)  # fixed decay magnitude

        # Q, K, V projections (same as Spikformer)
        # self.shortcut_lif = build_neuron(model_config)
        self.shortcut_lif = build_triton_neuron(model_config, output_mode="spike")

        self.q_linear = nn.Linear(dim, dim)
        self.q_bn = nn.BatchNorm1d(dim)
        # self.q_lif = build_neuron(model_config)
        self.q_lif = build_triton_neuron(model_config, output_mode="spike")

        self.k_linear = nn.Linear(dim, dim)
        self.k_bn = nn.BatchNorm1d(dim)
        # self.k_lif = build_neuron(model_config, output_mode="dual")
        self.k_lif = build_triton_neuron(model_config, output_mode="dual")

        self.v_linear = nn.Linear(dim, dim)
        self.v_bn = nn.BatchNorm1d(dim)
        # self.v_lif = build_neuron(model_config)
        self.v_lif = build_triton_neuron(model_config, output_mode="spike")

        # Output LIF (unchanged from baseline — preserves spike-driven output)
        # self.attn_lif = build_neuron(model_config, v_threshold=0.5)
        self.attn_lif = build_triton_neuron(model_config, v_threshold=0.5, output_mode="spike")

        # Output projection
        self.proj_linear = nn.Linear(dim, dim)
        self.proj_bn = nn.BatchNorm1d(dim)
        # self.proj_lif = build_neuron(model_config)
        self.proj_lif = build_triton_neuron(model_config, output_mode="spike")



        # --- SD-BGLA gate neurons ---
        # Decay gate: one PLIF per channel (d_k channels per head)
        decay_gate_lif_init_tau = float(getattr(model_config, "decay_gate_lif_init_tau", 2.0))
        
        self.decay_proj = nn.Linear(self.head_dim, self.head_dim, bias=True)
        self.decay_bn = nn.BatchNorm1d(self.head_dim)
        self.decay_gate = GateLIFNode(
            self.head_dim, 
            init_tau=decay_gate_lif_init_tau, 
            v_threshold=1.0
        )

        # # Precompute decay power table: table[m] = (1-δ)^m
        # max_T = 32  # generous upper bound
        # table = torch.tensor([(1.0 - self.delta) ** m for m in range(max_T + 1)])
        # self.register_buffer("decay_table", table)
        
        # NEW: make precision configurable 
        self.attn_precision = str(getattr(model_config, "attn_precision", "tf32")).lower() 
        assert self.attn_precision in ("ieee", "tf32", "tf32x3")

    def _get_qkv(self, x: torch.Tensor):
        """Produce binary Q, K, V and K's membrane potential.

        Args:
            x: [T, B, N, C]
        Returns:
            q, k, v: [T, B, H, N, D]  (binary spikes)
            u_k: [T, B, H, N, D]      (membrane potential before threshold)
        """
        T, B, N, C = x.shape
        H, D = self.num_heads, self.head_dim

        x = self.shortcut_lif(x)
 
        # nn.Linear broadcasts over leading dims; no need to flatten T,B.
        # _bn1d is shape-agnostic and handles the (T,B,N,C) -> flat reshape internally.
        q = self.q_linear(x)                         # [T, B, N, C]
        q = _bn1d(q, self.q_bn)
        q = self.q_lif(q)
        q = rearrange(q, "T B N (H D) -> T B H N D", H=H, D=D)
 
        k_pre = self.k_linear(x)                     # [T, B, N, C]
        k_pre = _bn1d(k_pre, self.k_bn)
 
        out = self.k_lif(k_pre)                      # DualOutput
        k = rearrange(out.spike, "T B N (H D) -> T B H N D", H=H, D=D)
        u_k_pooled = rearrange(out.v_seq, "T B N (H D) -> T B H N D", H=H, D=D)
        u_k_pooled = u_k_pooled.mean(dim=3)          # [T, B, H, D]
 
        v = self.v_linear(x)                         # [T, B, N, C]
        v = _bn1d(v, self.v_bn)
        v = self.v_lif(v)
        v = rearrange(v, "T B N (H D) -> T B H N D", H=H, D=D)
 
        return q, k, v, u_k_pooled

    def _compute_gate_spikes(self, k: torch.Tensor, u_k_pooled: torch.Tensor):
        """Compute decay and write gate spikes from k spikes and membrane potential.

        Args:
            k: [T, B, H, N, D]   (binary key spikes)
            u_k_pooled: [T, B, H, D] (key membrane potential, pooled over N)
        Returns:
            s_gamma: [T, B, H, D]  (binary decay spikes, per channel)
        """
        T, B, H, N, D = k.shape

        # Decay gate input: pooled membrane potential, mean over N
        # u_bar = u_k.mean(dim=3)  # [T, B, H, D]
        # u_bar = self.decay_proj(u_bar)
        
        # s_gamma = self.decay_gate(u_bar)  # [T, B, H, D], binary
        u_bar = self.decay_proj(u_k_pooled)   # [T, B, H, D]
        u_bar = _bn1d(u_bar, self.decay_bn)
        # u_bar = rearrange(u_k_pooled, "T B H D -> (T B H) D")
        # u_bar = self.decay_proj(u_bar)
        # u_bar = self.decay_bn(u_bar)
        # u_bar = rearrange(u_bar, "(T B H) D -> T B H D", T=T, B=B, H=H)
        s_gamma = self.decay_gate(u_bar)

        return s_gamma


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [T, B, N, C]
        Returns:
            out: [T, B, N, C]
        """
        T, B, N, C = x.shape
        H, D = self.num_heads, self.head_dim

        # Step 1: binary Q, K, V + membrane potential
        q, k, v, u_k = self._get_qkv(x)   # all [T, B, H, N, D]

        # Step 2: gate spikes
        s_gamma = self._compute_gate_spikes(k, u_k)
        
        o = sdbgla_attention( 
                             q, k, v, s_gamma, 
                             delta=self.delta, 
                             scale=self.scale, 
                             precision=self.attn_precision, 
                            )

        # Step 5: output — LIF threshold (spike-driven, binary output)
        o = rearrange(o, "T B H N D -> T B N (H D)")
        o = self.attn_lif(o)

        # Step 6: output projection
        o = self.proj_linear(rearrange(o, "T B N C -> (T B) N C"))
        o = _bn1d(rearrange(o, "(T B) N C -> T B N C", T=T, B=B), self.proj_bn)
        o = self.proj_lif(o)
        return o

class SDBGLAMlp(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, model_config=None):
        super().__init__()
        mlp_hidden = int(dim * mlp_ratio)

        self.fc1_linear = nn.Linear(dim, mlp_hidden)
        self.fc1_bn = nn.BatchNorm1d(mlp_hidden)
        # self.fc1_lif = build_neuron(model_config)
        self.fc1_lif = build_triton_neuron(model_config, output_mode="spike")

        self.fc2_linear = nn.Linear(mlp_hidden, dim)
        self.fc2_bn = nn.BatchNorm1d(dim)
        # self.fc2_lif = build_neuron(model_config)
        self.fc2_lif = build_triton_neuron(model_config, output_mode="spike")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, B, N, C]; Linear + _bn1d are shape-agnostic on last dim.
        h = self.fc1_lif(x)
        h = self.fc1_linear(h)
        h = _bn1d(h, self.fc1_bn)
 
        h = self.fc2_lif(h)
        h = self.fc2_linear(h)
        h = _bn1d(h, self.fc2_bn)
        return h

class SPS(nn.Module):
    """Spike-driven Patch Splitting with strided conv (earlier downsampling)."""

    def __init__(self, in_channels: int, embed_dim: int, img_size: int,
                 patch_size: int, model_config):
        super().__init__()
        # Stage 0: stride=2 → H/2. Downsamples immediately.
        self.proj_conv = nn.Conv2d(in_channels, embed_dim // 8, 3, stride=2, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dim // 8)
        self.proj_lif = build_triton_neuron(model_config, output_mode="spike")

        # Stage 1: stride=1
        self.proj_conv1 = nn.Conv2d(embed_dim // 8, embed_dim // 4, 3, stride=1, padding=1, bias=False)
        self.proj_bn1 = nn.BatchNorm2d(embed_dim // 4)
        self.proj_lif1 = build_triton_neuron(model_config, output_mode="spike")

        # Stage 2: stride=2 → H/4. No maxpool afterwards.
        self.proj_conv2 = nn.Conv2d(embed_dim // 4, embed_dim // 2, 3, stride=2, padding=1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(embed_dim // 2)
        self.proj_lif2 = build_triton_neuron(model_config, output_mode="spike")

        # Stage 3: stride=1
        self.proj_conv3 = nn.Conv2d(embed_dim // 2, embed_dim, 3, stride=1, padding=1, bias=False)
        self.proj_bn3 = nn.BatchNorm2d(embed_dim)
        self.proj_lif3 = build_triton_neuron(model_config, output_mode="spike")

        # RPE unchanged
        self.rpe_conv = nn.Conv2d(embed_dim, embed_dim, 3, stride=1, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dim)
        self.rpe_lif = build_triton_neuron(model_config, output_mode="spike")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, C, H, W = x.shape

        x = self.proj_conv(rearrange(x, "T B C H W -> (T B) C H W"))
        x = rearrange(self.proj_bn(x), "(T B) C H W -> T B C H W", T=T, B=B)
        x = rearrange(self.proj_lif(x), "T B C H W -> (T B) C H W")

        x = self.proj_conv1(x)
        x = rearrange(self.proj_bn1(x), "(T B) C H W -> T B C H W", T=T, B=B)
        x = rearrange(self.proj_lif1(x), "T B C H W -> (T B) C H W")

        x = self.proj_conv2(x)
        x = rearrange(self.proj_bn2(x), "(T B) C H W -> T B C H W", T=T, B=B)
        x = rearrange(self.proj_lif2(x), "T B C H W -> (T B) C H W")
        # [no maxpool2]

        x = self.proj_conv3(x)
        x = rearrange(self.proj_bn3(x), "(T B) C H W -> T B C H W", T=T, B=B)
        x = rearrange(self.proj_lif3(x), "T B C H W -> (T B) C H W")
        # [no maxpool3]

        x_feat = rearrange(x, "(T B) C H W -> T B C H W", T=T, B=B)
        x = self.rpe_conv(x)
        x = rearrange(self.rpe_bn(x), "(T B) C H W -> T B C H W", T=T, B=B)
        x = self.rpe_lif(x)
        x = x + x_feat

        return rearrange(x, "T B C H W -> T B (H W) C")

class FastStem(nn.Module):
    def __init__(self, embed_dim=384, model_config=None):
        super().__init__()
        # 3 → 48 → 96 → 192 → 384, 4× stride-2, 总 stride=16
        self.stem = nn.Sequential(
            nn.Conv2d(3, 48, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48), nn.GELU(),
            nn.Conv2d(48, 96, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96), nn.GELU(),
            nn.Conv2d(96, 192, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(192), nn.GELU(),
            nn.Conv2d(192, embed_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
        )
        self.entry_lif = build_triton_neuron(model_config, output_mode="spike")

    def forward(self, x):
        # x: [T, B, C, H, W]，但我们只用第 0 帧
        T, B = x.shape[:2]
        x0 = x[0]                               # [B, 3, 224, 224]，ImageNet 是静态图
        tokens = self.stem(x0)                  # [B, 384, 14, 14]
        tokens = rearrange(tokens, "B C H W -> B (H W) C")
        tokens = tokens.unsqueeze(0).expand(T, -1, -1, -1)   # 广播到 T
        return self.entry_lif(tokens)           # 仅一个 LIF，位于 body 入口
# ---------------------------------------------------------------------------
# SD-BGLA Block (Attention + MLP)
# ---------------------------------------------------------------------------

class SDBGLABlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float,
                 shift_k: int = 1, model_config=None):
        super().__init__()
        self.attn = SDBGLAttention(dim, num_heads, shift_k=shift_k,
                                   model_config=model_config)
        self.mlp = SDBGLAMlp(dim, mlp_ratio, model_config=model_config)
        # LayerScale: per-channel learnable scalar
        attn_layer_scale_init_values = getattr(model_config, "attn_layer_scale_init_values", 0.1)
        mlp_layer_scale_init_values = getattr(model_config, "mlp_layer_scale_init_values", 0.1)
        
        self.gamma_attn = nn.Parameter(attn_layer_scale_init_values * torch.ones(dim))
        self.gamma_mlp = nn.Parameter(mlp_layer_scale_init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, B, N, C]
        x = x + self.gamma_attn * self.attn(x)
        x = x + self.gamma_mlp * self.mlp(x)
        return x


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

@register_model("mem_gated_attn")
class SDBGLAFormer(nn.Module):
    """Memory Gated Attention Transformer.

    Architecture: SPS patch embedding + SD-BGLA blocks + linear head.
    Follows the Spikformer blueprint with SD-BGLA attention replacing SSA.
    """

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
        shift_k = int(getattr(model_config, "shift_k", 1))

        # self.patch_embed = SPS(in_channels, embed_dim, img_size, patch_size,
                            #    model_config)
        self.patch_embed = FastStem(embed_dim, model_config)

        self.blocks = nn.ModuleList([
            SDBGLABlock(embed_dim, num_heads, mlp_ratio, shift_k=shift_k,
                        model_config=model_config)
            for _ in range(depth)
        ])
        self.head = nn.Linear(embed_dim, num_classes)
        self.init_weights()
        
    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def init_weights(self):
        self.apply(self._init_weights)

        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

        # zero init output projection
        for blk in self.blocks:
            nn.init.zeros_(blk.attn.proj_linear.weight)
            nn.init.zeros_(blk.mlp.fc2_linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, B, C, H, W]
        x = self.patch_embed(x)   # [T, B, N, C]
        # x = checkpoint(self.patch_embed, x, use_reentrant=False)
        for blk in self.blocks:
            x = blk(x)
        x = x.mean(dim=2)         # [T, B, C]
        x = temporal_mean(x)      # [B, C]
        return self.head(x)
