from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..common.registry import register_model
from ..common.spike_ops import temporal_mean


class PatchEmbed(nn.Module):
    """Conv patchify: [B,C,H,W] -> [B,N,D]."""

    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.img_size = int(img_size)
        self.patch_size = int(patch_size)
        if self.img_size % self.patch_size != 0:
            raise ValueError(f"img_size must be divisible by patch_size: {img_size=} {patch_size=}")
        self.grid = self.img_size // self.patch_size
        self.num_patches = self.grid * self.grid
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # [B,D,gh,gw]
        x = x.flatten(2).transpose(1, 2).contiguous()  # [B,N,D]
        return x


def _sinusoidal_pos_embed(n_tokens: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Classic Vaswani sinusoidal position embedding, shape [1, n_tokens, dim]."""
    if dim % 2 != 0:
        raise ValueError(f"sinusoidal pos-embed requires even dim, got {dim}")
    pos = torch.arange(n_tokens, device=device, dtype=dtype).unsqueeze(1)  # [N,1]
    i = torch.arange(dim // 2, device=device, dtype=dtype).unsqueeze(0)    # [1,D/2]
    inv = torch.exp(-math.log(10000.0) * (2 * i) / dim)
    angles = pos * inv
    emb = torch.cat([angles.sin(), angles.cos()], dim=1)  # [N,D]
    return emb.unsqueeze(0)


class MultiHeadSelfAttention(nn.Module):
    """Scaled dot-product MHA (Attention Is All You Need)."""

    def __init__(self, dim: int, num_heads: int, attn_dropout: float = 0.0, proj_dropout: float = 0.0):
        super().__init__()
        dim = int(dim)
        num_heads = int(num_heads)
        if dim % num_heads != 0:
            raise ValueError(f"dim must be divisible by num_heads: {dim=} {num_heads=}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C]
        B, N, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,N,D]
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B,H,N,N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # [B,H,N,D]
        out = out.transpose(1, 2).reshape(B, N, C).contiguous()
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class FeedForward(nn.Module):
    """Position-wise FFN (Attention Is All You Need)."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """Pre-LN encoder block: LN -> MHA -> res, LN -> FFN -> res."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, attn_dropout=attn_dropout, proj_dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * float(mlp_ratio))
        self.ffn = FeedForward(dim, hidden, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


@dataclass(frozen=True)
class _Cfg:
    name: str
    T: int
    num_classes: int
    image_size: int
    patch_size: int
    in_channels: int
    embed_dim: int
    depth: int
    num_heads: int
    mlp_ratio: float
    dropout: float
    attn_dropout: float
    pos_embed: str
    pool: str


def _read_cfg(model_config) -> _Cfg:
    return _Cfg(
        name=str(getattr(model_config, "name", "vit")),
        T=int(getattr(model_config, "T", 1)),
        num_classes=int(getattr(model_config, "num_classes", 10)),
        image_size=int(getattr(model_config, "image_size", 32)),
        patch_size=int(getattr(model_config, "patch_size", 4)),
        in_channels=int(getattr(model_config, "in_channels", 3)),
        embed_dim=int(getattr(model_config, "embed_dim", 192)),
        depth=int(getattr(model_config, "depth", 4)),
        num_heads=int(getattr(model_config, "num_heads", 3)),
        mlp_ratio=float(getattr(model_config, "mlp_ratio", 4.0)),
        dropout=float(getattr(model_config, "dropout", 0.0)),
        attn_dropout=float(getattr(model_config, "attn_dropout", 0.0)),
        pos_embed=str(getattr(model_config, "pos_embed", "learned")).lower(),  # learned | sinusoidal
        pool=str(getattr(model_config, "pool", "cls")).lower(),  # cls | mean
    )


@register_model("vit")
class VisionTransformer(nn.Module):
    """A plain ViT-style Transformer encoder, aligned with 'Attention Is All You Need' blocks.

    Input:  [T, B, C, H, W]
    Output: [B, num_classes]
    """

    def __init__(self, model_config):
        super().__init__()
        cfg = _read_cfg(model_config)
        self.T = cfg.T
        self.num_classes = cfg.num_classes

        self.patch = PatchEmbed(cfg.image_size, cfg.patch_size, cfg.in_channels, cfg.embed_dim)
        n_tokens = self.patch.num_patches + 1  # + cls

        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        if cfg.pos_embed == "learned":
            self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, cfg.embed_dim))
            self._pos_embed_kind = "learned"
        elif cfg.pos_embed == "sinusoidal":
            self.register_buffer("pos_embed", torch.zeros(1, n_tokens, cfg.embed_dim), persistent=False)
            self._pos_embed_kind = "sinusoidal"
        else:
            raise ValueError(f"Unsupported pos_embed={cfg.pos_embed!r}")

        self.pos_drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    dim=cfg.embed_dim,
                    num_heads=cfg.num_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    dropout=cfg.dropout,
                    attn_dropout=cfg.attn_dropout,
                )
                for _ in range(cfg.depth)
            ]
        )
        self.norm = nn.LayerNorm(cfg.embed_dim)

        self.pool = cfg.pool
        if self.pool not in {"cls", "mean"}:
            raise ValueError(f"Unsupported pool={self.pool!r}")

        self.head = nn.Linear(cfg.embed_dim, cfg.num_classes) if cfg.num_classes > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        if getattr(self, "_pos_embed_kind", "") == "learned":
            nn.init.normal_(self.pos_embed, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0.0)
                nn.init.constant_(m.weight, 1.0)

    def _add_pos(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self, "_pos_embed_kind", "") == "sinusoidal":
            pe = _sinusoidal_pos_embed(x.shape[1], x.shape[2], x.device, x.dtype)
            return x + pe
        return x + self.pos_embed.to(dtype=x.dtype, device=x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T,B,C,H,W] (static inputs are expanded in LitVisionSNN)
        T, B = x.shape[0], x.shape[1]
        x = rearrange(x, "t b c h w -> (t b) c h w")
        x = self.patch(x)  # [TB,N,D]

        cls = self.cls_token.expand(x.shape[0], -1, -1)  # [TB,1,D]
        x = torch.cat([cls, x], dim=1)  # [TB,1+N,D]
        x = self._add_pos(x)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if self.pool == "cls":
            x = x[:, 0]  # [TB,D]
        else:
            x = x[:, 1:].mean(dim=1)  # [TB,D]

        x = self.head(x)  # [TB,C]
        x = x.view(T, B, -1)
        x = temporal_mean(x)  # [B,C]
        return x
