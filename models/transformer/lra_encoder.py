from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.registry import register_model


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_dim: int | None = None,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        dim = int(dim)
        num_heads = int(num_heads)
        qkv_dim = int(qkv_dim if qkv_dim is not None else dim)
        if qkv_dim % num_heads != 0:
            raise ValueError(f"qkv_dim must be divisible by num_heads: {qkv_dim=} {num_heads=}")
        self.num_heads = num_heads
        self.qkv_dim = qkv_dim
        self.head_dim = qkv_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # LRA uses qkv_features (qkv_dim) that can differ from emb_dim (dim).
        self.qkv = nn.Linear(dim, 3 * qkv_dim, bias=False)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(qkv_dim, dim, bias=False)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,N,Dh]
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # [B,H,N,Dh]
        out = out.transpose(1, 2).reshape(B, N, self.qkv_dim).contiguous()
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class FeedForward(nn.Module):
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
    """Pre-LN transformer encoder block (AIAYN style)."""

    def __init__(self, dim: int, num_heads: int, qkv_dim: int, mlp_dim: int, dropout: float, attn_dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, qkv_dim=qkv_dim, attn_dropout=attn_dropout, proj_dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


@dataclass(frozen=True)
class _Cfg:
    vocab_size: int
    max_len: int
    emb_dim: int
    num_heads: int
    num_layers: int
    qkv_dim: int
    mlp_dim: int
    dropout: float
    attn_dropout: float
    learn_pos_emb: bool
    classifier_pool: str
    num_classes: int


def _cfg(model_config) -> _Cfg:
    return _Cfg(
        vocab_size=int(getattr(model_config, "vocab_size", 256)),
        max_len=int(getattr(model_config, "max_len", 1024)),
        emb_dim=int(getattr(model_config, "emb_dim", 128)),
        num_heads=int(getattr(model_config, "num_heads", 8)),
        num_layers=int(getattr(model_config, "num_layers", 1)),
        qkv_dim=int(getattr(model_config, "qkv_dim", getattr(model_config, "qkv_features", 64))),
        mlp_dim=int(getattr(model_config, "mlp_dim", 128)),
        dropout=float(getattr(model_config, "dropout_rate", getattr(model_config, "dropout", 0.3))),
        attn_dropout=float(getattr(model_config, "attention_dropout_rate", getattr(model_config, "attn_dropout", 0.2))),
        learn_pos_emb=bool(getattr(model_config, "learn_pos_emb", True)),
        classifier_pool=str(getattr(model_config, "classifier_pool", "CLS")).upper(),
        num_classes=int(getattr(model_config, "num_classes", 10)),
    )


def _sinusoidal_pos_embed(n_tokens: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if dim % 2 != 0:
        raise ValueError(f"sinusoidal pos-embed requires even dim, got {dim}")
    pos = torch.arange(n_tokens, device=device, dtype=dtype).unsqueeze(1)
    i = torch.arange(dim // 2, device=device, dtype=dtype).unsqueeze(0)
    inv = torch.exp(-math.log(10000.0) * (2 * i) / dim)
    angles = pos * inv
    emb = torch.cat([angles.sin(), angles.cos()], dim=1)
    return emb.unsqueeze(0)


@register_model("lra_transformer")
class LRATransformerEncoder(nn.Module):
    """LRA-style Transformer encoder for token sequences (e.g. CIFAR10 flattened grayscale pixels).

    Input:  x [B, L] of int token ids (0..vocab_size-1).
    Output: logits [B, num_classes]
    """

    def __init__(self, model_config):
        super().__init__()
        cfg = _cfg(model_config)
        self.cfg = cfg
        self.expects_temporal_input = False

        self.embed = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.emb_dim))

        n_tokens = cfg.max_len + 1  # + CLS
        if cfg.learn_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, cfg.emb_dim))
            nn.init.normal_(self.pos_embed, std=0.02)
            self._pos_kind = "learned"
        else:
            self.register_buffer("pos_embed", torch.zeros(1, n_tokens, cfg.emb_dim), persistent=False)
            self._pos_kind = "sinusoidal"

        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    dim=cfg.emb_dim,
                    num_heads=cfg.num_heads,
                    qkv_dim=cfg.qkv_dim,
                    mlp_dim=cfg.mlp_dim,
                    dropout=cfg.dropout,
                    attn_dropout=cfg.attn_dropout,
                )
                for _ in range(cfg.num_layers)
            ]
        )
        self.norm = nn.LayerNorm(cfg.emb_dim)
        # LRA classifier_head: Dense(mlp_dim)->ReLU->Dense(num_classes)
        self.head = (
            nn.Sequential(
                nn.Linear(cfg.emb_dim, cfg.mlp_dim, bias=True),
                nn.ReLU(),
                nn.Linear(cfg.mlp_dim, cfg.num_classes, bias=True),
            )
            if cfg.num_classes > 0
            else nn.Identity()
        )

        nn.init.normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0.0)
                nn.init.constant_(m.weight, 1.0)

    def _pos(self, n_tokens: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self._pos_kind == "sinusoidal":
            return _sinusoidal_pos_embed(n_tokens, self.cfg.emb_dim, device, dtype)
        return self.pos_embed[:, :n_tokens].to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not (x.ndim == 2 or (x.ndim == 3 and x.shape[-1] == 1)):
            raise ValueError(f"LRATransformerEncoder expects [B, L] or [B, L, 1], got {tuple(x.shape)}")
        if x.ndim == 3:
            x = x.squeeze(-1)
        B, L = x.shape
        if L != self.cfg.max_len:
            raise ValueError(f"Expected sequence length {self.cfg.max_len}, got {L}")

        x = x.long()
        x = self.embed(x)  # [B,L,D]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # [B,1+L,D]
        x = x + self._pos(x.shape[1], x.device, x.dtype)
        x = self.drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        pool = self.cfg.classifier_pool
        if pool == "CLS":
            x = x[:, 0]
        elif pool == "MEAN":
            x = x[:, 1:].mean(dim=1)
        elif pool == "MAX":
            x = x[:, 1:].amax(dim=1)
        else:
            raise ValueError(f"Unsupported classifier_pool={pool!r}")

        return self.head(x)
