from __future__ import annotations

import torch
import torch.nn as nn

from ..common.registry import register_model
from .laminar_block import LaminarBlock
from .ff_block import FeedforwardBlock


def _conv_stem(in_channels: int, stem_dim: int) -> nn.Module:
    """Non-spiking conv stem: [B, C, H, W] -> [B, stem_dim].
    Run per-timestep over T.

    Uses max-pool (not avg) because the synthetic temporal-order task has
    spatially sparse cue blobs (~16 / 1024 pixels). Avg-pool over the full
    feature map crushes the signal by ~500x, which leaves downstream GLIF3
    populations silent at init regardless of projection scale.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(16),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.AdaptiveMaxPool2d(2),    # 8x8 -> 2x2 preserves blob peaks
        nn.Flatten(),                # 32 * 4 = 128
        nn.Linear(128, stem_dim, bias=True),
    )


@register_model("dyco_snn")
class DyCoSNN(nn.Module):
    """
    DyCo-SNN top-level model for Block A four-quadrant ablation.

    Config switches (from model_config):
      mode: 'd_only' | 'c_only' | 'dc' | 'ff'
      n_blocks: int (default 2)
      stem_dim: int (default 64)
      block_n_l4 / n_23e / n_23i / n_l5
      alpha_init: float in [0,1] (frozen if alpha_learnable=False)
      alpha_learnable: bool
      learnable_tau / learnable_asc / heterogeneous_init: bool
      recurrent_enabled: bool
      ff_hidden_mult, ff_depth: only used when mode='ff'
    """

    def __init__(self, model_config):
        super().__init__()
        cfg = model_config
        self.T = int(getattr(cfg, "T", 8))
        self.num_classes = int(getattr(cfg, "num_classes", 2))
        self.in_channels = int(getattr(cfg, "in_channels", 2))
        self.mode = str(getattr(cfg, "mode", "dc")).lower()

        self.stem_dim = int(getattr(cfg, "stem_dim", 64))
        self.n_blocks = int(getattr(cfg, "n_blocks", 2))
        # Stem outputs continuous values with std~0.05 on sparse event input,
        # whereas inter-block projections are calibrated for binary spike
        # inputs (std~0.2-0.3). Multiply stem output to match that scale so
        # block 0's L4 actually fires at T<=8 without over-driving block 1+.
        self.stem_out_scale = float(getattr(cfg, "stem_out_scale", 5.0))

        n_l4 = int(getattr(cfg, "n_l4", 48))
        n_23e = int(getattr(cfg, "n_23e", 64))
        n_23i = int(getattr(cfg, "n_23i", 16))
        n_l5 = int(getattr(cfg, "n_l5", 48))

        learnable_tau = bool(getattr(cfg, "learnable_tau", True))
        learnable_asc = bool(getattr(cfg, "learnable_asc", True))
        heterogeneous_init = bool(getattr(cfg, "heterogeneous_init", True))
        recurrent_enabled = bool(getattr(cfg, "recurrent_enabled", True))
        alpha_init = float(getattr(cfg, "alpha_init", 0.5))
        self.alpha_learnable = bool(getattr(cfg, "alpha_learnable", True))

        self.stem = _conv_stem(self.in_channels, self.stem_dim)

        # Block stack.
        self.blocks = nn.ModuleList()
        cur_in = self.stem_dim
        for _ in range(self.n_blocks):
            if self.mode == "ff":
                blk = FeedforwardBlock(
                    in_features=cur_in,
                    n_l5=n_l5,
                    hidden_mult=float(getattr(cfg, "ff_hidden_mult", 2.0)),
                    depth=int(getattr(cfg, "ff_depth", 2)),
                )
            else:
                blk = LaminarBlock(
                    in_features=cur_in,
                    n_l4=n_l4, n_23e=n_23e, n_23i=n_23i, n_l5=n_l5,
                    learnable_tau=learnable_tau,
                    learnable_asc=learnable_asc,
                    heterogeneous_init=heterogeneous_init,
                    recurrent_enabled=recurrent_enabled,
                )
            self.blocks.append(blk)
            cur_in = n_l5

        # Allocation alphas (one per block, learnable or frozen).
        # Stored as logit; alpha = sigmoid(logit) in [0,1].
        if self.mode != "ff":
            init_logit = torch.logit(torch.tensor(alpha_init, dtype=torch.float32).clamp(1e-4, 1 - 1e-4))
            init_logits = init_logit.repeat(self.n_blocks)
            self.alpha_logits = nn.Parameter(init_logits, requires_grad=self.alpha_learnable)
        else:
            self.register_parameter("alpha_logits", None)

        self.readout = nn.Linear(n_l5, self.num_classes, bias=True)

        # Stash hooks for LitVisionSNN.
        self.last_spikes = None
        self.latest_spike_rate_hz = None

    def get_alphas(self) -> torch.Tensor:
        if self.alpha_logits is None:
            return torch.zeros(self.n_blocks)
        return torch.sigmoid(self.alpha_logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, B, C, H, W] (event input, expanded by LitVisionSNN if static).
        T, B = x.shape[0], x.shape[1]
        # Apply stem per timestep.
        x_flat = x.reshape(T * B, *x.shape[2:])
        feat_flat = self.stem(x_flat) * self.stem_out_scale  # [T*B, stem_dim]
        feat = feat_flat.reshape(T, B, self.stem_dim)

        # Distribute alphas to blocks.
        alphas = self.get_alphas() if self.alpha_logits is not None else None
        for i, blk in enumerate(self.blocks):
            if alphas is not None:
                blk.set_alpha(alphas[i])
            else:
                blk.set_alpha(torch.tensor(0.0, device=feat.device))
            feat = blk(feat)                 # [T, B, n_l5]

        # Readout: mean over T -> linear.
        mean_feat = feat.mean(dim=0)         # [B, n_l5]
        logits = self.readout(mean_feat)     # [B, num_classes]

        # Hooks for LitVisionSNN logging.
        self.last_spikes = feat.detach()
        with torch.no_grad():
            # rough Hz: mean spike per timestep -> 1000 Hz cap if dt=1ms
            self.latest_spike_rate_hz = float(feat.float().mean().item() * 1000.0)
        return logits
