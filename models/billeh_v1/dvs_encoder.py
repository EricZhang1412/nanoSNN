from __future__ import annotations

import math

import torch
import torch.nn as nn


class DVSDirectEncoder(nn.Module):
    """Map DVS event frames directly to V1 input current, bypassing LGN.

    Input:  [T_dvs, B, C, H, W] (nanoSNN event-data convention)
    Output: [B, T_model, n_out] where n_out = n_receptors * n_neurons

    Architecture:
      Conv2d(C, h, 3, stride=2, padding=1) -> GELU      # H -> H/2
      Conv2d(h, 2h, 3, stride=2, padding=1) -> GELU     # -> H/4
      Conv2d(2h, 4h, 3, stride=2, padding=1) -> GELU    # -> H/8
      flatten -> Linear(4h * (H/8)^2, n_out)

    K-replay: if T_dvs < T_model and T_model % T_dvs == 0, each DVS frame's
    encoded vector is held for T_model // T_dvs consecutive timesteps so V1
    has time to settle on each frame.
    """

    def __init__(
        self,
        in_channels: int,
        image_size: int,
        n_out: int,
        T_model: int,
        hidden: int = 16,
        output_scale: float = 20.0,
    ):
        super().__init__()
        if image_size % 8 != 0:
            raise ValueError(
                f"DVSDirectEncoder needs image_size divisible by 8 (3x stride-2 conv), got {image_size}"
            )
        self.image_size = int(image_size)
        self.n_out = int(n_out)
        self.T_model = int(T_model)
        self.hidden = int(hidden)

        h = self.hidden
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, h,     kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(h,           h * 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(h * 2,       h * 4, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        spatial = self.image_size // 8
        self.feat_dim = h * 4 * spatial * spatial
        self.proj = nn.Linear(self.feat_dim, self.n_out)
        # LayerNorm + tanh + fixed scale: bounds encoder output magnitude so
        # training cannot push V1 into saturation (which kills the surrogate
        # gradient in SpikeGauss.backward). The previous learnable log scale
        # had nothing stopping it from blowing up.
        self.proj_norm = nn.LayerNorm(self.n_out)
        self.register_buffer("output_scale", torch.tensor(float(output_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"DVSDirectEncoder expects 5D input [T,B,C,H,W], got {tuple(x.shape)}")
        # nanoSNN event data is delivered as [T, B, C, H, W]; reorder to [B, T, C, H, W].
        x = x.permute(1, 0, 2, 3, 4)
        B, T_in, C, H, W = x.shape
        if H != self.image_size or W != self.image_size:
            raise ValueError(
                f"DVSDirectEncoder built for image_size={self.image_size}, got H={H} W={W}"
            )

        x = x.reshape(B * T_in, C, H, W)
        x = self.conv(x).flatten(1)
        x = self.proj(x)
        x = self.proj_norm(x)
        x = torch.tanh(x) * self.output_scale
        x = x.reshape(B, T_in, self.n_out)

        if T_in != self.T_model:
            if self.T_model <= 0 or self.T_model % T_in != 0:
                raise ValueError(
                    f"K-replay requires T_model ({self.T_model}) to be a positive multiple of "
                    f"T_dvs ({T_in})"
                )
            k = self.T_model // T_in
            x = x.repeat_interleave(k, dim=1)
        return x
