from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.registry import register_model
from .billeh_column import BillehColumnTorch
from .lgn_torch import TorchLGN
from .load_sparse_torch import load_billeh_torch


class RateReadout(nn.Module):
    def __init__(self, n_neurons: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(n_neurons, n_classes)

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        rates = spikes.float().mean(dim=1)
        return self.fc(rates)


@register_model("billeh_v1")
class BillehV1Classifier(nn.Module):
    """
    V1 GLIF backbone + linear rate readout for classification.
    Input supports:
      - static expanded temporal: [T, B, C, H, W]
      - direct temporal current/spike-like: [B, T, n_input]
    """

    def __init__(self, model_config):
        super().__init__()
        self.T = int(getattr(model_config, "T", 8))
        self.auto_n_input_from_in_channels = bool(
            getattr(model_config, "auto_n_input_from_in_channels", False)
        )
        in_channels = int(getattr(model_config, "in_channels", 1))
        if self.auto_n_input_from_in_channels:
            self.n_input = in_channels
        else:
            self.n_input = int(getattr(model_config, "n_input", 17400))
        self.n_neurons = int(getattr(model_config, "n_neurons", 1000))
        self.num_classes = int(getattr(model_config, "num_classes", 10))
        self.seed = int(getattr(model_config, "seed", 3000))
        self.full_core = bool(getattr(model_config, "full_core", False))
        self.train_v1 = bool(getattr(model_config, "train_v1", False))
        self.encoding = str(getattr(model_config, "encoding", "poisson")).lower()
        self.encoding_gain = float(getattr(model_config, "encoding_gain", 1.0))
        # If true, keep temporal length from input instead of forcing model_config.T.
        self.use_input_t = bool(getattr(model_config, "use_input_t", False))
        self.use_lgn = bool(getattr(model_config, "use_lgn", False))
        self.lgn_input_height = int(getattr(model_config, "lgn_input_height", 0) or 0)
        self.lgn_input_width = int(getattr(model_config, "lgn_input_width", 0) or 0)

        self.lgn = None
        if self.use_lgn:
            lgn_data_path = str(getattr(model_config, "lgn_data_path", "")).strip()
            lgn_temporal_kernels_path = str(getattr(model_config, "lgn_temporal_kernels_path", "")).strip() or None
            self.lgn = TorchLGN(
                lgn_data_path=lgn_data_path,
                temporal_kernels_path=lgn_temporal_kernels_path,
            )
            auto_n_input_from_lgn = bool(getattr(model_config, "auto_n_input_from_lgn", True))
            if auto_n_input_from_lgn:
                self.n_input = int(self.lgn.n_cells)

        data_dir = str(getattr(model_config, "billeh_data_dir", "")).strip()
        if not data_dir:
            raise ValueError("model_config.billeh_data_dir is required for billeh_v1")
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"billeh_data_dir not found: {data_dir}")

        loaded = load_billeh_torch(
            n_input=self.n_input,
            n_neurons=self.n_neurons,
            core_only=not self.full_core,
            data_dir=data_dir,
            seed=self.seed,
            connected_selection=self.full_core,
            localized_readout=False,
            neurons_per_output=4,
            device="cpu",
        )

        bkg_weights = loaded["bkg_weights"]
        if isinstance(bkg_weights, torch.Tensor):
            bkg_weights = bkg_weights.detach().cpu().numpy()

        self.v1 = BillehColumnTorch(
            loaded["network"],
            loaded["input_population"],
            bkg_weights,
            device="cpu",
        )
        self.readout = RateReadout(self.n_neurons, self.num_classes)
        self.latest_spike_rate_hz: torch.Tensor | None = None

        if not self.train_v1:
            self.v1.eval()
            for p in self.v1.parameters():
                p.requires_grad_(False)

    def _to_b_t_n(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_lgn:
            return self._to_b_t_n_via_lgn(x)

        # [T, B, C, H, W] (static expanded temporal) -> [B, T, n_input]
        if x.ndim == 5:
            if (not self.use_input_t) and x.shape[0] != self.T:
                # Keep the first T frames for consistency with model setting.
                t_keep = min(self.T, x.shape[0])
                x = x[:t_keep]
            t, b = x.shape[0], x.shape[1]
            x = x.permute(1, 0, 2, 3, 4).reshape(b, t, -1)
        elif x.ndim == 3:
            # already [B, T, n_input?]
            b, t = x.shape[:2]
            x = x.reshape(b, t, -1)
            if (not self.use_input_t) and t != self.T:
                t_keep = min(self.T, t)
                x = x[:, :t_keep]
        else:
            raise ValueError(f"Unsupported input shape for billeh_v1: {tuple(x.shape)}")

        # Resize feature dim to n_input if needed.
        if x.shape[-1] != self.n_input:
            bsz, tsz = x.shape[0], x.shape[1]
            x = x.reshape(bsz * tsz, 1, x.shape[-1])
            x = F.interpolate(x, size=self.n_input, mode="linear", align_corners=False)
            x = x.reshape(bsz, tsz, self.n_input)

        # By default, guarantee [B, T, n_input] with T=self.T.
        # When use_input_t=True, temporal length is kept as provided by input.
        if not self.use_input_t:
            if x.shape[1] < self.T:
                repeat_n = self.T - x.shape[1]
                x = torch.cat([x, x[:, -1:, :].repeat(1, repeat_n, 1)], dim=1)
            elif x.shape[1] > self.T:
                x = x[:, : self.T]
        return x

    def _to_b_t_n_via_lgn(self, x: torch.Tensor) -> torch.Tensor:
        if self.lgn is None:
            raise RuntimeError("use_lgn=True but LGN module is not initialized")

        if x.ndim == 5:
            # [T, B, C, H, W] -> [B, T, H, W]
            x = x.permute(1, 0, 2, 3, 4)
            if x.shape[2] > 1:
                movie = x.mean(dim=2)
            else:
                movie = x[:, :, 0]
        elif x.ndim == 4:
            # [B, C, H, W] -> [B, T=1, H, W]
            movie = x.mean(dim=1, keepdim=False).unsqueeze(1)
        elif x.ndim == 3:
            # Optional reshape path for sequence-like input: [B, T, H*W]
            h = int(getattr(self, "lgn_input_height", 0) or 0)
            w = int(getattr(self, "lgn_input_width", 0) or 0)
            if h <= 0 or w <= 0:
                raise ValueError(
                    "use_lgn with 3D input requires model_config.lgn_input_height and lgn_input_width"
                )
            if x.shape[-1] != h * w:
                raise ValueError(f"Expected input last dim={h*w}, got {x.shape[-1]}")
            movie = x.reshape(x.shape[0], x.shape[1], h, w)
        else:
            raise ValueError(f"Unsupported input shape for LGN path: {tuple(x.shape)}")

        x_btn = self.lgn(movie)
        if not self.use_input_t:
            if x_btn.shape[1] < self.T:
                repeat_n = self.T - x_btn.shape[1]
                x_btn = torch.cat([x_btn, x_btn[:, -1:, :].repeat(1, repeat_n, 1)], dim=1)
            elif x_btn.shape[1] > self.T:
                x_btn = x_btn[:, : self.T]

        if x_btn.shape[-1] != self.n_input:
            bsz, tsz = x_btn.shape[0], x_btn.shape[1]
            x_btn = x_btn.reshape(bsz * tsz, 1, x_btn.shape[-1])
            x_btn = F.interpolate(x_btn, size=self.n_input, mode="linear", align_corners=False)
            x_btn = x_btn.reshape(bsz, tsz, self.n_input)
        return x_btn

    def _encode(self, x_btn: torch.Tensor) -> torch.Tensor:
        if self.encoding == "identity":
            return x_btn.float()

        # Input to V1 is expected in [0,1]-like range.
        x_btn = x_btn.float()
        # For 1-D per-step features (n_input==1), normalize over time to avoid all-zero collapse.
        norm_dim = 1 if x_btn.shape[-1] == 1 else -1
        x_btn = x_btn - x_btn.amin(dim=norm_dim, keepdim=True)
        denom = x_btn.amax(dim=norm_dim, keepdim=True).clamp_min(1e-6)
        x_btn = x_btn / denom

        if self.encoding == "repeat":
            return x_btn
        if self.encoding == "poisson":
            p = (x_btn * self.encoding_gain).clamp(0.0, 1.0)
            return (torch.rand_like(p) < p).float()
        raise ValueError(f"Unknown encoding: {self.encoding}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_btn = self._to_b_t_n(x)
        x_btn = self._encode(x_btn)
        self.v1.device = x_btn.device
        if self.train_v1:
            spikes, _, _ = self.v1(x_btn)
        else:
            with torch.no_grad():
                spikes, _, _ = self.v1(x_btn)
        # Mean spike probability -> Hz (dt=1 ms => *1000).
        self.latest_spike_rate_hz = spikes.float().mean() * 1000.0
        return self.readout(spikes)
