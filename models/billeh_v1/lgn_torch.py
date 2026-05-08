from __future__ import annotations

import os
import pickle as pkl

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class TorchLGN(nn.Module):
    """
    PyTorch LGN front-end compatible with Billeh-V1 input.

    Inputs:
      movie: [B, T, H, W]
    Outputs:
      firing_rates: [B, T, N_lgn]
    """

    def __init__(
        self,
        lgn_data_path: str,
        temporal_kernels_path: str | None = None,
        spatial_bin_size: float = 1.0,
        max_spatial_size: float = 15.0,
    ):
        super().__init__()
        if not lgn_data_path or not os.path.isfile(lgn_data_path):
            raise FileNotFoundError(f"lgn_data_path not found: {lgn_data_path}")
        if temporal_kernels_path is None:
            temporal_kernels_path = os.path.join(os.path.dirname(lgn_data_path), "temporal_kernels.pkl")
        if not os.path.isfile(temporal_kernels_path):
            raise FileNotFoundError(
                "temporal_kernels.pkl not found. Please generate it first with the original LGN pipeline. "
                "You can run nanoSNN/scripts/prepare_lgn_kernels.py for offline cache generation. "
                f"Expected: {temporal_kernels_path}"
            )

        d = pd.read_csv(lgn_data_path, delimiter=" ")
        with open(temporal_kernels_path, "rb") as f:
            loaded = pkl.load(f)

        self.register_buffer(
            "spatial_sizes",
            torch.as_tensor(d["spatial_size"].to_numpy(copy=True), dtype=torch.float32),
        )
        self.register_buffer("x_raw", torch.as_tensor(d["x"].to_numpy(copy=True), dtype=torch.float32))
        self.register_buffer("y_raw", torch.as_tensor(d["y"].to_numpy(copy=True), dtype=torch.float32))

        self.register_buffer(
            "non_dominant_x_raw",
            torch.as_tensor(loaded["non_dominant_x"], dtype=torch.float32),
        )
        self.register_buffer(
            "non_dominant_y_raw",
            torch.as_tensor(loaded["non_dominant_y"], dtype=torch.float32),
        )
        self.register_buffer("amplitude", torch.as_tensor(loaded["amplitude"], dtype=torch.float32))
        self.register_buffer("non_dom_amplitude", torch.as_tensor(loaded["non_dom_amplitude"], dtype=torch.float32))
        self.register_buffer(
            "spontaneous_firing_rates",
            torch.as_tensor(loaded["spontaneous_firing_rates"], dtype=torch.float32),
        )
        model_id = d["model_id"].astype(str)
        is_composite = model_id.str.contains("ON") & model_id.str.contains("OFF")
        self.register_buffer(
            "is_composite",
            torch.as_tensor(is_composite.to_numpy(copy=True), dtype=torch.float32),
        )
        self.register_buffer(
            "dom_temporal_kernels",
            torch.as_tensor(loaded["dom_temporal_kernels"], dtype=torch.float32),
        )
        self.register_buffer(
            "non_dom_temporal_kernels",
            torch.as_tensor(loaded["non_dom_temporal_kernels"], dtype=torch.float32),
        )

        self.spatial_bin_size = float(spatial_bin_size)
        self.max_spatial_size = float(max_spatial_size)
        self.n_cells = int(self.spatial_sizes.numel())

    @staticmethod
    def _gaussian_kernel_2d(sigma: float, device: torch.device, dtype: torch.dtype):
        sigma = max(float(sigma), 1e-3)
        radius = max(1, int(round(3.0 * sigma)))
        coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        kernel = torch.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
        kernel = kernel / kernel.sum().clamp_min(1e-12)
        return kernel

    @staticmethod
    def _sample_at_coords(movie: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # movie: [B, T, H, W], x/y: [N] in image coordinates.
        bsz, tsz, h, w = movie.shape
        if x.numel() == 0:
            return movie.new_zeros((bsz, tsz, 0))

        inp = movie.reshape(bsz * tsz, 1, h, w)
        xn = (x / max(1, w - 1)) * 2.0 - 1.0
        yn = (y / max(1, h - 1)) * 2.0 - 1.0
        grid = torch.stack([xn, yn], dim=-1).view(1, -1, 1, 2).repeat(bsz * tsz, 1, 1, 1)
        sampled = F.grid_sample(inp, grid, mode="bilinear", padding_mode="border", align_corners=True)
        sampled = sampled[:, 0, :, 0].view(bsz, tsz, -1)
        return sampled

    @staticmethod
    def _temporal_filter(all_spatial_responses: torch.Tensor, temporal_kernels: torch.Tensor) -> torch.Tensor:
        # all_spatial_responses: [B, T, N], temporal_kernels: [N, K]
        bsz, tsz, n = all_spatial_responses.shape
        k = temporal_kernels.shape[1]
        x = all_spatial_responses.permute(0, 2, 1)  # [B, N, T]
        x = F.pad(x, (k - 1, 0))
        w = temporal_kernels.unsqueeze(1)  # [N, 1, K]
        y = F.conv1d(x, w, groups=n)  # [B, N, T]
        return y.permute(0, 2, 1)

    def _spatial_response(self, movie: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # movie: [B, T, H, W], returns [B, T, N]
        bsz, tsz, h, w = movie.shape
        device, dtype = movie.device, movie.dtype

        x_scaled = x * (w - 1) / 239.0
        y_scaled = y * (h - 1) / 119.0
        x_scaled = x_scaled.clamp(0.0, float(w - 1))
        y_scaled = y_scaled.clamp(0.0, float(h - 1))

        all_responses = movie.new_zeros((bsz, tsz, self.n_cells))
        spatial_range = torch.arange(0.0, self.max_spatial_size, self.spatial_bin_size, device=device, dtype=dtype)
        for i in range(spatial_range.numel() - 1):
            lo, hi = spatial_range[i], spatial_range[i + 1]
            sel = (self.spatial_sizes >= lo) & (self.spatial_sizes < hi)
            idx = torch.where(sel)[0]
            if idx.numel() == 0:
                continue

            sigma = float(torch.round((lo + hi) * 0.5).item()) / 3.0
            kernel = self._gaussian_kernel_2d(sigma=sigma, device=device, dtype=dtype)
            k = kernel.view(1, 1, kernel.shape[0], kernel.shape[1])
            convolved = F.conv2d(
                movie.reshape(bsz * tsz, 1, h, w),
                k,
                padding=(kernel.shape[0] // 2, kernel.shape[1] // 2),
            ).view(bsz, tsz, h, w)
            sampled = self._sample_at_coords(convolved, x_scaled[idx], y_scaled[idx])  # [B, T, M]
            all_responses[:, :, idx] = sampled
        return all_responses

    def forward(self, movie: torch.Tensor) -> torch.Tensor:
        # movie: [B, T, H, W]
        movie = movie.float()
        dom_spatial = self._spatial_response(movie, self.x_raw, self.y_raw)
        non_dom_spatial = self._spatial_response(movie, self.non_dominant_x_raw, self.non_dominant_y_raw)

        dom_filtered = self._temporal_filter(dom_spatial, self.dom_temporal_kernels)
        non_dom_filtered = self._temporal_filter(non_dom_spatial, self.non_dom_temporal_kernels)

        amp = self.amplitude.view(1, 1, -1)
        non_amp = self.non_dom_amplitude.view(1, 1, -1)
        spont = self.spontaneous_firing_rates.view(1, 1, -1)
        comp = self.is_composite.view(1, 1, -1)

        firing_rates = F.relu(dom_filtered * amp + spont)
        multi_firing_rates = firing_rates + F.relu(non_dom_filtered * non_amp + spont)
        firing_rates = firing_rates * (1.0 - comp) + multi_firing_rates * comp
        return firing_rates
