from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalizedPoolReadout(nn.Module):
    """
    Pool-prior readout: keep Chen-Maass L5e pool structure as a hard constraint
    (neurons outside pool_c never contribute to class c), but allow per-entry
    weights, per-class scale, and per-class bias to be learned.

    At init, output is numerically identical to the strict-reproduction version
    (per-class scale init -> 1 + ln(2), per-entry weight init -> 1/|pool_c|,
    bias init -> 0).

    Args
    ----
    pool_ids: list of LongTensor (or array-like), one per class. Each element is
              the neuron indices for that class' L5e readout pool. Pool sizes
              may differ.
    dampening_factor: gradient scaling for spikes (matches BillehColumnTorch).
    down_sample: number of time steps per output bin.
    n_classes: number of pools / classes.
    """

    def __init__(
        self,
        pool_ids,
        dampening_factor: float = 0.3,
        down_sample: int = 50,
        n_classes: int = 10,
    ):
        super().__init__()
        if len(pool_ids) != n_classes:
            raise ValueError(
                f"pool_ids must have length {n_classes}, got {len(pool_ids)}"
            )
        self.n_classes = int(n_classes)
        self.down_sample = int(down_sample)
        self.dampening_factor = float(dampening_factor)

        flat_indices, flat_class, init_weights = [], [], []
        for c, ids in enumerate(pool_ids):
            t = torch.as_tensor(ids, dtype=torch.long).reshape(-1)
            if t.numel() == 0:
                raise ValueError(f"Pool {c} is empty")
            flat_indices.append(t)
            flat_class.append(torch.full((t.numel(),), c, dtype=torch.long))
            init_weights.append(torch.full((t.numel(),), 1.0 / float(t.numel())))

        self.register_buffer("flat_indices", torch.cat(flat_indices))
        self.register_buffer("flat_class", torch.cat(flat_class))

        # Per-entry weight inside each pool; init = 1/|pool_c| reproduces the
        # original mean-over-pool exactly at step 0.
        self.pool_weights = nn.Parameter(torch.cat(init_weights))

        # Per-class affine. scale init 0 -> softplus(0) = ln(2) -> total
        # 1 + ln(2) ≈ 1.693, matching the original global scale init.
        self.scale = nn.Parameter(torch.zeros(n_classes))
        self.bias = nn.Parameter(torch.zeros(n_classes))

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        # spikes: [B, T, N]
        B, T, _ = spikes.shape
        if T % self.down_sample != 0:
            raise ValueError(
                f"T={T} must be divisible by down_sample={self.down_sample}"
            )

        df = self.dampening_factor
        ds = (1.0 / df) * spikes + (1.0 - 1.0 / df) * spikes.detach()

        # [B, T, P] where P = sum of pool sizes.
        gathered = ds.index_select(dim=2, index=self.flat_indices)
        weighted = gathered * self.pool_weights.view(1, 1, -1)

        # Sum entries within each class -> [B, T, n_classes].
        idx = self.flat_class.view(1, 1, -1).expand(B, T, -1)
        out = torch.zeros(
            B, T, self.n_classes, device=weighted.device, dtype=weighted.dtype
        )
        out = out.scatter_add(2, idx, weighted)

        # Per-class affine.
        out = out * (1.0 + F.softplus(self.scale)).view(1, 1, -1)
        out = out + self.bias.view(1, 1, -1)

        # Time downsample (mean over down_sample bins).
        n_chunks = T // self.down_sample
        out = out.view(B, n_chunks, self.down_sample, self.n_classes).mean(dim=2)
        return out
