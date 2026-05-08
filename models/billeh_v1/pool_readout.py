from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalizedPoolReadout(nn.Module):
    """
    Strict reproduction of the readout used in Chen-Scherr-Maass (Sci Adv 2022).

    For each class c, average the spike trains of a fixed L5e neuron pool, apply
    a "spike straight-through" gradient scaling, multiply by a learnable scalar,
    then collapse the time axis into ``T // down_sample`` non-overlapping bins.

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
        for c, ids in enumerate(pool_ids):
            t = torch.as_tensor(ids, dtype=torch.long).reshape(-1)
            if t.numel() == 0:
                raise ValueError(f"Pool {c} is empty")
            self.register_buffer(f"pool_{c}", t)
        self.scale_param = nn.Parameter(torch.zeros(1))

    def pool_indices(self, c: int) -> torch.Tensor:
        return getattr(self, f"pool_{c}")

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        # spikes: [B, T, N]
        B, T, _ = spikes.shape
        if T % self.down_sample != 0:
            raise ValueError(
                f"T={T} must be divisible by down_sample={self.down_sample}"
            )

        df = self.dampening_factor
        ds = (1.0 / df) * spikes + (1.0 - 1.0 / df) * spikes.detach()

        outs = []
        for c in range(self.n_classes):
            ids = self.pool_indices(c)
            t_out = ds.index_select(dim=2, index=ids).mean(dim=2)  # [B, T]
            outs.append(t_out)
        out = torch.stack(outs, dim=-1)  # [B, T, n_classes]
        out = out * (1.0 + F.softplus(self.scale_param))

        n_chunks = T // self.down_sample
        out = out.view(B, n_chunks, self.down_sample, self.n_classes).mean(dim=2)
        return out  # [B, n_chunks, n_classes] (pre-softmax logits)
