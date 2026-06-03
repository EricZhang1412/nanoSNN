from __future__ import annotations

import torch
from torch.utils.data import Dataset


class SyntheticTemporalOrderDataset(Dataset):
    """
    Two-cue temporal-order task.

    Each sample is a movie of shape [T, C=2, H, W].
    A 'cue-A' blob appears at one polarity channel in one timestep, and a
    'cue-B' blob at the other polarity in another timestep, at distinct
    spatial positions. Label = 0 if A precedes B, 1 if B precedes A.

    Important properties:
      - Shuffling frames destroys the label (used for Block B / gate 2).
      - Static-frame average has identical class statistics (no shortcut).
      - Cue presence/count is identical across classes.
    """

    def __init__(
        self,
        n_samples: int = 4096,
        T: int = 8,
        H: int = 32,
        W: int = 32,
        blob_size: int = 4,
        min_gap: int = 1,
        seed: int = 0,
        # Optional perturbations for Block B counterfactual evaluation.
        shuffle_time: bool = False,
        reverse_time: bool = False,
        first_last_only: bool = False,
    ):
        super().__init__()
        self.n_samples = int(n_samples)
        self.T = int(T)
        self.H = int(H)
        self.W = int(W)
        self.blob_size = int(blob_size)
        self.min_gap = int(min_gap)
        self.shuffle_time = bool(shuffle_time)
        self.reverse_time = bool(reverse_time)
        self.first_last_only = bool(first_last_only)
        if self.T < 2 * (self.min_gap + 1):
            raise ValueError(f"T={T} too small for min_gap={min_gap}")
        self.classes = ["A_before_B", "B_before_A"]
        self.gen = torch.Generator().manual_seed(seed)

        # Pre-sample every sample's parameters for determinism across epochs.
        self._params = []
        for _ in range(self.n_samples):
            label = int(torch.randint(0, 2, (1,), generator=self.gen).item())
            # Pick two distinct timesteps with order according to label.
            while True:
                t0 = int(torch.randint(0, self.T, (1,), generator=self.gen).item())
                t1 = int(torch.randint(0, self.T, (1,), generator=self.gen).item())
                if abs(t0 - t1) > self.min_gap:
                    break
            t_A, t_B = (min(t0, t1), max(t0, t1)) if label == 0 else (max(t0, t1), min(t0, t1))
            # Pick two distinct positions.
            posA = (int(torch.randint(0, self.H - self.blob_size, (1,), generator=self.gen).item()),
                    int(torch.randint(0, self.W - self.blob_size, (1,), generator=self.gen).item()))
            while True:
                posB = (int(torch.randint(0, self.H - self.blob_size, (1,), generator=self.gen).item()),
                        int(torch.randint(0, self.W - self.blob_size, (1,), generator=self.gen).item()))
                if abs(posB[0] - posA[0]) + abs(posB[1] - posA[1]) >= self.blob_size:
                    break
            self._params.append((label, t_A, t_B, posA, posB))

    def __len__(self):
        return self.n_samples

    def _draw_blob(self, frame: torch.Tensor, ch: int, pos):
        y, x = pos
        b = self.blob_size
        frame[ch, y:y + b, x:x + b] = 1.0

    def __getitem__(self, idx):
        label, t_A, t_B, posA, posB = self._params[idx]
        movie = torch.zeros(self.T, 2, self.H, self.W, dtype=torch.float32)
        # Cue A in channel 0, cue B in channel 1.
        self._draw_blob(movie[t_A], ch=0, pos=posA)
        self._draw_blob(movie[t_B], ch=1, pos=posB)

        if self.first_last_only:
            keep = torch.zeros(self.T, dtype=torch.bool)
            keep[0] = True
            keep[-1] = True
            movie = movie * keep.view(-1, 1, 1, 1)
        if self.reverse_time:
            movie = movie.flip(0)
        if self.shuffle_time:
            perm = torch.randperm(self.T)
            movie = movie[perm]

        return movie, int(label)


def build_synthetic_temporal_order(data_config, split: str) -> SyntheticTemporalOrderDataset:
    n_train = int(getattr(data_config, "n_train", 4096))
    n_val = int(getattr(data_config, "n_val", 512))
    n_test = int(getattr(data_config, "n_test", 1024))
    T = int(getattr(data_config, "T", 8))
    H = int(getattr(data_config, "image_size", 32))
    W = int(getattr(data_config, "image_size", 32))
    blob_size = int(getattr(data_config, "blob_size", 4))
    min_gap = int(getattr(data_config, "min_gap", 1))
    base_seed = int(getattr(data_config, "split_seed", 42))

    shuffle_time = bool(getattr(data_config, "shuffle_time", False))
    reverse_time = bool(getattr(data_config, "reverse_time", False))
    first_last_only = bool(getattr(data_config, "first_last_only", False))

    if split == "train":
        n, seed = n_train, base_seed
    elif split == "val":
        n, seed = n_val, base_seed + 1
    else:
        n, seed = n_test, base_seed + 2

    return SyntheticTemporalOrderDataset(
        n_samples=n, T=T, H=H, W=W,
        blob_size=blob_size, min_gap=min_gap, seed=seed,
        shuffle_time=shuffle_time, reverse_time=reverse_time,
        first_last_only=first_last_only,
    )
