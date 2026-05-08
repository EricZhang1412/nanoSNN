from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class NPZDatasetSpec:
    """Convention for NPZ classification datasets.

    Expected directory layout:
      root/
        train.npz
        val.npz
        test.npz

    Each *.npz must contain:
      - images: uint8 or float array, shape [N, H, W] or [N, H, W, C]
      - labels: int array, shape [N]
    """

    root: str
    split: str
    images_key: str = "images"
    labels_key: str = "labels"

    def path(self) -> str:
        return os.path.join(self.root, f"{self.split}.npz")


class NPZClassificationDataset(Dataset):
    def __init__(self, spec: NPZDatasetSpec):
        super().__init__()
        path = spec.path()
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing NPZ split file: {path}. "
                "Prepare it first (see scripts/prepare_lra_pathfinder_npz.py)."
            )
        blob = np.load(path, allow_pickle=False)
        if spec.images_key not in blob or spec.labels_key not in blob:
            raise ValueError(
                f"NPZ must contain keys {spec.images_key!r}, {spec.labels_key!r}; got {sorted(blob.files)}"
            )
        self.images = blob[spec.images_key]
        self.labels = blob[spec.labels_key].astype(np.int64)
        if len(self.images) != len(self.labels):
            raise ValueError(f"images/labels length mismatch: {len(self.images)} vs {len(self.labels)}")

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):
        img = self.images[idx]
        y = int(self.labels[idx])

        # img: [H,W] or [H,W,C] -> torch [C,H,W] float32 in [0,1]
        if img.ndim == 2:
            img = img[:, :, None]
        if img.ndim != 3:
            raise ValueError(f"Unsupported image shape in NPZ: {img.shape}")
        img = torch.from_numpy(np.ascontiguousarray(img))
        if img.dtype != torch.float32:
            img = img.float()
        if img.max() > 1.0:
            img = img / 255.0
        img = img.permute(2, 0, 1).contiguous()
        return img, y

