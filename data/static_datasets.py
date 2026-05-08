from __future__ import annotations

import os
import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import Dataset, Subset

from .transforms import build_eval_transform, build_train_transform
from .lra_datasets import NPZClassificationDataset, NPZDatasetSpec


STATIC_DATASETS = {"mnist", "cifar10", "cifar100", "imagenet", "imagefolder", "lra_pathfinder32_hard", "lra_cifar10"}


def _dataset_name(data_config) -> str:
    return str(getattr(data_config, "name", "")).lower()


def _resolve_imagefolder_root(root: str, split: str) -> str:
    candidates = []
    if split == "train":
        candidates.extend([os.path.join(root, "train"), root])
    else:
        candidates.extend([
            os.path.join(root, "val"),
            os.path.join(root, "test"),
            root,
        ])

    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(f"Could not find split directory for split={split} under root={root}")


def build_static_dataset(data_config, split: str):
    name = _dataset_name(data_config)
    root = os.path.expanduser(getattr(data_config, "root", "./datasets"))
    is_train = split == "train"
    transform = build_train_transform(data_config) if is_train else build_eval_transform(data_config)
    download = bool(getattr(data_config, "download", False))

    if name == "lra_pathfinder32_hard":
        # LRA Pathfinder is stored as npz splits to avoid TFDS dependency in runtime.
        # See scripts/prepare_lra_pathfinder_npz.py
        ds_root = os.path.join(root, "lra_pathfinder32_hard")
        return NPZClassificationDataset(NPZDatasetSpec(root=ds_root, split=split))

    if name == "lra_cifar10":
        # Follow LRA image task logic:
        # - dataset: CIFAR10
        # - preprocess: rgb -> grayscale, keep integer pixel values in [0,255]
        # - flatten to sequence length 32*32 = 1024
        # - split: train[:90%] / train[90%:] / test

        class _LRACIFAR10(Dataset):
            def __init__(self, base):
                self.base = base

            def __len__(self):
                return len(self.base)

            def __getitem__(self, idx):
                img, y = self.base[idx]  # img is PIL
                x = np.array(img, dtype=np.uint8)  # [32,32,3]
                # rgb_to_grayscale: TF uses 0.2989 R + 0.5870 G + 0.1140 B
                gray = (0.2989 * x[..., 0] + 0.5870 * x[..., 1] + 0.1140 * x[..., 2]).round().astype(np.int64)
                seq = torch.from_numpy(gray.reshape(-1).copy())  # [1024] int64
                return seq, int(y)

        if split == "test":
            base = datasets.CIFAR10(root=root, train=False, transform=None, download=download)
            return _LRACIFAR10(base)

        base = datasets.CIFAR10(root=root, train=True, transform=None, download=download)
        n = len(base)  # 50k
        n_train_full = int(0.9 * n)  # 45k (LRA train[:90%])
        n_val = n - n_train_full        # 5k  (LRA train[90%:])

        # Subsample the training split for low-resource comparisons.
        # train_fraction=0.1 means: use 10% of the *LRA training split* (i.e., 0.1 * 45k = 4.5k).
        train_fraction = float(getattr(data_config, "train_fraction", 0.1))
        train_fraction = max(0.0, min(1.0, train_fraction))
        n_train = max(1, int(round(train_fraction * n_train_full)))
        seed = int(getattr(data_config, "random_seed", 0))
        rng = np.random.RandomState(seed)
        idxs = rng.permutation(n).tolist()
        if split == "train":
            use = idxs[:n_train]
        elif split == "val":
            # Keep the original LRA validation split size, independent of train_fraction.
            use = idxs[n_train_full:n_train_full + n_val]
        else:
            raise ValueError(f"Unsupported split for lra_cifar10: {split}")
        return _LRACIFAR10(Subset(base, use))

    if name == "cifar10":
        return datasets.CIFAR10(root=root, train=is_train, transform=transform, download=download)
    if name == "cifar100":
        return datasets.CIFAR100(root=root, train=is_train, transform=transform, download=download)
    if name == "mnist":
        return datasets.MNIST(root=root, train=is_train, transform=transform, download=download)
    if name in {"imagenet", "imagefolder"}:
        dataset_root = _resolve_imagefolder_root(root, split)
        return datasets.ImageFolder(root=dataset_root, transform=transform)

    raise ValueError(f"Unsupported static dataset: {name}")
