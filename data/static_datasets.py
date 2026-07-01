from __future__ import annotations

import os
import torch
from torch.utils.data import Subset
from torchvision import datasets

from .transforms import build_eval_transform, build_train_transform


STATIC_DATASETS = {"mnist", "cifar10", "cifar100", "imagenet", "imagefolder"}


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


def _maybe_subset(dataset, data_config, split: str):
    size = getattr(data_config, f"{split}_subset_size", None)
    if size is None and split == "val":
        size = getattr(data_config, "eval_subset_size", None)
    if size is None and split == "test":
        size = getattr(data_config, "eval_subset_size", None)
    if size is None:
        return dataset

    size = int(size)
    if size <= 0 or size >= len(dataset):
        return dataset
    seed = int(getattr(data_config, "subset_seed", 42))
    generator = torch.Generator().manual_seed(seed + {"train": 0, "val": 1, "test": 2}[split])
    indices = torch.randperm(len(dataset), generator=generator)[:size].tolist()
    return Subset(dataset, indices)


def build_static_dataset(data_config, split: str):
    name = _dataset_name(data_config)
    root = os.path.expanduser(getattr(data_config, "root", "./datasets"))
    is_train = split == "train"
    transform = build_train_transform(data_config) if is_train else build_eval_transform(data_config)
    download = bool(getattr(data_config, "download", False))

    if name == "cifar10":
        dataset = datasets.CIFAR10(root=root, train=is_train, transform=transform, download=download)
        return _maybe_subset(dataset, data_config, split)
    if name == "cifar100":
        dataset = datasets.CIFAR100(root=root, train=is_train, transform=transform, download=download)
        return _maybe_subset(dataset, data_config, split)
    if name == "mnist":
        dataset = datasets.MNIST(root=root, train=is_train, transform=transform, download=download)
        return _maybe_subset(dataset, data_config, split)
    if name in {"imagenet", "imagefolder"}:
        dataset_root = _resolve_imagefolder_root(root, split)
        dataset = datasets.ImageFolder(root=dataset_root, transform=transform)
        return _maybe_subset(dataset, data_config, split)

    raise ValueError(f"Unsupported static dataset: {name}")
