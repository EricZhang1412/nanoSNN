from __future__ import annotations

import math
import os

import torch
from torch.utils.data import Dataset, Subset, random_split
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

from .transforms import build_event_transform


EVENT_DATASETS = {"cifar10dvs", "dvs128gesture"}


class TransformSubset(Dataset):
    def __init__(self, subset: Subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        frames, target = self.subset[index]
        if self.transform is not None:
            frames = self.transform(frames)
        return frames, target


def _dataset_name(data_config) -> str:
    return str(getattr(data_config, "name", "")).lower()


def _event_kwargs(data_config):
    return {
        "data_type": getattr(data_config, "event_data_type", "frame"),
        "frames_number": getattr(data_config, "frames_number", getattr(data_config, "T", None)),
        "split_by": getattr(data_config, "split_by", None),
        "duration": getattr(data_config, "duration", None),
        "custom_integrate_function": getattr(data_config, "custom_integrate_function", None),
        "custom_integrated_frames_dir_name": getattr(data_config, "custom_integrated_frames_dir_name", None),
    }


def _split_cifar10dvs(dataset, split: str, train_ratio: float, seed: int, transform=None):
    train_len = int(math.floor(len(dataset) * train_ratio))
    val_len = len(dataset) - train_len
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(dataset, [train_len, val_len], generator=generator)

    if split == "train":
        return TransformSubset(train_subset, transform=transform)
    return TransformSubset(val_subset, transform=transform)


def build_event_dataset(data_config, split: str):
    name = _dataset_name(data_config)
    root = os.path.expanduser(getattr(data_config, "root", "./datasets"))
    transform = build_event_transform(data_config)
    kwargs = _event_kwargs(data_config)

    if name == "cifar10dvs":
        dataset = CIFAR10DVS(root=root, transform=None, target_transform=None, **kwargs)
        train_ratio = float(getattr(data_config, "train_ratio", 0.9))
        seed = int(getattr(data_config, "split_seed", 42))
        return _split_cifar10dvs(dataset, split=split, train_ratio=train_ratio, seed=seed, transform=transform)

    if name == "dvs128gesture":
        is_train = split == "train"
        return DVS128Gesture(root=root, train=is_train, transform=transform, target_transform=None, **kwargs)

    raise ValueError(f"Unsupported event dataset: {name}")
