from __future__ import annotations

import math
import os

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, random_split
from torchvision import datasets
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

from .transforms import build_event_transform


EVENT_DATASETS = {"cifar10dvs", "dvs128gesture", "ucr", "sequential_mnist", "seqmnist"}


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


class UCRDataset(Dataset):
    def __init__(self, root: str, dataset_name: str, split: str, val_ratio: float = 0.1, split_seed: int = 42,
                 zscore_per_sample: bool = True):
        super().__init__()
        self.root = os.path.expanduser(root)
        self.dataset_name = str(dataset_name)
        self.zscore_per_sample = bool(zscore_per_sample)

        train_path = os.path.join(self.root, self.dataset_name, f"{self.dataset_name}_TRAIN.tsv")
        test_path = os.path.join(self.root, self.dataset_name, f"{self.dataset_name}_TEST.tsv")
        if not os.path.isfile(train_path):
            raise FileNotFoundError(f"UCR train split not found: {train_path}")
        if not os.path.isfile(test_path):
            raise FileNotFoundError(f"UCR test split not found: {test_path}")

        train_raw = np.genfromtxt(train_path, delimiter=None, dtype=np.float32)
        test_raw = np.genfromtxt(test_path, delimiter=None, dtype=np.float32)
        if train_raw.ndim != 2 or train_raw.shape[1] < 2:
            raise ValueError(f"Invalid UCR train file format: {train_path}")
        if test_raw.ndim != 2 or test_raw.shape[1] < 2:
            raise ValueError(f"Invalid UCR test file format: {test_path}")

        train_labels_raw, train_x = train_raw[:, 0], train_raw[:, 1:]
        test_labels_raw, test_x = test_raw[:, 0], test_raw[:, 1:]

        label_values = sorted(np.unique(np.concatenate([train_labels_raw, test_labels_raw])).tolist())
        label_to_id = {val: i for i, val in enumerate(label_values)}
        train_y = np.asarray([label_to_id[v] for v in train_labels_raw], dtype=np.int64)
        test_y = np.asarray([label_to_id[v] for v in test_labels_raw], dtype=np.int64)

        if split in {"train", "val"}:
            train_idx, val_idx = self._split_train_val_indices(len(train_x), val_ratio, split_seed)
            idx = train_idx if split == "train" else val_idx
            self.x = train_x[idx]
            self.y = train_y[idx]
        elif split == "test":
            self.x = test_x
            self.y = test_y
        else:
            raise ValueError(f"Unsupported UCR split: {split}")

        self.classes = [str(v) for v in label_values]

    @staticmethod
    def _split_train_val_indices(length: int, val_ratio: float, seed: int):
        if not (0.0 <= val_ratio < 1.0):
            raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
        n_val = int(math.floor(length * val_ratio))
        generator = np.random.default_rng(seed)
        indices = np.arange(length)
        generator.shuffle(indices)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]
        return train_idx, val_idx

    def __len__(self):
        return int(self.x.shape[0])

    def __getitem__(self, index):
        # UCR sample is 1-D time series. Return [T, 1] so collate gives [B, T, 1].
        series = self.x[index]
        if self.zscore_per_sample:
            mean = float(series.mean())
            std = float(series.std())
            series = (series - mean) / (std + 1e-6)
        series = torch.from_numpy(series).float().unsqueeze(-1)
        target = int(self.y[index])
        return series, target


class SequentialMNISTDataset(Dataset):
    def __init__(self, root: str, split: str, val_ratio: float = 0.1, split_seed: int = 42,
                 permute: bool = False, permute_seed: int = 0, normalize_to_01: bool = True, download: bool = True,
                 patch_size: int = 1):
        super().__init__()
        root = os.path.expanduser(root)
        self.permute = bool(permute)
        self.normalize_to_01 = bool(normalize_to_01)
        self.patch_size = int(patch_size)
        if self.patch_size <= 0 or 28 % self.patch_size != 0:
            raise ValueError(f"patch_size must be positive and divide 28, got {self.patch_size}")
        self.tokens_per_side = 28 // self.patch_size
        self.num_tokens = self.tokens_per_side * self.tokens_per_side

        if split in {"train", "val"}:
            base = datasets.MNIST(root=root, train=True, download=download)
            train_idx, val_idx = self._split_train_val_indices(len(base), val_ratio, split_seed)
            indices = train_idx if split == "train" else val_idx
            self.base = Subset(base, indices)
            self.classes = list(base.classes)
        elif split == "test":
            base = datasets.MNIST(root=root, train=False, download=download)
            self.base = base
            self.classes = list(base.classes)
        else:
            raise ValueError(f"Unsupported SequentialMNIST split: {split}")

        if self.permute:
            g = torch.Generator().manual_seed(int(permute_seed))
            self.pixel_permutation = torch.randperm(self.num_tokens, generator=g)
        else:
            self.pixel_permutation = None

    @staticmethod
    def _split_train_val_indices(length: int, val_ratio: float, seed: int):
        if not (0.0 <= val_ratio < 1.0):
            raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
        n_val = int(math.floor(length * val_ratio))
        generator = np.random.default_rng(seed)
        indices = np.arange(length)
        generator.shuffle(indices)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]
        return train_idx, val_idx

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, target = self.base[index]
        image = torch.as_tensor(np.array(image), dtype=torch.float32)
        if self.normalize_to_01:
            image = image / 255.0

        if self.patch_size == 1:
            sequence = image.reshape(-1, 1)  # [784, 1]
        else:
            # [28, 28] -> [num_patches, patch_size * patch_size]
            patches = image.unfold(0, self.patch_size, self.patch_size).unfold(1, self.patch_size, self.patch_size)
            sequence = patches.contiguous().view(self.num_tokens, self.patch_size * self.patch_size)

        if self.pixel_permutation is not None:
            sequence = sequence[self.pixel_permutation]
        return sequence, int(target)


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

    if name == "ucr":
        dataset_name = str(getattr(data_config, "dataset", "")).strip()
        if not dataset_name:
            raise ValueError("data_config.dataset is required for UCR dataset")
        val_ratio = float(getattr(data_config, "val_ratio", 0.1))
        split_seed = int(getattr(data_config, "split_seed", 42))
        zscore_per_sample = bool(getattr(data_config, "zscore_per_sample", True))
        return UCRDataset(
            root=root,
            dataset_name=dataset_name,
            split=split,
            val_ratio=val_ratio,
            split_seed=split_seed,
            zscore_per_sample=zscore_per_sample,
        )

    if name in {"sequential_mnist", "seqmnist"}:
        val_ratio = float(getattr(data_config, "val_ratio", 0.1))
        split_seed = int(getattr(data_config, "split_seed", 42))
        permute = bool(getattr(data_config, "permute", False))
        permute_seed = int(getattr(data_config, "permute_seed", 0))
        normalize_to_01 = bool(getattr(data_config, "normalize_to_01", True))
        download = bool(getattr(data_config, "download", True))
        patch_size = int(getattr(data_config, "patch_size", 1))
        return SequentialMNISTDataset(
            root=root,
            split=split,
            val_ratio=val_ratio,
            split_seed=split_seed,
            permute=permute,
            permute_seed=permute_seed,
            normalize_to_01=normalize_to_01,
            download=download,
            patch_size=patch_size,
        )

    raise ValueError(f"Unsupported event dataset: {name}")
