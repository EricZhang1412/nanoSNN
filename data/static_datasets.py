from __future__ import annotations

import os
from torchvision import datasets

from .transforms import build_eval_transform, build_train_transform

STATIC_DATASETS = {"cifar10", "cifar100", "imagenet", "imagefolder", "imagenet_hf"}


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


class HFImageDataset:
    """Wraps a HuggingFace Dataset to behave like a torchvision dataset."""

    def __init__(self, hf_dataset, transform=None, image_key="image", label_key="label"):
        self.dataset = hf_dataset
        self.transform = transform
        self.image_key = image_key
        self.label_key = label_key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = item[self.image_key].convert("RGB")
        label = item[self.label_key]
        if self.transform:
            img = self.transform(img)
        return img, label


def _build_hf_dataset(data_config, split: str, transform):
    from datasets import load_dataset

    # e.g. data_config.hf_path = "imagenet-1k" or a local parquet dir
    hf_path = getattr(data_config, "hf_path", "imagenet-1k")
    cache_dir = getattr(data_config, "cache_dir", None)
    hf_split = "validation" if split == "val" else split

    ds = load_dataset(hf_path, split=hf_split, cache_dir=cache_dir)

    image_key = getattr(data_config, "image_key", "image")
    label_key = getattr(data_config, "label_key", "label")
    return HFImageDataset(ds, transform=transform, image_key=image_key, label_key=label_key)


def build_static_dataset(data_config, split: str):
    name = _dataset_name(data_config)
    root = os.path.expanduser(getattr(data_config, "root", "./datasets"))
    is_train = split == "train"
    transform = build_train_transform(data_config) if is_train else build_eval_transform(data_config)
    download = bool(getattr(data_config, "download", False))

    if name == "cifar10":
        return datasets.CIFAR10(root=root, train=is_train, transform=transform, download=download)
    if name == "cifar100":
        return datasets.CIFAR100(root=root, train=is_train, transform=transform, download=download)
    if name in {"imagenet", "imagefolder"}:
        dataset_root = _resolve_imagefolder_root(root, split)
        return datasets.ImageFolder(root=dataset_root, transform=transform)
    if name == "imagenet_hf":
        return _build_hf_dataset(data_config, split, transform)

    raise ValueError(f"Unsupported static dataset: {name}")