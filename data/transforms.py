from __future__ import annotations

import numpy as np
import torch
from torchvision import transforms


DATASET_STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}


def _dataset_name(data_config) -> str:
    return str(getattr(data_config, "name", "")).lower()


def _normalize(name: str):
    mean, std = DATASET_STATS[name]
    return transforms.Normalize(mean=mean, std=std)


def build_train_transform(data_config):
    name = _dataset_name(data_config)
    image_size = int(getattr(data_config, "image_size", 224))

    if name in {"cifar10", "cifar100"}:
        return transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            _normalize(name),
        ])

    if name in {"imagenet", "imagefolder"}:
        resize_size = int(getattr(data_config, "resize_size", int(image_size / 0.875)))
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            _normalize("imagenet"),
        ])

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


def build_eval_transform(data_config):
    name = _dataset_name(data_config)
    image_size = int(getattr(data_config, "image_size", 224))

    if name in {"cifar10", "cifar100"}:
        return transforms.Compose([
            transforms.ToTensor(),
            _normalize(name),
        ])

    if name in {"imagenet", "imagefolder"}:
        resize_size = int(getattr(data_config, "resize_size", int(image_size / 0.875)))
        return transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            _normalize("imagenet"),
        ])

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


def build_event_transform(data_config):
    scale = float(getattr(data_config, "event_scale", 1.0))

    def _transform(frames):
        if isinstance(frames, np.ndarray):
            tensor = torch.from_numpy(frames)
        elif torch.is_tensor(frames):
            tensor = frames
        else:
            tensor = torch.tensor(frames)

        tensor = tensor.float()
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(1)
        if tensor.ndim == 4 and tensor.shape[-1] in {1, 2} and tensor.shape[1] not in {1, 2}:
            tensor = tensor.permute(0, 3, 1, 2)
        return tensor * scale

    return _transform
