from __future__ import annotations

from torch.utils.data._utils.collate import default_collate

from .event_datasets import EVENT_DATASETS, build_event_dataset
from .static_datasets import STATIC_DATASETS, build_static_dataset


EVENT_DATASET_NAMES = set(EVENT_DATASETS)
STATIC_DATASET_NAMES = set(STATIC_DATASETS)


def _dataset_name(data_config) -> str:
    return str(getattr(data_config, "name", "")).lower()


def is_event_dataset(data_config) -> bool:
    return _dataset_name(data_config) in EVENT_DATASET_NAMES


def build_dataset(data_config, split: str):
    if is_event_dataset(data_config):
        return build_event_dataset(data_config, split)
    return build_static_dataset(data_config, split)


def infer_num_classes(data_config, dataset=None) -> int:
    explicit = getattr(data_config, "num_classes", None)
    if explicit is not None:
        return int(explicit)
    if dataset is not None and hasattr(dataset, "classes"):
        return len(dataset.classes)

    name = _dataset_name(data_config)
    fallback = {
        "cifar10": 10,
        "cifar10dvs": 10,
        "cifar100": 100,
        "dvs128gesture": 11,
        "imagenet": 1000,
        "imagefolder": 1000,
    }
    if name not in fallback:
        raise ValueError(f"Cannot infer num_classes for dataset {name}")
    return fallback[name]


def event_collate_fn(batch):
    frames, targets = default_collate(batch)
    if frames.ndim == 5:
        frames = frames.transpose(0, 1).contiguous()
    return frames, targets


def build_collate_fn(data_config):
    return event_collate_fn if is_event_dataset(data_config) else None
