from __future__ import annotations

import lightning as L
from torch.utils.data import DataLoader

from .build import build_collate_fn, build_dataset, infer_num_classes, is_event_dataset


class VisionDataModule(L.LightningDataModule):
    def __init__(self, data_config, train_config):
        super().__init__()
        self.data_config = data_config
        self.train_config = train_config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.num_classes = None
        self.collate_fn = build_collate_fn(data_config)
        self.is_event_dataset = is_event_dataset(data_config)

    def setup(self, stage=None):
        if stage in (None, "fit"):
            self.train_dataset = build_dataset(self.data_config, "train")
            self.val_dataset = build_dataset(self.data_config, "val")
            self.num_classes = infer_num_classes(self.data_config, self.train_dataset)

        if stage in (None, "test"):
            self.test_dataset = build_dataset(self.data_config, "test")
            if self.num_classes is None:
                self.num_classes = infer_num_classes(self.data_config, self.test_dataset)

    def _loader_kwargs(self, shuffle: bool, drop_last: bool):
        num_workers = int(getattr(self.data_config, "num_workers", 4))
        return dict(
            batch_size=int(self.train_config.batch_size_per_gpu),
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=bool(getattr(self.data_config, "pin_memory", True)),
            persistent_workers=num_workers > 0,
            drop_last=drop_last,
            collate_fn=self.collate_fn,
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self._loader_kwargs(shuffle=True, drop_last=True))

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self._loader_kwargs(shuffle=False, drop_last=False))

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self._loader_kwargs(shuffle=False, drop_last=False))
