from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch.utils.data import Dataset

from data import event_datasets
from data.transforms import build_event_transform


class _FakeOfficialDataset(Dataset):
    calls: list[bool] = []

    def __init__(self, root, train, transform=None, target_transform=None, **kwargs):
        self.root = root
        self.train = bool(train)
        self.transform = transform
        self.length = 20 if self.train else 7
        self.calls.append(self.train)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        frames = torch.zeros(8, 1, 4)
        frames[index % 8] = 1.0
        if self.transform is not None:
            frames = self.transform(frames)
        return frames, index % 2


class EventDataProtocolTests(unittest.TestCase):
    def setUp(self):
        _FakeOfficialDataset.calls = []

    @staticmethod
    def _config(name: str):
        return SimpleNamespace(
            name=name,
            root="./datasets/test",
            T=8,
            frames_number=8,
            split_by="time",
            val_ratio=0.2,
            split_seed=17,
            augment_time_shift=True,
            time_shift=1,
            binarize=True,
        )

    def _assert_official_split_isolation(self, name: str, dataset_attr: str):
        cfg = self._config(name)
        patches = [mock.patch.object(event_datasets, dataset_attr, _FakeOfficialDataset)]
        if name == "shd":
            patches.append(mock.patch.object(event_datasets, "_HAS_SHD", True))

        with patches[0]:
            if len(patches) == 2:
                patches[1].start()
            try:
                train = event_datasets.build_event_dataset(cfg, "train")
                val = event_datasets.build_event_dataset(cfg, "val")
                test = event_datasets.build_event_dataset(cfg, "test")
            finally:
                if len(patches) == 2:
                    patches[1].stop()

        self.assertEqual(_FakeOfficialDataset.calls, [True, True, False])
        self.assertEqual(len(train), 16)
        self.assertEqual(len(val), 4)
        self.assertEqual(len(test), 7)

        train_indices = set(train.subset.indices)
        val_indices = set(val.subset.indices)
        self.assertFalse(train_indices & val_indices)
        self.assertEqual(train_indices | val_indices, set(range(20)))
        self.assertTrue(test.train is False)

    def test_dvs_validation_comes_from_official_training_set(self):
        self._assert_official_split_isolation("dvs128gesture", "DVS128Gesture")

    def test_shd_validation_comes_from_official_training_set(self):
        self._assert_official_split_isolation("shd", "SpikingHeidelbergDigits")

    def test_shd_time_shift_is_train_only(self):
        cfg = self._config("shd")
        frames = torch.zeros(7, 1, 4)
        frames[2] = 1.0

        with mock.patch("data.transforms.torch.randint", return_value=torch.tensor([1])):
            train = build_event_transform(cfg, "train")(frames)
            val = build_event_transform(cfg, "val")(frames)
            test = build_event_transform(cfg, "test")(frames)

        self.assertTrue(torch.equal(train[3], torch.ones_like(train[3])))
        self.assertTrue(torch.equal(train[2], torch.zeros_like(train[2])))
        self.assertTrue(torch.equal(val, frames))
        self.assertTrue(torch.equal(test, frames))

    def test_unknown_split_is_rejected(self):
        with self.assertRaises(ValueError):
            build_event_transform(self._config("shd"), "validation")


if __name__ == "__main__":
    unittest.main()
