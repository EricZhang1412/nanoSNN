"""Unit test for 10/11-class localized-readout pool selection."""
from __future__ import annotations
import os, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.load_config import load_config  # noqa: E402
from models.build_model import build_model  # noqa: E402


def _build_with_classes(num_classes: int):
    mcfg = load_config('configs/model_configs/billeh_v1_seqcifar10_lgn.yaml')
    mcfg.num_classes = num_classes
    dcfg = load_config('configs/data_configs/sequential_cifar10_fewshot.yaml')
    tcfg = load_config('configs/train_configs/default.yaml')
    ocfg = load_config('configs/optimizer_configs/billeh_seqcifar10.yaml')
    lit = build_model(mcfg, ocfg, tcfg, dcfg)
    return lit.model


def test_10_class_uses_pools_5_to_14():
    m = _build_with_classes(10)
    pool_sets = [set(m.readout.pool_indices(c).tolist()) for c in range(10)]
    for i in range(10):
        for j in range(i + 1, 10):
            assert pool_sets[i].isdisjoint(pool_sets[j]), (
                f"pools {i} and {j} overlap (10-class)"
            )


def test_11_class_no_pool_aliasing():
    """The bug: with num_classes=11, current code aliases class 5 and class 10
    to the same pool (pool 10). Fix must give 11 disjoint pools."""
    m = _build_with_classes(11)
    pool_sets = [set(m.readout.pool_indices(c).tolist()) for c in range(11)]
    for i in range(11):
        assert len(pool_sets[i]) > 0, f"class {i} has empty pool"
        for j in range(i + 1, 11):
            assert pool_sets[i].isdisjoint(pool_sets[j]), (
                f"pools {i} and {j} overlap (11-class) — readout aliasing bug"
            )


if __name__ == "__main__":
    test_10_class_uses_pools_5_to_14()
    test_11_class_no_pool_aliasing()
    print("[test_billeh_readout_pools] OK")
