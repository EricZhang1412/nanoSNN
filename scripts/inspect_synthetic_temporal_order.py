"""
Quick dataset sanity check for synthetic temporal-order configs.

It builds train/val/test splits and prints shape, label balance, and event
density before spending GPU time on model training.
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _summarize_split(data_cfg, split: str, n_preview: int):
    from data.build import build_dataset

    ds = build_dataset(data_cfg, split)
    labels = []
    event_counts = []
    for i in range(min(len(ds), n_preview)):
        x, y = ds[i]
        labels.append(int(y))
        event_counts.append(float((x > 0).float().sum().item()))

    x0, y0 = ds[0]
    counts = torch.bincount(torch.tensor(labels), minlength=int(getattr(data_cfg, "num_classes", 2)))
    print(
        f"{split:5s}: n={len(ds):5d} shape={tuple(x0.shape)} "
        f"preview_labels={counts.tolist()} "
        f"events/sample(mean preview)={sum(event_counts) / max(len(event_counts), 1):.1f} "
        f"first_label={int(y0)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect synthetic temporal-order dataset configs.")
    parser.add_argument("--data_config", type=str, required=True)
    parser.add_argument("--n_preview", type=int, default=128)
    args = parser.parse_args()

    root = _repo_root()
    sys.path.insert(0, str(root))

    from utils.load_config import load_config

    data_cfg = load_config(str(root / args.data_config))
    print(f"config={args.data_config}")
    for split in ("train", "val", "test"):
        _summarize_split(copy.deepcopy(data_cfg), split, args.n_preview)


if __name__ == "__main__":
    main()
