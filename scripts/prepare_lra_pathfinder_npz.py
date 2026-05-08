from __future__ import annotations

import argparse
import os
from typing import Literal

import numpy as np


Split = Literal["train", "val", "test"]


def _tfds_to_npz(out_dir: str, split: Split, tfds_name: str, data_dir: str | None) -> None:
    try:
        import tensorflow_datasets as tfds  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "This script needs tensorflow-datasets to download/parse LRA Pathfinder.\n"
            "Install it into the uv env, e.g.\n"
            "  uv add tensorflow-datasets\n"
            "Then rerun this script."
        ) from e

    # TFDS builder exposes difficulty splits: easy/intermediate/hard.
    # We map LRA's canonical 'hard' to our train/val/test by slicing.
    ds = tfds.load(tfds_name, split="hard", data_dir=data_dir, as_supervised=True)
    imgs, labels = [], []
    for x, y in tfds.as_numpy(ds):
        imgs.append(x)     # uint8 [H,W,1] or [H,W,3]
        labels.append(y)   # 0/1

    images = np.stack(imgs, axis=0)
    labels = np.asarray(labels, dtype=np.int64)

    # Deterministic split: 80/10/10.
    n = labels.shape[0]
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    if split == "train":
        sl = slice(0, n_train)
    elif split == "val":
        sl = slice(n_train, n_train + n_val)
    else:
        sl = slice(n_train + n_val, n)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{split}.npz")
    np.savez_compressed(out_path, images=images[sl], labels=labels[sl])
    print(f"Wrote {out_path}: images={images[sl].shape} labels={labels[sl].shape}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_root", default="./datasets/lra_pathfinder32_hard", help="Output directory with train/val/test.npz")
    p.add_argument("--tfds_name", default="pathfinder32", help="TFDS dataset name (e.g. pathfinder32/pathfinder64)")
    p.add_argument("--tfds_data_dir", default="", help="Optional TFDS data_dir cache (empty -> default)")
    args = p.parse_args()

    data_dir = args.tfds_data_dir.strip() or None
    for split in ("train", "val", "test"):
        _tfds_to_npz(args.out_root, split, args.tfds_name, data_dir)


if __name__ == "__main__":
    main()

