from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys

import numpy as np


def _load_module_from_path(module_name: str, py_path: str):
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from path: {py_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_args():
    parser = argparse.ArgumentParser(description="Prepare cached LGN temporal kernels for nanoSNN TorchLGN.")
    parser.add_argument(
        "--lgn_py",
        type=str,
        required=True,
        help="Path to original lgn.py (from Training-data-driven-V1-model).",
    )
    parser.add_argument(
        "--lgn_data_path",
        type=str,
        required=True,
        help="Path to lgn_full_col_cells_3.csv used by original LGN.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Target temporal_kernels.pkl path. If omitted, use <dirname(lgn_py)>/temporal_kernels.pkl.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-generate even if output file already exists.",
    )
    parser.add_argument(
        "--bmtk_root",
        type=str,
        default=None,
        help="Preferred bmtk source root (directory containing 'bmtk/' package).",
    )
    return parser.parse_args()


def _prepend_bmtk_root(bmtk_root: str | None):
    if not bmtk_root:
        return None
    root = os.path.abspath(os.path.expanduser(bmtk_root))
    pkg_dir = os.path.join(root, "bmtk")
    if not os.path.isdir(pkg_dir):
        raise FileNotFoundError(f"Invalid --bmtk_root, expected {pkg_dir} to exist.")
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
    return root


def _check_lgn_csv_compat(lgn_data_path: str):
    with open(lgn_data_path, "r", encoding="utf-8") as f:
        header = f.readline().strip()
    cols = header.split()
    required = {"spatial_size", "model_id", "x", "y"}
    if required.issubset(set(cols)):
        return
    hint = ""
    if {"model_id", "level_of_detail"}.issubset(set(cols)):
        hint = " It looks like you passed '...cell_models_3.csv'. Please use '...cells_3.csv' instead."
    raise ValueError(
        "lgn_data_path is not compatible with original lgn.py. "
        f"Missing required columns: {sorted(required - set(cols))}.{hint}"
    )


def _patch_numpy_legacy_aliases():
    """
    Compatibility shim for legacy code using removed aliases such as np.float/np.int.
    """
    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]
    if not hasattr(np, "int"):
        np.int = int  # type: ignore[attr-defined]


def main():
    args = _parse_args()
    lgn_py = os.path.abspath(os.path.expanduser(args.lgn_py))
    lgn_data_path = os.path.abspath(os.path.expanduser(args.lgn_data_path))
    if not os.path.isfile(lgn_py):
        raise FileNotFoundError(f"lgn.py not found: {lgn_py}")
    if not os.path.isfile(lgn_data_path):
        raise FileNotFoundError(f"lgn_data_path not found: {lgn_data_path}")
    _check_lgn_csv_compat(lgn_data_path)

    default_cache = os.path.join(os.path.dirname(lgn_py), "temporal_kernels.pkl")
    output = default_cache if args.output is None else os.path.abspath(os.path.expanduser(args.output))

    if os.path.isfile(output) and not args.force:
        print(f"[prepare_lgn_kernels] Found existing cache: {output}")
        return

    _patch_numpy_legacy_aliases()

    # Prefer user-specified bmtk root first. If not specified, try sibling
    # repository layout: <repo_root>/3rdparty/bmtk relative to lgn.py.
    effective_bmtk = args.bmtk_root
    if not effective_bmtk:
        candidate = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(lgn_py)), "3rdparty", "bmtk"))
        if os.path.isdir(os.path.join(candidate, "bmtk")):
            effective_bmtk = candidate
    added_bmtk = _prepend_bmtk_root(effective_bmtk)
    if added_bmtk:
        print(f"[prepare_lgn_kernels] Using bmtk root: {added_bmtk}")

    print(f"[prepare_lgn_kernels] Importing original LGN from: {lgn_py}")
    lgn_module = _load_module_from_path("external_lgn_module", lgn_py)
    if not hasattr(lgn_module, "LGN"):
        raise AttributeError(f"LGN class not found in module: {lgn_py}")

    print("[prepare_lgn_kernels] Building LGN cache (this may take a while)...")
    _ = lgn_module.LGN(lgn_data_path=lgn_data_path)

    if not os.path.isfile(default_cache):
        raise FileNotFoundError(
            "Original LGN finished but temporal_kernels.pkl was not produced at "
            f"{default_cache}. Please check lgn.py logic and dependencies."
        )

    if output != default_cache:
        os.makedirs(os.path.dirname(output), exist_ok=True)
        shutil.copy2(default_cache, output)
        print(f"[prepare_lgn_kernels] Copied cache to: {output}")
    else:
        print(f"[prepare_lgn_kernels] Cache ready: {output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[prepare_lgn_kernels] Error: {exc}", file=sys.stderr)
        raise
