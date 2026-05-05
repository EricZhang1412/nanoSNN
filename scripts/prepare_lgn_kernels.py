from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys


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
    return parser.parse_args()


def main():
    args = _parse_args()
    lgn_py = os.path.abspath(os.path.expanduser(args.lgn_py))
    lgn_data_path = os.path.abspath(os.path.expanduser(args.lgn_data_path))
    if not os.path.isfile(lgn_py):
        raise FileNotFoundError(f"lgn.py not found: {lgn_py}")
    if not os.path.isfile(lgn_data_path):
        raise FileNotFoundError(f"lgn_data_path not found: {lgn_data_path}")

    default_cache = os.path.join(os.path.dirname(lgn_py), "temporal_kernels.pkl")
    output = default_cache if args.output is None else os.path.abspath(os.path.expanduser(args.output))

    if os.path.isfile(output) and not args.force:
        print(f"[prepare_lgn_kernels] Found existing cache: {output}")
        return

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
