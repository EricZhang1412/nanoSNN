from __future__ import annotations

import argparse
import importlib.util
import os
import pickle as pkl
import shutil
import sys

import numpy as np
import pandas as pd


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
        default=None,
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
        help="Target temporal_kernels.pkl path. If omitted, use the directory of --lgn_py or --lgn_data_path.",
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
        help="Preferred bmtk source root (directory containing 'bmtk/' package). "
             "Defaults to nanoSNN/3rdparty/bmtk for direct cache generation.",
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


def _repo_root() -> str:
    return os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def _default_bmtk_root() -> str:
    return os.path.join(_repo_root(), "3rdparty", "bmtk")


def _metrics_dir(bmtk_root: str) -> str:
    path = os.path.join(
        bmtk_root,
        "bmtk",
        "simulator",
        "filternet",
        "lgnmodel",
        "cell_metrics",
    )
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"Cannot find BMTK LGN cell_metrics directory: {path}. "
            "Run `git submodule update --init --recursive` or pass --bmtk_root."
        )
    return path


def _spontaneous_lookup(metrics_dir: str, ctype: str) -> dict[str, float]:
    if "_sus" in ctype:
        path = os.path.join(metrics_dir, f"{ctype}_cells_v3.csv")
    else:
        path = os.path.join(metrics_dir, f"{ctype}_cell_data.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing BMTK cell metrics for {ctype!r}: {path}")

    prs_df = pd.read_csv(path)
    exp_df = prs_df.iloc[:, [13, 14, 17, 18, 28, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]].copy()
    sub_df = exp_df.iloc[:, [5, 6, 7, 8, 9]]
    exp_df["max_tf"] = sub_df.idxmax(axis=1).values
    exp_means = exp_df.groupby("max_tf").mean(numeric_only=True)
    return {str(idx)[3:]: float(row.iloc[4]) for idx, row in exp_means.iterrows()}


def _basis_batch(kpeaks: np.ndarray) -> np.ndarray:
    """Vectorized BMTK makeBasis_StimKernel for neye=0, ncos=2, nkt=600."""
    kpeaks = np.asarray(kpeaks, dtype=np.float64)
    n = kpeaks.shape[0]
    b = 0.3
    nkt = 600
    ylim = np.array([100.0, 200.0], dtype=np.float64)
    yrnge = np.log(ylim + b)
    db = yrnge[-1] - yrnge[0]
    ctrs = np.log(kpeaks)
    mxt = np.exp(yrnge[-1] + 2 * db) - b
    kt0 = np.arange(0, mxt, 1.0, dtype=np.float64)

    z = (np.log(kt0 + b)[None, :, None] - ctrs[:, None, :]) * np.pi / db / 2.0
    kbasis = (np.cos(np.clip(z, -np.pi, np.pi)) + 1.0) / 2.0
    kbasis = kbasis[:, ::-1, :]
    if kbasis.shape[1] < nkt:
        pad = np.zeros((n, nkt - kbasis.shape[1], 2), dtype=kbasis.dtype)
        kbasis = np.concatenate((pad, kbasis), axis=1)
    elif kbasis.shape[1] > nkt:
        kbasis = kbasis[:, -1 - nkt : -1, :]

    return kbasis / np.sqrt(np.sum(kbasis * kbasis, axis=1, keepdims=True))


def _combine_basis_delays(kbasis: np.ndarray, weights: np.ndarray, delays: np.ndarray) -> list[np.ndarray]:
    """Apply BMTK delay padding/cropping and weights to precomputed basis arrays."""
    weights = np.asarray(weights, dtype=np.float64)
    delays = np.asarray(delays, dtype=np.float64).astype(int)
    kernels: list[np.ndarray] = []
    for i in range(kbasis.shape[0]):
        d0, d1 = delays[i]
        diff = int(d1 - d0)
        length = int(600 + d0)
        basis_i = np.zeros((length, 2), dtype=np.float64)
        basis_i[:600, 0] = kbasis[i, :, 0]
        keep = min(length, max(0, 600 - diff))
        if keep > 0:
            basis_i[:keep, 1] = kbasis[i, diff : diff + keep, 1]
        basis_i = basis_i / np.sqrt(np.sum(basis_i * basis_i, axis=0, keepdims=True))
        kernels.append((basis_i @ weights[i]).astype(np.float32))
    return kernels


def _cross_from_above(x: np.ndarray, threshold: float) -> np.ndarray:
    x = np.asarray(x)
    return np.nonzero((x[:-1] >= threshold) & (x[1:] < threshold))[0] + 1


def _tcross_from_kernel(kernel: np.ndarray) -> int:
    max_ind = int(np.argmax(kernel))
    min_ind = int(np.argmin(kernel))
    crossing = _cross_from_above(kernel[max_ind:min_ind], 0.0)
    if crossing.size == 0:
        raise ValueError("Temporal kernel does not cross zero between max and min")
    return int(max_ind + crossing[0])


def _generate_with_bmtk(lgn_data_path: str, output: str, bmtk_root: str) -> None:
    metrics = _metrics_dir(bmtk_root)

    d = pd.read_csv(lgn_data_path, delimiter=" ")
    model_id = d["model_id"].astype(str).to_numpy()
    x = d["x"].to_numpy(dtype=np.float64)
    y = d["y"].to_numpy(dtype=np.float64)
    tuning_angle = d["tuning_angle"].to_numpy(dtype=np.float64)
    subfield_separation = d["sf_sep"].to_numpy(dtype=np.float64)

    amplitude = np.array([1.0 if "ON" in mid else -1.0 for mid in model_id], dtype=np.float32)
    non_dom_amplitude = np.zeros_like(amplitude)
    spontaneous_firing_rates = np.empty_like(amplitude)
    non_dominant_x = np.zeros_like(x, dtype=np.float32)
    non_dominant_y = np.zeros_like(y, dtype=np.float32)

    spont_cache: dict[str, dict[str, float]] = {}
    for i, mid in enumerate(model_id):
        if "ON" in mid and "OFF" in mid:
            spontaneous_firing_rates[i] = -1.0
            continue
        ctype = mid[: mid.find("_")]
        tf_str = mid[mid.find("_") + 1 :]
        if ctype not in spont_cache:
            spont_cache[ctype] = _spontaneous_lookup(metrics, ctype)
        spontaneous_firing_rates[i] = spont_cache[ctype][tf_str]

    temporal_peaks_dom = np.stack((d["kpeaks_dom_0"].to_numpy(), d["kpeaks_dom_1"].to_numpy()), -1)
    temporal_weights_dom = np.stack((d["weight_dom_0"].to_numpy(), d["weight_dom_1"].to_numpy()), -1)
    temporal_delays_dom = np.stack((d["delay_dom_0"].to_numpy(), d["delay_dom_1"].to_numpy()), -1)
    temporal_peaks_non_dom = np.stack((d["kpeaks_non_dom_0"].to_numpy(), d["kpeaks_non_dom_1"].to_numpy()), -1)
    temporal_weights_non_dom = np.stack((d["weight_non_dom_0"].to_numpy(), d["weight_non_dom_1"].to_numpy()), -1)
    temporal_delays_non_dom = np.stack((d["delay_non_dom_0"].to_numpy(), d["delay_non_dom_1"].to_numpy()), -1)

    kernel_length = 700
    dom_temporal_kernels = np.zeros((len(model_id), kernel_length), dtype=np.float32)
    non_dom_temporal_kernels = np.zeros((len(model_id), kernel_length), dtype=np.float32)

    print(f"[prepare_lgn_kernels] Computing {len(model_id)} temporal kernels from BMTK formulas...")
    dom_basis = _basis_batch(temporal_peaks_dom)
    dom_kernels = _combine_basis_delays(dom_basis, temporal_weights_dom, temporal_delays_dom)
    composite_idx = [i for i, mid in enumerate(model_id) if "ON" in mid and "OFF" in mid]
    non_dom_basis = _basis_batch(temporal_peaks_non_dom[composite_idx]) if composite_idx else None

    shifted_dom_delays = []
    shifted_non_delays = []
    for i, mid in enumerate(model_id):
        if "ON" in mid and "OFF" in mid:
            comp_pos = len(shifted_dom_delays)
            non_raw = _combine_basis_delays(
                non_dom_basis[comp_pos : comp_pos + 1],
                temporal_weights_non_dom[i : i + 1],
                temporal_delays_non_dom[i : i + 1],
            )[0][::-1]
            dom_raw = dom_kernels[i][::-1]

            non_dom_tcross = _tcross_from_kernel(non_raw)
            dom_tcross = _tcross_from_kernel(dom_raw)
            non_dom_sum = float(non_raw[:non_dom_tcross].sum())
            dom_sum = float(dom_raw[:dom_tcross].sum())

            non_ttp = 121.0 if "sONsOFF_001" in mid else 93.5
            dom_ttp = 115.0 if "sONsOFF_001" in mid else 64.8
            shifted_non_delays.append(temporal_delays_non_dom[i] + (non_ttp - non_dom_tcross))
            shifted_dom_delays.append(temporal_delays_dom[i] + (dom_ttp - dom_tcross))

            if "sONsOFF_001" in mid:
                spont, max_roff, max_ron, scale = 4.0, 35.0, 21.0, 1.0
            elif "sONtOFF_001" in mid:
                spont, max_roff, max_ron, scale = 5.5, 46.0, 31.0, 0.7
            else:
                raise ValueError(f"Unknown composite LGN cell type: {mid}")

            amp_on = 1.0
            amp_off = -scale * (max_roff / max_ron) * (non_dom_sum / dom_sum) * amp_on
            amp_off -= (spont * (max_roff - max_ron)) / (max_ron * dom_sum)
            non_dom_amplitude[i] = amp_on
            amplitude[i] = amp_off
            spontaneous_firing_rates[i] = spont / 2.0

            non_dominant_x[i] = np.cos(tuning_angle[i] * np.pi / 180.0) * subfield_separation[i] + x[i]
            non_dominant_y[i] = np.sin(tuning_angle[i] * np.pi / 180.0) * subfield_separation[i] + y[i]
        else:
            kernel = dom_kernels[i]
            dom_temporal_kernels[i, -len(kernel) :] = kernel

    if composite_idx:
        comp_dom_kernels = _combine_basis_delays(
            dom_basis[composite_idx],
            temporal_weights_dom[composite_idx],
            np.asarray(shifted_dom_delays),
        )
        comp_non_kernels = _combine_basis_delays(
            non_dom_basis,
            temporal_weights_non_dom[composite_idx],
            np.asarray(shifted_non_delays),
        )
        for row, dom_kernel, non_dom_kernel in zip(composite_idx, comp_dom_kernels, comp_non_kernels):
            dom_temporal_kernels[row, -len(dom_kernel) :] = dom_kernel
            non_dom_temporal_kernels[row, -len(non_dom_kernel) :] = non_dom_kernel

    to_save = {
        "dom_temporal_kernels": dom_temporal_kernels.astype(np.float32),
        "non_dom_temporal_kernels": non_dom_temporal_kernels.astype(np.float32),
        "non_dominant_x": non_dominant_x.astype(np.float32),
        "non_dominant_y": non_dominant_y.astype(np.float32),
        "amplitude": amplitude.astype(np.float32),
        "non_dom_amplitude": non_dom_amplitude.astype(np.float32),
        "spontaneous_firing_rates": spontaneous_firing_rates.astype(np.float32),
    }

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "wb") as f:
        pkl.dump(to_save, f)
    print(f"[prepare_lgn_kernels] Cache ready: {output}")


def main():
    args = _parse_args()
    lgn_data_path = os.path.abspath(os.path.expanduser(args.lgn_data_path))
    if not os.path.isfile(lgn_data_path):
        raise FileNotFoundError(f"lgn_data_path not found: {lgn_data_path}")
    _check_lgn_csv_compat(lgn_data_path)

    lgn_py = os.path.abspath(os.path.expanduser(args.lgn_py)) if args.lgn_py else None
    if lgn_py and not os.path.isfile(lgn_py):
        raise FileNotFoundError(f"lgn.py not found: {lgn_py}")

    default_cache = os.path.join(os.path.dirname(lgn_py or lgn_data_path), "temporal_kernels.pkl")
    output = default_cache if args.output is None else os.path.abspath(os.path.expanduser(args.output))

    if os.path.isfile(output) and not args.force:
        print(f"[prepare_lgn_kernels] Found existing cache: {output}")
        return

    _patch_numpy_legacy_aliases()

    if lgn_py is None:
        bmtk_root = os.path.abspath(os.path.expanduser(args.bmtk_root or _default_bmtk_root()))
        print(f"[prepare_lgn_kernels] Using BMTK root: {bmtk_root}")
        _generate_with_bmtk(lgn_data_path, output, bmtk_root)
        return

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
