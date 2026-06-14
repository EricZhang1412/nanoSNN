"""Generate Gate-1 recurrence/gate-path complexity tables.

This is intentionally config-only and does not import torch, so it can run in a
lightweight shell before launching remote jobs.  Counts exclude shared Q/K/V,
MLP, projection, and data movement; they cover only the condition-specific
attention recurrence/gate path used by the pilot decision.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIGS = {
    "dvs128/c0_sdla": "configs/model_configs/pilot/spikformer_dvs_c0_sdla.yaml",
    "dvs128/c1_lowrank": "configs/model_configs/pilot/spikformer_dvs_c1_lowrank.yaml",
    "dvs128/c2_oneminusk": "configs/model_configs/pilot/spikformer_dvs_c2_oneminusk.yaml",
    "dvs128/c3_mga": "configs/model_configs/pilot/spikformer_dvs_c3_mga.yaml",
    "shd/c0_sdla": "configs/model_configs/pilot/spikformer_shd_c0_sdla.yaml",
    "shd/c1_lowrank": "configs/model_configs/pilot/spikformer_shd_c1_lowrank.yaml",
    "shd/c2_oneminusk": "configs/model_configs/pilot/spikformer_shd_c2_oneminusk.yaml",
    "shd/c3_mga": "configs/model_configs/pilot/spikformer_shd_c3_mga.yaml",
}


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _condition(cfg: dict[str, Any], label: str) -> str:
    return str(cfg.get("attention_type") or label.split("/")[-1])


def _counts(cfg: dict[str, Any], label: str) -> dict[str, Any]:
    cond = _condition(cfg, label)
    depth = int(cfg.get("depth", 1))
    h = int(cfg.get("num_heads", 1))
    dim = int(cfg.get("embed_dim", cfg.get("dim", 256)))
    d = dim // h
    t = int(cfg.get("T", cfg.get("max_len", 1)))
    rank = int(cfg.get("gate_rank", 16))

    if cond == "c1_lowrank":
        fp_block_step = h * (2 * d * rank + d * d)
        gate_params_block = 2 * d * rank
    elif cond == "c2_oneminusk":
        fp_block_step = h * d * d
        gate_params_block = 0
    elif cond == "c3_mga":
        fp_block_step = 0
        use_write_scale = bool(cfg.get("mga_use_write_scale", True))
        # log_tau_gamma + V_gamma + log_tau_beta + V_beta_raw (+ write_scale)
        gate_params_block = 2 * h * d + 2 * h + (h if use_write_scale else 0)
    else:
        fp_block_step = 0
        gate_params_block = 0

    fp_model_step = fp_block_step * depth
    return {
        "label": label,
        "condition": cond,
        "T": t,
        "depth": depth,
        "num_heads": h,
        "head_dim": d,
        "gate_rank": rank if cond == "c1_lowrank" else "",
        "fp_mults_per_block_step": fp_block_step,
        "fp_mults_per_model_step": fp_model_step,
        "fp_mults_per_model_sequence": fp_model_step * t,
        "gate_params": gate_params_block * depth,
        "gate_params_M": round(gate_params_block * depth / 1e6, 6),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, default="pilot_results")
    p.add_argument(
        "--config",
        action="append",
        default=[],
        help="Optional label=path entry. If omitted, use canonical DVS/SHD pilot configs.",
    )
    args = p.parse_args()

    entries = {}
    if args.config:
        for item in args.config:
            label, path = item.split("=", 1)
            entries[label] = path
    else:
        entries = DEFAULT_CONFIGS

    rows = [_counts(_load_yaml(path), label) for label, path in entries.items()]
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "complexity.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "# Gate-path complexity",
        "",
        "Counts are per sample and exclude shared Q/K/V/proj/MLP work.",
        "",
        "| label | T | depth | H | D | FP-mults/block/step | FP-mults/model/step | FP-mults/model/seq | gate params |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['label']} | {r['T']} | {r['depth']} | {r['num_heads']} | {r['head_dim']} | "
            f"{r['fp_mults_per_block_step']} | {r['fp_mults_per_model_step']} | "
            f"{r['fp_mults_per_model_sequence']} | {r['gate_params']} |"
        )
    md_path = out_dir / "complexity.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
