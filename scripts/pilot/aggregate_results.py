"""Aggregate per-run JSONs into a CSV and print the Gate-1 decision table.

Reads `pilot_results/<task>/<condition>/seed<seed>.json` for each combination
listed below and emits:

  pilot_results/raw.csv     — flat per-run metrics
  pilot_results/summary.md  — decision table from PILOT §6 (mean ± 95% CI per task)

Usage:  uv run python -m scripts.pilot.aggregate_results --results_dir pilot_results
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

TASKS = ["dvs128", "shd"]
CONDITIONS = ["c0_sdla", "c1_lowrank", "c2_oneminusk", "c3_mga"]
SEEDS = [42, 123, 2024]


def _load_run(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"[warn] failed to load {path}: {e}")
        return None


def _ci95(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
    ci = 1.96 * sd / math.sqrt(len(values))
    return mean, ci


def aggregate(results_dir: Path) -> None:
    rows = []
    for task in TASKS:
        for cond in CONDITIONS:
            for seed in SEEDS:
                p = results_dir / task / cond / f"seed{seed}.json"
                payload = _load_run(p)
                if payload is None:
                    continue
                rows.append(payload)

    if not rows:
        print(f"No results found under {results_dir}")
        return

    # Raw CSV
    cols = sorted({k for r in rows for k in r.keys()})
    cols = ["task", "condition", "seed"] + [c for c in cols if c not in {"task", "condition", "seed"}]
    csv_path = results_dir / "raw.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {csv_path} ({len(rows)} rows)")

    # Decision table
    lines = ["# Gate-1 Pilot results\n\n## Decision table\n"]
    for task in TASKS:
        lines.append(f"### {task}\n")
        lines.append("| Condition | top1 mean ± 95%CI | best_epoch (mean) | params_M | gate_params_M | FP-mults/step | γ-rate | β-rate |")
        lines.append("|---|---|---|---|---|---|---|---|")
        per_cond: dict[str, list[float]] = {c: [] for c in CONDITIONS}
        per_cond_other: dict[str, dict[str, list[float]]] = {c: {} for c in CONDITIONS}
        for cond in CONDITIONS:
            for r in rows:
                if r.get("task") == task and r.get("condition") == cond:
                    if r.get("top1_last10_mean") is not None:
                        per_cond[cond].append(float(r["top1_last10_mean"]))
                    for k in ("best_epoch", "params_total_M", "params_gate_M",
                              "fp_mults_attention_path_per_step",
                              "gate_sparsity_gamma", "gate_sparsity_beta"):
                        per_cond_other[cond].setdefault(k, []).append(r.get(k))
        for cond in CONDITIONS:
            mean, ci = _ci95(per_cond[cond])
            be = _mean(per_cond_other[cond].get("best_epoch", []))
            pm = _mean(per_cond_other[cond].get("params_total_M", []))
            gp = _mean(per_cond_other[cond].get("params_gate_M", []))
            fp = _first_non_none(per_cond_other[cond].get("fp_mults_attention_path_per_step", []))
            gr = _mean(per_cond_other[cond].get("gate_sparsity_gamma", []))
            br = _mean(per_cond_other[cond].get("gate_sparsity_beta", []))
            lines.append(
                f"| {cond} | {mean:.2f} ± {ci:.2f} | "
                f"{_fmt(be)} | {_fmt(pm)} | {_fmt(gp)} | {_fmt(fp)} | "
                f"{_fmt(gr)} | {_fmt(br)} |"
            )

        # Decision logic per pilot §6
        a_c0 = _ci95(per_cond["c0_sdla"])[0]
        a_c1 = _ci95(per_cond["c1_lowrank"])[0]
        a_c2 = _ci95(per_cond["c2_oneminusk"])[0]
        a_c3, ci_c3 = _ci95(per_cond["c3_mga"])
        a_triv = max(a_c1, a_c2) if not (math.isnan(a_c1) and math.isnan(a_c2)) else float("nan")
        verdict = _gate_decision(a_c3, ci_c3, a_triv, per_cond)
        lines.append(f"\n**A_triv = {a_triv:.2f}** ; **A_C3 − A_triv = {a_c3 - a_triv:+.2f}** ; "
                     f"**verdict on {task}: {verdict}**\n")

    md_path = results_dir / "summary.md"
    md_path.write_text("\n".join(lines))
    print(f"wrote {md_path}")


def _gate_decision(a_c3, ci_c3, a_triv, per_cond) -> str:
    """Per pilot §6: PASS / PIVOT-eligible / FAIL."""
    if math.isnan(a_c3) or math.isnan(a_triv):
        return "INSUFFICIENT DATA"
    delta = a_c3 - a_triv
    # Non-overlap test (rough): require lower-bound of C3 CI ≥ upper-bound of best trivial.
    best_triv_name = "c1_lowrank" if (_ci95(per_cond["c1_lowrank"])[0] or 0) > (_ci95(per_cond["c2_oneminusk"])[0] or 0) else "c2_oneminusk"
    a_best, ci_best = _ci95(per_cond[best_triv_name])
    non_overlap = (a_c3 - ci_c3) > (a_best + ci_best)
    if delta >= 0.5 and non_overlap:
        return "PASS"
    if abs(delta) < 0.5:
        return "PIVOT-eligible (check T_eff / FP-mults)"
    return "FAIL"


def _mean(values):
    vs = [float(v) for v in values if v is not None]
    if not vs:
        return None
    return sum(vs) / len(vs)


def _first_non_none(values):
    for v in values:
        if v is not None:
            return v
    return None


def _fmt(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3g}"
    return str(v)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, default="pilot_results")
    args = p.parse_args()
    aggregate(Path(args.results_dir).resolve())
