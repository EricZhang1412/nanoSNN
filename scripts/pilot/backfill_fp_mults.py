"""Backfill `fp_mults_attention_path_per_step` in existing pilot JSONs.

Use this after pulling old remote results produced before the PilotJSONLogger
started recording real gate-path FP-mult counts.  It updates only JSON files
that already exist; no training or torch import is required.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.pilot.complexity_table import DEFAULT_CONFIGS, _counts, _load_yaml


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, default="pilot_results")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite non-placeholder values too")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    complexity = {
        label: _counts(_load_yaml(path), label)["fp_mults_per_model_step"]
        for label, path in DEFAULT_CONFIGS.items()
    }

    n = 0
    for label, fp in complexity.items():
        task, cond = label.split("/", 1)
        for path in (results_dir / task / cond).glob("seed*.json"):
            payload = json.loads(path.read_text())
            old = payload.get("fp_mults_attention_path_per_step")
            if args.overwrite or old in (None, -1, "-1"):
                payload["fp_mults_attention_path_per_step"] = int(fp)
                path.write_text(json.dumps(payload, indent=2) + "\n")
                n += 1
                print(f"updated {path}: {old} -> {fp}")
    print(f"updated {n} JSON files")


if __name__ == "__main__":
    main()
