"""Thin wrapper around `train.py` for Gate-1 pilot runs.

Differences vs. `train.py`:

  * Adds CLI flags `--task`, `--condition`, `--seed`, `--results_dir`.
  * Forces the random seed (overriding `random_seed` in train_config).
  * Injects a PilotJSONLogger callback that writes
    `<results_dir>/<task>/<condition>/seed<seed>.json` on fit-end.
  * Names the checkpoint dir
    `<results_dir>/checkpoints/<task>/<condition>/seed<seed>/`.

Usage example (single run):
    uv run python -m scripts.pilot.train_pilot \\
        --task dvs128 --condition c3_mga --seed 42 \\
        --project_config configs/default_project_configs.yaml \\
        --data_config configs/data_configs/dvs128gesture_pilot.yaml \\
        --train_config configs/train_configs/pilot_dvs.yaml \\
        --model_config configs/model_configs/pilot/spikformer_dvs_c3_mga.yaml \\
        --optimizer_config configs/optimizer_configs/pilot_dvs.yaml \\
        --results_dir pilot_results

Use `scripts/pilot/run_all.sh` to dispatch all 24 runs across GPUs.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make the repo importable when run as a script.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _REPO_ROOT)

# Import train.py's primitives.
import train as train_mod  # noqa: E402

from scripts.pilot.pilot_logger import PilotJSONLogger  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Gate-1 pilot training wrapper")
    p.add_argument("--project_config", type=str, required=True)
    p.add_argument("--data_config", type=str, required=True)
    p.add_argument("--train_config", type=str, required=True)
    p.add_argument("--model_config", type=str, required=True)
    p.add_argument("--optimizer_config", type=str, required=True)
    p.add_argument("--resume", type=str, default="none")
    p.add_argument("--ckpt_dir", type=str, default=None)
    # Pilot-specific
    # Keep these labels free-form so follow-up sweeps can write to
    # task-specific namespaces such as `shd_T200` or condition labels such as
    # `c3_mga_gamma_spike` while still using the same training wrapper.
    p.add_argument("--task", type=str, required=True)
    p.add_argument("--condition", type=str, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--results_dir", type=str, default="pilot_results")
    return p.parse_args()


def main():
    args = parse_args()
    pilot_results_dir = Path(args.results_dir).resolve()

    # Override checkpoint dir to a deterministic per-run path.
    ckpt_dir = (pilot_results_dir / "checkpoints" / args.task / args.condition
                / f"seed{args.seed}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.ckpt_dir = str(ckpt_dir)

    # Override the seed in train_config by writing a patched YAML to a temp file.
    import yaml, tempfile
    with open(args.train_config, "r", encoding="utf-8") as f:
        train_cfg_dict = yaml.safe_load(f)
    train_cfg_dict["random_seed"] = int(args.seed)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(train_cfg_dict, f, sort_keys=False)
        args.train_config = f.name

    # Hook to add our PilotJSONLogger to the Trainer's callbacks.
    json_dir = pilot_results_dir / args.task / args.condition
    json_dir.mkdir(parents=True, exist_ok=True)
    pilot_callback = PilotJSONLogger(
        out_dir=str(json_dir),
        task=args.task,
        condition=args.condition,
        seed=args.seed,
    )

    # Monkey-patch Trainer init in train.py to inject the callback.
    import lightning.pytorch as L
    _orig_trainer_init = L.Trainer.__init__

    def _patched_init(self, *t_args, callbacks=None, **t_kwargs):
        callbacks = list(callbacks) if callbacks is not None else []
        callbacks.append(pilot_callback)
        return _orig_trainer_init(self, *t_args, callbacks=callbacks, **t_kwargs)

    L.Trainer.__init__ = _patched_init
    try:
        train_mod.train(args)
    finally:
        L.Trainer.__init__ = _orig_trainer_init


if __name__ == "__main__":
    main()
