"""
Eval-only temporal-control runner for DyCo-SNN Block A checkpoints.

This script loads already trained checkpoints and tests them on clean and
counterfactual versions of synthetic_temporal_order without calling fit().

Example:
  uv run python -m scripts.eval_dyco_temporal_controls \
    --ckpt_root exp/dyco_synth/blockA_seed42 \
    --out_csv exp/dyco_synth/blockA_seed42/temporal_controls.csv
"""
from __future__ import annotations

import argparse
import csv
import copy
import re
import sys
from pathlib import Path

import lightning as L
from lightning.pytorch import Trainer, seed_everything


CONTROL_FLAGS = {
    "clean": dict(shuffle_time=False, reverse_time=False, first_last_only=False),
    "shuffle": dict(shuffle_time=True, reverse_time=False, first_last_only=False),
    "reverse": dict(shuffle_time=False, reverse_time=True, first_last_only=False),
    "first_last": dict(shuffle_time=False, reverse_time=False, first_last_only=True),
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_ckpt(mode_dir: Path, ckpt_file: str | None) -> Path:
    if ckpt_file:
        ckpt = mode_dir / ckpt_file
        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        return ckpt

    last = mode_dir / "last.ckpt"
    if last.is_file():
        return last

    candidates = sorted(mode_dir.glob("*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No .ckpt files found under {mode_dir}")

    def key(path: Path):
        m = re.search(r"epoch=(\d+)-step=(\d+)", path.name)
        if m:
            return (int(m.group(1)), int(m.group(2)), path.stat().st_mtime)
        return (-1, -1, path.stat().st_mtime)

    return max(candidates, key=key)


def _set_control_flags(data_cfg, flags: dict[str, bool]):
    for name in ("shuffle_time", "reverse_time", "first_last_only"):
        setattr(data_cfg, name, bool(flags.get(name, False)))


def _model_config_path(mode: str) -> Path:
    return _repo_root() / "configs" / "model_configs" / f"dyco_snn_{mode}.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval DyCo-SNN temporal controls from checkpoints.")
    parser.add_argument("--data_config", type=str, default="configs/data_configs/synthetic_temporal_order.yaml")
    parser.add_argument("--train_config", type=str, default="configs/train_configs/dyco_pilot.yaml")
    parser.add_argument("--optimizer_config", type=str, default="configs/optimizer_configs/dyco.yaml")
    parser.add_argument("--ckpt_root", type=str, default="exp/dyco_synth/blockA_seed42")
    parser.add_argument("--ckpt_file", type=str, default=None,
                        help="Optional exact checkpoint filename inside each mode dir, e.g. epoch=003-step=0001000.ckpt")
    parser.add_argument("--modes", nargs="+", default=["d_only", "c_only", "dc", "ff"],
                        choices=["d_only", "c_only", "dc", "ff"])
    parser.add_argument("--controls", nargs="+", default=["clean", "shuffle", "reverse", "first_last"],
                        choices=sorted(CONTROL_FLAGS))
    parser.add_argument("--out_csv", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "gpu"])
    args = parser.parse_args()

    root = _repo_root()
    sys.path.insert(0, str(root))

    from data import VisionDataModule
    from models.build_model import build_model
    from utils.load_config import load_config
    from utils.resume import register_checkpoint_safe_globals

    seed_everything(args.seed, workers=True)
    register_checkpoint_safe_globals()

    data_base = load_config(str(root / args.data_config))
    train_cfg = load_config(str(root / args.train_config))
    opt_cfg = load_config(str(root / args.optimizer_config))
    ckpt_root = root / args.ckpt_root

    rows = []
    for mode in args.modes:
        model_cfg = load_config(str(_model_config_path(mode)))
        ckpt_path = _resolve_ckpt(ckpt_root / mode, args.ckpt_file)
        print(f"\n===== mode={mode} ckpt={ckpt_path} =====")

        for control in args.controls:
            data_cfg = copy.deepcopy(data_base)
            _set_control_flags(data_cfg, CONTROL_FLAGS[control])
            print(f"\n--- control={control} flags={CONTROL_FLAGS[control]} ---")

            datamodule = VisionDataModule(data_config=data_cfg, train_config=train_cfg)
            lit_model = build_model(
                model_config=model_cfg,
                optimizer_config=opt_cfg,
                train_config=train_cfg,
                data_config=data_cfg,
            )
            trainer = Trainer(
                accelerator=args.device,
                devices=1,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=True,
            )
            metrics = trainer.test(lit_model, datamodule=datamodule, ckpt_path=str(ckpt_path))[0]
            row = {
                "mode": mode,
                "control": control,
                "ckpt": str(ckpt_path),
                "test_top1": float(metrics.get("test/top1", float("nan"))),
                "test_loss": float(metrics.get("test/loss", float("nan"))),
                "test_cls_loss": float(metrics.get("test/cls_loss", float("nan"))),
                "test_spike_rate_hz": float(metrics.get("test/spike_rate_hz", float("nan"))),
            }
            rows.append(row)

    print("\n===== summary =====")
    header = ["mode", "control", "test_top1", "test_loss", "test_spike_rate_hz"]
    print(",".join(header))
    for row in rows:
        print(",".join(str(row[h]) for h in header))

    if args.out_csv:
        out_path = root / args.out_csv
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else header)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
