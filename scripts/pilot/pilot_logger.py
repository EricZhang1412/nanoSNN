"""Per-run JSON logger for the Gate-1 pilot.

Used as a Lightning callback that, at the end of training, writes a single
JSON file with the metrics defined in `docs/PILOT_GATE1.md` §4:

    {
      "task": "dvs128",
      "condition": "C3_MGA",
      "seed": 42,
      "top1_best": 96.4,
      "top1_last10_mean": 95.9,
      "top1_last10_std": 0.3,
      "best_epoch": 187,
      "epochs_to_90pct": 42,
      "total_train_time_h": 2.6,
      "params_total_M": 1.81,
      "params_gate_M": 0.001,
      "spike_rate_attn_avg": 0.27,
      "fp_mults_attention_path_per_step": 0,
      "gate_sparsity_gamma": 0.31,
      "gate_sparsity_beta": 0.42
    }
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_only


class PilotJSONLogger(Callback):
    """Records per-epoch val/top1 and writes a final summary JSON on fit-end."""

    def __init__(
        self,
        out_dir: str,
        task: str,
        condition: str,
        seed: int,
        last_n: int = 10,
        accuracy_target: float = 0.90,
    ):
        super().__init__()
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.task = task
        self.condition = condition
        self.seed = int(seed)
        self.last_n = int(last_n)
        self.accuracy_target = float(accuracy_target)

        self._epoch_top1: list[float] = []
        self._epoch_train_loss: list[float] = []
        self._epoch_attn_spike_rate: list[float] = []
        self._epoch_gate_gamma_rate: list[float] = []
        self._epoch_gate_beta_rate: list[float] = []
        self._start_t: float | None = None

    # ------------------------------------------------------------------

    @rank_zero_only
    def on_fit_start(self, trainer, pl_module) -> None:
        self._start_t = time.perf_counter()

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        metric = trainer.callback_metrics.get("val/top1")
        if metric is None:
            return
        v = float(metric.detach().cpu().item() if torch.is_tensor(metric) else metric)
        # Lightning logs probabilities in [0, 1]; convert to %.
        if v <= 1.5:
            v *= 100.0
        self._epoch_top1.append(v)

        # Optional auxiliary metrics, all best-effort.
        for src_name, sink in (
            ("train/loss", self._epoch_train_loss),
            ("val/attn_spike_rate", self._epoch_attn_spike_rate),
            ("val/gate_gamma_rate", self._epoch_gate_gamma_rate),
            ("val/gate_beta_rate", self._epoch_gate_beta_rate),
        ):
            m = trainer.callback_metrics.get(src_name)
            if m is not None:
                sink.append(float(m.detach().cpu().item() if torch.is_tensor(m) else m))

    @rank_zero_only
    def on_fit_end(self, trainer, pl_module) -> None:
        end_t = time.perf_counter()
        total_h = (end_t - (self._start_t or end_t)) / 3600.0

        top1 = self._epoch_top1
        if not top1:
            payload = self._fallback_payload(total_h)
        else:
            best_top1 = max(top1)
            best_epoch = int(top1.index(best_top1))
            last_n = top1[-self.last_n:]
            mean_last = sum(last_n) / len(last_n)
            std_last = (
                math.sqrt(sum((v - mean_last) ** 2 for v in last_n) / len(last_n))
                if len(last_n) > 1 else 0.0
            )
            # First epoch reaching the target accuracy (%).
            tgt = self.accuracy_target * 100.0
            try:
                epochs_to_target = next(i for i, v in enumerate(top1) if v >= tgt)
            except StopIteration:
                epochs_to_target = -1

            # Param counts from the underlying model (Lightning wraps it as .model).
            inner = getattr(pl_module, "model", pl_module)
            total_params = sum(p.numel() for p in inner.parameters())
            gate_params = self._gate_param_count(inner)

            payload = {
                "task": self.task,
                "condition": self.condition,
                "seed": self.seed,
                "top1_best": round(best_top1, 4),
                "top1_last10_mean": round(mean_last, 4),
                "top1_last10_std": round(std_last, 4),
                "best_epoch": best_epoch,
                "epochs_to_90pct": epochs_to_target,
                "total_train_time_h": round(total_h, 3),
                "params_total_M": round(total_params / 1e6, 4),
                "params_gate_M": round(gate_params / 1e6, 6),
                "spike_rate_attn_avg": round(
                    sum(self._epoch_attn_spike_rate) / max(1, len(self._epoch_attn_spike_rate)),
                    4,
                ) if self._epoch_attn_spike_rate else None,
                "fp_mults_attention_path_per_step": self._fp_mults_estimate(inner),
                "gate_sparsity_gamma": round(
                    sum(self._epoch_gate_gamma_rate) / max(1, len(self._epoch_gate_gamma_rate)),
                    4,
                ) if self._epoch_gate_gamma_rate else None,
                "gate_sparsity_beta": round(
                    sum(self._epoch_gate_beta_rate) / max(1, len(self._epoch_gate_beta_rate)),
                    4,
                ) if self._epoch_gate_beta_rate else None,
            }

        out_path = self.out_dir / f"seed{self.seed}.json"
        out_path.write_text(json.dumps(payload, indent=2))
        try:
            trainer.print(f"[PilotJSONLogger] wrote {out_path}")
        except Exception:
            print(f"[PilotJSONLogger] wrote {out_path}")

    # ------------------------------------------------------------------

    def _fallback_payload(self, total_h: float) -> dict[str, Any]:
        return {
            "task": self.task,
            "condition": self.condition,
            "seed": self.seed,
            "top1_best": None,
            "top1_last10_mean": None,
            "top1_last10_std": None,
            "best_epoch": -1,
            "epochs_to_90pct": -1,
            "total_train_time_h": round(total_h, 3),
            "params_total_M": None,
            "params_gate_M": None,
            "spike_rate_attn_avg": None,
            "fp_mults_attention_path_per_step": None,
            "gate_sparsity_gamma": None,
            "gate_sparsity_beta": None,
            "note": "no val/top1 metric observed",
        }

    @staticmethod
    def _gate_param_count(model) -> int:
        """Sum of parameters whose qualified name suggests they belong to a gate."""
        gate_keywords = (
            "gate_up", "gate_down",              # C1
            "log_tau_gamma", "log_tau_beta",     # C3
            "V_gamma", "V_beta_raw",             # C3 (V_gamma unconstrained, V_beta via softplus)
        )
        total = 0
        for name, p in model.named_parameters():
            if any(k in name for k in gate_keywords):
                total += p.numel()
        return total

    @staticmethod
    def _fp_mults_estimate(model) -> int | None:
        """Pick up `estimate_attn_fp_mults_per_step` from the first attention module."""
        for mod in model.modules():
            est = getattr(mod, "estimate_attn_fp_mults_per_step", None)
            if callable(est):
                # We can't know K_shape without a forward; report -1 as placeholder.
                return -1
        return None
