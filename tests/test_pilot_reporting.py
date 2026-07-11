from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
import numpy as np

from scripts.pilot.aggregate_results import _ci95, _gate_decision, aggregate
from scripts.pilot.complexity_table import _counts
from scripts.pilot.pilot_logger import PilotJSONLogger
from scripts.pilot.st_erf_diag import summarize


class _GateParameters(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_tau_gamma_raw = nn.Parameter(torch.zeros(4, 64))
        self.log_tau_beta_raw = nn.Parameter(torch.zeros(4))
        self.V_gamma = nn.Parameter(torch.zeros(4, 64))
        self.V_beta_raw = nn.Parameter(torch.zeros(4))
        self.gate_input_norm = nn.LayerNorm(64)
        self.log_write_scale = nn.Parameter(torch.zeros(4))
        self.unrelated = nn.Linear(64, 64)


class PilotReportingTests(unittest.TestCase):
    def test_single_seed_has_no_confidence_interval_or_verdict(self):
        _, ci = _ci95([86.0])
        self.assertTrue(math.isnan(ci))

        per_cond = {
            "c1_lowrank": [85.0],
            "c2_oneminusk": [84.0],
            "c3_mga": [86.0],
        }
        verdict = _gate_decision(86.0, ci, 85.0, per_cond, {}, "shd", min_seeds=3)
        self.assertTrue(verdict.startswith("INSUFFICIENT DATA"))

    def test_three_seed_clear_margin_can_pass(self):
        per_cond = {
            "c1_lowrank": [85.0, 85.0, 85.0],
            "c2_oneminusk": [84.0, 84.0, 84.0],
            "c3_mga": [86.0, 86.0, 86.0],
        }
        a_c3, ci_c3 = _ci95(per_cond["c3_mga"])
        verdict = _gate_decision(a_c3, ci_c3, 85.0, per_cond, {}, "shd", min_seeds=3)
        self.assertEqual(verdict, "PASS")

    def test_three_seed_ci_uses_student_t_and_avoids_false_pass(self):
        per_cond = {
            "c1_lowrank": [84.0, 85.0, 86.0],
            "c2_oneminusk": [83.0, 84.0, 85.0],
            "c3_mga": [87.0, 88.0, 89.0],
        }
        a_c1, ci_c1 = _ci95(per_cond["c1_lowrank"])
        a_c3, ci_c3 = _ci95(per_cond["c3_mga"])
        self.assertAlmostEqual(ci_c1, 2.4841, places=3)
        verdict = _gate_decision(a_c3, ci_c3, a_c1, per_cond, {}, "shd", min_seeds=3)
        self.assertEqual(verdict, "FAIL")

    def test_single_seed_summary_is_marked_exploratory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for condition, score in (
                ("c1_lowrank", 85.0),
                ("c2_oneminusk", 84.0),
                ("c3_mga", 86.0),
            ):
                out = root / "shd" / condition
                out.mkdir(parents=True)
                (out / "seed42.json").write_text(
                    json.dumps(
                        {
                            "task": "shd",
                            "condition": condition,
                            "seed": 42,
                            "top1_last10_mean": score,
                        }
                    )
                )

            aggregate(
                root,
                tasks=["shd"],
                conditions=["c1_lowrank", "c2_oneminusk", "c3_mga"],
                seeds=[42],
                min_seeds=3,
            )
            summary = (root / "summary.md").read_text()
            self.assertIn("CI unavailable", summary)
            self.assertIn("INSUFFICIENT DATA", summary)
            self.assertNotIn("± 0.00", summary)
            self.assertNotIn("verdict on shd: PASS", summary)

    def test_c3_parameter_accounting_includes_layer_norm(self):
        runtime_count = PilotJSONLogger._gate_param_count(_GateParameters())
        self.assertEqual(runtime_count, 652)

        config_count = _counts(
            {
                "attention_type": "c3_mga",
                "T": 100,
                "depth": 2,
                "num_heads": 4,
                "embed_dim": 256,
                "mga_gate_input_norm": True,
                "mga_use_write_scale": True,
            },
            "shd/c3_mga",
        )
        self.assertEqual(config_count["gate_params"], 1304)

    def test_st_erf_reports_cross_time_lag_metrics(self):
        matrix = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
                [0.0, 3.0, 1.0],
            ]
        )
        stats = summarize(matrix)
        self.assertGreater(stats["E_past"], 0.0)
        self.assertAlmostEqual(stats["mean_past_lag"], 1.0)
        self.assertAlmostEqual(stats["T_eff_past"], 1.0)


if __name__ == "__main__":
    unittest.main()
