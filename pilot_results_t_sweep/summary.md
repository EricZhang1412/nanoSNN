# Gate-1 Pilot results

## Decision table

### shd_T25

| Condition | top1 mean ± 95%CI | best_epoch (mean) | params_M | gate_params_M | FP-mults/step | T_eff | E_diag | E_past | mean past lag | γ-rate | β-rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| c0_sdla | 81.55 (n=1; CI unavailable) | 85 | 1.92 | 0 | 0 | — | — | — | — | — | — |
| c1_lowrank | 83.57 (n=1; CI unavailable) | 84 | 1.92 | 0.0041 | 49152 | — | — | — | — | — | — |
| c2_oneminusk | 84.38 (n=1; CI unavailable) | 82 | 1.92 | 0 | 32768 | — | — | — | — | — | — |
| c3_mga | 84.79 (n=1; CI unavailable) | 90 | 1.92 | 0.0013 | 0 | — | — | — | — | 0.596 | 0.751 |

**A_triv = 84.38** ; **A_C3 − A_triv = +0.41** ; **verdict on shd_T25: INSUFFICIENT DATA (c1_lowrank=1/3, c2_oneminusk=1/3, c3_mga=1/3)**

### shd_T50

| Condition | top1 mean ± 95%CI | best_epoch (mean) | params_M | gate_params_M | FP-mults/step | T_eff | E_diag | E_past | mean past lag | γ-rate | β-rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| c0_sdla | 85.51 (n=1; CI unavailable) | 78 | 1.92 | 0 | 0 | — | — | — | — | — | — |
| c1_lowrank | 87.56 (n=1; CI unavailable) | 95 | 1.92 | 0.0041 | 49152 | — | — | — | — | — | — |
| c2_oneminusk | 86.67 (n=1; CI unavailable) | 73 | 1.92 | 0 | 32768 | — | — | — | — | — | — |
| c3_mga | 85.80 (n=1; CI unavailable) | 94 | 1.92 | 0.0013 | 0 | — | — | — | — | 0.569 | 0.647 |

**A_triv = 87.56** ; **A_C3 − A_triv = -1.77** ; **verdict on shd_T50: INSUFFICIENT DATA (c1_lowrank=1/3, c2_oneminusk=1/3, c3_mga=1/3)**

### shd_T100

| Condition | top1 mean ± 95%CI | best_epoch (mean) | params_M | gate_params_M | FP-mults/step | T_eff | E_diag | E_past | mean past lag | γ-rate | β-rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| c0_sdla | 84.56 (n=1; CI unavailable) | 96 | 1.92 | 0 | 0 | — | — | — | — | — | — |
| c1_lowrank | 85.70 (n=1; CI unavailable) | 66 | 1.92 | 0.0041 | 49152 | — | — | — | — | — | — |
| c2_oneminusk | 85.40 (n=1; CI unavailable) | 81 | 1.92 | 0 | 32768 | — | — | — | — | — | — |
| c3_mga | 86.38 (n=1; CI unavailable) | 55 | 1.92 | 0.0013 | 0 | — | — | — | — | 0.538 | 0.627 |

**A_triv = 85.70** ; **A_C3 − A_triv = +0.68** ; **verdict on shd_T100: INSUFFICIENT DATA (c1_lowrank=1/3, c2_oneminusk=1/3, c3_mga=1/3)**

### shd_T200

| Condition | top1 mean ± 95%CI | best_epoch (mean) | params_M | gate_params_M | FP-mults/step | T_eff | E_diag | E_past | mean past lag | γ-rate | β-rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| c0_sdla | 80.37 (n=1; CI unavailable) | 90 | 1.92 | 0 | 0 | — | — | — | — | — | — |
| c1_lowrank | — | — | — | — | — | — | — | — | — | — | — |
| c2_oneminusk | — | — | — | — | — | — | — | — | — | — | — |
| c3_mga | — | — | — | — | — | — | — | — | — | — | — |

**A_triv = —** ; **A_C3 − A_triv = —** ; **verdict on shd_T200: INSUFFICIENT DATA (c1_lowrank=0/3, c2_oneminusk=0/3, c3_mga=0/3)**
