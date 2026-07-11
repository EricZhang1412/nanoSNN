# Gate-1 Pilot Decision And Follow-Up Plan

This document records the legacy Gate-1 results and commands for the next
experiment wave. Run commands from the repository root:

```bash
cd /path/to/nanoSNN
```

## 1. Frozen Pilot Decision

The original per-task numbers are retained below, but the preregistered overall
decision was STOP: DVS has `C3-C1=-0.5902 pp`, which is neither PASS nor
PIVOT-eligible, and the protocol specifies STOP if any task fails. Treat the
long-horizon work as a new pivot hypothesis rather than a passed Gate-1.

The legacy pipeline also used the official test split for validation and applied
random SHD time shift during evaluation. These values are exploratory and must
be rerun under `docs/MGA_V2_SPEC.md` before publication.

Headline metric is `top1_last10_mean`, as pre-registered in
`docs/PILOT_GATE1.md`.

### DVS128-Gesture: short-horizon control

T=16 is short and the task is close to saturated: C0 already reaches 96.11%.
MGA is competitive but does not beat the strongest low-rank gate.

| condition | mean top1_last10 | note |
|---|---:|---|
| C0 SDLA | 96.1111 | memoryless baseline is already strong |
| C1 lowrank | 96.7708 | strongest on DVS |
| C2 one-minus-k | 92.9514 | unstable, one seed collapses |
| C3 MGA | 96.1806 | competitive, not a DVS PASS |

Interpretation: DVS is retained as a short-horizon/saturation control, not the
main evidence for MGA's long-horizon advantage.

### SHD: PIVOT-eligible

| condition | mean top1_last10 | mean top1_best |
|---|---:|---:|
| C0 SDLA | 84.3080 | 85.9231 |
| C1 lowrank | 86.0295 | 87.8431 |
| C2 one-minus-k | 85.0382 | 86.5844 |
| C3 MGA | 86.3445 | 87.7542 |

MGA does not meet the PASS threshold over C1 (`+0.315 pp < +0.5 pp`). It meets
the original SHD-only PIVOT criterion relative to C2, but this does not override
the overall STOP decision or the evaluation-protocol limitations.

| condition | E_diag | T_eff |
|---|---:|---:|
| C0 SDLA | 0.077550 | 79.5496 |
| C1 lowrank | 0.001193 | 72.2688 |
| C2 one-minus-k | 0.000684 | 69.2915 |
| C3 MGA | 0.000444 | 79.3646 |

ST-ERF pivot check:

```text
T_eff(C3) - T_eff(C2) = 79.3646 - 69.2915 = +10.0731 > +2
```

`T_eff` is computed only from `M[t,t]`; it measures diagonal coverage, not the
length of cross-time influence. New diagnostics also report lower-triangular
past energy and its lag distribution. The legacy pivot check is retained for
auditability but is not sufficient evidence of temporal memory by itself.

## 2. Complexity / FP-Mult Accounting

The code now records `fp_mults_attention_path_per_step` as gate-path FP
multiplications per sample per timestep, summed over attention blocks.  It
excludes shared Q/K/V/proj/MLP work.

For the canonical H=4, D=64, depth=2 pilot configs:

| condition | FP-mults/block/step | FP-mults/model/step | gate params |
|---|---:|---:|---:|
| C0 SDLA | 0 | 0 | 0 |
| C1 lowrank | 24576 | 49152 | 4096 |
| C2 one-minus-k | 16384 | 32768 | 0 |
| C3 MGA | 0 | 0 | 1304 |

The C3 count includes two per-head LayerNorm affine parameter sets and the
per-head write scale. The zero FP-multiply count applies only to a fixed-point
mapping of recurrent state decay, not the gate LIFs or the full model.

Generate the table:

```bash
uv run python -m scripts.pilot.complexity_table --results_dir pilot_results
```

Backfill old JSONs that still have `-1`:

```bash
uv run python -m scripts.pilot.backfill_fp_mults --results_dir pilot_results
uv run python -m scripts.pilot.aggregate_results --results_dir pilot_results
```

## 3. DVS ST-ERF

Run the DVS seed-42 ST-ERF diagnostic to test whether T=16 has a temporal
horizon ceiling.

```bash
TASKS=dvs128 CONDITIONS="c0_sdla c1_lowrank c2_oneminusk c3_mga" \
GPU=0 N_SAMPLES=256 RESULTS_DIR=pilot_results \
bash scripts/pilot/run_st_erf_all.sh
```

Quick smoke version:

```bash
TASKS=dvs128 CONDITIONS="c2_oneminusk c3_mga" \
GPU=0 N_SAMPLES=16 RESULTS_DIR=pilot_results \
bash scripts/pilot/run_st_erf_all.sh
```

Read the summary:

```bash
cat pilot_results/figs/st_erf_summary.json
```

## 4. SHD Temporal-Horizon Sweep

The existing seed-42 screening is incomplete and non-monotonic: C3 is +0.4108,
-1.7667, and +0.6846 pp versus the strongest C1/C2 baseline at T=25, 50, and
100 respectively; T=200 has only C0 completed. The T=100 result is promising,
but no horizon has enough seeds for a confirmatory verdict.

First run seed 42 across T=25/50/100/200:

```bash
GPUS=0,1,2,3 RESULTS_DIR=pilot_results_t_sweep \
TS="25 50 100 200" SEEDS="42" \
bash scripts/pilot/run_temporal_sweep.sh
```

If C3 advantage grows with T, rerun selected horizons with 3 seeds:

```bash
GPUS=0,1,2,3 RESULTS_DIR=pilot_results_t_sweep_3seed \
TS="100 200" SEEDS="42 123 2024" \
bash scripts/pilot/run_temporal_sweep.sh
```

Useful overrides:

```bash
DATA_ROOT=/path/to/SHD
BATCH_SIZE=96
MAX_EPOCHS=100
DRY_RUN=1
```

## 5. C3 Ablations

Run the default ablation pack on SHD T=100:

```bash
GPUS=0,1,2,3 RESULTS_DIR=pilot_results_c3_ablation \
T=100 SEEDS="42" \
bash scripts/pilot/run_c3_ablation.sh
```

The default pack covers:

```text
full
gamma_spike
gamma_rate
gamma_only
beta_only
no_gates
k2
k4
no_write_scale
```

If a variant is promising, rerun it with 3 seeds:

```bash
GPUS=0,1,2 RESULTS_DIR=pilot_results_c3_ablation_3seed \
T=100 SEEDS="42 123 2024" \
ABLATIONS="full gamma_spike k4 no_write_scale" \
bash scripts/pilot/run_c3_ablation.sh
```

## 6. External Long-Horizon Benchmark

Sequential MNIST / Permuted Sequential MNIST uses the new
`spikformer_sequence` model and the same C0-C3 attention modules.

Sequential MNIST:

```bash
GPUS=0,1,2,3 RESULTS_DIR=pilot_results_seqmnist \
SEEDS="42" PERMUTE=0 \
bash scripts/pilot/run_seqmnist_long.sh
```

Permuted Sequential MNIST:

```bash
GPUS=0,1,2,3 RESULTS_DIR=pilot_results_pseqmnist \
SEEDS="42" PERMUTE=1 \
bash scripts/pilot/run_seqmnist_long.sh
```

If the quick run is useful, rerun with 3 seeds:

```bash
GPUS=0,1,2,3 RESULTS_DIR=pilot_results_pseqmnist_3seed \
SEEDS="42 123 2024" PERMUTE=1 \
bash scripts/pilot/run_seqmnist_long.sh
```

## 7. Main Hypothesis For The Follow-Up

The next wave should test temporal scaling rather than treating all tasks as
equally diagnostic:

```text
MGA is a hardware-friendly temporal memory gate.  Its advantage should grow
with temporal horizon and long-range credit assignment pressure.  On short,
saturated tasks it should remain competitive; on longer tasks it should improve
temporal ERF and match or exceed low-rank FP gates with fewer gate parameters
and no recurrence gate-path FP multiplications.
```
