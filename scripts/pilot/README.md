# Gate-1 Pilot — MGA vs SpikingBrain trivial gates

This directory implements the pilot protocol in `docs/PILOT_GATE1.md`:
4 attention conditions × 3 seeds × 2 tasks (DVS128-Gesture + SHD), plus an
ST-ERF diagnostic on seed-42 checkpoints.

## What was added

| File / path                                                       | Purpose                                                      |
|-------------------------------------------------------------------|--------------------------------------------------------------|
| `models/spikformer/gate_attention.py`                             | C0/C1/C2/C3 attention modules + dispatcher                   |
| `models/spikformer/model.py` (`SpikformerAudio`, `AudioConvStem`) | 1D Spikformer backbone for SHD                               |
| `data/event_datasets.py` (SHD branch)                             | SpikingHeidelbergDigits loader                               |
| `data/transforms.py` (SHD branch)                                 | 1D event transform + random time shift                       |
| `data/build.py` (`shd_collate_fn`)                                | `[B,T,1,700] → [T,B,1,700]` collate                          |
| `configs/model_configs/pilot/*.yaml`                              | 4 conditions × 2 tasks = 8 model configs                     |
| `configs/data_configs/{dvs128gesture_pilot,shd_pilot}.yaml`       | Pilot data configs                                           |
| `configs/optimizer_configs/pilot_{dvs,shd}.yaml`                  | AdamW recipes per pilot §3                                   |
| `configs/train_configs/pilot_{dvs,shd}.yaml`                      | Epoch / batch / precision per pilot §3                       |
| `scripts/pilot/sanity_attention.py`                               | Forward/isolation/param-count/FP-mult unit tests             |
| `scripts/pilot/pilot_logger.py`                                   | Per-run JSON logger (pilot §4 metrics)                       |
| `scripts/pilot/train_pilot.py`                                    | Thin training wrapper with seed override + JSON callback     |
| `scripts/pilot/run_all.sh`                                        | Multi-GPU launcher for the 24 runs                           |
| `scripts/pilot/aggregate_results.py`                              | CSV + decision-table aggregator (pilot §6)                   |
| `scripts/pilot/st_erf_diag.py`                                    | ST-ERF heatmap + E_diag / T_eff summary (pilot §5)           |
| `scripts/pilot/run_st_erf_all.sh`                                 | Sequence all eight ST-ERF runs                               |

## Quick smoke test (CPU, no datasets needed)

```bash
uv run python -m scripts.pilot.sanity_attention
```

Expected:
- All four conditions forward without NaN.
- Shared backbone keys identical; only gate keys differ.
- `Δ(C2 - C0) == 0`, `Δ(C3 - C0) == 2 H D + 2 H`,
  `Δ(C1 - C0) ≈ 2 C r` (where `r = 16`).
- FP-mults on the attention path: C0 = 0, C2 = 0, C3 = 0; C1 > 0.

## Full pilot on a single H100

```bash
GPUS=0 RESULTS_DIR=pilot_results bash scripts/pilot/run_all.sh
```

## Full pilot on 4–8× H100

```bash
GPUS=0,1,2,3 RESULTS_DIR=pilot_results bash scripts/pilot/run_all.sh
GPUS=0,1,2,3,4,5,6,7 RESULTS_DIR=pilot_results bash scripts/pilot/run_all.sh
```

The launcher round-robins (task, condition) pairs across GPUs and sequentializes
the 3 seeds within each (gpu, task, condition).  Expected wall-clock with 4×H100
≈ 5–8 h (pilot §7).

## Aggregate and decide

`run_all.sh` calls `aggregate_results.py` at the end.  You can re-run it
manually:

```bash
uv run python -m scripts.pilot.aggregate_results --results_dir pilot_results
```

Output:
- `pilot_results/raw.csv`     — per-run JSON merged
- `pilot_results/summary.md`  — decision table (PASS / PIVOT-eligible / FAIL)

If you are aggregating old JSONs generated before FP-mult accounting was
implemented, backfill them first:

```bash
uv run python -m scripts.pilot.backfill_fp_mults --results_dir pilot_results
uv run python -m scripts.pilot.complexity_table --results_dir pilot_results
uv run python -m scripts.pilot.aggregate_results --results_dir pilot_results
```

## ST-ERF diagnostic (after training)

```bash
GPU=0 RESULTS_DIR=pilot_results bash scripts/pilot/run_st_erf_all.sh
```

Writes 8 heatmaps to `pilot_results/figs/` and a JSON of E_diag/T_eff stats.
Limit to DVS only:

```bash
TASKS=dvs128 GPU=0 RESULTS_DIR=pilot_results bash scripts/pilot/run_st_erf_all.sh
```

## Follow-up sweeps

The follow-up plan and command cookbook are in
`docs/PILOT_FOLLOWUP_PLAN.md`.  Main entrypoints:

```bash
# SHD temporal horizon sweep: T=25/50/100/200, seed 42
GPUS=0,1,2,3 RESULTS_DIR=pilot_results_t_sweep bash scripts/pilot/run_temporal_sweep.sh

# C3 ablations: membrane-vs-spike, gamma/beta, k_bits, write_scale
GPUS=0,1,2,3 RESULTS_DIR=pilot_results_c3_ablation bash scripts/pilot/run_c3_ablation.sh

# External long-horizon benchmark: Sequential MNIST
GPUS=0,1,2,3 RESULTS_DIR=pilot_results_seqmnist bash scripts/pilot/run_seqmnist_long.sh
```

## Datasets

- **DVS128-Gesture**: spikingjelly auto-downloads to `data/dvs128gesture/`
  when first instantiated (already supported by the repo).
- **SHD**: spikingjelly auto-downloads to `data/shd/` on first run.  Frames
  are integrated to T=100 equal-time bins (`split_by=time`) on first use,
  then cached as `.npz` under `data/shd/frames_number_100_split_by_time/`.

## Reproducibility / deviations

- Precision: `bf16-mixed` on both tasks (H100-friendly).
- For C3 (MGA), the K-LIF in the attention block is implemented manually
  (Python loop) to expose the pre-threshold membrane.  Hyperparameters
  match `build_neuron(model_config)` for the other LIFs (same `tau`,
  `v_threshold`, surrogate); the spike output of this manual LIF matches
  the spikingjelly LIF up to reset semantics — we use **soft reset** so
  that the pre-threshold membrane can be recovered for the γ-gate, per
  pilot §1 (which explicitly requires `U_k^{(t)}`, not the spike output).
- All four conditions share an identical Q/V/proj backbone (param-equal
  modulo gate params).  Verify by inspecting `state_dict()` keys via
  `scripts/pilot/sanity_attention.py::test_isolation_state_dict`.
