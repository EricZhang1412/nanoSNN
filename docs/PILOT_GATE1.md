# Gate-1 Pilot Spec (canonical copy)

> Historical preregistration: this file records the original v1 protocol and is
> intentionally not rewritten to match the evolved implementation. The legacy
> runs used the official test split for validation and applied stochastic SHD
> time shift during evaluation, so their metrics are exploratory. Current runs
> must follow `docs/MGA_V2_SPEC.md`.

This file is a verbatim copy of the user's pilot plan, saved into the repo
for archival reference.  See `scripts/pilot/README.md` for the implementation
overview.

---

## 0. Purpose

We are testing whether **Membrane-Gated Attention (MGA)** beats the two
SpikingBrain-style trivial gates (low-rank linear gate; shared `1−k` gate)
on DVS128-Gesture and SHD, at matched capacity.

Decisions are pre-committed; do not re-litigate them.

## 1. Four attention conditions

All four are drop-in replacements of a single linear-attention module:

* **C0 — `c0_sdla`** : memoryless SDLA baseline.
  `S_t = K_t^T V_t` (no recurrence).
* **C1 — `c1_lowrank`** : SpikingBrain-7B low-rank gate (rank `r=16`).
  `g_t = sigmoid(W_down(W_up(mean_N K_t))); S_t = diag(g_t) S_{t-1} + K_t^T V_t`.
  Init `W_down = 0` so `g_t = 0.5` at start.
* **C2 — `c2_oneminusk`** : SpikingBrain-76B shared `1 − k` gate.
  `g_t = 1 − mean_N K_t; S_t = diag(g_t) S_{t-1} + K_t^T V_t`.
  No learnable gate params.
* **C3 — `c3_mga`** : Membrane-Gated Attention (this paper).
  Two STP-inspired gate LIFs:
  * γ LIF input = `mean_N U_k^{(t)}` (pooled K **pre-threshold membrane**, NOT K spikes).
  * β LIF input = scalar `mean(K_t)` (K spike rate per sample × head).
  * `α_eff = 1 − s_γ · 2^{-k_bits}`, `k_bits = 3` (hardware: `S − (S >> 3)`).
  * `S_t = α_eff ⊙ S_{t-1} + s_β · K_t^T V_t`.
  Learnable: `log_tau_γ ∈ ℝ^{H×D}, V_γ_raw ∈ ℝ^{H×D}, log_tau_β ∈ ℝ^H, V_β_raw ∈ ℝ^H`.
  Init `log_tau_raw = log(4.0)`, `V_raw = log(e − 1)` so `softplus(V_raw) = 1.0`.

## 2. Isolation discipline

* Spikformer-2-256 backbone, 4 heads, head_dim=64.
* Same Q/K/V LIF projections; same MLP; same data/aug/optimizer; same precision.
* Verify by diffing `state_dict()` keys across conditions (only attention-module keys should differ).
* Bf16 mixed precision on H100; fp32 accumulation.

## 3. Datasets and recipes

### DVS128-Gesture (primary)
* T = 16, frame-integration: spikingjelly default (`split_by=number`).
* 128×128 input, patchify 16 (Spikformer's SPS does 4×4 = 4× downsampling).
* Loss = CE on time-averaged logits.
* AdamW lr=1e-3, wd=1e-2, cosine, warmup=10, total=200, batch=16/GPU.
* Seeds: 42, 123, 2024.

### SHD (secondary)
* T = 100 bins (10 ms each), 700 input channels → 20 classes.
* 1D conv stem (700 → 70 tokens × 256 dim).
* Aug: random temporal shift ±5 bins.
* AdamW lr=5e-4, wd=1e-2, cosine, warmup=5, total=100, batch=128/GPU.
* Seeds: 42, 123, 2024.

## 4. Per-run metrics (JSON)

See `scripts/pilot/pilot_logger.py`. Headline numbers: `top1_last10_mean`
and `fp_mults_attention_path_per_step`.

## 5. ST-ERF diagnostic (seed 42 only)

256 test samples. Compute `M[t, τ] = ‖∂(‖S_block[t]‖_F²) / ∂X[τ]‖`
via `torch.autograd.grad`. Report:
* `E_diag = sum_t M[t,t]² / sum_{t,τ} M[t,τ]²`
* `T_eff = (sum_t M[t,t])² / sum_t M[t,t]²`

## 6. Go / no-go (per task)

* **PASS** if `A_C3 − max(A_C1, A_C2) ≥ 0.5` and CIs don't overlap.
* **PIVOT-eligible** if `|A_C3 − A_triv| < 0.5` but `T_eff(C3) > T_eff(C2) + 2` and FP-mult savings ≥ 30%.
* **FAIL** otherwise.

Overall: GO if PASS on ≥ 1 task and no FAIL; PIVOT if any PIVOT-eligible
and no FAIL; STOP if any FAIL.

## 7. Implementation notes / deviations

* Pre-threshold K membrane: K LIF in C3 is implemented manually so we can
  read `u_pre[t]` before reset.  Soft reset is used in that path; spike
  output is otherwise equivalent to the spikingjelly LIF.  Q and V keep
  the standard spikingjelly LIF (default hard reset).
* `k_bits = 3` fixed.  C1 `r = 16` fixed.
* Best metric reported as `top1_last10_mean` (avoids epoch-best snooping);
  `top1_best` provided for context only.

## 8. Pitfalls (do NOT)

* DO NOT use K spikes as the γ-gate input in C3 — that collapses C3 into a
  variant of C2 and kills the paper.
* DO NOT retune lr per condition.  Use `lr=1e-3` (DVS) / `5e-4` (SHD) for
  all conditions in the headline number.
* DO NOT report a single seed.
* DO NOT silently change T/N/D across conditions.
