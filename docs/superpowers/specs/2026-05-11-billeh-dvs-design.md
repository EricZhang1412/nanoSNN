# Billeh-V1 + LGN on DVS datasets — Design

**Date:** 2026-05-11
**Scope:** Train `billeh_v1` (V1 column + LGN front-end + L5e localized
readout) on CIFAR10-DVS and DVS128Gesture, following spikingjelly +
mainstream SNN data-processing conventions.
**Goal:** observe how event-based temporal data behaves under a biologically
calibrated visual-cortex model, end-to-end with the same Chen-Maass rate
regularization used for the static / sequential image pipelines.

## Background

`billeh_v1` already has a working LGN path used by:
- `billeh_v1_mnist_lgn.yaml` — strict-reproduction MNIST trial-timed movie.
- `billeh_v1_seqcifar10_lgn.yaml` — sequential CIFAR-10 (added 2026-05-10);
  folds the row-major pixel scan back into a 32×32 static frame.

DVS data differs in two material ways:

1. **Polarity.** spikingjelly returns `[T_dvs, 2, H, W]` event-count frames,
   with channel 0 = ON, channel 1 = OFF. The current `_to_b_t_n_via_lgn`
   ndim=5 branch averages the two channels (`x.mean(dim=2)`), which is
   physically incoherent because LGN's ON-cell and OFF-cell sub-populations
   internally model the signed luminance gradient.
2. **Time axis.** spikingjelly defaults to `T_dvs ∈ {10, 16, 20}` integrated
   frames per sample; LGN temporal kernels are calibrated for ms-resolution
   sequences (model `T = 1024` in our seq-CIFAR config). Direct feed of
   `T_input = 16` would leave LGN's 30-50 ms temporal kernel unable to act
   and would make the per-neuron rate regularization (mean over 16 frames)
   excessively noisy.

Additionally, DVS128Gesture has 11 classes, which collides with the
existing localized-readout indexing convention (pools 5..14 — only 10 pools
available in that range, current code silently aliases two classes to the
same pool).

## Approach

**Recommended (Approach A — adopted): data-side polarity collapse + model-side
K-replay, with config-driven per-dataset variation.**

Rejected alternatives (briefly):
- **B — all in data:** transform produces `[T_lgn=1024, 1, H, W]` including
  replay. Couples dataset config to model's T; awkward when reusing the same
  data with a different `model.T`.
- **C — all in model:** `_to_b_t_n_via_lgn` takes raw `[T_dvs, B, 2, H, W]`
  and does polarity + normalization + replay. Pushes dataset-specific
  preprocessing inside the model class.

Approach A keeps polarity (dataset property) in the dataset transform layer
and `T_lgn` (model property) in the model layer, matching spikingjelly's
convention of doing polarity collapse in dataset transforms.

### Component diagram

```
spikingjelly DVS dataset             → [T_dvs, 2, H, W]  uint16 event counts
   │  (build_dvs_lgn_transform)
   ▼
signed-diff + per-sample magnitude
normalization                        → [T_dvs, 1, H, W]  float in ~[-1, 1]
   │  (event_collate_fn — unchanged)
   ▼
batched event movie                  → [T_dvs, B, 1, H, W]
   │  (model._to_b_t_n_via_lgn ndim=5, K-replay extension)
   ▼
LGN-ready movie                      → [B, T_lgn=1024, H, W]
   │  (TorchLGN — unchanged)
   ▼
LGN firing rates                     → [B, 1024, 17400]  Hz
   │  × lgn_input_scale
   ▼
V1 column (3000 GLIF) → spikes       → [B, 1024, n_neurons]
   │  L5e localized pool readout (with 11-class fix)
   ▼
logits                               → [B, num_classes]
```

## Components

### 1. `data/transforms.py:build_dvs_lgn_transform(data_config)` (new)

**Purpose.** Polarity collapse + magnitude normalization for spikingjelly DVS
frames.

**Input.** `np.ndarray` or `torch.Tensor` of shape `[T_dvs, 2, H, W]`, integer
or float event counts.

**Output.** `torch.float32` of shape `[T_dvs, 1, H_out, W_out]`, values
approximately in `[-1, 1]`, where `H_out, W_out` come from
`data_config.image_size` (defaults to native sensor resolution).

**Logic.**
```python
tensor = ensure_tensor(frames).float()      # [T_dvs, 2, H_native, W_native]
signed = tensor[:, 0] - tensor[:, 1]        # [T_dvs, H, W]

# Spatial resize to image_size if requested. spikingjelly's CIFAR10DVS /
# DVS128Gesture return native 128x128 — they do NOT auto-resize.
if (H_out, W_out) != (H, W):
    signed = F.interpolate(
        signed.unsqueeze(1),                # [T_dvs, 1, H, W] for interpolate
        size=(H_out, W_out),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)

denom  = signed.abs().amax(dim=(0, 1, 2)).clamp_min(1e-3)
signed = (signed / denom).unsqueeze(1)      # [T_dvs, 1, H_out, W_out]
return signed
```

Per-sample (not per-batch) magnitude normalization to keep all samples in a
similar dynamic range without requiring the data loader to know batch
statistics.

Optional `polarity_mode: signed | mean` data-config knob — defaults to
`signed`, retains existing `mean` behavior for ablation.

### 2. `data/event_datasets.py:build_event_dataset` (modify)

For `cifar10dvs` and `dvs128gesture`, switch transform selection on
`data_config.transform_type` (default `'default'`):

```python
if str(getattr(data_config, "transform_type", "default")) == "dvs_lgn":
    transform = build_dvs_lgn_transform(data_config)
else:
    transform = build_event_transform(data_config)
```

This preserves existing non-LGN model usage of these datasets unchanged.

### 3. `models/billeh_v1/model.py:_to_b_t_n_via_lgn` ndim=5 branch (extend)

**Goal.** Allow `T_input < self.T` via integer K-replay.

**Diff (conceptual):**
```python
if x.ndim == 5:
    x = x.permute(1, 0, 2, 3, 4)                              # [B, T_in, C, H, W]
    movie = x.mean(dim=2) if x.shape[2] > 1 else x[:, :, 0]   # [B, T_in, H, W]
    t_in, t_lgn = movie.shape[1], int(self.T)
    if t_lgn != t_in:
        if t_lgn % t_in != 0:
            raise ValueError(
                f"model.T={t_lgn} must be a multiple of input T={t_in} for K-replay"
            )
        k = t_lgn // t_in
        movie = movie.repeat_interleave(k, dim=1)              # [B, T_lgn, H, W]
```

`repeat_interleave` materializes the K-replayed tensor in one allocation;
memory cost is `O(B × T_lgn × H × W)` (≈ 38 MB at B=4, T=1024, H=W=48,
fp32) — comparable to the existing static-frame path.

The `mean(dim=2)` on a 1-channel input is a no-op aside from squeezing C — so
post-transform single-channel signed-diff input passes through unchanged.

### 4. Localized-readout 11-class fix in `models/billeh_v1/model.py` (modify)

**Current bug** (`model.py:138-144`): for `num_classes=11` the loop tries
pool keys `_5 .. _15`. Pool `_15` is missing → falls back to pool `_10`,
which is *also* used by class index 5 (key `_10`). Two classes alias to one
pool.

**Fix:**
```python
if self.num_classes == 10 and "localized_readout_neuron_ids_5" in network:
    pool_offset = 5     # legacy 10-class image-task convention
elif self.num_classes <= 15:
    pool_offset = 0     # use first num_classes pools
else:
    raise ValueError(f"num_classes={self.num_classes} > 15 not supported by localized readout")
pool_ids = [
    np.asarray(network[f"localized_readout_neuron_ids_{i + pool_offset}"]).reshape(-1)
    for i in range(self.num_classes)
]
```

CIFAR10-DVS (10) keeps the legacy offset=5 path. DVS128Gesture (11) uses
pools 0..10 with no aliasing.

### 5. New configs

**`configs/data_configs/cifar10dvs_lgn.yaml`:**
```yaml
name: cifar10dvs
root: /data2/dataset/cifar10dvs
num_classes: 10
image_size: 48
in_channels: 1            # post signed-diff
is_event: true
T: 16
train_ratio: 0.9
split_seed: 42
event_data_type: frame
frames_number: 16
split_by: number
transform_type: dvs_lgn
polarity_mode: signed
num_workers: 4
pin_memory: true
```

**`configs/data_configs/dvs128gesture_lgn.yaml`:** same shape as above with
`name: dvs128gesture`, `num_classes: 11`, `image_size: 128`,
`root: /data2/dataset/dvs128gesture`. Comment notes: spikingjelly cannot
auto-download DVS128Gesture; user must place the IBM zip under
`<root>/download/` before first run, then spikingjelly extracts and
integrates frames once.

**`configs/model_configs/billeh_v1_cifar10dvs_lgn.yaml`:** based on
`billeh_v1_seqcifar10_lgn.yaml`, with:
- `T: 1024` (16 × 64)
- `in_channels: 1`
- `lgn_input_height: 48, lgn_input_width: 48`
- `lgn_input_scale: 1.0e-3` (initial guess; recalibrate via smoke test)
- `pre_delay: 64, post_delay: 0, response_window_len: 0, down_sample: 64`
- `n_neurons: 3000, neurons_per_output: 30, localized_readout: false`
- `billeh_data_dir: /data2/dataset/LGN_GLIF_Models/new_0505/GLIF_network2`
- `lgn_data_path / lgn_temporal_kernels_path`: same as MNIST-LGN config

**`configs/model_configs/billeh_v1_dvsgesture_lgn.yaml`:** built second,
after CIFAR10-DVS calibration. Identical except `num_classes: 11`,
`lgn_input_height/width: 128`, `chunk_size: 32` (memory budget).

**Optimizer:** reuse existing `billeh_seqcifar10.yaml` (rate_cost=0.1,
voltage_cost=1e-5, huber_kappa=0.002, target_rates_path defaulted to
`<billeh_data_dir>/garrett_firing_rates.pkl`).

## Data flow examples

CIFAR10-DVS sample at training time:
1. spikingjelly returns `[16, 2, 128, 128]` event counts at native sensor
   resolution.
2. `build_dvs_lgn_transform`: `signed = on - off`, bilinear-resize to 48×48,
   per-sample magnitude normalization to `[-1, 1]`, shape `[16, 1, 48, 48]`.
3. `event_collate_fn` stacks → `[B, 16, 1, 48, 48]` then transposes ndim==5
   → `[16, B, 1, 48, 48]`.
4. Model `_to_b_t_n_via_lgn`: permute → `[B, 16, 1, 48, 48]`, squeeze C →
   `[B, 16, 48, 48]`, K-replay K=64 → `[B, 1024, 48, 48]`.
5. LGN → `[B, 1024, 17400]` Hz.
6. × `lgn_input_scale` → V1 column → `[B, 1024, 3000]` spikes → readout →
   `[B, 10]`.

## Validation plan

For each new model+data config combination, before launching full training,
run a smoke test patterned after the `billeh_v1_seqcifar10_lgn` smoke test:

1. **Construction:** `n_input == 17400`, `target_firing_rates.numel() == n_neurons`,
   `_resp_chunk_mask.sum() > 0`.
2. **Forward shape:** logits `[B, num_classes]`, `last_spikes [B, T, n_neurons]`.
3. **Initial population rate:** `5 < spike_rate_hz < 30`. If not, recalibrate
   `lgn_input_scale` (DVS magnitudes can differ from CIFAR by 1-2 orders).
4. **Backward:** loss is finite, sample gradients are non-zero on
   `input_weight_values`, `recurrent_weight_values`, and readout params.

### Calibration order

1. Implement code changes (transform + model patch + readout fix).
2. Create `cifar10dvs_lgn.yaml` and `billeh_v1_cifar10dvs_lgn.yaml`.
3. Run smoke test, tune `lgn_input_scale` until rate ∈ [10, 30] Hz.
4. Launch full CIFAR10-DVS training; verify rate stays in band over the
   first epoch.
5. Once stable, write `dvs128gesture_lgn.yaml` and the matching model config;
   smoke test, then full training.

## Out of scope

- Augmentation (random temporal crop, flips, mixup-on-events). Worth adding
  later but defer until the basic pipeline is validated; the current LGN +
  V1 stack has not been verified against augmented event streams.
- DVS-specific learning rate / schedule tuning. Reuse `billeh_seqcifar10.yaml`
  optimizer initially.
- LGN ON/OFF subregion explicit dual-channel feed (Approach C from earlier
  brainstorm). Could be revisited if signed-diff underperforms.

## Risks

- **`lgn_input_scale` calibration drift.** DVS event magnitudes vary a lot
  by sample; per-sample normalization in the transform should bound this,
  but if rates remain unstable, fall back to a fixed scale + clamp.
- **DVS128Gesture spikingjelly setup friction.** Manual zip placement may
  trip up first-time users; document clearly in config comments.
- **`F.grid_sample` at H=48.** LGN's spatial coordinates are calibrated for
  240×120 retinotopic field. With grid_sample's bilinear interpolation +
  border padding the spatial sampling is well-defined but undersampled. If
  smoke-test rates look pathological, try image_size=64 as a fallback.
