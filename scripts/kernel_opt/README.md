# Kernel Optimization Utilities

These scripts profile and benchmark Spikformer large-T inference on Ascend NPU without changing the model topology or state dict.

One numerical/performance smoke run is recorded in
[`docs/KERNEL_SMOKE_RESULTS.md`](../../docs/KERNEL_SMOKE_RESULTS.md).

## 1. Profile top ops

```bash
.venv/bin/python scripts/kernel_opt/profile_spikformer_npu.py \
  --model_config configs/model_configs/pilot/spikformer_shd_c0_sdla.yaml \
  --T 100 --batch_size 4 --mode eval --steps 1 --warmup 1 --row_limit 25 \
  --npu_trace --trace_dir kernel_profiles/shd_T100_c0
```

The script prints a PyTorch profiler table and, with `--npu_trace`, exports a torch_npu/CANN trace and aggregates `ASCEND_PROFILER_OUTPUT/operator_details.csv` by operator name.

## 2. Benchmark fused LIF kernel

```bash
.venv/bin/python scripts/kernel_opt/benchmark_lif_kernel.py \
  --T 100 200 --batch_size 8 --tokens 70 --channels 256 --repeats 3
```

## 3. Benchmark streaming attention kernel

```bash
.venv/bin/python scripts/kernel_opt/benchmark_attention_kernel.py \
  --mode diag --T 100 200 --batch_size 2 --heads 4 --tokens 70 --head_dim 64
```

Use `--mode diag` for C1/C2-style diagonal gates and `--mode mga` for C3-style gamma/beta gates.

## 4. Benchmark end-to-end inference

```bash
.venv/bin/python scripts/kernel_opt/benchmark_spikformer_inference.py \
  --model_config configs/model_configs/pilot/spikformer_shd_c0_sdla.yaml \
  --T 100 200 --batch_size 4 --repeats 2
```

Add `--streaming_attention` to patch C0/C1/C2/C3 attention modules with a forward-only streaming path that avoids materializing the full `S_seq` recurrence state during eval:

```bash
.venv/bin/python scripts/kernel_opt/benchmark_spikformer_inference.py \
  --model_config configs/model_configs/pilot/spikformer_shd_c2_oneminusk.yaml \
  --T 100 200 --batch_size 4 --repeats 2 --streaming_attention
```

Add `--triton_attention` to use the Triton-Ascend streaming attention kernel instead of the PyTorch streaming fallback:

```bash
.venv/bin/python scripts/kernel_opt/benchmark_spikformer_inference.py \
  --model_config configs/model_configs/pilot/spikformer_shd_c2_oneminusk.yaml \
  --T 100 200 --batch_size 4 --repeats 2 --streaming_attention --triton_attention
```

## Notes

- The fused Triton LIF path supports eval forward and optional training forward/backward when `NANOSNN_TRITON_LIF_TRAIN=1` or benchmark `--mode train` is used.
- The streaming attention patch is forward-only and intended for eval/inference memory reduction.
- Both patches monkey-patch existing modules at runtime; they do not alter checkpoint keys.

## 5. Benchmark training-step fused LIF forward/backward

```bash
.venv/bin/python scripts/kernel_opt/benchmark_spikformer_training.py \
  --model_config configs/model_configs/pilot/spikformer_shd_c0_sdla.yaml \
  --T 100 200 --batch_size 2 --repeats 2
```

This patches SpikingJelly LIF nodes with the Triton LIF forward and ATan-surrogate backward when `detach_reset=True`; unsupported LIF variants fall back.

## Use in normal `train.py`

The patches are opt-in by environment variable and keep checkpoint keys unchanged:

```bash
NANOSNN_TRITON_LIF=1 NANOSNN_TRITON_LIF_TRAIN=1 \
.venv/bin/python train.py ...
```

For eval/inference-only attention memory testing:

```bash
NANOSNN_TRITON_LIF=1 NANOSNN_STREAMING_ATTN=1 NANOSNN_TRITON_ATTN=1 \
.venv/bin/python train.py ...
```

`NANOSNN_STREAMING_ATTN` falls back to the original attention forward during training; it is an eval/inference memory path.
