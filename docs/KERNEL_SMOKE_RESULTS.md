# Ascend kernel smoke results

These synthetic measurements verify numerical equivalence and integration on a
single Ascend 910B3 with PyTorch/torch-npu 2.6.0 and CANN 9.0.0. They are local
smoke results, not a paper-quality latency or energy study.

## Fused LIF eval microbenchmark

Shape is `[T, B=8, N=70, C=256]`, fp32, 2 warmups and 3 repeats.

| T | PyTorch ms | Triton ms | speedup | PyTorch peak MB | Triton peak MB | max abs diff |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 13.801 | 0.570 | 24.212x | 58.0 | 55.8 | 0 |
| 200 | 26.671 | 1.084 | 24.608x | 113.3 | 111.1 | 0 |

## MGA streaming-attention microbenchmark

Shape is `[T, B=2, H=4, N=70, D=64]`, 2 warmups and 3 repeats.

| T | materialized ms | streaming ms | speedup | materialized peak MB | streaming peak MB | max abs diff |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 24.018 | 0.836 | 28.737x | 42.0 | 14.9 | 0 |
| 200 | 49.742 | 1.568 | 31.714x | 83.7 | 28.9 | 0 |

## End-to-end C3 inference

SHD C3, batch 2, fused LIF plus Triton streaming attention, 1 warmup and 2
repeats.

| T | reference ms | optimized ms | speedup | max abs diff |
|---:|---:|---:|---:|---:|
| 100 | 406.151 | 187.089 | 2.171x | 0 |
| 200 | 823.958 | 373.572 | 2.206x | 0 |

## End-to-end C3 training smoke

SHD C3, batch 2, fused LIF forward/backward, no warmup and 1 measured step.

| T | reference ms | fused-LIF ms | speedup | reference peak MB | fused peak MB | max grad diff |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 1730.522 | 602.287 | 2.873x | 1580.2 | 1279.8 | 9.16e-8 |
| 200 | 3585.180 | 1226.536 | 2.923x | 3166.7 | 2565.0 | 7.08e-8 |

The streaming-attention patch is inference-only. The training result fuses LIF
nodes but retains the materialized recurrent attention history, so it does not
demonstrate that the T=200 full-batch training OOM is solved. Repeat with more
warmups/runs and record power before using these numbers in a paper.
