# Membrane-Gated Attention v2

This document freezes the implementation used for the next confirmatory
experiments. `docs/PILOT_GATE1.md` remains the historical v1 preregistration and
must not be read as an exact description of the current module.

## State update

For each attention head, MGA-v2 computes

```text
x_gamma_t = LayerNorm(mean_tokens(K_membrane_pre_threshold_t))
s_gamma_t = LIF_gamma(x_gamma_t)
s_beta_t  = LIF_beta(mean_tokens_and_channels(K_spike_t))
w_h       = exp(log_write_scale_h)

S_t = (1 - s_gamma_t * 2^(-k_bits)) * S_(t-1)
      + w_h * s_beta_t * K_t^T V_t
```

The K projection uses a manually unrolled LIF only to expose its pre-threshold
membrane. Its spike and hard-reset dynamics match the K-LIF used by C0-C2.

## Frozen defaults

The canonical C3 configs explicitly record these values:

| setting | value |
|---|---:|
| `mga_gamma_input` | `membrane` |
| `mga_use_gamma` / `mga_use_beta` | `true` / `true` |
| `mga_init_tau` | `4.0` |
| `mga_init_V_gamma` | `0.0` (unconstrained) |
| `mga_init_V_beta` | `0.05` (softplus parameterization) |
| `mga_gate_input_norm` | `true` |
| `mga_k_bits` | `3` |
| `mga_use_write_scale` | `true` |
| `mga_init_write_scale` | `0.125` |

For `H=4`, `D=64`, and depth 2, C3 has 1,304 condition-specific parameters:
1,024 gate-LIF vector parameters, 16 scalar gate-LIF parameters, 256 LayerNorm
parameters, and 8 write-scale parameters.

## Hardware accounting boundary

The zero-multiply claim is limited to a fixed-point deployment mapping of the
recurrent state update. Binary masks require no general multiply, and
`2^(-k_bits)` decay can be implemented as shift and subtract. The current
PyTorch and Triton reference paths use floating-point tensors and multiplications;
the gate LIF dynamics and shared Q/K/V/projection/MLP work are also excluded from
the state-update count. Latency, memory, and energy must be reported separately.

## Confirmatory protocol

- Split validation data deterministically from the official training set.
- Keep the official test split untouched until one final evaluation of the best
  validation checkpoint.
- Apply random temporal shift to training samples only.
- Use seeds 42, 123, and 2024 with identical effective batch size, update count,
  optimizer, precision, and temporal horizon across C0-C3.
- Report a decision only when all compared conditions have at least three runs.
- Treat legacy diagonal `T_eff` as coverage rather than memory length; report
  off-diagonal past-energy and lag summaries for temporal propagation.
- Treat results from the pre-cleanup data pipeline as exploratory rather than
  confirmatory.
