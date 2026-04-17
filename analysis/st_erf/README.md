# ST-ERF Analysis for nanoSNN

Computes and visualizes the **Spatio-Temporal Effective Receptive Field (ST-ERF)** to diagnose temporal memory in spiking attention mechanisms.

## Setup

Place this `analysis/` folder at the root of the nanoSNN project:
```
nanoSNN/
├── analysis/          ← this folder
│   ├── __init__.py
│   ├── st_erf.py      # Core computation (PyTorch autograd)
│   ├── visualize.py   # Publication-quality plots
│   └── run_st_erf.py  # CLI entry point
├── models/
├── configs/
├── train.py
└── ...
```

## Quick Start

### 1. Random-init analysis (no training needed)

Diagnose a single model's attention structure:
```bash
python -m analysis.run_st_erf \
    --model_config configs/model_configs/spikformer_tiny.yaml \
    --mode single \
    --level attention \
    --num_samples 10 \
    --output_dir analysis_outputs/baseline_random
```

### 2. Trained model analysis

```bash
python -m analysis.run_st_erf \
    --model_config configs/model_configs/spikformer_tiny.yaml \
    --ckpt exp/outputs/checkpoints/spikformer/last.ckpt \
    --mode single \
    --level attention \
    --num_samples 20 \
    --output_dir analysis_outputs/baseline_trained
```

### 3. Compare baseline vs SD-BGLA (Fig 2)

With trained checkpoints (recommended for paper):
```bash
python -m analysis.run_st_erf \
    --model_config configs/model_configs/spikformer_tiny.yaml \
    --model_config_gated configs/model_configs/mem_gated_attn_tiny.yaml \
    --ckpt_baseline exp/outputs/checkpoints/baseline/last.ckpt \
    --ckpt_gated exp/outputs/checkpoints/sdbgla/last.ckpt \
    --mode compare \
    --block_idx -1 \
    --num_samples 20 \
    --output_dir analysis_outputs/fig2
```

Without training (random init, for structural analysis):
```bash
python -m analysis.run_st_erf \
    --model_config configs/model_configs/spikformer_tiny.yaml \
    --model_config_gated configs/model_configs/mem_gated_attn_tiny.yaml \
    --mode compare \
    --num_samples 10 \
    --output_dir analysis_outputs/fig2_random
```

### 4. Full model level ST-ERF

```bash
python -m analysis.run_st_erf \
    --model_config configs/model_configs/spikformer_tiny.yaml \
    --ckpt exp/outputs/checkpoints/baseline/last.ckpt \
    --mode single \
    --level model \
    --num_samples 10 \
    --output_dir analysis_outputs/model_level
```

## Outputs

```
analysis_outputs/
├── st_erf_*.npy                  # Raw ST-ERF matrices
├── st_erf_*_per_layer.pdf        # Per-layer heatmap grid
├── st_erf_*_last_block.pdf       # Detailed 3-panel plot
└── fig2_st_erf_block*.pdf/png    # Paper Fig 2 comparison
```

## API Usage

```python
import torch
from spikingjelly.activation_based import functional
from analysis.st_erf import compute_st_erf_attention, print_erf_stats
from analysis.visualize import plot_fig2_compact

# Load your model
model = ...
model.eval()

# Compute ST-ERF for a specific attention block
erf = compute_st_erf_attention(
    model.blocks[0].attn,
    T=4, B=2, N=64, C=256,
    num_samples=10, device="cuda",
)
print_erf_stats(erf, label="Block 0")

# Compare two models
erf_baseline = ...
erf_gated = ...
plot_fig2_compact(erf_baseline, erf_gated, save_path="fig2.pdf")
```

## Key Metrics

| Metric | Formula | Memoryless | Memorable |
|--------|---------|------------|-----------|
| Diagonal energy | Σ ERF(t,t)² / Σ ERF(t,τ)² | ~99% | ~70% |
| T_eff | Σ_τ ERF(t,τ) / max_τ ERF(t,τ) | ~1.0 | >2.0 |
| 1-step decay | ERF(t,t-1) / ERF(t,t) | ~β² | >β |

## Notes

- **Surrogate gradient**: ST-ERF uses the same surrogate gradient as training (via spikingjelly's autograd). This measures the gradient landscape the optimizer actually sees.
- **Batch norm**: Models are set to `eval()` mode, so BN uses running statistics.
- **Neuron reset**: `functional.reset_net()` is called before each forward pass to clear membrane potential history.
- **GPU memory**: For large models (embed_dim=512+), reduce `batch_size` to 1 and `num_samples` to 5.