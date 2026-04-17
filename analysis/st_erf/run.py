"""
ST-ERF Analysis for nanoSNN models.

Usage:
    # 1. Random-init analysis (no checkpoint needed)
    python -m analysis.st_erf.run \
        --model_config configs/model_configs/spikformer_tiny.yaml \
        --mode random_init

    # 2. Trained model analysis (from checkpoint)
    python -m analysis.st_erf.run \
        --model_config configs/model_configs/spikformer_tiny.yaml \
        --ckpt path/to/last.ckpt \
        --mode trained

    # 3. Compare baseline vs SD-BGLA
    python -m analysis.st_erf.run \
        --model_config configs/model_configs/spikformer_tiny.yaml \
        --ckpt_baseline path/to/baseline/last.ckpt \
        --ckpt_gated path/to/sdbgla/last.ckpt \
        --mode compare

    # 4. Compare two configs (random init, no training)
    python -m analysis.st_erf.run \
        --model_config configs/model_configs/spikformer_tiny.yaml \
        --model_config_gated configs/model_configs/mem_gated_attn_tiny.yaml \
        --mode compare_random
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

from spikingjelly.activation_based import functional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.load_config import load_config
from models.common.registry import get_model_cls
from models.build_model import init_weights

# trigger model registration
import models.spikformer  # noqa: F401
try:
    import models.mem_gated_attention  # noqa: F401
except ImportError:
    pass

from analysis.st_erf.st_erf import (
    compute_st_erf_attention,
    compute_st_erf_model,
    compute_st_erf_all_layers,
    compute_kv_erf,
    compute_kv_erf_all_layers,
    print_erf_stats,
    compute_erf_stats,
)
from analysis.st_erf.visualize import (
    plot_fig2_compact,
    plot_single_heatmap,
    plot_per_layer_grid,
)

def _reinit_for_analysis(model: nn.Module):
    """Re-initialize Linear layers with Kaiming init for ST-ERF analysis.
 
    Some models (e.g. SDBGLAFormer) use very small weight init (std=0.02)
    which causes near-zero firing rates in deep LIF chains, making the
    bilinear gradient dS/dX = d(K^T V)/dX ≈ 0 (because K≈0 and V≈0).
 
    For random-init structural analysis, we want all models to have
    comparable firing rates so the ST-ERF comparison is fair.
    This does NOT affect trained-checkpoint analysis.
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

def build_model_from_config(config_path: str, device: str = "cuda"):
    """Build a model from a YAML config file."""
    cfg = load_config(config_path)
    name = str(cfg.name).lower()
    model_cls = get_model_cls(name)
    model = model_cls(cfg)
    init_weights(model)
    # Re-init Linear layers with Kaiming for fair comparison in random-init mode
    _reinit_for_analysis(model)
    model = model.to(device)
    # Use train() mode: spikingjelly needs it to build the surrogate
    # gradient computation graph. BN uses batch stats, which is fine
    # for ST-ERF analysis on random inputs.
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()
    return model, cfg


def load_trained_model(config_path: str, ckpt_path: str, device: str = "cuda"):
    """Load a trained model from a Lightning checkpoint."""
    model, cfg = build_model_from_config(config_path, device="cpu")

    # Lightning checkpoint: state_dict is under "state_dict" key
    # with "model." prefix
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    # Strip "model." prefix if present (Lightning wraps in LitVisionSNN)
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            cleaned[k[len("model."):]] = v
        else:
            cleaned[k] = v

    model.load_state_dict(cleaned, strict=False)
    model = model.to(device)
    model.train()  # train mode for surrogate gradient graph
    print(f"  Loaded checkpoint: {ckpt_path}")
    return model, cfg


def get_model_dims(model, cfg) -> dict:
    """Extract T, N, C from model/config."""
    T = int(getattr(cfg, "T", getattr(model, "T", 4)))
    C = int(getattr(cfg, "embed_dim", 256))
    img_size = int(getattr(cfg, "image_size", 32))
    # N = (img_size // 4)^2 for SPS with 4x downsampling
    N = (img_size // 4) ** 2
    return {"T": T, "N": N, "C": C, "img_size": img_size}


def run_single(args):
    """Analyze a single model."""
    device = args.device

    if args.ckpt:
        model, cfg = load_trained_model(args.model_config, args.ckpt, device)
        tag = "trained"
    else:
        model, cfg = build_model_from_config(args.model_config, device)
        tag = "random_init"

    dims = get_model_dims(model, cfg)
    T, N, C = dims["T"], dims["N"], dims["C"]

    print(f"\nModel: {model.__class__.__name__} ({tag})")
    print(f"  T={T}, N={N}, C={C}, blocks={len(model.blocks)}")
    print(f"  Attention type: {model.blocks[0].attn.__class__.__name__}")
    print(f"  Target: {args.target}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Select compute function based on target
    if args.target == "kv":
        compute_fn = compute_kv_erf_all_layers
        prefix = "kv_erf"
        title_prefix = "KV-ERF"
    else:
        compute_fn = compute_st_erf_all_layers
        prefix = "st_erf"
        title_prefix = "ST-ERF"

    if args.level == "attention":
        erf_dict = compute_fn(
            model, T=T, B=args.batch_size, N=N, C=C,
            num_samples=args.num_samples, device=device, seed=args.seed,
        )
        for i, erf in erf_dict.items():
            np.save(os.path.join(args.output_dir, f"{prefix}_{tag}_block{i}.npy"), erf)
            print_erf_stats(erf, label=f"Block {i}")

        plot_per_layer_grid(
            erf_dict,
            save_path=os.path.join(args.output_dir, f"{prefix}_{tag}_per_layer.pdf"),
            title=f"Per-layer {title_prefix} ({tag})",
        )

        last_erf = erf_dict[max(erf_dict.keys())]
        beta = float(getattr(cfg, "tau", 2.0))
        beta_val = 1.0 - 1.0 / beta if beta > 0 else 0.5
        plot_single_heatmap(
            last_erf, title=f"{title_prefix} — Last Block ({tag})",
            save_path=os.path.join(args.output_dir, f"{prefix}_{tag}_last_block.pdf"),
            beta=beta_val,
        )

    elif args.level == "model":
        erf = compute_st_erf_model(
            model, T=T, B=args.batch_size,
            img_size=dims["img_size"],
            in_channels=int(getattr(cfg, "in_channels", 3)),
            num_samples=args.num_samples, device=device, seed=args.seed,
        )
        np.save(os.path.join(args.output_dir, f"{prefix}_{tag}_model.npy"), erf)
        print_erf_stats(erf, label="Full model")
        plot_single_heatmap(
            erf, title=f"{title_prefix} — Full Model ({tag})",
            save_path=os.path.join(args.output_dir, f"{prefix}_{tag}_model.pdf"),
        )


def run_compare(args):
    """Compare baseline vs gated model."""
    device = args.device
    os.makedirs(args.output_dir, exist_ok=True)

    # Select compute function
    if args.target == "kv":
        compute_fn = compute_kv_erf
        prefix = "kv_erf"
        title_prefix = "KV-ERF"
    else:
        compute_fn = compute_st_erf_attention
        prefix = "st_erf"
        title_prefix = "ST-ERF"

    # --- Baseline ---
    if args.ckpt_baseline:
        model_b, cfg_b = load_trained_model(args.model_config, args.ckpt_baseline, device)
        tag = "trained"
    else:
        model_b, cfg_b = build_model_from_config(args.model_config, device)
        tag = "random_init"
    dims_b = get_model_dims(model_b, cfg_b)

    # --- Gated ---
    cfg_gated = args.model_config_gated or args.model_config
    if args.ckpt_gated:
        model_g, cfg_g = load_trained_model(cfg_gated, args.ckpt_gated, device)
    else:
        model_g, cfg_g = build_model_from_config(cfg_gated, device)
    dims_g = get_model_dims(model_g, cfg_g)

    # Use the target block
    block_idx = args.block_idx if args.block_idx >= 0 else len(model_b.blocks) - 1

    print(f"\nTarget: {args.target} | Block: {block_idx}")

    print(f"\n--- Baseline: {model_b.blocks[block_idx].attn.__class__.__name__} ---")
    erf_b = compute_fn(
        model_b.blocks[block_idx].attn,
        T=dims_b["T"], B=args.batch_size, N=dims_b["N"], C=dims_b["C"],
        num_samples=args.num_samples, device=device, seed=args.seed,
        block_idx=block_idx,
    )
    np.save(os.path.join(args.output_dir, f"{prefix}_baseline_block{block_idx}.npy"), erf_b)
    print_erf_stats(erf_b, label="Baseline")

    print(f"\n--- Gated: {model_g.blocks[block_idx].attn.__class__.__name__} ---")
    erf_g = compute_fn(
        model_g.blocks[block_idx].attn,
        T=dims_g["T"], B=args.batch_size, N=dims_g["N"], C=dims_g["C"],
        num_samples=args.num_samples, device=device, seed=args.seed,
        block_idx=block_idx,
    )
    np.save(os.path.join(args.output_dir, f"{prefix}_gated_block{block_idx}.npy"), erf_g)
    print_erf_stats(erf_g, label="SD-BGLA")

    # Beta for decay reference line
    tau_val = float(getattr(cfg_b, "tau", 2.0))
    beta_val = 1.0 - 1.0 / tau_val if tau_val > 0 else 0.5

    # Plot Fig 2
    plot_fig2_compact(
        erf_b, erf_g, beta=beta_val,
        save_path=os.path.join(args.output_dir, f"fig2_{prefix}_block{block_idx}.pdf"),
        base_label=f"Baseline {title_prefix}",
        gated_label=f"SD-BGLA {title_prefix}",
    )
    plot_fig2_compact(
        erf_b, erf_g, beta=beta_val,
        save_path=os.path.join(args.output_dir, f"fig2_{prefix}_block{block_idx}.png"),
        base_label=f"Baseline {title_prefix}",
        gated_label=f"SD-BGLA {title_prefix}",
    )

    # Print comparison table
    sb = compute_erf_stats(erf_b)
    sg = compute_erf_stats(erf_g)
    print(f"\n{'=' * 55}")
    print(f"  Comparison: Block {block_idx}")
    print(f"{'=' * 55}")
    print(f"  {'Metric':<22} {'Baseline':>10} {'SD-BGLA':>10}")
    print(f"  {'-' * 44}")
    print(f"  {'Diag energy (%)':<22} {sb['diag_energy']:>10.1f} {sg['diag_energy']:>10.1f}")
    print(f"  {'Diag/off ratio':<22} {sb['diag_off_ratio']:>10.1f} {sg['diag_off_ratio']:>10.1f}")
    print(f"  {'T_eff':<22} {sb['T_eff_mean']:>10.2f} {sg['T_eff_mean']:>10.2f}")
    print(f"  {'1-step decay':<22} {sb['one_step_decay']:>10.4f} {sg['one_step_decay']:>10.4f}")
    print(f"{'=' * 55}")


def main():
    parser = argparse.ArgumentParser(description="ST-ERF analysis for nanoSNN")

    parser.add_argument("--model_config", type=str, required=True,
                        help="Path to model config YAML (baseline)")
    parser.add_argument("--model_config_gated", type=str, default=None,
                        help="Path to gated model config YAML (for compare mode)")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Checkpoint path (single model mode)")
    parser.add_argument("--ckpt_baseline", type=str, default=None,
                        help="Baseline checkpoint (compare mode)")
    parser.add_argument("--ckpt_gated", type=str, default=None,
                        help="Gated model checkpoint (compare mode)")

    parser.add_argument("--mode", type=str, default="single",
                        choices=["single", "compare"],
                        help="Analysis mode")
    parser.add_argument("--level", type=str, default="attention",
                        choices=["attention", "model"],
                        help="Compute ST-ERF at attention or full model level")
    parser.add_argument("--target", type=str, default="full",
                        choices=["full", "kv"],
                        help="What to measure: 'full' = entire attention output, "
                             "'kv' = KV state S[t]=K^T V only (isolates attention memory)")
    parser.add_argument("--block_idx", type=int, default=-1,
                        help="Block index for compare mode (-1 = last)")

    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of random inputs to average")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size per sample")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="analysis_outputs")

    args = parser.parse_args()

    if args.mode == "single":
        run_single(args)
    elif args.mode == "compare":
        run_compare(args)

    print("\nDone!")


if __name__ == "__main__":
    main()