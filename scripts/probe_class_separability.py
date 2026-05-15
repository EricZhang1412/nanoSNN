#!/usr/bin/env python
"""
Probe LGN -> V1 -> pool-readout class separability.

For each true class label c (0..C-1), feed N validation samples through the
model and collect the per-pool average firing rate. Output a [C, C] matrix
M[true_class, pool] = avg pool firing rate (post dampening straight-through,
pre per-class affine).

Interpretation:
  - If diag(M) is much larger than off-diag entries -> V1 already routes
    class-specific signal to the correct pool; readout just needs to learn
    the right scale/bias.
  - If M is nearly row-constant (every class produces the same pool
    pattern) -> LGN/V1 output is class-INVARIANT, root cause confirmed.
  - If diag is no larger than off-diag but rows differ across classes ->
    V1 encodes class info but pool indices don't align with task classes,
    fix is "let readout pick its own pools" (option C).

Optional checkpoint: --ckpt path/to/ckpt.ckpt
Without --ckpt the model runs at fresh init so we measure the structural
prior alone.

Usage (DDP-free, single GPU):
    python -m scripts.probe_class_separability \\
        --data_config configs/data_configs/cifar10dvs_lgn.yaml \\
        --train_config configs/train_configs/transformer.yaml \\
        --model_config configs/model_configs/billeh_v1_cifar10dvs_lgn.yaml \\
        --optimizer_config configs/optimizer_configs/billeh_seqcifar10.yaml \\
        --n_per_class 20
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.load_config import load_config  # noqa: E402
from models.build_model import build_model  # noqa: E402
from data import VisionDataModule  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_config", required=True)
    p.add_argument("--train_config", required=True)
    p.add_argument("--model_config", required=True)
    p.add_argument("--optimizer_config", required=True)
    p.add_argument("--ckpt", default=None, help="Optional Lightning ckpt path; omit for fresh init.")
    p.add_argument("--n_per_class", type=int, default=20)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--split", choices=["val", "train"], default="val")
    p.add_argument("--no_cuda", action="store_true")
    args = p.parse_args()

    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True

    data_config = load_config(args.data_config)
    train_config = load_config(args.train_config)
    model_config = load_config(args.model_config)
    optimizer_config = load_config(args.optimizer_config)

    # Build model.
    lit_model = build_model(
        model_config=model_config,
        optimizer_config=optimizer_config,
        train_config=train_config,
        data_config=data_config,
    )

    if args.ckpt is not None:
        print(f"Loading checkpoint: {args.ckpt}")
        state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        missing, unexpected = lit_model.load_state_dict(state["state_dict"], strict=False)
        if missing:
            print(f"  [warn] missing keys: {len(missing)} (first 5: {missing[:5]})")
        if unexpected:
            print(f"  [warn] unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")
    else:
        print("No checkpoint -> probing fresh init.")

    lit_model.eval().to(device)

    # Build dataloader directly (skip Lightning DataModule's DDP-aware machinery).
    dm = VisionDataModule(data_config=data_config, train_config=train_config)
    dm.setup("fit")
    ds = dm.train_dataset if args.split == "train" else dm.val_dataset
    loader = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=2,
        collate_fn=dm.collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    n_classes = int(getattr(model_config, "num_classes", 10))
    readout = lit_model.model.readout
    flat_indices = readout.flat_indices.to(device)
    flat_class = readout.flat_class.to(device)

    # Collect per (true_class, pool) pool firing rate.
    per_class_rates = defaultdict(list)   # true_class -> list of [n_classes] tensors
    spike_rates_hz = defaultdict(list)    # true_class -> list of scalar mean rates
    pool_sizes = torch.zeros(n_classes, dtype=torch.long)
    for c in range(n_classes):
        pool_sizes[c] = int((flat_class == c).sum().item())
    print(f"pool sizes per class: {pool_sizes.tolist()}")

    seen = defaultdict(int)
    done = False
    for x, y in loader:
        if done:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        _ = lit_model(x)
        spikes = lit_model.model.last_spikes   # [B, T, N], 0/1
        B, T, _N = spikes.shape

        gathered = spikes.index_select(2, flat_indices)            # [B, T, P]
        # Equal weighting (1/|pool_c|) to match a clean firing-rate measure,
        # independent of the learnable pool_weights, so the metric isn't
        # contaminated by what the readout already learned.
        per_entry_w = torch.empty_like(gathered[0, 0])
        for c in range(n_classes):
            mask = (flat_class == c)
            per_entry_w[mask] = 1.0 / float(pool_sizes[c].item())
        weighted = gathered * per_entry_w.view(1, 1, -1)

        idx = flat_class.view(1, 1, -1).expand(B, T, -1)
        rates = torch.zeros(B, T, n_classes, device=device, dtype=weighted.dtype)
        rates = rates.scatter_add(2, idx, weighted)
        rates_mean = rates.mean(dim=1)                              # [B, n_classes] avg over time

        # Hz approximation: mean spike over (T, N) at dt=1ms.
        rate_hz = spikes.float().mean(dim=(1, 2)) * 1000.0          # [B]

        for bi in range(B):
            c_true = int(y[bi].item())
            if seen[c_true] >= args.n_per_class:
                continue
            per_class_rates[c_true].append(rates_mean[bi].cpu())
            spike_rates_hz[c_true].append(float(rate_hz[bi].item()))
            seen[c_true] += 1
        done = all(seen[c] >= args.n_per_class for c in range(n_classes))
        print(f"progress: collected per class = {dict(sorted(seen.items()))}")

    # Build [n_classes, n_classes] matrix.
    M = torch.zeros(n_classes, n_classes)
    M_std = torch.zeros(n_classes, n_classes)
    spike_hz = torch.zeros(n_classes)
    spike_hz_std = torch.zeros(n_classes)
    for c in range(n_classes):
        if not per_class_rates[c]:
            print(f"[warn] no samples for class {c}")
            continue
        stk = torch.stack(per_class_rates[c])    # [n_per_class, n_classes]
        M[c] = stk.mean(0)
        M_std[c] = stk.std(0)
        rates_h = torch.tensor(spike_rates_hz[c])
        spike_hz[c] = rates_h.mean()
        spike_hz_std[c] = rates_h.std()

    np.set_printoptions(formatter={"float": lambda v: f"{v:7.4f}"}, linewidth=200)
    print("\n=== Per-class average pool firing rate matrix ===")
    print("Rows: true class. Columns: pool index. Diag-dominant -> good signal.\n")
    print("         " + "  ".join([f"pool_{c:02d}" for c in range(n_classes)]))
    for c in range(n_classes):
        row = "  ".join(f"{M[c, p].item():7.4f}" for p in range(n_classes))
        print(f"cls_{c:02d}:  {row}")

    print("\n=== Separability metrics ===")
    diag = torch.diagonal(M)
    off_diag_mean_per_row = (M.sum(dim=1) - diag) / max(1, n_classes - 1)
    diag_minus_offdiag = diag - off_diag_mean_per_row
    print(f"diag - mean(off-diag) per row:")
    for c in range(n_classes):
        print(f"  cls_{c:02d}: diag={diag[c].item():.4f}, off_mean={off_diag_mean_per_row[c].item():.4f}, gap={diag_minus_offdiag[c].item():+.4f}")
    print(f"\nOverall diag mean = {diag.mean().item():.4f}")
    print(f"Overall off-diag mean = {(M.sum() - diag.sum()).item() / (n_classes * (n_classes - 1)):.4f}")
    print(f"Argmax-row accuracy (if readout were pure max-pool selector): "
          f"{(M.argmax(dim=1) == torch.arange(n_classes)).float().mean().item() * 100:.1f}%")

    # Per-row variance: how much does each true class's pool pattern differ
    # from the global mean over classes?
    row_mean = M.mean(dim=0, keepdim=True)
    row_dev = (M - row_mean).pow(2).mean(dim=1).sqrt()
    print(f"\nPer-class L2 deviation from global mean pool pattern:")
    for c in range(n_classes):
        print(f"  cls_{c:02d}: ||row - global_mean|| = {row_dev[c].item():.4e}")
    print(f"Mean row deviation = {row_dev.mean().item():.4e}")
    print(f"Within-class noise (mean std over samples) = {M_std.mean().item():.4e}")
    print(f"Signal-to-noise (row_dev / within_class_std) = "
          f"{(row_dev.mean() / M_std.mean().clamp_min(1e-12)).item():.3f}")

    print("\n=== Population firing rate per class (Hz) ===")
    for c in range(n_classes):
        print(f"  cls_{c:02d}: {spike_hz[c].item():6.2f} Hz  (std {spike_hz_std[c].item():.2f})")

    # -------- Per-neuron discriminability and linear separability --------
    print("\n=== Per-neuron analysis (all V1 neurons, not just pools) ===")
    print("Re-collecting per-neuron firing rates...")
    seen2 = defaultdict(int)
    all_rates = []   # list of [n_neurons]
    all_labels = []
    loader2 = DataLoader(
        ds, batch_size=args.batch, shuffle=True, num_workers=2,
        collate_fn=dm.collate_fn, pin_memory=(device.type == "cuda"),
    )
    n_neurons = None
    done = False
    for x, y in loader2:
        if done:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        _ = lit_model(x)
        spikes = lit_model.model.last_spikes      # [B, T, N]
        rate_per_neuron = spikes.float().mean(dim=1)   # [B, N]
        B_actual, n_neurons = rate_per_neuron.shape
        for bi in range(B_actual):
            c_true = int(y[bi].item())
            if seen2[c_true] >= args.n_per_class:
                continue
            all_rates.append(rate_per_neuron[bi].cpu())
            all_labels.append(c_true)
            seen2[c_true] += 1
        done = all(seen2[c] >= args.n_per_class for c in range(n_classes))

    X = torch.stack(all_rates).numpy()              # [N_samp, n_neurons]
    y_arr = np.asarray(all_labels)
    print(f"collected X shape = {X.shape}, n_classes = {n_classes}")

    # Per-neuron discriminability: per-class mean firing rate, then std across classes.
    per_class_rate_n = np.zeros((n_classes, n_neurons), dtype=np.float32)
    for c in range(n_classes):
        per_class_rate_n[c] = X[y_arr == c].mean(0)
    discrim = per_class_rate_n.std(axis=0)          # [n_neurons]
    discrim_norm = discrim / (per_class_rate_n.mean(0) + 1e-9)   # relative

    topk = 20
    top_idx = np.argsort(-discrim)[:topk]
    print(f"\nTop-{topk} most class-discriminative V1 neurons (by per-class std):")
    for k, n_idx in enumerate(top_idx):
        rates_c = per_class_rate_n[:, n_idx]
        ratio = rates_c.max() / max(rates_c.min(), 1e-9)
        print(f"  rank {k+1}: neuron #{int(n_idx)}, "
              f"per-class rate min={rates_c.min():.4f}, max={rates_c.max():.4f}, "
              f"max/min ratio={ratio:.2f}x, std={discrim[n_idx]:.4f}")

    print(f"\nDiscriminability stats over all {n_neurons} neurons:")
    print(f"  std distribution: min={discrim.min():.4e}, median={np.median(discrim):.4e}, "
          f"max={discrim.max():.4e}, mean={discrim.mean():.4e}")

    # Linear separability test: train a closed-form linear discriminant on
    # the per-neuron rates and report train accuracy. This is the upper bound
    # of what an option-C (fully-learnable) readout could achieve on these
    # raw V1 spike rates without any further training of V1 itself.
    print("\n=== Linear-separability ceiling (LDA on raw V1 rates) ===")
    try:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split

        Xtr, Xte, ytr, yte = train_test_split(X, y_arr, test_size=0.3, random_state=0, stratify=y_arr)

        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        lda.fit(Xtr, ytr)
        print(f"  LDA (shrinkage='auto') train acc = {lda.score(Xtr, ytr) * 100:.1f}%, "
              f"test acc = {lda.score(Xte, yte) * 100:.1f}%")

        # Sweep regularization strength so we see the gap between memorization
        # and generalization. Smaller C = stronger L2 regularization.
        for C in (1e-3, 1e-2, 1e-1, 1.0, 1e1):
            lr = LogisticRegression(max_iter=5000, C=C, penalty="l2")
            lr.fit(Xtr, ytr)
            tr_acc = lr.score(Xtr, ytr) * 100
            te_acc = lr.score(Xte, yte) * 100
            print(f"  LogReg C={C:>5.0e}: train={tr_acc:5.1f}%, test={te_acc:5.1f}%, gap={tr_acc - te_acc:+5.1f}%")
    except ImportError:
        print("  scikit-learn not installed -> skipping LDA/LogReg ceiling")
        print("  install with: .venv/bin/pip install scikit-learn")


if __name__ == "__main__":
    main()
