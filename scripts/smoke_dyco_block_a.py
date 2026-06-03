"""
CPU smoke test for DyCo-SNN Block A (four-quadrant).

Builds each of {d_only, c_only, dc, ff} variants, runs one forward + backward
pass on the synthetic temporal-order task, asserts shapes and that gradients
flow through the parameters that should be trainable in each mode.

Usage:
  uv run python -m scripts.smoke_dyco_block_a

Expected runtime: under 1 minute on CPU.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F


def _make_data_config(T=8, image_size=32, n_train=64):
    return SimpleNamespace(
        name="synthetic_temporal_order",
        is_event=True,
        T=T, image_size=image_size, in_channels=2,
        num_classes=2,
        n_train=n_train, n_val=16, n_test=16,
        blob_size=4, min_gap=1, split_seed=0,
        shuffle_time=False, reverse_time=False, first_last_only=False,
        num_workers=0, pin_memory=False,
    )


def _make_model_config(mode, T=8):
    base = dict(
        name="dyco_snn", T=T, num_classes=2, image_size=32, in_channels=2,
        n_blocks=2, stem_dim=64, n_l4=48, n_23e=64, n_23i=16, n_l5=48,
        learnable_tau=True, learnable_asc=True, heterogeneous_init=True,
        recurrent_enabled=True, alpha_init=0.5, alpha_learnable=True,
        ff_hidden_mult=2.0, ff_depth=2, mode=mode,
    )
    if mode == "d_only":
        base.update(dict(recurrent_enabled=False, alpha_init=0.0, alpha_learnable=False))
    elif mode == "c_only":
        base.update(dict(learnable_tau=False, learnable_asc=False, heterogeneous_init=False,
                         recurrent_enabled=True, alpha_init=1.0, alpha_learnable=False))
    elif mode == "ff":
        pass
    elif mode == "dc":
        pass
    else:
        raise ValueError(mode)
    return SimpleNamespace(**base)


def _build_loader(data_cfg, batch_size=4):
    from data.synthetic_temporal_order import build_synthetic_temporal_order
    from data.build import build_collate_fn
    from torch.utils.data import DataLoader
    ds = build_synthetic_temporal_order(data_cfg, split="train")
    collate = build_collate_fn(data_cfg)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)


def _trainable_summary(model):
    rows = []
    total, trainable = 0, 0
    for name, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
        rows.append((name, tuple(p.shape), p.requires_grad))
    return rows, total, trainable


def _check_grads(model, key_substrings, msg, must_be_nonzero=True):
    found = []
    for name, p in model.named_parameters():
        if any(s in name for s in key_substrings) and p.grad is not None:
            found.append((name, float(p.grad.abs().mean().item())))
    if must_be_nonzero:
        assert found, f"{msg}: no matching params found"
        assert any(g > 0 for _, g in found), f"{msg}: all matching grads are zero ({found})"
    return found


def _run_mode(mode, batch_size=4, T=8, verbose=False):
    print(f"\n========== mode = {mode} ==========")
    data_cfg = _make_data_config(T=T)
    model_cfg = _make_model_config(mode, T=T)

    from models.build_model import build_model
    train_cfg = SimpleNamespace(batch_size_per_gpu=batch_size,
                                 trainer=SimpleNamespace(max_epochs=1))
    opt_cfg = SimpleNamespace(lr=1e-3, weight_decay=0.0)
    lit = build_model(model_config=model_cfg, optimizer_config=opt_cfg,
                      train_config=train_cfg, data_config=data_cfg)
    model = lit.model

    rows, total, trainable = _trainable_summary(model)
    print(f"  params: total={total:,}  trainable={trainable:,}")
    if verbose:
        for name, shape, req in rows[:25]:
            print(f"    {'T' if req else '.'} {name:50s} {shape}")
        if len(rows) > 25:
            print(f"    ... and {len(rows) - 25} more")

    loader = _build_loader(data_cfg, batch_size=batch_size)
    x, y = next(iter(loader))
    print(f"  input shape: {tuple(x.shape)}  label distribution: {torch.bincount(y, minlength=2).tolist()}")
    assert x.ndim == 5 and x.shape[0] == T, f"expected [T,B,C,H,W] with T={T}, got {x.shape}"

    logits = lit(x)
    assert logits.shape == (batch_size, 2), f"bad logits shape {logits.shape}"
    loss = F.cross_entropy(logits, y)
    loss.backward()
    spike_rate = getattr(model, "latest_spike_rate_hz", None)
    print(f"  loss={float(loss.item()):.4f}  spike_rate_hz={spike_rate}")
    # Sanity: network must actually fire at init, otherwise no learning signal
    # for the recurrent path and α gate. Threshold is conservative.
    assert spike_rate is not None and spike_rate > 0.5, (
        f"[{mode}] network is silent at init (spike_rate_hz={spike_rate}); "
        f"increase proj_scale_init or lower v_th"
    )

    # Mode-specific gradient sanity checks.
    if mode == "d_only":
        _check_grads(model, ["log_tau_m", "log_tau_s", "asc_amps", "log_param_k"],
                     "[d_only] D-channel params should have nonzero grad")
        # Recurrent weight either absent (FF) or frozen at zero.
        for name, p in model.named_parameters():
            if "rec_23.weight" in name:
                assert not p.requires_grad, f"[d_only] rec_23 must be frozen, got requires_grad on {name}"
        # Alpha must be frozen.
        for name, p in model.named_parameters():
            if name == "alpha_logits":
                assert not p.requires_grad
        print("  ✓ d_only: D params learn, rec frozen, alpha frozen at 0")

    elif mode == "c_only":
        _check_grads(model, ["rec_23.weight"], "[c_only] recurrent weights should learn")
        for name, p in model.named_parameters():
            if any(s in name for s in ["log_tau_m", "log_tau_s", "asc_amps", "log_param_k"]):
                assert not p.requires_grad, f"[c_only] D params must be frozen, got requires_grad on {name}"
            if name == "alpha_logits":
                assert not p.requires_grad
        print("  ✓ c_only: rec learns, D params frozen, alpha frozen at 1")

    elif mode == "dc":
        _check_grads(model, ["log_tau_m", "log_tau_s", "asc_amps", "log_param_k"],
                     "[dc] D params should learn")
        _check_grads(model, ["rec_23.weight"], "[dc] rec weights should learn")
        # Alpha learnable.
        alpha_grads = _check_grads(model, ["alpha_logits"], "[dc] alpha_logits should learn")
        print(f"  ✓ dc: D+C+alpha all learning   alpha_logit_grad_mean={alpha_grads}")

    elif mode == "ff":
        # FF: no recurrent, no per-pop learnable D params (we forced learnable_tau=False etc).
        assert getattr(model, "alpha_logits", None) is None
        # Just make sure conv stem + linears + readout learn.
        _check_grads(model, ["readout.weight", "linears.0.weight", "stem"],
                     "[ff] readout/linears/stem should learn")
        print("  ✓ ff: feedforward baseline trains")

    return total, trainable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="all", choices=["all", "d_only", "c_only", "dc", "ff"])
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    # Make the project root importable when run from repo root.
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    modes = ["d_only", "c_only", "dc", "ff"] if args.mode == "all" else [args.mode]
    totals = {}
    for m in modes:
        totals[m] = _run_mode(m, batch_size=args.batch, T=args.T, verbose=args.verbose)

    print("\n========== matched-param summary ==========")
    for m, (total, trainable) in totals.items():
        print(f"  {m:8s}  total={total:>10,}  trainable={trainable:>10,}")
    if {"dc", "ff"}.issubset(set(modes)):
        ratio = totals["ff"][0] / max(totals["dc"][0], 1)
        print(f"\n  ff / dc total-param ratio: {ratio:.3f}  (tune ff_hidden_mult to bring close to 1.0 for matched-param)")

    print("\nALL SMOKE TESTS PASSED.")


if __name__ == "__main__":
    main()
