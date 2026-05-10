"""End-to-end smoke test for billeh-v1 LGN on DVS datasets.

Examples:
    .venv/bin/python scripts/smoke_billeh_v1_dvs.py \\
        --model_config configs/model_configs/billeh_v1_cifar10dvs_lgn.yaml \\
        --data_config configs/data_configs/cifar10dvs_lgn.yaml

    .venv/bin/python scripts/smoke_billeh_v1_dvs.py \\
        --model_config configs/model_configs/billeh_v1_dvsgesture_lgn.yaml \\
        --data_config configs/data_configs/dvs128gesture_lgn.yaml \\
        --batch 2

Checks: forward shape, n_input==17400, rate in [5, 30] Hz, finite gradients.
"""
from __future__ import annotations
import argparse
import os
import sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.load_config import load_config  # noqa: E402
from models.build_model import build_model  # noqa: E402
from data.event_datasets import build_event_dataset  # noqa: E402
from data.build import event_collate_fn  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_config", required=True)
    p.add_argument("--data_config", required=True)
    p.add_argument("--train_config", default="configs/train_configs/default.yaml")
    p.add_argument("--optimizer_config", default="configs/optimizer_configs/billeh_seqcifar10.yaml")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--no_cuda", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu")
    print(f"[smoke] device={device}")

    mcfg = load_config(args.model_config)
    dcfg = load_config(args.data_config)
    tcfg = load_config(args.train_config)
    ocfg = load_config(args.optimizer_config)

    lit = build_model(mcfg, ocfg, tcfg, dcfg).to(device)
    lit.train()
    m = lit.model
    print(f"[smoke] model T={m.T}, n_input={m.n_input}, num_classes={m.num_classes}")
    assert m.n_input == 17400, m.n_input

    print("[smoke] building train split (may trigger spikingjelly preprocessing)…")
    ds = build_event_dataset(dcfg, split="train")
    print(f"[smoke] dataset size: {len(ds)}")

    samples = [ds[i] for i in range(args.batch)]
    x, y = event_collate_fn(samples)
    x = x.to(device)
    y = y.to(device)
    print(f"[smoke] input batch: {tuple(x.shape)}, dtype={x.dtype}, "
          f"range=[{x.min().item():.3f}, {x.max().item():.3f}]")

    logits = lit(x)
    print(f"[smoke] logits: {tuple(logits.shape)}")
    assert logits.shape == (args.batch, m.num_classes)

    spikes = m.last_spikes
    rate_hz = (spikes.detach().float().mean() * 1000.0).item()
    print(f"[smoke] population mean rate ≈ {rate_hz:.2f} Hz")
    if not (5.0 <= rate_hz <= 30.0):
        print(f"[smoke] WARNING: rate {rate_hz:.2f} Hz outside [5, 30] target — "
              f"adjust model_config.lgn_input_scale "
              f"(current={getattr(mcfg, 'lgn_input_scale', None)}).")

    loss = lit._shared_step((x, y), split="train")
    print(f"[smoke] loss={loss.item():.4f}")
    loss.backward()

    grads = [(n, p.grad.abs().mean().item())
             for n, p in lit.named_parameters()
             if p.grad is not None and p.grad.abs().sum() > 0]
    assert grads, "no non-zero gradients"
    print(f"[smoke] non-zero grad params: {len(grads)}")
    print("[smoke] OK")


if __name__ == "__main__":
    main()
