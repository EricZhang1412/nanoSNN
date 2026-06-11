"""
Diagnose where the d_only network goes silent.

Tracks stem -> L4 input drive -> L4 spikes -> L2/3 drive -> L2/3 spikes ->
L5 drive -> L5 spikes, per timestep, per block.

Usage:
  uv run python -m scripts.diag_dyco_silent
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch


def main():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    from data.synthetic_temporal_order import build_synthetic_temporal_order
    from data.build import build_collate_fn
    from torch.utils.data import DataLoader
    from models.build_model import build_model

    data_cfg = SimpleNamespace(
        name="synthetic_temporal_order", is_event=True, T=8, image_size=32,
        in_channels=2, num_classes=2, n_train=64, n_val=16, n_test=16,
        blob_size=4, min_gap=1, split_seed=0,
        shuffle_time=False, reverse_time=False, first_last_only=False,
        num_workers=0, pin_memory=False,
    )
    model_cfg = SimpleNamespace(
        name="dyco_snn", T=8, num_classes=2, image_size=32, in_channels=2,
        n_blocks=2, stem_dim=64, n_l4=48, n_23e=64, n_23i=16, n_l5=48,
        learnable_tau=True, learnable_asc=True, heterogeneous_init=True,
        recurrent_enabled=False, alpha_init=0.0, alpha_learnable=False,
        ff_hidden_mult=2.0, ff_depth=2, mode="d_only",
    )
    train_cfg = SimpleNamespace(batch_size_per_gpu=4, trainer=SimpleNamespace(max_epochs=1))
    opt_cfg = SimpleNamespace(lr=1e-3, weight_decay=0.0)

    lit = build_model(model_config=model_cfg, optimizer_config=opt_cfg,
                      train_config=train_cfg, data_config=data_cfg)
    model = lit.model

    ds = build_synthetic_temporal_order(data_cfg, split="train")
    loader = DataLoader(ds, batch_size=4, shuffle=False,
                        collate_fn=build_collate_fn(data_cfg))
    x, y = next(iter(loader))
    print(f"\ninput: shape={tuple(x.shape)}  nonzero_frac={(x != 0).float().mean().item():.4f}  "
          f"max={x.max().item():.3f}")
    print(f"label distribution: {torch.bincount(y, minlength=2).tolist()}")

    model.eval()
    with torch.no_grad():
        T, B, C, H, W = x.shape
        # 1) Stem stats.
        stem_out = model.stem(x.reshape(T * B, C, H, W))
        print(f"\n[stem] out shape={tuple(stem_out.shape)}  mean={stem_out.mean():.4f}  "
              f"std={stem_out.std():.4f}  max={stem_out.max():.4f}  "
              f"nonzero_frac={(stem_out.abs() > 1e-6).float().mean():.4f}")
        feat = stem_out.reshape(T, B, -1)

        # 2) Walk through block 0 manually.
        blk = model.blocks[0]
        blk.set_alpha(torch.tensor(1e-4))
        state = blk.zero_state(B, x.device)

        l5_spikes_all = []
        for t in range(T):
            l4_drive = blk.proj_in_to_l4(feat[t])
            z_l4, s_l4 = blk.pop_l4(l4_drive, state.s_l4)

            l23_ff = blk.proj_l4_to_23(z_l4)
            z_23e, s_23e = blk.pop_23e(l23_ff[:, :blk.n_23e], state.s_23e)
            z_23i, s_23i = blk.pop_23i(l23_ff[:, blk.n_23e:], state.s_23i)
            z_23 = torch.cat([z_23e, z_23i], dim=-1)

            l5_drive = blk.proj_23_to_l5(z_23)
            z_l5, s_l5 = blk.pop_l5(l5_drive, state.s_l5)

            state = type(state)(
                s_l4=s_l4, s_23e=s_23e, s_23i=s_23i, s_l5=s_l5,
                z_23e_prev=z_23e,
            )
            l5_spikes_all.append(z_l5)
            print(
                f"  t={t}: "
                f"l4_drive[std={l4_drive.std():.2f} max={l4_drive.max():.2f}] "
                f"l4_v[max={s_l4.v.max():.2f}] z_l4[frac={z_l4.mean():.3f}] | "
                f"l23_ff[std={l23_ff.std():.2f}] "
                f"l23_v[max={max(s_23e.v.max(), s_23i.v.max()):.2f}] "
                f"z_23[frac={z_23.mean():.3f}] | "
                f"l5_drive[std={l5_drive.std():.2f}] "
                f"l5_v[max={s_l5.v.max():.2f}] z_l5[frac={z_l5.mean():.3f}]"
            )

        z_l5_seq = torch.stack(l5_spikes_all, dim=0)
        print(f"\n[block 0] total L5 spikes over T=8: {z_l5_seq.sum().item()}  "
              f"avg frac: {z_l5_seq.mean().item():.4f}")

    print("\n--- HINT ---")
    print("If l4_v.max never approaches v_th=1.0 -> input drive too weak.")
    print("If z_l4 fires but z_23 stays 0  -> proj_l4_to_23 attenuates signal.")
    print("If z_23 fires but z_l5 stays 0  -> proj_23_to_l5 attenuates signal.")


if __name__ == "__main__":
    main()
