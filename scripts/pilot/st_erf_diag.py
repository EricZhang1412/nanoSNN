"""ST-ERF (Spatio-Temporal Effective Receptive Field) diagnostic.

For a trained Spikformer/SpikformerAudio checkpoint, compute

    M[t, τ] = E_sample  ‖ ∂ ( S_block[t] ) / ∂ ( X[τ] ) ‖_F   (Frobenius)

on a small subset of the test set (default 256 samples), where S_block[t] is
the KV-state of the LAST gated-attention block at step t and X[τ] is the
patch-embed output at step τ.

Reports two scalar summaries per checkpoint:

  * E_diag = sum_t M[t, t]^2 / sum_{t,τ} M[t, τ]^2  (diagonal energy concentration)
  * T_eff = (sum_t M[t, t])^2 / sum_t M[t, t]^2     (participation ratio, time)

And saves a heatmap PNG (matplotlib, no GUI).

Usage:
  uv run python -m scripts.pilot.st_erf_diag \\
      --ckpt pilot_results/checkpoints/dvs128/c3_mga/seed42/last.ckpt \\
      --task dvs128 \\
      --condition c3_mga \\
      --data_config configs/data_configs/dvs128gesture_pilot.yaml \\
      --model_config configs/model_configs/pilot/spikformer_dvs_c3_mga.yaml \\
      --out_dir pilot_results/figs \\
      --n_samples 256
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from spikingjelly.activation_based import functional  # noqa: E402


def load_model_and_data(ckpt_path: str, model_config_path: str, data_config_path: str):
    from utils.load_config import load_config
    from models.build_model import build_model
    from data import VisionDataModule

    model_config = load_config(model_config_path)
    data_config = load_config(data_config_path)

    # Dummy optimizer/train cfg — we only need the model + datamodule.
    optim_cfg = load_config(os.path.join(_REPO_ROOT, "configs/optimizer_configs/pilot_dvs.yaml"))
    train_cfg = load_config(os.path.join(_REPO_ROOT, "configs/train_configs/pilot_dvs.yaml"))
    lit = build_model(model_config, optim_cfg, train_cfg, data_config)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = state["state_dict"] if "state_dict" in state else state
    missing, unexpected = lit.load_state_dict(sd, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)} (first 5): {missing[:5]}")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)} (first 5): {unexpected[:5]}")

    dm = VisionDataModule(data_config=data_config, train_config=train_cfg)
    dm.setup("test")
    return lit, dm


def collect_intermediate_S(model, x):
    """Run forward and return the per-step S of the LAST attention block,
    plus the patch-embed output `feat[t]` whose gradient we want.

    We add hooks to capture both.
    """
    captured = {"feat": None, "S_seq": None}

    inner = model.model

    # Hook on patch_embed (works for both Spikformer and SpikformerAudio).
    pe = inner.patch_embed
    feat_handle = pe.register_forward_hook(lambda mod, inp, out: captured.__setitem__("feat", out))

    # Hook on last gated attention module: override _run_state_update to capture S_seq.
    last_block = inner.blocks[-1]
    last_attn = last_block.attn
    if not hasattr(last_attn, "_run_state_update"):
        feat_handle.remove()
        raise RuntimeError("Last block is not a gated linear attention; ST-ERF only "
                           "supports c0/c1/c2/c3 attention modules.")
    orig = last_attn._run_state_update

    def wrapped(K, V, K_membrane=None):
        S = orig(K, V, K_membrane)
        captured["S_seq"] = S
        return S

    last_attn._run_state_update = wrapped
    try:
        functional.reset_net(inner)
        _ = model(x)
    finally:
        feat_handle.remove()
        last_attn._run_state_update = orig

    return captured["feat"], captured["S_seq"]


def compute_st_erf(model, x_sample) -> np.ndarray:
    """Return T x T matrix of ‖∂ ‖S_block[t]‖_F^2 / ∂ X[τ]‖ via autograd.

    We treat ‖S[t]‖_F^2 as a scalar and back-prop w.r.t. X[τ] (the patch-embed
    output) — implemented via `torch.autograd.grad`.  Shape is [T, T].
    """
    model.eval()
    x_sample = x_sample.detach()
    x_sample.requires_grad_(False)

    feat, S_seq = collect_intermediate_S(model, x_sample)
    if feat is None or S_seq is None:
        raise RuntimeError("Failed to capture intermediate tensors")
    # feat: [T, B, N, C]   S_seq: [T, B, H, D, D]
    T = S_seq.shape[0]
    if feat.shape[0] != T:
        raise RuntimeError(f"feat T ({feat.shape[0]}) != S_seq T ({T})")

    feat = feat.detach().requires_grad_(True)
    # Redo the block stack on `feat` so that S_seq depends on `feat`.
    # We assume `inner.blocks[-1]` is the gated attention block.
    inner = model.model
    h = feat
    for blk in inner.blocks:
        h = blk(h)
    # h is unused; we need S_seq from the LAST block, captured via the hook.
    # But the previous call already populated S_seq — it was from the LAST forward.
    # We need to redo forward with `feat` as input so grad flows.
    captured = {"S_seq": None}
    last_attn = inner.blocks[-1].attn
    orig = last_attn._run_state_update

    def wrapped(K, V, K_membrane=None):
        S = orig(K, V, K_membrane)
        captured["S_seq"] = S
        return S

    last_attn._run_state_update = wrapped
    try:
        functional.reset_net(inner)
        h = feat
        for blk in inner.blocks:
            h = blk(h)
    finally:
        last_attn._run_state_update = orig

    S_seq = captured["S_seq"]
    matrix = np.zeros((T, T), dtype=np.float64)
    for t in range(T):
        scalar = (S_seq[t] ** 2).sum()
        grads = torch.autograd.grad(
            outputs=scalar,
            inputs=feat,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0]  # [T, B, N, C]
        # Frobenius norm per τ.
        for tau in range(T):
            matrix[t, tau] = float(grads[tau].pow(2).sum().sqrt())
    return matrix


def summarize(M: np.ndarray) -> dict[str, float]:
    diag = np.diag(M)
    total = (M ** 2).sum()
    diag_e = (diag ** 2).sum() / max(total, 1e-30)
    s = diag.sum()
    t_eff = (s * s) / max((diag ** 2).sum(), 1e-30)
    return {"E_diag": float(diag_e), "T_eff": float(t_eff)}


def plot_heatmap(M: np.ndarray, out_path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(M, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xlabel("τ (source step)")
    ax.set_ylabel("t (KV-state step)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--task", type=str, required=True, choices=["dvs128", "shd"])
    p.add_argument("--condition", type=str, required=True)
    p.add_argument("--data_config", type=str, required=True)
    p.add_argument("--model_config", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="pilot_results/figs")
    p.add_argument("--n_samples", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    lit, dm = load_model_and_data(args.ckpt, args.model_config, args.data_config)
    lit = lit.to(args.device).eval()

    test_loader = dm.test_dataloader()
    matrices = []
    seen = 0
    for x, _ in test_loader:
        if seen >= args.n_samples:
            break
        x = x.to(args.device)
        # Iterate over the batch one sample at a time to keep autograd cheap.
        for i in range(x.shape[1] if x.ndim >= 4 else x.shape[0]):
            if seen >= args.n_samples:
                break
            x_i = x[:, i:i + 1] if x.ndim >= 4 else x[i:i + 1]
            try:
                M = compute_st_erf(lit, x_i)
            except RuntimeError as e:
                print(f"[skip sample {seen}] {e}")
                continue
            matrices.append(M)
            seen += 1
    if not matrices:
        print("No samples processed — abort.")
        return

    M_mean = np.mean(matrices, axis=0)
    stats = summarize(M_mean)
    print(f"[{args.task}/{args.condition}]  E_diag={stats['E_diag']:.3f}  T_eff={stats['T_eff']:.2f}  "
          f"  (n_samples={seen})")

    out_dir = Path(args.out_dir)
    np.save(out_dir / f"st_erf_{args.task}_{args.condition}.npy", M_mean)
    plot_heatmap(
        M_mean,
        out_dir / f"st_erf_{args.task}_{args.condition}.png",
        title=f"ST-ERF  {args.task}  {args.condition}\nE_diag={stats['E_diag']:.3f}  T_eff={stats['T_eff']:.2f}",
    )

    # Also append summary stats to a JSON.
    import json
    summary_path = out_dir / "st_erf_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
    else:
        summary = {}
    summary[f"{args.task}/{args.condition}"] = stats
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"updated {summary_path}")


if __name__ == "__main__":
    main()
