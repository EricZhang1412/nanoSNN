"""
ST-ERF computation for spiking attention modules.

Uses PyTorch autograd to compute exact surrogate gradients — no finite
differences or Hutchinson estimation needed.

Usage:
    from analysis.st_erf import compute_st_erf_attention, compute_st_erf_model
    erf = compute_st_erf_attention(attn_module, T=4, B=2, N=64, C=256)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from spikingjelly.activation_based import functional


def _bn1d(x: torch.Tensor, bn: nn.BatchNorm1d) -> torch.Tensor:
    """Apply BN1d on [T, B, N, C] — same as model.py."""
    T, B, N, C = x.shape
    x = rearrange(x, "T B N C -> (T B) N C")
    x = rearrange(bn(rearrange(x, "TB N C -> TB C N")), "TB C N -> TB N C")
    return rearrange(x, "(T B) N C -> T B N C", T=T, B=B).contiguous()


def _freeze_bn(module: nn.Module):
    """Set all BatchNorm layers to eval mode while keeping rest in train.

    This prevents cross-timestep gradient leakage: in train mode,
    BN computes statistics over the merged (T*B) dimension, so
    perturbing X[tau] changes the normalization at ALL timesteps,
    creating a spurious non-decaying gradient floor.
    """
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()


def _extract_spikes(x):
    """Extract spike tensor from LIF output.

    Some custom neurons (e.g. DualOutput from tl_neuron_ops) return
    a named tuple or wrapper containing both spikes and membrane
    potentials. This helper extracts the spike tensor regardless
    of the return type.
    """
    if isinstance(x, torch.Tensor):
        return x
    # DualOutput or similar: try common attribute names
    for attr in ("spikes", "spike", "s", "output", "data"):
        if hasattr(x, attr):
            val = getattr(x, attr)
            if isinstance(val, torch.Tensor):
                return val
    # Tuple/list: first element is usually spikes
    if isinstance(x, (tuple, list)) and len(x) > 0:
        return x[0]
    # Indexable: try [0]
    try:
        return x[0]
    except (TypeError, IndexError, KeyError):
        pass
    raise TypeError(
        f"Cannot extract spike tensor from LIF output of type {type(x)}. "
        f"Available attributes: {[a for a in dir(x) if not a.startswith('_')]}"
    )


# ==============================================================
# Attention-level ST-ERF
# ==============================================================

@torch.no_grad()
def _get_firing_rates(attn_module: nn.Module, X: torch.Tensor) -> dict:
    """Quick diagnostic: measure firing rates of Q/K/V LIF neurons."""
    functional.reset_net(attn_module)
    attn_module.eval()
    # run forward to populate spike stats
    _ = attn_module(X)
    return {}  # placeholder; can be extended


def _make_input(shape, device, scale=1.0):
    """Create a leaf tensor with requires_grad=True at the desired scale.

    IMPORTANT: do NOT use `torch.randn(...) * scale` — this creates a
    non-leaf tensor (with MulBackward grad_fn), and .requires_grad_(True)
    on a non-leaf silently fails to make it a gradient target for
    torch.autograd.grad(). Instead, sample directly at the desired std.
    """
    return torch.empty(*shape, device=device).normal_(0, scale).requires_grad_(True)


def _get_input_scale(attn_module: nn.Module) -> float:
    """Auto-calibrate input scale for models with pre-attention LIF.

    SD-BGLA has a shortcut_lif before the linear projections. With
    N(0,1) input and tau=2, threshold=1, only ~2% of neurons fire,
    causing K≈0, V≈0, gradient≈0. We scale up the input so ~25% fire.
    """
    if not hasattr(attn_module, 'shortcut_lif'):
        return 1.0

    tau_val = 2.0
    thr = 1.0
    try:
        node = getattr(attn_module.shortcut_lif, 'node', attn_module.shortcut_lif)
        if hasattr(node, 'tau_tensor'):
            tau_val = node.tau_tensor.item()
        thr = float(node.v_threshold)
    except Exception:
        pass

    # For LIF: fires when input/tau >= threshold, i.e. input >= tau*threshold.
    # For N(0, scale): P(x >= tau*thr) ≈ 25% when scale ≈ tau*thr / 0.67
    scale = tau_val * thr / 0.67
    print(f"  [auto-scale] input std={scale:.1f} "
          f"(shortcut_lif tau={tau_val}, thr={thr})")
    return scale


def compute_st_erf_attention(
    attn_module: nn.Module,
    T: int = 4,
    B: int = 2,
    N: int = 64,
    C: int = 256,
    num_samples: int = 10,
    num_projections: int = 5,
    device: str = "cuda",
    seed: int = 42,
    block_idx: int | None = None,
) -> np.ndarray:
    """
    Compute ST-ERF matrix for a single attention module.

    ST-ERF(t, tau) = || dO[t] / dX[tau] ||_F

    Estimated via Hutchinson's identity:
        ||J||_F^2 = E_r[ ||J^T r||^2 ]  where r ~ N(0, I)

    Each sample draws a new random input X; for each X, we use
    `num_projections` random vectors r to estimate the Frobenius norm.
    """
    torch.manual_seed(seed)
    attn_module = attn_module.to(device)
    attn_module.train()
    _freeze_bn(attn_module)

    erf_accum = np.zeros((T, T))
    label = f"block {block_idx}" if block_idx is not None else "attn"
    input_scale = _get_input_scale(attn_module)

    for s in range(num_samples):
        with torch.enable_grad():
            X_list = [
                _make_input((B, N, C), device, scale=input_scale)
                for _ in range(T)
            ]
            X = torch.stack(X_list, dim=0)
            functional.reset_net(attn_module)
            O = _extract_spikes(attn_module(X))

            if s == 0:
                if not isinstance(O, torch.Tensor) or not O.requires_grad:
                    raise RuntimeError(
                        f"Output issue: type={type(O)}, "
                        f"requires_grad={O.requires_grad if isinstance(O, torch.Tensor) else 'N/A'}"
                    )

            # Hutchinson Frobenius norm estimation
            erf_sq = np.zeros((T, T))
            for p in range(num_projections):
                for t in range(T):
                    # Random projection: r ~ N(0, I), same shape as O[t]
                    r = torch.randn_like(O[t])
                    scalar = (O[t] * r).sum()  # = r^T @ vec(O[t])

                    for tau in range(t + 1):
                        # grad = J^T r, where J = dO[t]/dX[tau]
                        grad = torch.autograd.grad(
                            scalar, X_list[tau],
                            retain_graph=True,
                            allow_unused=True,
                        )[0]
                        if grad is not None:
                            # ||J^T r||^2 → average gives ||J||_F^2
                            erf_sq[t, tau] += grad.norm().item() ** 2 / num_projections

            # ||J||_F = sqrt(E[||J^T r||^2])
            erf_accum += np.sqrt(np.maximum(erf_sq, 0)) / num_samples

        if (s + 1) % max(1, num_samples // 5) == 0 or s == 0:
            print(f"  [{label}] sample {s+1}/{num_samples}", flush=True)

    return erf_accum


# ==============================================================
# Full-model ST-ERF (input image → pre-classifier features)
# ==============================================================

def compute_st_erf_model(
    model: nn.Module,
    T: int = 4,
    B: int = 2,
    img_size: int = 32,
    in_channels: int = 3,
    num_samples: int = 10,
    device: str = "cuda",
    seed: int = 42,
) -> np.ndarray:
    """
    Compute ST-ERF for the full model (image input → temporal features).

    The model must accept [T, B, C, H, W] input.
    We measure sensitivity of the per-step features (before temporal_mean)
    to each input timestep.

    Args:
        model: a Spikformer or similar model with .patch_embed and .blocks
        T, B, img_size, in_channels: input dimensions
        num_samples, device, seed: estimation parameters

    Returns:
        erf_matrix: [T, T] numpy array
    """
    torch.manual_seed(seed)
    model = model.to(device)
    model.train()  # train mode for surrogate gradient graph
    _freeze_bn(model)

    erf_accum = np.zeros((T, T))

    for s in range(num_samples):
        with torch.enable_grad():
            # Create T separate input frames
            X_list = [
                torch.randn(B, in_channels, img_size, img_size,
                             device=device, requires_grad=True)
                for _ in range(T)
            ]
            X = torch.stack(X_list, dim=0)  # [T, B, C, H, W]

            # Reset neuron states
            functional.reset_net(model)

            # Forward through patch_embed + blocks (skip head + temporal_mean)
            feat = model.patch_embed(X)  # [T, B, N, embed_dim]
            for blk in model.blocks:
                feat = blk(feat)
            # feat: [T, B, N, embed_dim] — per-timestep features

            for t in range(T):
                scalar = feat[t].sum()
                for tau in range(t + 1):
                    grad = torch.autograd.grad(
                        scalar, X_list[tau],
                        retain_graph=True,
                        allow_unused=True,
                    )[0]
                    if grad is not None:
                        erf_accum[t, tau] += grad.norm().item() / num_samples

        if (s + 1) % max(1, num_samples // 5) == 0 or s == 0:
            print(f"  [model] sample {s+1}/{num_samples}", flush=True)

    return erf_accum


# ==============================================================
# Per-layer ST-ERF (all attention blocks)
# ==============================================================

def compute_st_erf_all_layers(
    model: nn.Module,
    T: int = 4,
    B: int = 2,
    N: int | None = None,
    C: int | None = None,
    num_samples: int = 10,
    device: str = "cuda",
    seed: int = 42,
) -> dict[int, np.ndarray]:
    """
    Compute ST-ERF for every attention block in the model.

    Args:
        model: Spikformer-like model with .blocks[i].attn
        N, C: token count and embedding dim (auto-detected if None)

    Returns:
        dict mapping block_idx → [T, T] ST-ERF matrix
    """
    if C is None:
        C = int(getattr(model, "head", None) and model.head.in_features or 256)
    if N is None:
        img_size = int(getattr(model, "patch_embed", None) and
                       getattr(model.patch_embed, "num_patches", 64) or 64)
        N = img_size

    results = {}
    for i, blk in enumerate(model.blocks):
        print(f"\n--- Block {i} ({blk.attn.__class__.__name__}) ---")
        erf = compute_st_erf_attention(
            blk.attn, T=T, B=B, N=N, C=C,
            num_samples=num_samples, device=device,
            seed=seed, block_idx=i,
        )
        results[i] = erf
    return results


# ==============================================================
# KV-state ST-ERF (isolates attention's temporal memory)
# ==============================================================

def compute_kv_erf(
    attn_module: nn.Module,
    T: int = 4,
    B: int = 2,
    N: int = 64,
    C: int = 256,
    num_samples: int = 10,
    num_projections: int = 5,
    device: str = "cuda",
    seed: int = 42,
    block_idx: int | None = None,
) -> np.ndarray:
    """
    Compute ST-ERF for the KV state only:

        KV-ERF(t, tau) = || dS[t] / dX[tau] ||_F

    This isolates the temporal dependency of the attention's associative
    memory (KV state) from Q readout and output LIF contributions.

    Supports two attention types:
      - Baseline (SpikeSelfAttention / PSPGated): S[t] = K[t]^T V[t]
      - SD-BGLA (SDBGLAttention): S[t] via gated recurrence
    """
    torch.manual_seed(seed)
    attn_module = attn_module.to(device)
    attn_module.train()
    _freeze_bn(attn_module)

    H = attn_module.num_heads
    D = attn_module.head_dim
    is_sdbgla = hasattr(attn_module, "decay_gate")

    erf_accum = np.zeros((T, T))
    label = f"kv-block{block_idx}" if block_idx is not None else "kv"
    if is_sdbgla:
        label += " (SD-BGLA recurrent)"
    else:
        label += " (memoryless)"

    input_scale = _get_input_scale(attn_module)

    for s in range(num_samples):
        with torch.enable_grad():
            X_list = [
                _make_input((B, N, C), device, scale=input_scale)
                for _ in range(T)
            ]
            X = torch.stack(X_list, dim=0)  # [T, B, N, C]

            functional.reset_net(attn_module)

            if is_sdbgla:
                # ---- SD-BGLA: replicate forward up to S_all ----
                # Step 1: get Q, K, V, membrane potential
                q, k, v, u_k = attn_module._get_qkv(X)

                # DEBUG: trace gradient chain on first sample
                if s == 0:
                    print(f"  [DEBUG] X.requires_grad={X.requires_grad}")
                    print(f"  [DEBUG] k: type={type(k)}, requires_grad={k.requires_grad if isinstance(k, torch.Tensor) else 'N/A'}, grad_fn={k.grad_fn if isinstance(k, torch.Tensor) else 'N/A'}")
                    print(f"  [DEBUG] v: requires_grad={v.requires_grad}, grad_fn={v.grad_fn}")
                    print(f"  [DEBUG] u_k: requires_grad={u_k.requires_grad}, grad_fn={u_k.grad_fn}")

                # Step 2: gate spikes
                s_gamma = attn_module._compute_gate_spikes(k, u_k)

                if s == 0:
                    print(f"  [DEBUG] s_gamma: requires_grad={s_gamma.requires_grad}, grad_fn={s_gamma.grad_fn}")

                # Step 3: per-timestep KV outer product
                kv = torch.einsum("TBHND,TBHNE->TBHDE", k.float(), v.float())

                if s == 0:
                    print(f"  [DEBUG] kv: requires_grad={kv.requires_grad}, grad_fn={kv.grad_fn}")

                # Step 4: recurrent KV state (includes decay gate)
                S_all = attn_module._forward_recurrent(kv, s_gamma)

                if s == 0:
                    print(f"  [DEBUG] S_all: requires_grad={S_all.requires_grad}, grad_fn={S_all.grad_fn}")
                    # Quick gradient test: can we get ANY gradient?
                    test_grad = torch.autograd.grad(
                        S_all.sum(), X_list[0],
                        retain_graph=True, allow_unused=True
                    )[0]
                    print(f"  [DEBUG] test grad X_list[0]: {test_grad is not None}, "
                          f"norm={test_grad.norm().item() if test_grad is not None else 0}")

                S_list = [S_all[t] for t in range(T)]

            else:
                # ---- Baseline: S[t] = K[t]^T V[t], no recurrence ----
                x_flat = rearrange(X, "T B N C -> (T B) N C")

                k = attn_module.k_linear(x_flat)
                k = _bn1d(rearrange(k, "(T B) N C -> T B N C", T=T, B=B),
                          attn_module.k_bn)
                k = _extract_spikes(attn_module.k_lif(k))
                k = rearrange(k, "T B N (H D) -> T B H N D", H=H, D=D)

                v = attn_module.v_linear(x_flat)
                v = _bn1d(rearrange(v, "(T B) N C -> T B N C", T=T, B=B),
                          attn_module.v_bn)
                v = _extract_spikes(attn_module.v_lif(v))
                v = rearrange(v, "T B N (H D) -> T B H N D", H=H, D=D)

                # Apply K gate if present (PSPGatedLinear)
                if hasattr(attn_module, "k_gate") and attn_module.k_gate is not None:
                    k = _extract_spikes(attn_module.k_gate(k))

                S_list = []
                for t in range(T):
                    kv_t = rearrange(k[t], "B H N D -> B H D N") @ v[t]
                    S_list.append(kv_t)

            # ---- Compute Frobenius norm via Hutchinson: dS[t] / dX[tau] ----
            erf_sq = np.zeros((T, T))
            for p in range(num_projections):
                for t in range(T):
                    r = torch.randn_like(S_list[t])
                    scalar = (S_list[t] * r).sum()
                    for tau in range(t + 1):
                        grad = torch.autograd.grad(
                            scalar, X_list[tau],
                            retain_graph=True,
                            allow_unused=True,
                        )[0]
                        if grad is not None:
                            erf_sq[t, tau] += grad.norm().item() ** 2 / num_projections

            erf_accum += np.sqrt(np.maximum(erf_sq, 0)) / num_samples

        if (s + 1) % max(1, num_samples // 5) == 0 or s == 0:
            print(f"  [{label}] sample {s+1}/{num_samples}", flush=True)

    return erf_accum


def compute_kv_erf_all_layers(
    model: nn.Module,
    T: int = 4,
    B: int = 2,
    N: int | None = None,
    C: int | None = None,
    num_samples: int = 10,
    device: str = "cuda",
    seed: int = 42,
) -> dict[int, np.ndarray]:
    """Compute KV-ERF for every attention block."""
    if C is None:
        C = int(getattr(model, "head", None) and model.head.in_features or 256)
    if N is None:
        N = getattr(model.patch_embed, "num_patches", 64)

    results = {}
    for i, blk in enumerate(model.blocks):
        print(f"\n--- KV-ERF Block {i} ({blk.attn.__class__.__name__}) ---")
        erf = compute_kv_erf(
            blk.attn, T=T, B=B, N=N, C=C,
            num_samples=num_samples, device=device,
            seed=seed, block_idx=i,
        )
        results[i] = erf
    return results


# ==============================================================
# Statistics
# ==============================================================

def compute_erf_stats(erf: np.ndarray) -> dict:
    """Compute summary statistics from an ST-ERF matrix."""
    T = erf.shape[0]
    diag = np.diag(erf)

    # Diagonal energy
    total_e = sum(erf[t, tau]**2 for t in range(T) for tau in range(t+1))
    diag_e = sum(diag[t]**2 for t in range(T))
    diag_energy = diag_e / (total_e + 1e-12) * 100

    # Off-diagonal
    off = [erf[t, tau] for t in range(T) for tau in range(t)]
    off_mean = np.mean(off) if off else 0
    ratio = diag.mean() / (off_mean + 1e-12)

    # T_eff
    T_effs = []
    for t in range(1, T):
        row = erf[t, :t+1]
        if row.max() > 1e-12:
            T_effs.append(row.sum() / row.max())

    # One-step decay
    one_step = []
    for t in range(1, T):
        if diag[t] > 1e-12:
            one_step.append(erf[t, t-1] / diag[t])

    return {
        "diag_energy": diag_energy,
        "diag_off_ratio": ratio,
        "T_eff_mean": np.mean(T_effs) if T_effs else 1.0,
        "one_step_decay": np.mean(one_step) if one_step else 0.0,
    }


def print_erf_stats(erf: np.ndarray, label: str = ""):
    """Print formatted statistics."""
    T = erf.shape[0]
    stats = compute_erf_stats(erf)
    print(f"\n{'=' * 55}")
    print(f"  ST-ERF Statistics {label}")
    print(f"{'=' * 55}")
    print(f"  Diagonal energy:    {stats['diag_energy']:.1f}%")
    print(f"  Diag/off-diag:      {stats['diag_off_ratio']:.1f}×")
    print(f"  T_eff (mean):       {stats['T_eff_mean']:.2f}")
    print(f"  1-step decay:       {stats['one_step_decay']:.4f}")

    print(f"\n  Normalized rows:")
    diag = np.diag(erf)
    for t in range(1, min(T, 7)):
        if diag[t] > 1e-12:
            row = erf[t, :t+1] / diag[t]
            vals = " ".join(f"{v:.4f}" for v in row)
            print(f"    t={t+1}: [{vals}]")
    print(f"{'=' * 55}")