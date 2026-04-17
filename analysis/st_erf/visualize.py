"""
Publication-quality ST-ERF visualization.

Generates:
  - Fig 2 compact: (a) Baseline  (b) SD-BGLA  (c) Decay curves
  - Fig 2 full: 2×3 layout with log-scale heatmaps + statistics
  - Per-layer comparison grids
  - Beta sweep panels
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm
from matplotlib.gridspec import GridSpec
import os


# ---- Global style ----
def _set_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Helvetica', 'Times New Roman'],
        'mathtext.fontset': 'stix',
        'font.size': 18,
        'axes.linewidth': 0.8,
    })


def _row_normalize(erf: np.ndarray) -> np.ndarray:
    T = erf.shape[0]
    out = erf.copy()
    for t in range(T):
        rm = erf[t, :t+1].max()
        if rm > 1e-12:
            out[t] = erf[t] / rm
        else:
            out[t] = 0
    return out


def _causal_mask(T: int) -> np.ndarray:
    return np.triu(np.ones((T, T), dtype=bool), k=1)


def _mean_decay(erf: np.ndarray) -> np.ndarray:
    T = erf.shape[0]
    max_dist = T - 1
    sums = np.zeros(max_dist + 1)
    counts = np.zeros(max_dist + 1)
    for t in range(1, T):
        dv = erf[t, t]
        if dv > 1e-12:
            for tau in range(t + 1):
                d = t - tau
                sums[d] += erf[t, tau] / dv
                counts[d] += 1
    out = np.ones(max_dist + 1)
    for d in range(max_dist + 1):
        if counts[d] > 0:
            out[d] = sums[d] / counts[d]
    return out


# ==============================================================
# Fig 2 Compact: 1 row × 3 panels (recommended for paper)
# ==============================================================

def plot_fig2_compact(
    erf_base: np.ndarray,
    erf_gated: np.ndarray,
    beta: float = 0.25,
    save_path: str = "fig2_st_erf.pdf",
    base_label: str = "Baseline (memoryless)",
    gated_label: str = "SD-BGLA (membrane-gated)",
):
    """
    Paper Fig 2 — three panels:
      (a) Baseline row-normalized heatmap
      (b) SD-BGLA row-normalized heatmap
      (c) Temporal decay comparison
    """
    _set_style()
    T = erf_base.shape[0]

    base_norm = _row_normalize(erf_base)
    gated_norm = _row_normalize(erf_gated)
    mask = _causal_mask(T)
    ticks = range(T)
    labels = [str(i + 1) for i in range(T)]

    fig, axes = plt.subplots(
        1, 3, figsize=(17, 5),
        gridspec_kw={'width_ratios': [1, 1, 1], 'wspace': 0.28},
    )

    # ---- (a) Baseline ----
    ax = axes[0]
    bp = base_norm.copy(); bp[mask] = np.nan
    im_a = ax.imshow(bp, cmap='plasma', aspect='equal',
                      vmin=0, vmax=1, origin='upper', interpolation='nearest')
    ax.set_title(rf'$\mathbf{{(a)}}$ {base_label}', fontsize=13, pad=8)
    ax.set_xlabel(r'Input time step $\tau$', fontsize=13)
    ax.set_ylabel(r'Output time step $t$', fontsize=13)
    # ax.set_xticks(ticks); ax.set_xticklabels(labels)
    # ax.set_yticks(ticks); ax.set_yticklabels(labels)
    step = max(1, T // 5)  # T=16 → step=3, 标注 1,4,7,10,13,16
    major_ticks = list(range(0, T, step))
    if (T - 1) not in major_ticks:
        major_ticks.append(T - 1)  # 确保最后一个有标注
    ax.set_xticks(major_ticks)
    ax.set_xticklabels([str(i + 1) for i in major_ticks])
    ax.set_yticks(major_ticks)
    ax.set_yticklabels([str(i + 1) for i in major_ticks])

    fig.colorbar(im_a, ax=ax, shrink=0.78, pad=0.03).ax.tick_params(labelsize=14)

    # Stats inset
    from analysis.st_erf.st_erf import compute_erf_stats
    sb = compute_erf_stats(erf_base)
    # ax.text(0.02, 0.02,
    #         f"diag energy: {sb['diag_energy']:.1f}%\n"
    #         f"$T_{{\\mathrm{{eff}}}}$ = {sb['T_eff_mean']:.2f}",
    #         transform=ax.transAxes, fontsize=9, va='bottom',
    #         color='white', fontweight='bold',
    #         bbox=dict(facecolor='black', alpha=0.5, pad=2,
    #                   boxstyle='round,pad=0.3'))

    # ---- (b) SD-BGLA ----
    ax = axes[1]
    gp = gated_norm.copy(); gp[mask] = np.nan
    im_b = ax.imshow(gp, cmap='plasma', aspect='equal',
                      vmin=0, vmax=1, origin='upper', interpolation='nearest')
    ax.set_title(rf'$\mathbf{{(b)}}$ {gated_label}', fontsize=13, pad=8)
    ax.set_xlabel(r'Input time step $\tau$', fontsize=13)
    ax.set_ylabel(r'Output time step $t$', fontsize=13)
    # ax.set_xticks(ticks); ax.set_xticklabels(labels)
    # ax.set_yticks(ticks); ax.set_yticklabels(labels)
    step = max(1, T // 5)  # T=16 → step=3, 标注 1,4,7,10,13,16
    major_ticks = list(range(0, T, step))
    if (T - 1) not in major_ticks:
        major_ticks.append(T - 1)  # 确保最后一个有标注
    ax.set_xticks(major_ticks)
    ax.set_xticklabels([str(i + 1) for i in major_ticks])
    ax.set_yticks(major_ticks)
    ax.set_yticklabels([str(i + 1) for i in major_ticks])
    
    fig.colorbar(im_b, ax=ax, shrink=0.78, pad=0.03).ax.tick_params(labelsize=14)

    sg = compute_erf_stats(erf_gated)
    # ax.text(0.02, 0.02,
    #         f"diag energy: {sg['diag_energy']:.1f}%\n"
    #         f"$T_{{\\mathrm{{eff}}}}$ = {sg['T_eff_mean']:.2f}",
    #         transform=ax.transAxes, fontsize=9, va='bottom',
    #         color='white', fontweight='bold',
    #         bbox=dict(facecolor='black', alpha=0.5, pad=2,
    #                   boxstyle='round,pad=0.3'))

    # ---- (c) Decay curves ----
    ax = axes[2]

    # Individual curves (faded)
    for t in range(3, T):
        dv = erf_base[t, t]
        if dv > 1e-12:
            row = erf_base[t, :t+1] / dv
            deltas = t - np.arange(t + 1)
            order = np.argsort(deltas)
            ax.plot(deltas[order], row[order], 's-',
                    color='#4A90D9', alpha=0.15, markersize=3, linewidth=0.8)
        dv = erf_gated[t, t]
        if dv > 1e-12:
            row = erf_gated[t, :t+1] / dv
            deltas = t - np.arange(t + 1)
            order = np.argsort(deltas)
            ax.plot(deltas[order], row[order], 'o-',
                    color='#E85D4A', alpha=0.15, markersize=3, linewidth=0.8)

    # Mean decay (bold)
    bd = _mean_decay(erf_base)
    gd = _mean_decay(erf_gated)
    ax.plot(range(len(bd)), bd, 's-', color='#2563EB', linewidth=2.5,
            markersize=7, alpha=0.9, label='Baseline', zorder=10)
    ax.plot(range(len(gd)), gd, 'o-', color='#DC2626', linewidth=2.5,
            markersize=7, alpha=0.9, label='SD-BGLA', zorder=10)

    # Theoretical beta^(t-tau)
    d = np.linspace(0, T - 1, 100)
    ax.plot(d, beta**d, 'k:', linewidth=2, alpha=0.4,
            label=rf'$\beta^{{t-\tau}}$')

    ax.set_xlabel(r'Temporal distance $t - \tau$', fontsize=13)
    ax.set_ylabel(r'Normalized ST-ERF', fontsize=13)
    ax.set_title(r'$\mathbf{(c)}$ Temporal decay', fontsize=13, pad=8)
    ax.set_yscale('log')
    ax.set_ylim(1e-5, 5.0)
    ax.set_xlim(-0.3, T - 0.5)
    # ax.legend(
    #     fontsize=10, 
    #     bbox_to_anchor=(0.02, 0.02),
    #     loc='lower left',
    #     framealpha=0.92,
    #     edgecolor='#ccc'
    # )
    ax.grid(True, alpha=0.2, linestyle='--')

    # Fill between
    ml = min(len(bd), len(gd))
    ax.fill_between(range(ml), bd[:ml], gd[:ml],
                    alpha=0.06, color='red', zorder=0)

    plt.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.12)
    plt.close()
    print(f"  Saved: {save_path}")


# ==============================================================
# Single model heatmap (for standalone analysis)
# ==============================================================

def plot_single_heatmap(
    erf: np.ndarray,
    title: str = "ST-ERF",
    save_path: str = "st_erf.pdf",
    beta: float | None = 0.25,
):
    """Three-panel figure: (a) log heatmap  (b) row-normalized  (c) decay."""
    _set_style()
    T = erf.shape[0]
    erf_norm = _row_normalize(erf)
    mask = _causal_mask(T)
    ticks = range(T)
    labels = [str(i+1) for i in range(T)]

    fig, axes = plt.subplots(
        1, 3, figsize=(18, 5.5),
        gridspec_kw={'width_ratios': [1, 1, 1.2], 'wspace': 0.32},
    )

    # (a) Log heatmap
    ax = axes[0]
    erf_log = erf.copy(); erf_log[mask] = np.nan; erf_log[erf_log <= 0] = np.nan
    pos = erf[~mask & (erf > 0)]
    vmin = pos.min() * 0.3 if len(pos) else 1e-4
    vmax = pos.max() * 1.5 if len(pos) else 1.0
    im = ax.imshow(erf_log, cmap='plasma', aspect='equal',
                    norm=LogNorm(vmin=vmin, vmax=vmax),
                    origin='upper', interpolation='nearest')
    ax.set_title(r'$\mathbf{(a)}$ ST-ERF (log)', fontsize=12, pad=8)
    ax.set_xlabel(r'$\tau$', fontsize=16); ax.set_ylabel(r'$t$', fontsize=13)
    # ax.set_xticks(ticks); ax.set_xticklabels(labels)
    # ax.set_yticks(ticks); ax.set_yticklabels(labels)
    # 只标注每隔几个的刻度
    step = max(1, T // 5)  # T=16 → step=3, 标注 1,4,7,10,13,16
    major_ticks = list(range(0, T, step))
    if (T - 1) not in major_ticks:
        major_ticks.append(T - 1)  # 确保最后一个有标注
    ax.set_xticks(major_ticks)
    ax.set_xticklabels([str(i + 1) for i in major_ticks])
    ax.set_yticks(major_ticks)
    ax.set_yticklabels([str(i + 1) for i in major_ticks])

    # fig.colorbar(im, ax=ax, shrink=0.82, pad=0.03).ax.tick_params(labelsize=14)

    # (b) Row-normalized
    ax = axes[1]
    ep = erf_norm.copy(); ep[mask] = np.nan
    im2 = ax.imshow(ep, cmap='plasma', aspect='equal', vmin=0, vmax=1,
                     origin='upper', interpolation='nearest')
    ax.set_title(r'$\mathbf{(b)}$ Row-normalized', fontsize=12, pad=8)
    ax.set_xlabel(r'$\tau$', fontsize=16); ax.set_ylabel(r'$t$', fontsize=13)
    # ax.set_xticks(ticks); ax.set_xticklabels(labels)
    # ax.set_yticks(ticks); ax.set_yticklabels(labels)
    # 只标注每隔几个的刻度
    step = max(1, T // 5)  # T=16 → step=3, 标注 1,4,7,10,13,16
    major_ticks = list(range(0, T, step))
    if (T - 1) not in major_ticks:
        major_ticks.append(T - 1)  # 确保最后一个有标注
    ax.set_xticks(major_ticks)
    ax.set_xticklabels([str(i + 1) for i in major_ticks])
    ax.set_yticks(major_ticks)
    ax.set_yticklabels([str(i + 1) for i in major_ticks])

    # fig.colorbar(im2, ax=ax, shrink=0.82, pad=0.03).ax.tick_params(labelsize=14)

    # (c) Decay
    ax = axes[2]
    colors = cm.viridis(np.linspace(0.15, 0.85, T))
    for t in range(2, T):
        dv = erf[t, t]
        if dv > 1e-12:
            row = erf[t, :t+1] / dv
            deltas = t - np.arange(t+1)
            order = np.argsort(deltas)
            ax.plot(deltas[order], row[order], 'o-', color=colors[t],
                    alpha=0.65, markersize=5, linewidth=1.5,
                    label=rf'$t={t+1}$')

    if beta is not None:
        d = np.linspace(0, T-1, 100)
        ax.plot(d, beta**d, 'r--', linewidth=2.5, alpha=0.9,
                label=rf'$\beta^{{t-\tau}}$ ($\beta$={beta})')

    ax.set_xlabel(r'Temporal distance $t - \tau$', fontsize=18)
    ax.set_ylabel(r'Normalized ST-ERF', fontsize=18)
    ax.set_title(r'$\mathbf{(c)}$ Temporal decay', fontsize=18, pad=8)
    ax.set_yscale('log'); ax.set_ylim(1e-5, 5.0); ax.set_xlim(-0.3, T-0.5)
    ax.legend(fontsize=16, loc='upper right', framealpha=0.92)
    ax.grid(True, alpha=0.2, linestyle='--')


    dummy_cb = fig.colorbar(im_b, ax=axes[2], shrink=0.78, pad=0.03)
    dummy_cb.ax.set_visible(False)

    plt.suptitle(title, fontsize=16, y=1.03, fontweight='bold')
    plt.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.12)
    plt.close()
    print(f"  Saved: {save_path}")


# ==============================================================
# Per-layer grid
# ==============================================================

def plot_per_layer_grid(
    erf_dict: dict[int, np.ndarray],
    save_path: str = "st_erf_per_layer.pdf",
    title: str = "Per-layer ST-ERF",
):
    """Row-normalized heatmap grid for all layers."""
    _set_style()
    n = len(erf_dict)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.2),
                              gridspec_kw={'wspace': 0.25})
    if n == 1:
        axes = [axes]

    for idx, (block_i, erf) in enumerate(sorted(erf_dict.items())):
        T = erf.shape[0]
        mask = _causal_mask(T)
        enorm = _row_normalize(erf); enorm[mask] = np.nan
        ax = axes[idx]
        im = ax.imshow(enorm, cmap='plasma', aspect='equal',
                        vmin=0, vmax=1, origin='upper', interpolation='nearest')
        ax.set_title(f'Block {block_i}', fontsize=13)
        ax.set_xlabel(r'$\tau$', fontsize=12)
        if idx == 0:
            ax.set_ylabel(r'$t$', fontsize=12)
        ticks = range(T); labs = [str(i+1) for i in range(T)]
        ax.set_xticks(ticks); ax.set_xticklabels(labs, fontsize=8)
        ax.set_yticks(ticks); ax.set_yticklabels(labs, fontsize=8)

        from analysis.st_erf.st_erf import compute_erf_stats
        s = compute_erf_stats(erf)
        # ax.text(0.5, -0.18, f"$T_{{\\mathrm{{eff}}}}$={s['T_eff_mean']:.2f}",
        #         transform=ax.transAxes, ha='center', fontsize=9, color='#666')

    fig.colorbar(im, ax=axes, shrink=0.75, pad=0.02,
                 label='Row-normalized ST-ERF')
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.04)
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    
    fig.text(
        0.5, -0.03,
        f'Baseline: diag energy {sb["diag_energy"]:.1f}%, '
        f'$T_{{\\mathrm{{eff}}}}$ = {sb["T_eff_mean"]:.2f}, '
        f'$\\rho$ = {sb["one_step_decay"]:.3f}'
        f'    |    '
        f'SD-BGLA: diag energy {sg["diag_energy"]:.1f}%, '
        f'$T_{{\\mathrm{{eff}}}}$ = {sg["T_eff_mean"]:.2f}, '
        f'$\\rho$ = {sg["one_step_decay"]:.3f}',
        ha='center', fontsize=10, color='#444',
    )

    plt.close()
    print(f"  Saved: {save_path}")