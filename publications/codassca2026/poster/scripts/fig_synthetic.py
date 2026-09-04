"""Synthetic-benchmark row: (a) dataset explainer, (b) accuracy vs K, (c) what each selector picks."""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
sys.path.insert(0, str(Path(__file__).resolve().parent))
from poster_style import *   # noqa

d = np.load(DATA / "synthetic_attribution.npz")
X, y, factor_of, corr, chan_var, order = d["X"], d["y"], d["factor_of"], d["corr"], d["chan_var"], d["order"]
NF = 4
role_order = np.concatenate([np.where(factor_of == f)[0] for f in range(NF)] + [np.where(factor_of < 0)[0]])
roles_sorted = factor_of[role_order]
role_color = lambda r: C_CLUSTERS[r] if r >= 0 else C_NOISE
CLUSTER_LABELS = [f"F{i+1}" for i in range(NF)] + ["noise"]


def role_strip(ax, roles, orientation="h"):
    """Colored strip showing the role of each (sorted) channel."""
    for i, r in enumerate(roles):
        if orientation == "h":
            ax.add_patch(Rectangle((i, 0), 1, 1, fc=role_color(r), ec="none"))
        else:
            ax.add_patch(Rectangle((0, i), 1, 1, fc=role_color(r), ec="none"))
    n = len(roles)
    ax.set_xlim(0, n if orientation == "h" else 1); ax.set_ylim(0, 1 if orientation == "h" else n)
    if orientation == "v": ax.invert_yaxis()
    despine_all(ax)


def bracket_labels(ax, roles, y, fontsize=FS_SMALL, color=INK):
    """Label contiguous role runs along x (channel index space)."""
    start = 0
    for i in range(1, len(roles) + 1):
        if i == len(roles) or roles[i] != roles[start]:
            r = roles[start]
            ax.text((start + i) / 2, y, CLUSTER_LABELS[r] if r >= 0 else f"noise ({i-start} ch)",
                    ha="center", va="top", fontsize=fontsize, color=color, clip_on=False)
            start = i


def fig_dataset():
    fig = slot(780, 305)
    gs = fig.add_gridspec(2, 3, width_ratios=[0.035, 1.0, 1.0], height_ratios=[1, 0.06],
                          wspace=0.12, hspace=0.06, left=0.09, right=0.93, top=0.86, bottom=0.2)
    ax_strip = fig.add_subplot(gs[0, 0]); ax_win = fig.add_subplot(gs[0, 1]); ax_corr = fig.add_subplot(gs[0, 2])
    ax_hs = fig.add_subplot(gs[1, 1]); ax_hs2 = fig.add_subplot(gs[1, 2])
    W = X[0][:, role_order].T                                   # (64 ch, 64 t)
    vmax = np.abs(W).max()
    ax_win.imshow(W, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax_win.set_title("One window: 64 channels × 64 time steps", fontsize=FS_LABEL, pad=10)
    ax_win.set_xticks([]); ax_win.set_yticks([])
    ax_win.grid(False); ax_win.tick_params(labelsize=FS_SMALL)
    role_strip(ax_strip, roles_sorted, "v")
    for f in range(NF):
        idx = np.where(roles_sorted == f)[0]
        ax_strip.text(-0.6, idx.mean() + 0.5, f"F{f+1}", ha="right", va="center", fontsize=FS_SMALL, color=INK)
    idx = np.where(roles_sorted < 0)[0]
    ax_strip.text(-0.6, idx.mean() + 0.5, "noise", ha="right", va="center", fontsize=FS_SMALL, color=INK, rotation=90)
    C = corr[np.ix_(role_order, role_order)]
    im = ax_corr.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    ax_corr.set_title("Channel correlation", fontsize=FS_LABEL, pad=10)
    ax_corr.set_xticks([]); ax_corr.set_yticks([]); ax_corr.grid(False)
    cb = fig.colorbar(im, ax=ax_corr, fraction=0.045, pad=0.02, ticks=[-1, 0, 1]); cb.ax.tick_params(labelsize=FS_SMALL)
    cb.outline.set_visible(False)
    ax_hs.axis("off")
    role_strip(ax_hs2, roles_sorted, "h"); ax_hs2.set_xlim(0, 64)
    bracket_labels(ax_hs2, roles_sorted, -0.35, fontsize=FS_SMALL - 2)
    ax_win.text(0, 66.0, "rows sorted by role (F1 loudest … F4 quietest)", ha="left", va="top",
                fontsize=FS_SMALL - 2, color=INK2, clip_on=False)
    save(fig, "fig_syn_dataset.png")


def fig_curve():
    K = np.array([2, 4, 6, 8, 12, 16, 20])
    S = {"Proposed (label-free)": [0.267, 0.278, 0.494, 0.944, 0.994, 1.000, 1.000],
         "Supervised MI":         [0.272, 0.450, 0.994, 0.994, 0.950, 1.000, 1.000],
         "PCA":                   [0.144, 0.517, 0.956, 0.950, 0.994, 0.994, 1.000],
         "Variance":              [0.106, 0.139, 0.283, 0.244, 0.583, 0.911, 1.000],
         "Random-K":              [0.100, 0.140, 0.234, 0.254, 0.295, 0.493, 0.536]}
    CEIL = 0.872
    fig = slot(700, 305); ax = fig.add_subplot(111)
    ax.axvspan(7.3, 8.7, color=GOLD, alpha=0.35, lw=0, zorder=0)
    ax.axhline(CEIL, ls=(0, (6, 4)), color=C_FULL, lw=2, zorder=1)
    ax.text(20.3, CEIL - 0.02, "full set = 0.87", ha="right", va="top", fontsize=FS_SMALL, color=INK)
    for name, vals in S.items():
        c, mk, ls = METHOD_STYLE[name]
        ax.plot(K, vals, marker=mk, color=c, ls=ls, label=name, zorder=3, markeredgecolor="white", markeredgewidth=1.5)
    ax.annotate("K = 8: 0.94 with\n8 of 64 channels", xy=(8, 0.944), xytext=(8.6, 0.66), fontsize=FS_SMALL,
                color=C_OURS, ha="left", arrowprops=dict(arrowstyle="-", color=C_OURS, lw=1.5))
    ax.set_xlabel("Number of retained channels, K (of 64)"); ax.set_ylabel("kNN accuracy")
    ax.set_xticks(K); ax.set_ylim(0, 1.06); ax.set_xlim(1, 21)
    fig.legend(loc="lower center", ncol=3, handlelength=2.0, fontsize=FS_SMALL - 1, frameon=False, columnspacing=1.4, handletextpad=0.5, bbox_to_anchor=(0.55, 0.0))
    fig.subplots_adjust(left=0.13, right=0.98, top=0.95, bottom=0.40)
    save(fig, "fig_syn_curve.png")


def selectors_k8():
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import mutual_info_classif
    Fm = X.mean(axis=1)                                     # (N, 64) channel means
    pca_load = np.abs(PCA(n_components=8).fit(Fm - Fm.mean(0)).components_).sum(0)
    mi = mutual_info_classif(Fm, y, random_state=0)
    rng = np.random.default_rng(0)
    return {"Proposed (label-free)": list(order[:8]),
            "Supervised MI": list(np.argsort(-mi)[:8]),
            "PCA": list(np.argsort(-pca_load)[:8]),
            "Variance": list(np.argsort(-chan_var)[:8]),
            "Random-K": list(rng.choice(64, 8, replace=False))}


def fig_selection():
    picks = selectors_k8()
    fig = slot(784, 305)
    ax = fig.add_axes([0.24, 0.2, 0.60, 0.72])
    pos = {ch: i for i, ch in enumerate(role_order)}
    rows = list(picks.keys())
    for r, name in enumerate(rows):
        yy = len(rows) - 1 - r
        for i, role in enumerate(roles_sorted):
            ax.add_patch(Rectangle((i + 0.08, yy + 0.12), 0.84, 0.76, fc=role_color(role), ec="none", alpha=0.45))
        sel = picks[name]
        for ch in sel:
            i = pos[ch]
            ax.add_patch(Rectangle((i + 0.14, yy + 0.2), 0.72, 0.6, fc=INK, ec="none"))
        cov = len({factor_of[c] for c in sel if factor_of[c] >= 0}); noise = sum(factor_of[c] < 0 for c in sel)
        col = C_OURS if name.startswith("Proposed") else INK
        ax.text(64.8, yy + 0.5, f"{cov}/4 factors · {noise} noise", va="center", ha="left", fontsize=FS_SMALL, color=col, clip_on=False)
        ax.text(-0.8, yy + 0.5, name, va="center", ha="right", fontsize=FS_TICK, color=col,
                weight="bold" if name.startswith("Proposed") else "normal", clip_on=False)
    ax.set_xlim(0, 64); ax.set_ylim(0, len(rows)); despine_all(ax)
    bracket_labels(ax, roles_sorted, -0.12, fontsize=FS_SMALL - 2)
    ax.set_title("Channels picked at K = 8   (black = selected, background = channel role)", fontsize=FS_LABEL, pad=8)
    save(fig, "fig_syn_selection.png")


if __name__ == "__main__":
    fig_dataset(); fig_curve(); fig_selection()
