"""Generate figures for generalization/EXPLAINER.md.

Numbers are the recorded full-LOSO results (see generalization/reports/*.txt and
RESEARCH_LOG.md). KNN-5 macro-F1, leave-one-subject-out over PAMAP2 subjects 1-8.

Run: python experiments/general_make_figures.py
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path("publications/generalization/figures"); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
})

K = np.array([5, 7, 10])
# full-LOSO macro-F1 (mean, std). AE from general_pamap2_loso.py; MI/var/rand from
# general_pamap2_baseline_diag.py (both KNN-5, LOSO subjects 1-8).
SERIES = {
    "Proposed method (label-free)": ([0.5368, 0.6274, 0.6788], [0.090, 0.073, 0.081], "#c0392b", "o"),
    "Mutual-info (SUPERVISED)":      ([0.5374, 0.6055, 0.6844], [0.090, 0.069, 0.109], "#2c7fb8", "s"),
    "Variance (unsupervised)":       ([0.5103, 0.5397, 0.5914], [0.142, 0.162, 0.174], "#f39c12", "^"),
    "Random-K":                      ([0.5539, 0.5935, 0.6492], [0.075, 0.076, 0.078], "#7f8c8d", "D"),
}
CEILING = 0.722  # all 27 channels, KNN-5 LOSO


def fig_acc_vs_k():
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    ax.axhline(CEILING, ls="--", color="#2ecc71", lw=1.5, label=f"All 27 ch. (ceiling) = {CEILING:.2f}")
    for name, (m, s, color, mk) in SERIES.items():
        ax.errorbar(K, m, yerr=s, marker=mk, color=color, capsize=3, lw=1.6, ms=5, label=name)
    ax.set_xlabel("Number of selected channels (K)")
    ax.set_ylabel("Macro-F1 (LOSO)")
    ax.set_title("PAMAP2 HAR: selection vs. baselines")
    ax.set_xticks(K); ax.grid(alpha=0.3); ax.legend(fontsize=6, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "fig_acc_vs_k.png", dpi=200); plt.close(fig)


def fig_stability():
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    labels = ["K=5", "K=7", "K=10"]
    ae_std = [0.090, 0.073, 0.081]
    var_std = [0.142, 0.162, 0.174]
    x = np.arange(3); w = 0.36
    ax.bar(x - w/2, ae_std, w, color="#c0392b", label="Proposed method")
    ax.bar(x + w/2, var_std, w, color="#f39c12", label="Variance")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Std of macro-F1 across subjects\n(lower = more stable)")
    ax.set_title("Selection stability across subjects (LOSO)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "fig_stability.png", dpi=200); plt.close(fig)


def _box(ax, x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                                fc=fc, ec="#333", lw=1.2))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=9)


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                 color="#333", lw=1.4))


def fig_method():
    fig, ax = plt.subplots(figsize=(10.5, 3.6)); ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.axis("off")
    ax.text(6, 3.8, "Dependency-aware, label-free channel selection", ha="center",
            fontsize=12, weight="bold")
    _box(ax, 0.1, 2.4, 1.5, 1.0, "Group 1\nchannels", "#dfeefb")
    _box(ax, 0.1, 0.5, 1.5, 1.0, "Group G\nchannels", "#dfeefb")
    ax.text(0.85, 2.0, "·  ·  ·", ha="center", fontsize=12)
    _box(ax, 2.1, 2.4, 1.6, 1.0, "per-group\nencoder", "#e8f6ef")
    _box(ax, 2.1, 0.5, 1.6, 1.0, "per-group\nencoder", "#e8f6ef")
    _box(ax, 4.2, 1.45, 1.4, 1.0, "mean\nfusion", "#fdecd2")
    _box(ax, 6.1, 1.45, 1.5, 1.0, "latent z\n(perturb)", "#fde0dc")
    _box(ax, 8.1, 1.45, 1.7, 1.0, "decode →\nper-channel\nrecon. change", "#e8f6ef")
    _box(ax, 10.3, 1.45, 1.5, 1.0, "MMR\nselect K", "#ead9f2")
    for y in (2.9, 1.0):
        _arrow(ax, 1.6, y, 2.1, y if y > 1.5 else y)
    _arrow(ax, 3.7, 2.9, 4.5, 2.2); _arrow(ax, 3.7, 1.0, 4.5, 1.7)
    _arrow(ax, 5.6, 1.95, 6.1, 1.95); _arrow(ax, 7.6, 1.95, 8.1, 1.95)
    _arrow(ax, 9.8, 1.95, 10.3, 1.95)
    ax.text(6, 0.08, "Engine (perturbation → influence → MMR) is identical across domains; "
            "only the encoder/decoder changes with the data's axis.",
            ha="center", fontsize=12, color="#555", style="italic")
    fig.tight_layout(); fig.savefig(OUT / "fig_method.png", dpi=150); plt.close(fig)


def fig_crossdomain():
    fig, ax = plt.subplots(figsize=(8.5, 2.8)); ax.axis("off")
    rows = [
        ["concept", "Wearable sensors", "Biomedical imaging"],
        ["group", "body-worn IMU (hand/chest/ankle)", "excitation wavelength"],
        ["channel", "IMU axis (acc/gyro/mag x,y,z)", "emission band"],
        ["regular axis", "1D time", "2D space (H×W)"],
        ["encoder", "Conv1D", "Conv2D / Conv3D"],
        ["selection engine", "identical", "identical"],
    ]
    n = len(rows); colw = [0.22, 0.36, 0.42]
    for i, row in enumerate(rows):
        y = 1 - (i + 1) / (n + 0.5)
        x = 0
        for j, cell in enumerate(row):
            fc = "#34495e" if i == 0 else ("#eafaf1" if (j == 0 or "identical" in cell) else "#f7f9fa")
            tc = "white" if i == 0 else "#222"
            ax.add_patch(plt.Rectangle((x, y), colw[j], 0.14, fc=fc, ec="#ccc"))
            ax.text(x + colw[j]/2, y + 0.07, cell, ha="center", va="center",
                    fontsize=9, color=tc, weight="bold" if i == 0 else "normal")
            x += colw[j]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("One method, two verification domains — only the encoder changes", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "fig_crossdomain.png", dpi=150); plt.close(fig)


def fig_biomed():
    """Biomedical verification: drastic channel reduction maintains/improves accuracy.
    Band counts are placed ON the columns (all vs selected). Numbers are existing
    verified results (Lichens, Collagen sponges)."""
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    groups = ["Lichens", "Collagen sponges"]
    full = [88.2, 79.8]; full_bands = [192, 158]   # all channels
    sel = [89.4, 85.6];  sel_bands = [9, 30]        # selected subset
    x = np.arange(2); w = 0.36
    ax.bar(x - w/2, full, w, color="#95a5a6", label="All channels")
    ax.bar(x + w/2, sel, w, color="#c0392b", label="Selected subset")
    for i in range(2):
        ax.text(x[i] - w/2, full[i] + 0.6, f"{full[i]:.1f}%", ha="center", fontsize=8)
        ax.text(x[i] + w/2, sel[i] + 0.6, f"{sel[i]:.1f}%", ha="center", fontsize=8, color="#c0392b")
        # band counts ON the columns (mid-bar, clear of the legend)
        ax.text(x[i] - w/2, 72, f"{full_bands[i]}\nbands", ha="center", va="center",
                fontsize=8, color="white", weight="bold")
        ax.text(x[i] + w/2, 72, f"{sel_bands[i]}\nbands", ha="center", va="center",
                fontsize=8, color="white", weight="bold")
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=8)
    ax.set_ylabel("Classification accuracy (%)"); ax.set_ylim(60, 100)
    ax.set_title("Biomedical imaging: fewer channels, same-or-better accuracy", fontsize=9)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.9); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "fig_biomed.png", dpi=200); plt.close(fig)


def fig_har_selection():
    """Representative K=10 selection on PAMAP2, shown on the 3 IMU x 9 axis grid.
    Illustrates that the method spreads picks across body locations and sensor types."""
    groups = ["Hand", "Chest", "Ankle"]
    chans = ["acc x", "acc y", "acc z", "gyro x", "gyro y", "gyro z", "mag x", "mag y", "mag z"]
    # representative K=10 selection (group, channel-index)
    sel = {("Hand", 6), ("Hand", 7), ("Chest", 0), ("Chest", 3), ("Chest", 4),
           ("Chest", 6), ("Chest", 7), ("Chest", 8), ("Ankle", 1), ("Ankle", 8)}
    grid = np.zeros((3, 9))
    for gi, g in enumerate(groups):
        for c in range(9):
            grid[gi, c] = 1.0 if (g, c) in sel else 0.0
    fig, ax = plt.subplots(figsize=(7.6, 2.7))
    ax.imshow(grid, cmap="Greens", vmin=0, vmax=1.4, aspect="auto")
    ax.set_xticks(range(9)); ax.set_xticklabels(chans, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(3)); ax.set_yticklabels(groups)
    for gi in range(3):
        for c in range(9):
            if grid[gi, c]:
                ax.text(c, gi, "✓", ha="center", va="center", color="#145a32",
                        fontsize=12, weight="bold")
    ax.set_xticks(np.arange(-0.5, 9, 1), minor=True); ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax.grid(which="minor", color="#ccc", lw=1); ax.tick_params(which="minor", length=0)
    ax.set_title("Representative K=10 selection (spread across IMUs and sensor types)", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "fig_har_selection.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    fig_acc_vs_k(); fig_stability(); fig_method(); fig_crossdomain(); fig_biomed()
    fig_har_selection()
    print("wrote figures to", OUT)
    for p in sorted(OUT.glob("*.png")):
        print(" ", p, p.stat().st_size, "bytes")
