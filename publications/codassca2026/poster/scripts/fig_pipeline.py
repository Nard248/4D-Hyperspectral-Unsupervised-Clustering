"""Fig. 1: the three-stage pipeline with real mini-visuals from the synthetic benchmark.
Simplified layout: every element sits inside its stage band with padding; no decorative line."""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from poster_style import *   # noqa
from fig_synthetic import role_order, roles_sorted, role_color, factor_of

d = np.load(DATA / "synthetic_attribution.npz")
X0, latent, per_dim, dim_scores, acc, order = (d["X0"], d["latent"], d["per_dim"], d["dim_scores"], d["accumulated"], d["order"])
W, H = 1334 / 72, 336 / 72
C_ENC, C_FUS, C_LAT, C_MMR = "#e8f6ef", "#fdecd2", "#fde0dc", "#ead9f2"
FS_BOX, FS_LAB, FS_HEAD = 18, 17, 20


def box(ax, x, y, w, h, text, fc, fs=FS_BOX, ec="#333", lw=1.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.1", fc=fc, ec=ec, lw=lw, zorder=2))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=3, linespacing=1.15)


def arrow(ax, x1, y1, x2, y2, lw=2.0, color="#333"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15, color=color, lw=lw, zorder=4))


def inset(fig, x, y, w, h):
    ax = fig.add_axes([x / W, y / H, w / W, h / H]); ax.set_zorder(5); return ax


def tile(ax, M):
    v = np.abs(M).max(); ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
    despine_all(ax); ax.set_facecolor("white")
    for s in ax.spines.values(): s.set_visible(True); s.set_edgecolor("#555"); s.set_linewidth(1.0)


def bars(ax, v, colors, ylim=None):
    ax.bar(np.arange(len(v)), v, color=colors, width=0.9 if len(v) > 20 else 0.7, lw=0); despine_all(ax)
    if ylim: ax.set_ylim(*ylim)


def main():
    fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    bands = [(0.10, 8.85, "Stage 1 · learn the joint structure of all channels (no labels)", SAGE_LIGHT),
             (9.10, 5.20, "Stage 2 · perturbation attribution", GOLD),
             (14.45, 3.98, "Stage 3 · MMR shortlist", "#e6dff0")]
    for x, w, t, c in bands:
        ax.add_patch(FancyBboxPatch((x, 0.12), w, 4.48, boxstyle="round,pad=0.0,rounding_size=0.15", fc=c, ec="none", alpha=0.45, zorder=0))
        ax.text(x + w / 2, 4.36, t, ha="center", va="center", fontsize=FS_HEAD, weight="bold", zorder=3)
    # ---------------- Stage 1
    for yc, g, lab in [(3.2, 0, "Group 1 · $C_1$ channels × $T$"), (1.25, 3, "Group $G$")]:
        a = inset(fig, 0.4, yc - 0.425, 1.3, 0.85); tile(a, X0[:, g * 16:(g + 1) * 16].T)
        ax.text(0.4, yc + 0.47, lab, ha="left", va="bottom", fontsize=FS_LAB)
        box(ax, 2.0, yc - 0.425, 1.6, 0.85, "Encoder $E_g$\nConv1D / Conv2D", C_ENC, fs=16)
        arrow(ax, 1.72, yc, 1.98, yc)
        arrow(ax, 3.62, yc, 3.93, 2.22 + (0.2 if yc > 2 else -0.2))
    for yy in (2.06, 2.22, 2.38):
        ax.add_patch(plt.Circle((1.05, yy), 0.028, color="#444", zorder=3))
    box(ax, 3.95, 1.8, 1.15, 0.85, "mean\nfusion", C_FUS)
    arrow(ax, 5.12, 2.22, 5.43, 2.22)
    box(ax, 5.45, 1.47, 1.4, 1.5, "", C_LAT)
    ax.text(6.15, 3.02, "latent $z\\in\\mathbb{R}^d$", ha="center", va="bottom", fontsize=FS_LAB)
    a = inset(fig, 5.55, 1.57, 1.2, 1.3); bars(a, latent[0], "#3b4a63"); a.axhline(0, color="#777", lw=0.8)
    ax.text(6.15, 1.28, "no time / space extent", ha="center", va="center", fontsize=15, color=INK2, style="italic")
    for yc in (3.2, 1.25):
        arrow(ax, 6.87, 2.22 + (0.2 if yc > 2 else -0.2), 7.18, yc)
        box(ax, 7.15, yc - 0.425, 1.72, 0.85, "Decoder $D_g$ → $\\hat{X}_g$", C_ENC, fs=16)
    ax.text(4.5, 0.42, "trained on reconstruction only, no labels:   $\\mathcal{L}=\\sum_g \\| X_g - D_g(z)\\|^2$", ha="center", va="center", fontsize=FS_LAB)
    # ---------------- Stage 2
    j = int(np.argmax(dim_scores))
    cols = ["#3b4a63"] * 8; cols[j] = C_OURS
    a = inset(fig, 9.35, 2.65, 1.2, 1.0); bars(a, latent[0], cols); a.axhline(0, color="#777", lw=0.8)
    a.annotate("", xy=(j, latent[0][j] + 0.9 * np.abs(latent[0]).max()), xytext=(j, latent[0][j]),
               arrowprops=dict(arrowstyle="-|>", color=C_OURS, lw=2))
    ax.text(9.95, 3.72, "shift $z_j$ by $\\delta$", ha="center", va="bottom", fontsize=FS_LAB)
    arrow(ax, 10.6, 3.15, 10.85, 3.15); box(ax, 10.87, 2.87, 0.85, 0.56, "decode", C_ENC, fs=16); arrow(ax, 11.74, 3.15, 12.0, 3.15)
    a = inset(fig, 12.05, 2.65, 2.0, 1.0); v = per_dim[j][role_order]; bars(a, v / v.max(), [role_color(r) for r in roles_sorted])
    ax.text(13.05, 3.72, "$|\\Delta\\hat{X}|$ per channel", ha="center", va="bottom", fontsize=FS_LAB)
    ax.text(11.7, 2.33, "accumulate over $j$ and $\\delta$", ha="center", va="center", fontsize=FS_LAB)
    arrow(ax, 11.7, 2.17, 11.7, 1.95)
    a = inset(fig, 9.4, 0.75, 4.6, 1.15); A = acc[role_order] / acc.max(); bars(a, A, [role_color(r) for r in roles_sorted], (0, 1.12))
    ax.text(11.7, 0.45, "relevance $I_{g,c}$: informative high, noise low", ha="center", va="center", fontsize=16)
    # ---------------- Stage 3
    box(ax, 14.65, 3.1, 3.58, 0.95, "greedy MMR:\n$c^\\star=\\arg\\max\\,[\\tilde I_{g,c}-\\lambda\\cdot\\max\\,\\mathrm{sim}]$", C_MMR, fs=16)
    arrow(ax, 16.44, 3.08, 16.44, 2.86)
    for k, ch in enumerate(order[:8]):
        yy = 2.7 - k * 0.3
        ax.add_patch(FancyBboxPatch((14.7, yy - 0.13), 1.65, 0.26, boxstyle="round,pad=0.0,rounding_size=0.08",
                                    fc=role_color(factor_of[ch]), ec="none", alpha=0.95, zorder=2))
        ax.text(14.8, yy, f"{k+1}", ha="left", va="center", fontsize=15, color="white", weight="bold", zorder=3)
        ax.text(15.12, yy, f"g{ch // 16 + 1} · ch {ch % 16 + 1}", ha="left", va="center", fontsize=15, color="white", zorder=3)
    ax.text(17.4, 1.9, "K = 8 →\nall 4 factors,\nno noise", ha="center", va="center", fontsize=FS_LAB, linespacing=1.25)
    ax.text(17.4, 0.75, "ordered list:\nany budget K", ha="center", va="center", fontsize=15, style="italic", linespacing=1.2)
    for x in (8.95, 14.3):
        arrow(ax, x - 0.07, 2.36, x + 0.22, 2.36, lw=3.0, color=INK)
    save(fig, "fig_pipeline.png", pad=0.0)


if __name__ == "__main__":
    main()
