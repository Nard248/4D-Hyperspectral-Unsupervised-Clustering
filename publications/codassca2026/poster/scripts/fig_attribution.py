"""Fig. 2: perturbation attribution, shown on the synthetic benchmark (real model output)."""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from poster_style import *   # noqa
from fig_synthetic import role_order, roles_sorted, role_color, bracket_labels, role_strip, CLUSTER_LABELS, factor_of

d = np.load(DATA / "synthetic_attribution.npz")
per_dim, dim_scores, acc, order = d["per_dim"], d["dim_scores"], d["accumulated"], d["order"]
rank = np.argsort(-dim_scores)                        # latent coordinates by variance
NAVY = "#3b4a63"


def main():
    fig = slot(954, 336)
    gs = fig.add_gridspec(2, 3, width_ratios=[0.75, 2.1, 2.0], height_ratios=[1, 0.07], wspace=0.28, hspace=0.05,
                          left=0.06, right=0.99, top=0.80, bottom=0.2)
    ax1 = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[0, 1]); ax3 = fig.add_subplot(gs[0, 2])
    s2 = fig.add_subplot(gs[1, 1]); s3 = fig.add_subplot(gs[1, 2])
    # (1) latent variance ranking
    ax1.bar(np.arange(8), dim_scores[rank], color=NAVY, width=0.72)
    ax1.set_xticks([0, 3, 7]); ax1.set_xticklabels(["1", "4", "8"], fontsize=FS_SMALL)
    ax1.set_ylabel("variance across windows", fontsize=FS_SMALL); ax1.tick_params(axis="y", labelsize=FS_SMALL - 2)
    ax1.set_title("1 · rank latent coordinates\nby variance, keep top m", fontsize=FS_TICK, pad=8)
    ax1.set_xlabel("coordinate rank", fontsize=FS_SMALL)
    # (2) per-coordinate reconstruction change
    M = per_dim[rank][:, role_order]
    ax2.imshow(M / M.max(), aspect="auto", cmap=SEQ_CMAP, interpolation="nearest", vmin=0, vmax=1)
    ax2.set_yticks(np.arange(8)); ax2.set_yticklabels([f"$z_{{{j+1}}}$" for j in range(8)], fontsize=FS_SMALL)
    ax2.set_xticks([]); ax2.grid(False)
    ax2.set_title("2 · shift each coordinate by δ, decode,\nmeasure $|\\Delta\\hat{X}|$ per channel", fontsize=FS_TICK, pad=8)
    ax2.set_ylabel("perturbed coordinate", fontsize=FS_SMALL)
    role_strip(s2, roles_sorted, "h"); bracket_labels(s2, roles_sorted, -0.4, fontsize=FS_SMALL - 2)
    # (3) accumulated relevance + MMR picks
    A = acc[role_order] / acc.max()
    ax3.bar(np.arange(64) + 0.5, A, width=0.85, color=[role_color(r) for r in roles_sorted], lw=0)
    pos = {ch: i for i, ch in enumerate(role_order)}
    for k, ch in enumerate(order[:8]):
        i = pos[ch]
        ax3.plot(i + 0.5, A[i] + 0.07, marker="v", color=INK, ms=11, lw=0, zorder=4)
    ax3.set_xlim(0, 64); ax3.set_ylim(0, 1.3); ax3.set_xticks([]); ax3.set_yticks([0, 0.5, 1.0])
    ax3.tick_params(axis="y", labelsize=FS_SMALL - 2); ax3.set_ylabel("relevance $I_{g,c}$ (norm.)", fontsize=FS_SMALL)
    ax3.set_title("3 · accumulate → relevance; greedy MMR\npicks K = 8 (triangles)", fontsize=FS_TICK, pad=8)
    handles = [Patch(fc=C_CLUSTERS[i], label=CLUSTER_LABELS[i]) for i in range(4)] + [Patch(fc=C_NOISE, label="noise")]
    ax3.legend(handles=handles, loc="upper right", ncol=5, fontsize=FS_SMALL - 3, frameon=False, handlelength=1.0,
               columnspacing=0.8, handletextpad=0.4, borderaxespad=0.0)
    role_strip(s3, roles_sorted, "h"); bracket_labels(s3, roles_sorted, -0.4, fontsize=FS_SMALL - 2)
    save(fig, "fig_attribution.png")


if __name__ == "__main__":
    main()
