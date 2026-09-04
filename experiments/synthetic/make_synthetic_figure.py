"""Figure for the controlled synthetic benchmark (numbers from run_synthetic_recovery.py,
seed 0, locked config: bottleneck CAE, signed latent, latent=8). Across-seed mean at K=8
is 0.70+/-0.29 with 0.88 mean factor coverage -- stated in the caption, not the plot.

Run: python experiments/synthetic/make_synthetic_figure.py
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("publications/generalization/figures"); OUT.mkdir(parents=True, exist_ok=True)
PAPER = Path("publications/generalization/paper/figures"); PAPER.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5})

K = np.array([2, 4, 6, 8, 12, 16, 20])
SERIES = {
    "Proposed (label-free CAE)":  ([0.267, 0.278, 0.494, 0.944, 0.994, 1.000, 1.000], "#c0392b", "o"),
    "Mutual-info (SUPERVISED)":   ([0.272, 0.450, 0.994, 0.994, 0.950, 1.000, 1.000], "#2c7fb8", "s"),
    "Variance (unsupervised)":    ([0.106, 0.139, 0.283, 0.244, 0.583, 0.911, 1.000], "#f39c12", "^"),
    "Random-K":                   ([0.100, 0.140, 0.234, 0.254, 0.295, 0.493, 0.536], "#7f8c8d", "D"),
}
CEILING = 0.872


def main():
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    ax.axhline(CEILING, ls="--", color="#2ecc71", lw=1.5, label=f"All 64 ch. = {CEILING:.2f}")
    for name, (m, color, mk) in SERIES.items():
        ax.plot(K, m, marker=mk, color=color, lw=1.6, ms=4, label=name)
    ax.set_xlabel("Number of retained channels (K, of 64)")
    ax.set_ylabel("Classification accuracy")
    ax.set_title("Synthetic benchmark: 4 factors + 44 noise")
    ax.set_xticks(K); ax.grid(alpha=0.3); ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    fig.tight_layout()
    for d in (OUT, PAPER):
        fig.savefig(d / "fig_synthetic.png", dpi=200)
    plt.close(fig)
    print("wrote fig_synthetic.png to", OUT, "and", PAPER)


if __name__ == "__main__":
    main()
