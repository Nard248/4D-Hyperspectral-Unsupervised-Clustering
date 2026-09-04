"""Figure for the EEG (BCI-IV-2a) verification. Numbers from run_bci_loso.py
(within-subject, session 1 -> 2, CSP+LDA macro-F1, mean over 9 subjects).

Run: python experiments/eeg/make_eeg_figure.py
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

K = np.array([3, 4, 6, 8, 10])
SERIES = {
    "Proposed (label-free CAE)": ([0.399, 0.426, 0.517, 0.533, 0.544], "#c0392b", "o"),
    "Mutual-info (SUPERVISED)":  ([0.419, 0.456, 0.527, 0.540, 0.561], "#2c7fb8", "s"),
    "PCA (unsupervised)":        ([0.340, 0.412, 0.484, 0.538, 0.568], "#8e44ad", "v"),
    "Variance (unsupervised)":   ([0.372, 0.402, 0.466, 0.525, 0.524], "#f39c12", "^"),
    "Random-K":                  ([0.396, 0.427, 0.486, 0.501, 0.535], "#7f8c8d", "D"),
}
CEILING = 0.578


def main():
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    ax.axhline(CEILING, ls="--", color="#2ecc71", lw=1.5, label=f"All 22 electrodes = {CEILING:.2f}")
    for name, (m, color, mk) in SERIES.items():
        ax.plot(K, m, marker=mk, color=color, lw=1.6, ms=4, label=name)
    ax.set_xlabel("Number of retained electrodes (K, of 22)")
    ax.set_ylabel("Macro-F1 (within-subject)")
    ax.set_title("BCI-IV-2a motor imagery")
    ax.set_xticks(K); ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    for d in (OUT, PAPER):
        fig.savefig(d / "fig_eeg.png", dpi=200)
    plt.close(fig)
    print("wrote fig_eeg.png")


if __name__ == "__main__":
    main()
