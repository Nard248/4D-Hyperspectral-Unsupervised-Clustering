"""EEG row: (a) electrode montage with the 4 region groups, (b) power vs relevance vs picks
topomaps, (c) macro-F1 vs K."""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Polygon, Patch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from poster_style import *   # noqa
import mne; mne.set_log_level("ERROR")
from mne.channels.layout import _find_topomap_coords

d = json.load(open(DATA / "eeg_selections.json"))
CH, GROUPS, names, subs = d["channels"], d["groups"], d["flat_names"], d["subjects"]
SPH = 0.095
info = mne.create_info(CH, 250.0, "eeg"); info.set_montage("standard_1020")
POS = _find_topomap_coords(info, picks=list(range(22)), sphere=SPH)      # true positions (interpolated maps)
# evenly spaced schematic grid for the dot maps, so neighbouring circles never touch
_COL = {"Fz": 0, "FC3": -2, "FC1": -1, "FCz": 0, "FC2": 1, "FC4": 2, "C5": -3, "C3": -2, "C1": -1, "Cz": 0, "C2": 1, "C4": 2, "C6": 3,
        "CP3": -2, "CP1": -1, "CPz": 0, "CP2": 1, "CP4": 2, "P1": -1, "Pz": 0, "P2": 1, "POz": 0}
_ROW = {"Fz": 0.058, "FC3": 0.03, "FC1": 0.03, "FCz": 0.03, "FC2": 0.03, "FC4": 0.03, "C5": 0.0, "C3": 0.0, "C1": 0.0, "Cz": 0.0,
        "C2": 0.0, "C4": 0.0, "C6": 0.0, "CP3": -0.03, "CP1": -0.03, "CPz": -0.03, "CP2": -0.03, "CP4": -0.03, "P1": -0.058, "Pz": -0.058,
        "P2": -0.058, "POz": -0.084}
GRID = np.array([[_COL[c] * 0.0235, _ROW[c]] for c in CH])
R_DOT = 0.0105
flat_to_canon = [CH.index(n) for n in names]
GROUP_LABEL = {"frontal": "frontal: Fz, FC3–FC4", "central": "central: C5–C6",
               "centro_parietal": "centro-parietal: CP3–CP4", "parietal": "parietal: P1, Pz, P2, POz"}
group_of = {i: gi for gi, (g, idxs) in enumerate(GROUPS.items()) for i in idxs}


def head(ax, r=SPH, lw=2.2):
    ax.add_patch(Circle((0, 0), r, fc="white", ec=INK, lw=lw, zorder=1))
    ax.add_patch(Polygon([(-0.17 * r, 0.985 * r), (0, 1.15 * r), (0.17 * r, 0.985 * r)], closed=False,
                         fc="none", ec=INK, lw=lw, zorder=1))
    for s in (-1, 1):
        ax.add_patch(Ellipse((s * 1.04 * r, 0), 0.13 * r, 0.34 * r, fc="none", ec=INK, lw=lw, zorder=1))
    ax.set_xlim(-1.17 * r, 1.17 * r); ax.set_ylim(-1.12 * r, 1.2 * r); ax.set_aspect("equal"); despine_all(ax)


def to_canon(v):
    out = np.zeros(22); out[flat_to_canon] = np.asarray(v, float); return out


def norm01(v):
    v = np.asarray(v, float); return (v - v.min()) / (v.max() - v.min() + 1e-12)


def fig_montage():
    fig = slot(470, 305)
    ax = fig.add_axes([0.0, 0.0, 0.56, 1.0]); head(ax)
    for i, (x, y) in enumerate(GRID):
        c = C_CLUSTERS[group_of[i]]
        ax.add_patch(Circle((x, y), R_DOT, fc=c, ec="white", lw=1.2, zorder=3))
        ax.text(x, y, CH[i], ha="center", va="center", fontsize=14, color="white", weight="bold", zorder=4)
    ax.text(0, 1.155 * SPH, "front", ha="center", va="bottom", fontsize=FS_SMALL - 2, color=INK2)
    lx = fig.add_axes([0.56, 0.0, 0.44, 1.0]); despine_all(lx)
    handles = [Patch(fc=C_CLUSTERS[gi], label=GROUP_LABEL[g]) for gi, g in enumerate(GROUPS)]
    lx.legend(handles=handles, loc="center left", fontsize=FS_SMALL - 2, frameon=False, handlelength=1.1,
              labelspacing=1.0, borderaxespad=0, handletextpad=0.5)
    save(fig, "fig_eeg_montage.png")


def fig_topomaps():
    power = np.mean([norm01(np.log(np.asarray(s["scores"]["variance"]))) for s in subs.values()], axis=0)
    relev = np.mean([norm01(s["scores"]["ours"]) for s in subs.values()], axis=0)
    def freq(m, k=6):
        f = np.zeros(22)
        for s in subs.values():
            for n in s["orders"][m][:k]: f[CH.index(n)] += 1
        return f
    f_ours, f_pca = freq("ours"), freq("pca")
    fig = slot(900, 305)
    titles = ["Signal power (log variance)\nwhat PCA & variance see", "Label-free relevance\n(reconstruction change)",
              "Picked by Proposed, K = 6\n(darker = more subjects)", "Picked by PCA, K = 6\n(darker = more subjects)"]
    axes = [fig.add_axes([0.005 + i * 0.25, 0.0, 0.24, 0.80]) for i in range(4)]
    for ax, t in zip(axes, titles):
        ax.set_title(t, fontsize=FS_SMALL - 1, pad=6, weight="normal")
    for ax, vals in zip(axes[:2], [to_canon(power), to_canon(relev)]):
        mne.viz.plot_topomap(vals, info, axes=ax, show=False, cmap=SEQ_CMAP, contours=4, sensors=False,
                             sphere=SPH, outlines="head", res=96, vlim=(0, 1))
        ax.scatter(POS[:, 0], POS[:, 1], s=14, c=INK, zorder=5, lw=0)
        for line in ax.lines: line.set_linewidth(2.0); line.set_color(INK)
    for ax, f, col in zip(axes[2:], [f_ours, f_pca], [C_OURS, C_PCA]):
        head(ax)
        for i, (x, y) in enumerate(GRID):
            a = float(np.clip(0.08 + 0.92 * f[i] / 9.0, 0, 1))
            ax.add_patch(Circle((x, y), R_DOT, fc=col, alpha=a, ec=INK2, lw=0.8, zorder=3))
            if f[i] >= 5:      # name the electrodes picked for most subjects, inside their circle
                ax.text(x, y, CH[i], ha="center", va="center", fontsize=13, weight="bold",
                        color="white" if a >= 0.55 else INK, zorder=4)
    save(fig, "fig_eeg_topomaps.png")


def fig_curve():
    K = np.array([3, 4, 6, 8, 10])
    S = {"Proposed (label-free)": [0.399, 0.426, 0.517, 0.533, 0.544],
         "Supervised MI":         [0.419, 0.456, 0.527, 0.540, 0.561],
         "PCA":                   [0.340, 0.412, 0.484, 0.538, 0.568],
         "Variance":              [0.372, 0.402, 0.466, 0.525, 0.524],
         "Random-K":              [0.396, 0.427, 0.486, 0.501, 0.535]}
    CEIL = 0.578
    fig = slot(654, 305); ax = fig.add_subplot(111)
    ax.axvspan(5.6, 6.4, color=GOLD, alpha=0.35, lw=0, zorder=0)
    ax.axhline(CEIL, ls=(0, (6, 4)), color=C_FULL, lw=2, zorder=1)
    ax.text(2.75, CEIL + 0.006, "all 22 electrodes = 0.58", ha="left", va="bottom", fontsize=FS_SMALL, color=INK)
    for name, vals in S.items():
        c, mk, ls = METHOD_STYLE[name]
        ax.plot(K, vals, marker=mk, color=c, ls=ls, label=name, zorder=3, markeredgecolor="white", markeredgewidth=1.5)
    ax.annotate("K = 6: 0.52 = 89% of\nthe full-set score", xy=(6, 0.517), xytext=(6.4, 0.385), fontsize=FS_SMALL,
                color=C_OURS, ha="left", arrowprops=dict(arrowstyle="-", color=C_OURS, lw=1.5))
    ax.set_xlabel("Number of retained electrodes, K (of 22)"); ax.set_ylabel("Macro-F1")
    ax.set_xticks(K); ax.set_ylim(0.32, 0.64); ax.set_xlim(2.6, 10.4); ax.set_yticks([0.4, 0.5, 0.6])
    fig.legend(loc="lower center", ncol=3, handlelength=2.0, fontsize=FS_SMALL - 1, frameon=False, columnspacing=1.4, handletextpad=0.5, bbox_to_anchor=(0.56, 0.0))
    fig.subplots_adjust(left=0.16, right=0.98, top=0.95, bottom=0.40)
    save(fig, "fig_eeg_curve.png")


if __name__ == "__main__":
    fig_montage(); fig_topomaps(); fig_curve()
