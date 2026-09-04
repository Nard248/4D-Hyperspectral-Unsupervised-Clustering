"""Biomedical ME-HSI row: (a) dataset explainer tiles, (b) EEM grids with selected bands,
(c) full-set vs selected accuracy bars."""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Patch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
from poster_style import *   # noqa

REPO = ROOT.parents[2]
L = np.load(DATA / "hsi_lichens.npz"); C = np.load(DATA / "hsi_collagen.npz")
PHOTO = REPO / "publications/iasim_poster/02_lichens_TPAMI/01_lichens_sample_RGB.png"
SEL_L = json.load(open(REPO / "results/Lichens_Dataset_1_MasterRun/experiments/bands_9_pca_dim_3_abso_mag_medium_max/wavelengths.json"))
SEL_C = json.load(open(REPO / "results/Collagen_Pepsin_Normalized/experiments/bands_30_var_dim_3_perc_mag_medium_non/wavelengths.json"))
CLASS_COLORS = C_CLUSTERS


def content_box(D, pad=6):
    """Bounding box (r0, r1, c0, c1) of the valid pixels in the downsampled frame."""
    step = int(D["ds_step"]); valid = D["valid_mask"][::step, ::step] > 0
    rows, cols = np.where(valid)
    return (max(rows.min() - pad, 0), min(rows.max() + pad, valid.shape[0]), max(cols.min() - pad, 0), min(cols.max() + pad, valid.shape[1]))


def band_image(D, ex, em):
    cube, wl = D[f"cube_{int(ex)}"], D[f"wl_{int(ex)}"]
    j = int(np.argmin(np.abs(wl - em)))
    img = cube[:, :, j].astype(float)
    step = int(D["ds_step"]); valid = D["valid_mask"][::step, ::step] > 0
    img[~valid] = np.nan
    lo, hi = np.nanpercentile(img, [1, 99.5])
    r0, r1, c0, c1 = content_box(D)
    return np.clip((img - lo) / (hi - lo + 1e-9), 0, 1)[r0:r1, c0:c1], float(wl[j])


def class_image(D):
    step = int(D["ds_step"]); cls = D["class_mask"][::step, ::step]
    rgb = np.ones(cls.shape + (3,))
    for cid in range(1, cls.max() + 1):
        rgb[cls == cid] = np.array([int(CLASS_COLORS[cid - 1][i:i + 2], 16) / 255 for i in (1, 3, 5)])
    r0, r1, c0, c1 = content_box(D)
    return rgb[r0:r1, c0:c1]


def false_color(D, ex, ems):
    chans = [band_image(D, ex, e)[0] for e in ems]           # blue, green, red
    rgb = np.dstack(chans[::-1]); rgb[np.isnan(rgb)] = 1.0
    return rgb


def fig_dataset():
    fig = slot(1180, 305)
    groups = [("Lichens · 192 bands (8 excitations × 22–28 emissions) · 4 classes", L,
               [("photo", np.asarray(Image.open(PHOTO).convert("RGB"))), ("class map", class_image(L)),
                ("band ex 385 · em 500", band_image(L, 385, 500)[0])]),
              ("Collagen sponges · 158 bands (6 × 24–31) · 3 classes", C,
               [("false colour (ex 365)", false_color(C, 365, [450, 520, 600])), ("class map", class_image(C)),
                ("band ex 365 · em 660", band_image(C, 365, 660)[0])])]
    for gi, (header, D, tiles) in enumerate(groups):
        x0 = 0.005 + gi * 0.515
        fig.text(x0 + 0.24, 0.985, header, ha="center", va="top", fontsize=FS_SMALL, weight="bold", color=INK)
        for c, (t, img) in enumerate(tiles):
            ax = fig.add_axes([x0 + c * 0.16, 0.02, 0.15, 0.76]); despine_all(ax)
            ax.set_anchor("N")
            if img.ndim == 2:
                ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            else:
                ax.imshow(img, interpolation="nearest")
            fig.text(x0 + c * 0.16 + 0.075, 0.815, t, ha="center", va="bottom", fontsize=FS_SMALL - 2)
    save(fig, "fig_hsi_dataset.png")


def draw_eem(ax, D, sel, title):
    exs = [float(e) for e in D["excitations"]]
    for i, ex in enumerate(exs):
        wl, cube = D[f"wl_{int(ex)}"], D[f"cube_{int(ex)}"]
        step = int(D["ds_step"]); valid = D["valid_mask"][::step, ::step] > 0
        spec = cube[valid].mean(axis=0); spec = spec / (spec.max() + 1e-12)
        for w, v in zip(wl, spec):
            ax.add_patch(Rectangle((w - 5, i - 0.5), 10, 1, fc=plt.get_cmap(EEM_CMAP)(0.08 + 0.92 * v), ec="white", lw=0.6))
    for s in sel:
        i = exs.index(float(s["excitation"]))
        ax.plot(s["emission"], i, marker="o", ms=11, mfc="white", mec=INK, mew=1.8, zorder=5)
    ax.set_xlim(412, 728); ax.set_ylim(-0.6, len(exs) - 0.4)
    ax.set_yticks(range(len(exs))); ax.set_yticklabels([f"{int(e)}" for e in exs], fontsize=FS_SMALL - 1)
    ax.set_xticks(range(420, 721, 50)); ax.tick_params(axis="x", labelsize=FS_SMALL - 1)
    ax.grid(False); ax.set_facecolor("#f2f2ef")
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title(title, fontsize=FS_SMALL, pad=4, weight="normal", loc="left")
    ax.set_ylabel("excitation (nm)", fontsize=FS_SMALL - 2)


def fig_eem():
    fig = slot(640, 305)
    ax1 = fig.add_axes([0.13, 0.56, 0.85, 0.34]); ax2 = fig.add_axes([0.13, 0.13, 0.85, 0.27])
    draw_eem(ax1, L, SEL_L, "Lichens · white dots = the 9 selected bands")
    draw_eem(ax2, C, SEL_C, "Collagen sponges · white dots = the 30 selected bands")
    ax1.set_xticklabels([]); ax2.set_xlabel("emission (nm)", fontsize=FS_SMALL - 1)
    save(fig, "fig_hsi_eem.png")


def fig_bars():
    fig = slot(444, 305); ax = fig.add_subplot(111)
    groups = ["Lichens", "Collagen sponges"]; full = [88.2, 79.8]; sel = [89.4, 85.6]
    fb = [192, 158]; sb = [9, 30]
    x = np.arange(2); w = 0.36
    ax.bar(x - w / 2 - 0.02, full, w, color="#b9b9b4", label="all bands")
    ax.bar(x + w / 2 + 0.02, sel, w, color=C_OURS, label="selected subset")
    for i in range(2):
        ax.text(x[i] - w / 2 - 0.02, full[i] + 0.4, f"{full[i]:.1f}%", ha="center", va="bottom", fontsize=FS_SMALL, color=INK)
        ax.text(x[i] + w / 2 + 0.02, sel[i] + 0.4, f"{sel[i]:.1f}%", ha="center", va="bottom", fontsize=FS_SMALL, color=C_OURS, weight="bold")
        ax.text(x[i] - w / 2 - 0.02, 72.5, f"{fb[i]}\nbands", ha="center", va="center", fontsize=FS_SMALL - 1, color="white", weight="bold")
        ax.text(x[i] + w / 2 + 0.02, 72.5, f"{sb[i]}\nbands", ha="center", va="center", fontsize=FS_SMALL - 1, color="white", weight="bold")
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=FS_TICK)
    ax.set_ylabel("per-pixel accuracy (%)"); ax.set_ylim(70, 96.5); ax.set_yticks([70, 75, 80, 85, 90, 95]); ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", fontsize=FS_SMALL - 1, ncol=2, borderaxespad=0.1, handlelength=1.2, columnspacing=1.2)
    ax.set_title("95% / 81% of the bands removed,\naccuracy kept or improved", fontsize=FS_TICK, pad=8)
    fig.subplots_adjust(left=0.22, right=0.98, top=0.82, bottom=0.14)
    save(fig, "fig_hsi_bars.png")


if __name__ == "__main__":
    fig_dataset(); fig_eem(); fig_bars()
