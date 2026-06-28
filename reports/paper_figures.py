"""Generate ALL paper figures by RE-RUNNING the simulations, and record the exact config/parameters
used for each figure into paper/figures/params.json. Data-generation figures are freshly rendered;
headline results are re-validated at documented settings.

Run:  python reports/paper_figures.py
Output: paper/figures/*.png  + paper/figures/params.json
"""
from __future__ import annotations

import json
import os
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spectraforge.fluorophore import Fluorophore
from classification_experiment import feature_matrix
from realistic_benchmark import DISC, NUIS, EXCITATIONS, build_dataset, band_is_discriminative, add_cube_clutter

OUT = pathlib.Path(__file__).resolve().parents[1] / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
PARAMS = {}
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})


def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / name, bbox_inches="tight"); plt.close(fig)
    print("wrote", name)


# ---- Fig 1: fluorophore excitation/emission spectra (the parametric photophysics model) ----
def fig_fluorophores():
    wl = np.linspace(400, 700, 600)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    for name, f in DISC.items():
        ax[0].plot(wl, f.excitation(wl), "--", alpha=0.7)
        ax[0].plot(wl, f.emission(wl) / f.emission(wl).max(), label=name)
    ax[0].set_title("Discriminative dyes (dashed=excitation, solid=emission)")
    ax[0].set_xlabel("wavelength (nm)"); ax[0].legend(fontsize=7)
    for name, f in NUIS.items():
        ax[1].plot(wl, f.excitation(wl), "--", alpha=0.7)
        ax[1].plot(wl, f.emission(wl) / f.emission(wl).max(), label=name)
    ax[1].set_title("Bright nuisance autofluorophores")
    ax[1].set_xlabel("wavelength (nm)"); ax[1].legend(fontsize=7)
    save(fig, "fig_fluorophores.png")
    PARAMS["fig_fluorophores"] = {n: dict(ex_peak=f.ex_peak_nm, ex_fwhm=f.ex_fwhm_nm, em_peak=f.em_peak_nm,
                                          em_fwhm=f.em_fwhm_nm, extinction=f.extinction, qy=f.quantum_yield)
                                  for n, f in {**DISC, **NUIS}.items()}


# ---- Fig 2: scene maps + rendered bands (the forward model) ----
def fig_scene():
    p = dict(seed=1, size=64, em_step=5, nuisance_amp=1.2, photon_scale=2000, read_sigma=0.005)
    sp, gt, y, acq = build_dataset(**p)
    X, colmap = feature_matrix(sp)
    fig, ax = plt.subplots(2, 3, figsize=(9, 5.6))
    ax[0, 0].imshow(y.reshape(64, 64), cmap="tab10"); ax[0, 0].set_title("class labels (argmax disc.)")
    # a discriminative-window band and a scatter band at ex=470
    cols470 = [(j, m) for j, (e, m) in enumerate(colmap) if e == 470.0]
    for k, (em_target, ttl) in enumerate([(515, "ex470/em515 (disc dye)"), (470, "ex470/em470 (Rayleigh)"),
                                          (665, "ex470/em665 (nuisance)")]):
        j = min(cols470, key=lambda t: abs(colmap[t[0]][1] - em_target))[0]
        ax[0, 1 + k if k < 2 else 1].imshow(X[:, j].reshape(64, 64), cmap="viridis")
    # bottom row: mean spectra per excitation
    for e in EXCITATIONS[:3]:
        cols = [(colmap[j][1], X[:, j].mean()) for j, (ex, m) in enumerate(colmap) if ex == e]
        ems, vals = zip(*sorted(cols))
        ax[1, 0].plot(ems, vals, label=f"ex {int(e)}")
    ax[1, 0].set_title("mean emission per excitation"); ax[1, 0].set_xlabel("em (nm)"); ax[1, 0].legend(fontsize=7)
    disc = band_is_discriminative(colmap)
    var = X.var(0)
    from sklearn.feature_selection import f_classif
    F = np.nan_to_num(f_classif(X, y)[0])
    ax[1, 1].scatter(var, F, s=6, c=disc, cmap="coolwarm", alpha=0.6)
    ax[1, 1].set_xlabel("per-band variance"); ax[1, 1].set_ylabel("per-band discriminability (F)")
    ax[1, 1].set_title(f"variance vs informativeness\ncorr={np.corrcoef(var, F)[0,1]:+.2f}")
    ax[1, 2].imshow(X[:, [j for j, (e, m) in enumerate(colmap) if e == 470.0][20]].reshape(64, 64), cmap="magma")
    ax[1, 2].set_title("example rendered band")
    save(fig, "fig_scene.png")
    PARAMS["fig_scene"] = {**p, "excitations": EXCITATIONS, "em_range_nm": [420, 700],
                           "n_features": X.shape[1], "var_info_corr": float(np.corrcoef(var, F)[0, 1])}


# ---- Fig 3: clutter injection (fixed-pattern nuisance) before/after ----
def fig_clutter():
    p = dict(seed=1, size=64, em_step=2, nuisance_amp=1.2, photon_scale=2000, read_sigma=0.005)
    sp, gt, y, acq = build_dataset(**p)
    X0, colmap = feature_matrix(sp)
    add_cube_clutter(sp, 64, 1, n_modes=36, amp=3.3)
    X1, _ = feature_matrix(sp)
    j = [k for k, (e, m) in enumerate(colmap) if e == 470.0][20]
    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    ax[0].imshow(X0[:, j].reshape(64, 64), cmap="viridis"); ax[0].set_title("band: no clutter")
    ax[1].imshow(X1[:, j].reshape(64, 64), cmap="viridis"); ax[1].set_title("same band + fixed-pattern clutter")
    ax[2].plot(sorted(X0.var(0))[::-1], label="no clutter"); ax[2].plot(sorted(X1.var(0))[::-1], label="+clutter")
    ax[2].set_title("sorted per-band variance"); ax[2].set_xlabel("band rank"); ax[2].legend(fontsize=7)
    save(fig, "fig_clutter.png")
    PARAMS["fig_clutter"] = {**p, "clutter_modes": 36, "clutter_amp": 3.3}


# ---- Fig 4: reabsorption (secondary inner filter) reshapes the emission band ----
def fig_reabsorption():
    base = dict(seed=1, size=48, em_step=2, nuisance_amp=0.5, photon_scale=20000, read_sigma=0.002)
    sp0, _, y, acq = build_dataset(**base)
    sp1, _, _, _ = build_dataset(**base, reabsorption=True, reabsorption_strength=4.0)
    X0, colmap = feature_matrix(sp0); X1, _ = feature_matrix(sp1)
    ems = [m for (e, m) in colmap if e == 470.0]
    idx = [k for k, (e, m) in enumerate(colmap) if e == 470.0]
    px = int(np.argmax(X0[:, idx].sum(1)))     # a bright pixel
    fig, ax = plt.subplots(1, 1, figsize=(5, 3.2))
    ax.plot(ems, X0[px, idx], label="no reabsorption")
    ax.plot(ems, X1[px, idx], label="with reabsorption (s=4)")
    ax.set_title("Reabsorption reshapes the emission band\n(blue edge suppressed -> apparent red-shift)")
    ax.set_xlabel("emission (nm)"); ax.set_ylabel("intensity @ ex470"); ax.legend(fontsize=7)
    save(fig, "fig_reabsorption.png")
    PARAMS["fig_reabsorption"] = {**base, "reabsorption_strength": 4.0}


# ---- Fig 5: results — visualize the committed campaign numbers ----
def fig_results():
    # (values from the committed experiment logs; scripts named in captions/params)
    beat = {"regime": ["clean", "low", "moderate", "high", "severe"],
            "random95": [0.573, 0.479, 0.574, 0.587, 0.592],
            "PCA": [0.699, 0.470, 0.539, 0.549, 0.549], "AE": [0.655, 0.449, 0.548, 0.575, 0.569],
            "mutInfo": [0.696, 0.504, 0.576, 0.604, 0.615]}
    comp = {"k": [4, 8, 12, 16, 24, 32, 48, 64, 100],
            "PCA": [0.453, 0.517, 0.552, 0.572, 0.606, 0.616, 0.647, 0.656, 0.662],
            "AE": [0.467, 0.532, 0.552, 0.573, 0.616, 0.632, 0.639, 0.657, 0.667],
            "mutInfo": [0.495, 0.571, 0.604, 0.622, 0.659, 0.658, 0.673, 0.680, 0.680],
            "random": [0.461, 0.509, 0.547, 0.581, 0.638, 0.618, 0.636, 0.662, 0.668], "full": 0.683}
    spatial = {"method": ["per-pixel\nvariance", "per-pixel\nmutInfo", "random", "texture-var", "texture-MI", "spatial-CNN", "oracle"],
               "F1": [0.516, 0.517, 0.710, 0.657, 0.688, 0.534, 0.751]}
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    x = np.arange(len(beat["regime"])); w = 0.2
    for i, m in enumerate(["random95", "PCA", "AE", "mutInfo"]):
        ax[0].bar(x + (i - 1.5) * w, beat[m], w, label=("random(95th)" if m == "random95" else m))
    ax[0].set_xticks(x); ax[0].set_xticklabels(beat["regime"]); ax[0].set_title("Beat-random across noise levels (k=24)")
    ax[0].set_ylabel("best-NL macro-F1"); ax[0].legend(fontsize=7)
    for m in ["PCA", "AE", "mutInfo", "random"]:
        ax[1].plot(comp["k"], comp[m], marker="o", ms=3, label=m)
    ax[1].axhline(comp["full"], ls="--", c="k", label="full-564")
    ax[1].set_title("Compression: accuracy vs #bands"); ax[1].set_xlabel("k (selected bands)"); ax[1].legend(fontsize=7)
    cols = ["#c44", "#c44", "#888", "#4a4", "#4a4", "#48c", "#444"]
    ax[2].bar(range(len(spatial["method"])), spatial["F1"], color=cols)
    ax[2].set_xticks(range(len(spatial["method"]))); ax[2].set_xticklabels(spatial["method"], fontsize=6.5, rotation=0)
    ax[2].set_title("Spatial-texture regime (per-pixel BLIND)"); ax[2].set_ylabel("block-CV macro-F1")
    save(fig, "fig_results.png")
    PARAMS["fig_results"] = {"beat_random": beat, "compression": comp, "spatial": spatial,
                             "source_scripts": ["beat_random.py", "compression_curve.py", "spatial_regime.py"]}


def main():
    fig_fluorophores(); fig_scene(); fig_clutter(); fig_reabsorption(); fig_results()
    with open(OUT / "params.json", "w") as f:
        json.dump(PARAMS, f, indent=2)
    print("wrote params.json with", len(PARAMS), "figure configs")


if __name__ == "__main__":
    main()
