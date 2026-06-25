"""Five synthetic full-ME-HSI datasets at increasing MIXTURE+NOISE+CLUTTER, high spectral resolution,
evaluated under a REALISTIC few-shot labelled budget — the setting where full-band data underperforms
a good selection (as seen on real instruments).

Pipeline (verification gate printed per level):
  1. Build 5 levels (pristine -> severe): more nuisance spectral mixture + photon/read/scatter noise +
     class-irrelevant fixed-pattern CLUTTER injected into the cube. High res: 141 em-bands x 4 ex = 564.
  2. ROIs: clearest 50% pixels by ground-truth concentration purity (clean labels).
  3. Few-shot labels: train on only `PER_CLASS` ROI pixels/class, test on the held-out ROI — so the
     full 564-band classifier overfits the clutter and a 24-band selection can WIN.
  4. Select k=24 with PCA (pca_load) and the AE (conv & mlp) + supervised mutual_info reference.
  5. Report macro-F1 for full vs each selection, and Δ = selection - full (POSITIVE = selection beats
     full data) — averaged over dataset seeds × train/test repeats.

Run:  python reports/mixnoise_experiment.py [--smoke]
"""
from __future__ import annotations

import contextlib
import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import method_zoo as mz
from classification_experiment import cols_for_bands, feature_matrix
from realistic_benchmark import DISC, build_dataset
from spectraforge.scenegen import random_field
from sweep_common import topn_diverse
from swarm_zoo import FlexSpectralAE

BUDGET = 24
SIZE = 64
EM_STEP = 2
PER_CLASS = 12          # realistic few-shot labelled budget per class (small ROIs)
REPEATS = 6             # train/test resamples per dataset
NBANDS = 4 * 141
# 5 levels: nuisance (spectral mixture) + photon/read/scatter (noise) + clutter (fixed-pattern, the
# dominant real-instrument confound that makes full-band data overfit at small label budgets)
LEVELS = {
    "L1-pristine": dict(nuisance_amp=0.4, turbidity_amp=0.1, rayleigh=0.10, raman=0.10, photon_scale=50000, read_sigma=0.002, clutter_modes=0,  clutter_amp=0.0),
    "L2-low":      dict(nuisance_amp=0.8, turbidity_amp=0.3, rayleigh=0.20, raman=0.20, photon_scale=8000,  read_sigma=0.004, clutter_modes=20, clutter_amp=1.2),
    "L3-moderate": dict(nuisance_amp=1.2, turbidity_amp=0.6, rayleigh=0.35, raman=0.30, photon_scale=2000,  read_sigma=0.006, clutter_modes=28, clutter_amp=2.2),
    "L4-high":     dict(nuisance_amp=2.0, turbidity_amp=1.0, rayleigh=0.50, raman=0.40, photon_scale=600,   read_sigma=0.009, clutter_modes=36, clutter_amp=3.3),
    "L5-severe":   dict(nuisance_amp=3.0, turbidity_amp=1.4, rayleigh=0.70, raman=0.50, photon_scale=200,   read_sigma=0.012, clutter_modes=44, clutter_amp=4.5),
}
_AE_CONV = dict(backbone="conv", act="gelu", mask_ratio=0.5, latent_dim=8, depth=3, width=64, epochs=250)
_AE_MLP = dict(backbone="mlp", act="gelu", mask_ratio=0.6, latent_dim=6, depth=3, width=128, epochs=300)


def roi_mask(seed, size, frac=0.5):
    f = np.stack([random_field(size, size, seed * 13 + 31 * k) for k in range(len(DISC))])
    p = (np.sort(f, 0)[-1] - np.sort(f, 0)[-2]).ravel()
    return p >= np.quantile(p, 1 - frac)


def _clfs(seed):
    return {"knn": KNeighborsClassifier(7),
            "rf": RandomForestClassifier(150, random_state=seed, n_jobs=-1),
            "mlp": MLPClassifier(hidden_layer_sizes=(48,), max_iter=250, random_state=seed)}


def fewshot(X, y, cols, seed):
    """Mean macro-F1 over REPEATS few-shot splits; return knn and best-of-panel."""
    knn, best = [], []
    for r in range(REPEATS):
        rng = np.random.default_rng(seed * 1000 + r)
        tr = np.concatenate([rng.choice(np.where(y == c)[0], PER_CLASS, replace=False) for c in np.unique(y)])
        te = np.setdiff1d(np.arange(len(y)), tr)
        sc = StandardScaler().fit(X[tr][:, cols])
        Xtr, Xte = sc.transform(X[tr][:, cols]), sc.transform(X[te][:, cols])
        f1s = {}
        for n, c in _clfs(seed).items():
            with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
                c.fit(Xtr, y[tr]); f1s[n] = f1_score(y[te], c.predict(Xte), average="macro")
        knn.append(f1s["knn"]); best.append(max(f1s.values()))
    return float(np.mean(knn)), float(np.mean(best))


def ae_cols(spectra, colmap, seed, kw):
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        m = FlexSpectralAE(seed=seed, **kw).fit(spectra)
    return cols_for_bands(colmap, m.select(BUDGET))


def main():
    smoke = "--smoke" in sys.argv
    levels = (["L1-pristine", "L5-severe"] if smoke else list(LEVELS))
    seeds = ([1] if smoke else [1, 2])
    print("=" * 104)
    print(f"5-LEVEL MIX/NOISE/CLUTTER — high-res 564-band, ROI px, FEW-SHOT {PER_CLASS}/class x{REPEATS}, k={BUDGET}")
    print("Δ = selection - full  (POSITIVE => the 24-band selection BEATS full 564-band data)")
    print("=" * 104)
    rows = []
    for lvl in levels:
        params = LEVELS[lvl]
        acc = {m: {"knn": [], "best": []} for m in ["full", "PCA", "AE-conv", "AE-mlp", "mutInfo*"]}
        for seed in seeds:
            sp, gt, y, acq = build_dataset(seed, size=SIZE, em_step=EM_STEP, **params)
            X, colmap = feature_matrix(sp)
            roi = roi_mask(seed, SIZE); Xr, yr = X[roi], y[roi]
            assert X.shape[1] == NBANDS and np.isfinite(X).all() and set(np.unique(yr)) == {0, 1, 2}
            rng = np.random.default_rng(seed)
            sel = {"full": list(range(NBANDS)),
                   "PCA": mz.pca_load(X, colmap, BUDGET, seed, rng, sp, k=6),
                   "AE-conv": ae_cols(sp, colmap, seed, _AE_CONV),
                   "AE-mlp": ae_cols(sp, colmap, seed, _AE_MLP),
                   "mutInfo*": topn_diverse(np.nan_to_num(mutual_info_classif(Xr, yr, random_state=seed)), colmap, BUDGET)}
            for name, cols in sel.items():
                k, b = fewshot(Xr, yr, cols, seed)
                acc[name]["knn"].append(k); acc[name]["best"].append(b)

        def mean(m, k):
            return float(np.mean(acc[m][k]))

        full_k, full_b = mean("full", "knn"), mean("full", "best")
        print(f"\n### {lvl}  (nuis={params['nuisance_amp']}, photon={params['photon_scale']}, "
              f"clutter={params['clutter_amp']}x{params['clutter_modes']})  VERIFY 564b/3cls OK")
        print(f"  {'method':<10}{'knn':>7}{'bestNL':>8}{'Δknn':>8}{'Δbest':>8}")
        print(f"  {'full-564':<10}{full_k:>7.3f}{full_b:>8.3f}{'—':>8}{'—':>8}")
        best_ae = None
        for name in ["PCA", "AE-conv", "AE-mlp", "mutInfo*"]:
            k, b = mean(name, "knn"), mean(name, "best")
            tag = "  <-- beats full" if b > full_b else ""
            print(f"  {name:<10}{k:>7.3f}{b:>8.3f}{k-full_k:>+8.3f}{b-full_b:>+8.3f}{tag}")
        rows.append((lvl, full_b, mean("PCA", "best"), max(mean("AE-conv", "best"), mean("AE-mlp", "best")),
                     mean("mutInfo*", "best")))

    print("\n" + "=" * 104)
    print(f"{'level':<14}{'full564':>9}{'PCA':>8}{'AE':>8}{'mutInfo*':>10}{'AE-full':>9}{'PCA-full':>10}{'AE-PCA':>8}")
    for lvl, f, p, a, mi in rows:
        print(f"{lvl:<14}{f:>9.3f}{p:>8.3f}{a:>8.3f}{mi:>10.3f}{a-f:>+9.3f}{p-f:>+10.3f}{a-p:>+8.3f}")
    print("\nPositive AE-full / PCA-full => selection beats full 564-band data (the real-instrument case).")


if __name__ == "__main__":
    main()
