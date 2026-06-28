"""Mechanism test: is "blind selection ≈ random" caused by HIGH SPECTRAL RESOLUTION (redundancy)?
Re-run the beat-random check at the cluttered L4 regime across emission resolutions (em_step = 2/5/10
nm => 564/228/116 bands). Hypothesis: at coarse resolution (fewer, less-redundant bands) blind AE/PCA
beat random; at fine resolution they collapse to random.

Run:  python reports/resolution_effect.py [--smoke]
"""
from __future__ import annotations

import contextlib
import os
import sys

import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

import method_zoo as mz
from classification_experiment import cols_for_bands, feature_matrix
from mixnoise_experiment import BUDGET, LEVELS, SIZE, _AE_CONV, _clfs, roi_mask
from realistic_benchmark import build_dataset
from sweep_common import topn_diverse
from swarm_zoo import FlexSpectralAE

PER_CLASS = 30
REPEATS = 4
N_RANDOM = 12
PARAMS = LEVELS["L4-high"]


def fit_ae(spectra, seed):
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        return FlexSpectralAE(seed=seed, **_AE_CONV).fit(spectra)


def fewshot(X, y, cols, seed):
    out = []
    for r in range(REPEATS):
        rng = np.random.default_rng(seed * 100 + r)
        tr = np.concatenate([rng.choice(np.where(y == c)[0], PER_CLASS, replace=False) for c in np.unique(y)])
        te = np.setdiff1d(np.arange(len(y)), tr)
        sc = StandardScaler().fit(X[tr][:, cols])
        Xtr, Xte = sc.transform(X[tr][:, cols]), sc.transform(X[te][:, cols])
        fs = []
        for n, c in _clfs(seed).items():
            with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
                c.fit(Xtr, y[tr]); fs.append(f1_score(y[te], c.predict(Xte), average="macro"))
        out.append(max(fs))
    return float(np.mean(out))


def main():
    smoke = "--smoke" in sys.argv
    steps = ([10] if smoke else [2, 5, 10])
    seeds = ([1] if smoke else [1, 2])
    print("=" * 92)
    print(f"RESOLUTION EFFECT (L4-high) — does fine resolution make blind selection ~ random? k={BUDGET}")
    print("=" * 92)
    print(f"{'em_step':>8}{'nbands':>8}{'random µ':>10}{'rand95':>8}{'PCA':>8}{'AE':>8}{'mutInfo*':>10}{'AE>rnd':>8}{'PCA>rnd':>8}")
    for step in steps:
        ae_v, pca_v, mi_v, rnd_v, nb = [], [], [], [], None
        for seed in seeds:
            sp, gt, y, acq = build_dataset(seed, size=SIZE, em_step=step, **PARAMS)
            X, colmap = feature_matrix(sp); nb = X.shape[1]
            roi = roi_mask(seed, SIZE); Xr, yr = X[roi], y[roi]
            ae = fit_ae(sp, seed)
            MI = np.nan_to_num(mutual_info_classif(Xr, yr, random_state=seed))
            ae_v.append(fewshot(Xr, yr, cols_for_bands(colmap, ae.select(BUDGET)), seed))
            pca_v.append(fewshot(Xr, yr, mz.pca_load(X, colmap, BUDGET, seed, np.random.default_rng(seed), sp, k=6), seed))
            mi_v.append(fewshot(Xr, yr, topn_diverse(MI, colmap, BUDGET), seed))
            for j in range(N_RANDOM):
                rng = np.random.default_rng(seed * 1000 + j)
                rnd_v.append(fewshot(Xr, yr, list(rng.choice(nb, BUDGET, replace=False)), seed))
        rnd = np.array(rnd_v); r95 = np.percentile(rnd, 95)
        ae_m, pca_m, mi_m = np.mean(ae_v), np.mean(pca_v), np.mean(mi_v)
        print(f"{step:>8}{nb:>8}{rnd.mean():>10.3f}{r95:>8.3f}{pca_m:>8.3f}{ae_m:>8.3f}{mi_m:>10.3f}"
              f"{('YES' if ae_m > r95 else 'no'):>8}{('YES' if pca_m > r95 else 'no'):>8}")
    print("-" * 92)
    print("If 'YES' appears at coarse steps (10/5) but not fine (2): redundancy from high resolution is")
    print("why blind selection collapses to random on realistic data.")


if __name__ == "__main__":
    main()
