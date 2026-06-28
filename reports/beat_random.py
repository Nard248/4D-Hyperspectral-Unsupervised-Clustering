"""Honesty check (from H4): does BLIND selection (AE, PCA) significantly beat RANDOM band selection,
and in which regimes? If random is competitive, blind selection adds little; if the AE beats random
where PCA doesn't, that's a kept win. k=24, few-shot, tight random CI from many random draws.

Run:  python reports/beat_random.py [--smoke]
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
from mixnoise_experiment import BUDGET, EM_STEP, LEVELS, NBANDS, SIZE, _AE_CONV, _clfs, roi_mask
from realistic_benchmark import build_dataset
from sweep_common import topn_diverse
from swarm_zoo import FlexSpectralAE

PER_CLASS = 30
REPEATS = 4
N_RANDOM = 12


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
    levels = (["L1-pristine", "L5-severe"] if smoke else list(LEVELS))
    seeds = ([1] if smoke else [1, 2])
    print("=" * 96)
    print(f"BEAT RANDOM? — blind AE/PCA vs random (k={BUDGET}), few-shot {PER_CLASS}/class best-NL F1")
    print("=" * 96)
    print(f"{'level':<13}{'random µ':>9}{'rand95':>8}{'PCA':>8}{'AE':>8}{'mutInfo*':>10}{'AE>rand':>9}{'PCA>rand':>9}")
    for lvl in levels:
        params = LEVELS[lvl]
        ae_v, pca_v, mi_v, rnd_v = [], [], [], []
        for seed in seeds:
            sp, gt, y, acq = build_dataset(seed, size=SIZE, em_step=EM_STEP, **params)
            X, colmap = feature_matrix(sp)
            roi = roi_mask(seed, SIZE); Xr, yr = X[roi], y[roi]
            ae = fit_ae(sp, seed)
            MI = np.nan_to_num(mutual_info_classif(Xr, yr, random_state=seed))
            ae_v.append(fewshot(Xr, yr, cols_for_bands(colmap, ae.select(BUDGET)), seed))
            pca_v.append(fewshot(Xr, yr, mz.pca_load(X, colmap, BUDGET, seed, np.random.default_rng(seed), sp, k=6), seed))
            mi_v.append(fewshot(Xr, yr, topn_diverse(MI, colmap, BUDGET), seed))
            for j in range(N_RANDOM):
                rng = np.random.default_rng(seed * 1000 + j)
                rnd_v.append(fewshot(Xr, yr, list(rng.choice(NBANDS, BUDGET, replace=False)), seed))
        rnd = np.array(rnd_v); r95 = np.percentile(rnd, 95)
        ae_m, pca_m, mi_m = np.mean(ae_v), np.mean(pca_v), np.mean(mi_v)
        print(f"{lvl:<13}{rnd.mean():>9.3f}{r95:>8.3f}{pca_m:>8.3f}{ae_m:>8.3f}{mi_m:>10.3f}"
              f"{('YES' if ae_m > r95 else 'no'):>9}{('YES' if pca_m > r95 else 'no'):>9}")
    print("-" * 96)
    print("YES = method mean exceeds the 95th percentile of random subsets (genuinely beats random).")


if __name__ == "__main__":
    main()
