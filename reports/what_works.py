"""Constructive counterpart to the night's negatives: WHAT selector actually works, per regime?
Compare the floor (random), ceiling (full-564), blind (PCA, AE), and supervised (mutInfo = linear,
RF-importance = nonlinear) at k=24, few-shot, across clean / clutter / nonlinear-reabsorption regimes.
Goal: a clear, actionable recommendation + does a supervised NONLINEAR selector win on nonlinear data?

Run:  python reports/what_works.py [--smoke]
"""
from __future__ import annotations

import contextlib
import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
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
N_RANDOM = 10
REGIMES = {
    "clean":     dict(nuisance_amp=0.4, turbidity_amp=0.1, rayleigh=0.10, raman=0.10, photon_scale=50000, read_sigma=0.002, clutter_modes=0, clutter_amp=0.0),
    "clutter":   LEVELS["L4-high"],
    "nonlinear": dict(nuisance_amp=1.0, turbidity_amp=0.5, rayleigh=0.3, raman=0.25, photon_scale=2000, read_sigma=0.005, reabsorption=True, reabsorption_strength=3.0),
}


def fit_ae(spectra, seed):
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        return FlexSpectralAE(seed=seed, **_AE_CONV).fit(spectra)


def fewshot(X, y, cols, seed, return_imp=False):
    out = []
    imp = None
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


def rf_importance_cols(X, y, L, colmap, seed):
    rf = RandomForestClassifier(200, random_state=seed, n_jobs=-1)
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        rf.fit(StandardScaler().fit_transform(X[L]), y[L])
    return topn_diverse(rf.feature_importances_, colmap, BUDGET)


def main():
    smoke = "--smoke" in sys.argv
    regimes = (["clutter"] if smoke else list(REGIMES))
    seeds = ([1] if smoke else [1, 2])
    print("=" * 100)
    print(f"WHAT WORKS — floor/ceiling/blind/supervised at k={BUDGET}, few-shot {PER_CLASS}/class best-NL F1")
    print("=" * 100)
    print(f"{'regime':<11}{'random':>8}{'rand95':>8}{'full564':>9}{'PCA':>7}{'AE':>7}{'mutInfo*':>9}{'RFimp*':>8}{'best>rand':>16}")
    for reg in regimes:
        params = REGIMES[reg]
        acc = {m: [] for m in ["random", "full", "PCA", "AE", "mutInfo*", "RFimp*"]}
        rnd_all = []
        for seed in seeds:
            sp, gt, y, acq = build_dataset(seed, size=SIZE, em_step=EM_STEP, **params)
            X, colmap = feature_matrix(sp)
            roi = roi_mask(seed, SIZE); Xr, yr = X[roi], y[roi]
            ae = fit_ae(sp, seed)
            MI = np.nan_to_num(mutual_info_classif(Xr, yr, random_state=seed))
            rng0 = np.random.default_rng(seed)
            L0 = np.concatenate([rng0.choice(np.where(yr == c)[0], PER_CLASS, replace=False) for c in np.unique(yr)])
            acc["full"].append(fewshot(Xr, yr, list(range(NBANDS)), seed))
            acc["PCA"].append(fewshot(Xr, yr, mz.pca_load(X, colmap, BUDGET, seed, rng0, sp, k=6), seed))
            acc["AE"].append(fewshot(Xr, yr, cols_for_bands(colmap, ae.select(BUDGET)), seed))
            acc["mutInfo*"].append(fewshot(Xr, yr, topn_diverse(MI, colmap, BUDGET), seed))
            acc["RFimp*"].append(fewshot(Xr, yr, rf_importance_cols(Xr, yr, L0, colmap, seed), seed))
            for j in range(N_RANDOM):
                rng = np.random.default_rng(seed * 1000 + j)
                rnd_all.append(fewshot(Xr, yr, list(rng.choice(NBANDS, BUDGET, replace=False)), seed))
        rnd = np.array(rnd_all); r95 = np.percentile(rnd, 95)
        m = {k: float(np.mean(v)) for k, v in acc.items()}
        winners = [k for k in ["PCA", "AE", "mutInfo*", "RFimp*"] if m[k] > r95]
        print(f"{reg:<11}{rnd.mean():>8.3f}{r95:>8.3f}{m['full']:>9.3f}{m['PCA']:>7.3f}{m['AE']:>7.3f}"
              f"{m['mutInfo*']:>9.3f}{m['RFimp*']:>8.3f}{','.join(winners) or 'none':>16}")
    print("-" * 100)
    print("Actionable: which selectors beat random per regime; does RFimp* (supervised nonlinear) win on")
    print("the nonlinear regime where mutInfo* (linear) may not?")


if __name__ == "__main__":
    main()
