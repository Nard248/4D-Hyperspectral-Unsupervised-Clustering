"""Round-2 H15 — can UNSUPERVISED selection be rescued under clutter by suppressing the clutter
subspace? Clutter is high-variance but low-rank/structured. Project out the top-K principal components
(the clutter+nuisance subspace), then select bands by loadings on the RESIDUAL (the signal subspace).
Does clutter-suppressed selection beat random where vanilla variance/PCA selection failed?

Run:  python reports/clutter_robust.py [--smoke]
"""
from __future__ import annotations

import contextlib
import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from classification_experiment import feature_matrix
from mixnoise_experiment import BUDGET, EM_STEP, LEVELS, NBANDS, SIZE, _clfs, roi_mask
from realistic_benchmark import build_dataset
from sweep_common import topn_diverse

PER_CLASS = 30
REPEATS = 4
N_RANDOM = 12
KS = [10, 20, 40, 80]


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


def pca_load_residual(X, K):
    """Sum |loadings| of the top-6 PCs of X AFTER projecting out the top-K (clutter) PCs."""
    Xc = StandardScaler().fit_transform(X)
    if K > 0:
        p = PCA(K, random_state=0).fit(Xc)
        Xc = Xc - p.inverse_transform(p.transform(Xc))
    p2 = PCA(6, random_state=0).fit(Xc)
    return np.sum(np.abs(p2.components_), axis=0)


def var_residual(X, K):
    Xc = StandardScaler().fit_transform(X)
    if K > 0:
        p = PCA(K, random_state=0).fit(Xc)
        Xc = Xc - p.inverse_transform(p.transform(Xc))
    return Xc.var(0)


def main():
    smoke = "--smoke" in sys.argv
    levels = (["L4-high"] if smoke else ["L4-high", "L5-severe"])
    seeds = ([1] if smoke else [1, 2])
    ks = ([20] if smoke else KS)
    methods = ["random", "variance", "pca_load"] + [f"pcaRes{K}" for K in ks] + [f"varRes{ks[-1]}", "mutInfo*"]
    print("=" * 104)
    print(f"H15 CLUTTER-ROBUST UNSUPERVISED SELECTION — k={BUDGET}, few-shot {PER_CLASS}/class best-NL F1")
    print("project out top-K (clutter) PCs, select on residual; does it beat random where vanilla failed?")
    print("=" * 104)
    for lvl in levels:
        params = LEVELS[lvl]
        acc = {m: [] for m in methods}
        rnd_all = []
        for seed in seeds:
            sp, gt, y, acq = build_dataset(seed, size=SIZE, em_step=EM_STEP, **params)
            X, colmap = feature_matrix(sp)
            roi = roi_mask(seed, SIZE); Xr, yr = X[roi], y[roi]
            MI = np.nan_to_num(mutual_info_classif(Xr, yr, random_state=seed))
            sel = {"variance": topn_diverse(Xr.var(0), colmap, BUDGET),
                   "pca_load": topn_diverse(pca_load_residual(Xr, 0), colmap, BUDGET),
                   "mutInfo*": topn_diverse(MI, colmap, BUDGET)}
            for K in ks:
                sel[f"pcaRes{K}"] = topn_diverse(pca_load_residual(Xr, K), colmap, BUDGET)
            sel[f"varRes{ks[-1]}"] = topn_diverse(var_residual(Xr, ks[-1]), colmap, BUDGET)
            for m, cols in sel.items():
                acc[m].append(fewshot(Xr, yr, cols, seed))
            for j in range(N_RANDOM):
                rng = np.random.default_rng(seed * 1000 + j)
                rnd_all.append(fewshot(Xr, yr, list(rng.choice(NBANDS, BUDGET, replace=False)), seed))
        rnd = np.array(rnd_all); r95 = np.percentile(rnd, 95)
        acc["random"] = [rnd.mean()]
        print(f"\n### {lvl} (clutter={params['clutter_amp']}x{params['clutter_modes']}) | random µ={rnd.mean():.3f} 95th={r95:.3f}")
        for m in methods:
            v = float(np.mean(acc[m]))
            tag = "  <-- beats random" if v > r95 and m not in ("random",) else ""
            print(f"  {m:<12}{v:>8.3f}{tag}")
    print("\nWin = a clutter-suppressed (residual) method beats random's 95th pct where vanilla var/pca did not.")


if __name__ == "__main__":
    main()
