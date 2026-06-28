"""H4 — Compression: how few bands can a selector use while RETAINING full-data accuracy, and which
selector compresses best? Fixed regime, sweep the band budget k; report best-NL F1 and % of full-564
accuracy retained. The real value proposition of band selection (acquire/process few bands cheaply).

Run:  python reports/compression_curve.py [--smoke]
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
from mixnoise_experiment import EM_STEP, LEVELS, NBANDS, SIZE, _AE_CONV, _clfs, roi_mask
from realistic_benchmark import build_dataset
from sweep_common import topn_diverse
from swarm_zoo import FlexSpectralAE

PER_CLASS = 50          # enough labels that full-564 is a strong ceiling
REPEATS = 4
KS = [4, 8, 12, 16, 24, 32, 48, 64, 100]
LEVEL = "L3-moderate"


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
    ks = ([8, 24, 64] if smoke else KS)
    seeds = ([1] if smoke else [1, 2])
    params = LEVELS[LEVEL]
    print("=" * 92)
    print(f"H4 COMPRESSION CURVE — {LEVEL} (clutter={params['clutter_amp']}x{params['clutter_modes']}), "
          f"{PER_CLASS} labels/class, best-NL F1")
    print("=" * 92)
    data = []
    full_vals = []
    for seed in seeds:
        sp, gt, y, acq = build_dataset(seed, size=SIZE, em_step=EM_STEP, **params)
        X, colmap = feature_matrix(sp)
        roi = roi_mask(seed, SIZE); Xr, yr = X[roi], y[roi]
        ae = fit_ae(sp, seed)
        MI = np.nan_to_num(mutual_info_classif(Xr, yr, random_state=seed))
        full_vals.append(fewshot(Xr, yr, list(range(NBANDS)), seed))
        data.append((seed, X, Xr, yr, colmap, sp, ae, MI))
    full = float(np.mean(full_vals))
    print(f"full-564 best-NL = {full:.3f}  (the ceiling; selections reported as % of this)\n")
    print(f"{'k':>5}{'PCA':>8}{'AE':>8}{'mutInfo*':>10}{'random':>8}   {'AE%full':>8}{'PCA%full':>9}")
    res = {m: [] for m in ["PCA", "AE", "mutInfo*", "random"]}
    for k in ks:
        vals = {m: [] for m in res}
        for seed, X, Xr, yr, colmap, sp, ae, MI in data:
            rng = np.random.default_rng(seed)
            sels = {"PCA": mz.pca_load(X, colmap, k, seed, rng, sp, k=6),
                    "AE": cols_for_bands(colmap, ae.select(k)),
                    "mutInfo*": topn_diverse(MI, colmap, k),
                    "random": list(rng.choice(NBANDS, k, replace=False))}
            for m, cols in sels.items():
                vals[m].append(fewshot(Xr, yr, cols, seed))
        for m in res:
            res[m].append(float(np.mean(vals[m])))
        a, p = res["AE"][-1], res["PCA"][-1]
        print(f"{k:>5}{p:>8.3f}{a:>8.3f}{res['mutInfo*'][-1]:>10.3f}{res['random'][-1]:>8.3f}   "
              f"{a/full:>7.0%}{p/full:>9.0%}")
    print("-" * 92)
    for m in ["AE", "PCA", "mutInfo*"]:
        hit = next((ks[i] for i, v in enumerate(res[m]) if v >= 0.95 * full), None)
        print(f"{m:<9} reaches 95% of full at k = {hit if hit else '>'+str(ks[-1])}  (of {NBANDS} bands)")
    print("\nValue prop: smallest k retaining ~full accuracy = the compression; lower is better.")


if __name__ == "__main__":
    main()
