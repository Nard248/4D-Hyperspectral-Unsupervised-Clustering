"""H2 — Is the AE's advantage over PCA monotonic in clutter/noise? Fix the base regime and sweep ONLY
the fixed-pattern clutter strength; measure AE vs PCA selection accuracy (few-shot best-NL) with a
bootstrap CI on the AE-PCA margin per clutter level. Find the crossover where the AE overtakes PCA.

Run:  python reports/noise_margin_sweep.py [--smoke]
"""
from __future__ import annotations

import contextlib
import os
import sys

import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

import method_zoo as mz
from classification_experiment import cols_for_bands, feature_matrix
from mixnoise_experiment import BUDGET, EM_STEP, NBANDS, SIZE, _AE_CONV, _clfs, roi_mask
from realistic_benchmark import build_dataset
from sweep_common import topn_diverse
from swarm_zoo import FlexSpectralAE

PER_CLASS = 25
REPEATS = 5
BASE = dict(nuisance_amp=1.0, turbidity_amp=0.5, rayleigh=0.3, raman=0.25, photon_scale=2000, read_sigma=0.005)
CLUTTER = [0.0, 0.5, 1.0, 2.0, 3.0, 4.5]
MODES = 28


def ae_cols(spectra, colmap, seed):
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        m = FlexSpectralAE(seed=seed, **_AE_CONV).fit(spectra)
    return cols_for_bands(colmap, m.select(BUDGET))


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
    return out


def main():
    smoke = "--smoke" in sys.argv
    clutter = ([0.0, 4.5] if smoke else CLUTTER)
    seeds = ([1] if smoke else [1, 2, 3])
    print("=" * 92)
    print(f"H2 AE-PCA MARGIN vs CLUTTER — base fixed, sweep clutter; few-shot {PER_CLASS}/class best-NL F1")
    print("=" * 92)
    print(f"{'clutter':>8}{'AE':>8}{'PCA':>8}{'margin':>9}{'95% CI':>18}")
    rows = []
    for camp in clutter:
        ae_all, pca_all = [], []
        for seed in seeds:
            sp, gt, y, acq = build_dataset(seed, size=SIZE, em_step=EM_STEP, clutter_modes=MODES, clutter_amp=camp, **BASE)
            X, colmap = feature_matrix(sp)
            roi = roi_mask(seed, SIZE); Xr, yr = X[roi], y[roi]
            pca = mz.pca_load(X, colmap, BUDGET, seed, np.random.default_rng(seed), sp, k=6)
            ae = ae_cols(sp, colmap, seed)
            ae_all += fewshot(Xr, yr, ae, seed)
            pca_all += fewshot(Xr, yr, pca, seed)
        ae_all, pca_all = np.array(ae_all), np.array(pca_all)
        diff = ae_all - pca_all
        boot = [np.mean(np.random.default_rng(s).choice(diff, len(diff))) for s in range(2000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append((camp, ae_all.mean(), pca_all.mean(), diff.mean(), lo, hi))
        print(f"{camp:>8.1f}{ae_all.mean():>8.3f}{pca_all.mean():>8.3f}{diff.mean():>+9.3f}{f'[{lo:+.3f},{hi:+.3f}]':>18}")
    print("-" * 92)
    pos = [r[0] for r in rows if r[4] > 0]
    print(f"AE>PCA (CI excludes 0) at clutter: {pos or 'none'}")
    mono = all(rows[i][3] <= rows[i + 1][3] + 0.01 for i in range(len(rows) - 1))
    print(f"margin roughly monotonic in clutter: {mono}")


if __name__ == "__main__":
    main()
