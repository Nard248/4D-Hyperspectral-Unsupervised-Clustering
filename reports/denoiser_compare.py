"""Round-2 H16b — the AE's best lead: as a DENOISER it helps a lot on clean data (H16). But does its
NONLINEARITY beat a LINEAR denoiser (PCA reconstruction)? Compare classification on raw vs PCA-recon
(rank-K) vs AE-recon, across regimes. If AE-recon > PCA-recon on clean/moderate, nonlinear denoising is
a genuine AE contribution.

Run:  python reports/denoiser_compare.py [--smoke]
"""
from __future__ import annotations

import contextlib
import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from classification_experiment import feature_matrix
from mixnoise_experiment import EM_STEP, LEVELS, NBANDS, SIZE, _AE_CONV, _clfs, roi_mask
from realistic_benchmark import build_dataset
from swarm_zoo import FlexSpectralAE

PER_CLASS = 30
REPEATS = 4
REGIMES = {"clean": "L1-pristine", "low": "L2-low", "moderate": "L3-moderate", "clutter": "L4-high"}


def fewshot(X, y, seed):
    cols = list(range(X.shape[1]))
    out = []
    for r in range(REPEATS):
        rng = np.random.default_rng(seed * 100 + r)
        tr = np.concatenate([rng.choice(np.where(y == c)[0], PER_CLASS, replace=False) for c in np.unique(y)])
        te = np.setdiff1d(np.arange(len(y)), tr)
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        fs = []
        for n, c in _clfs(seed).items():
            with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
                c.fit(Xtr, y[tr]); fs.append(f1_score(y[te], c.predict(Xte), average="macro"))
        out.append(max(fs))
    return float(np.mean(out))


def pca_recon(X, K):
    Xs = StandardScaler().fit_transform(X)
    p = PCA(K, random_state=0).fit(Xs)
    return p.inverse_transform(p.transform(Xs))


def main():
    smoke = "--smoke" in sys.argv
    regimes = (["clean", "clutter"] if smoke else list(REGIMES))
    seeds = ([1] if smoke else [1, 2])
    print("=" * 92)
    print("H16b DENOISER COMPARE — raw vs PCA-recon(linear) vs AE-recon(nonlinear), full-data best-NL F1")
    print("=" * 92)
    print(f"{'regime':<10}{'raw':>8}{'pcaR8':>8}{'pcaR16':>8}{'AErec':>8}{'AE-raw':>9}{'AE-pcaR8':>10}")
    for reg in regimes:
        params = LEVELS[REGIMES[reg]]
        raw, p8, p16, ae = [], [], [], []
        for seed in seeds:
            sp, gt, y, acq = build_dataset(seed, size=SIZE, em_step=EM_STEP, **params)
            X, colmap = feature_matrix(sp)
            roi = roi_mask(seed, SIZE); Xr, yr = X[roi], y[roi]
            with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
                m = FlexSpectralAE(seed=seed, **_AE_CONV).fit(sp)
            R = np.asarray(m._recon.cpu().numpy(), float)[roi]
            raw.append(fewshot(Xr, yr, seed))
            p8.append(fewshot(pca_recon(Xr, 8), yr, seed))
            p16.append(fewshot(pca_recon(Xr, 16), yr, seed))
            ae.append(fewshot(R, yr, seed))
        rawm, p8m, p16m, aem = map(lambda v: float(np.mean(v)), (raw, p8, p16, ae))
        print(f"{reg:<10}{rawm:>8.3f}{p8m:>8.3f}{p16m:>8.3f}{aem:>8.3f}{aem-rawm:>+9.3f}{aem-p8m:>+10.3f}")
    print("\nAE-pcaR8 = AE-recon − PCA-recon(8). Positive on clean/moderate => nonlinear denoising beats")
    print("linear denoising (a genuine AE win). Under clutter all denoisers expected to fail.")


if __name__ == "__main__":
    main()
