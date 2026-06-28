"""Round-2 H16 — a NEW role for the AE: not a selector but a DENOISER (preprocessing). Does classifying
on the AE's reconstruction (which a small bottleneck might strip of clutter) beat classifying on raw
data, under clutter? Tests whether the AE adds value in a different role than band selection.

Run:  python reports/ae_denoiser.py [--smoke]
"""
from __future__ import annotations

import contextlib
import os
import sys

import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

import method_zoo as mz
from classification_experiment import feature_matrix
from mixnoise_experiment import BUDGET, EM_STEP, LEVELS, NBANDS, SIZE, _AE_CONV, _clfs, roi_mask
from realistic_benchmark import build_dataset
from swarm_zoo import FlexSpectralAE

PER_CLASS = 30
REPEATS = 4
REGIMES = {"clean": LEVELS["L1-pristine"], "clutter": LEVELS["L4-high"]}


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
    regimes = (["clutter"] if smoke else list(REGIMES))
    seeds = ([1] if smoke else [1, 2])
    print("=" * 80)
    print(f"H16 AE AS DENOISER — classify on raw vs AE-reconstruction, full & PCA-24 (best-NL F1)")
    print("=" * 80)
    print(f"{'regime':<10}{'raw-full':>10}{'AErec-full':>11}{'raw-PCA':>9}{'AErec-PCA':>10}{'Δfull':>8}")
    for reg in regimes:
        params = REGIMES[reg]
        rf, af, rp, ap = [], [], [], []
        for seed in seeds:
            sp, gt, y, acq = build_dataset(seed, size=SIZE, em_step=EM_STEP, **params)
            X, colmap = feature_matrix(sp)
            roi = roi_mask(seed, SIZE); Xr, yr = X[roi], y[roi]
            with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
                m = FlexSpectralAE(seed=seed, **_AE_CONV).fit(sp)
            R = np.asarray(m._recon.cpu().numpy(), float)[roi]               # AE reconstruction (denoised?)
            pca = mz.pca_load(X, colmap, BUDGET, seed, np.random.default_rng(seed), sp, k=6)
            rf.append(fewshot(Xr, yr, list(range(NBANDS)), seed))
            af.append(fewshot(R, yr, list(range(NBANDS)), seed))
            rp.append(fewshot(Xr, yr, pca, seed))
            ap.append(fewshot(R, yr, pca, seed))
        rfm, afm, rpm, apm = map(lambda v: float(np.mean(v)), (rf, af, rp, ap))
        tag = "  <-- AE denoise helps" if afm > rfm else ""
        print(f"{reg:<10}{rfm:>10.3f}{afm:>11.3f}{rpm:>9.3f}{apm:>10.3f}{afm-rfm:>+8.3f}{tag}")
    print("\nΔfull = AErec-full − raw-full. Positive => the AE reconstruction denoises (a NEW AE value).")


if __name__ == "__main__":
    main()
