"""H1b — AE as a LABEL-EFFICIENCY tool. The bottleneck (H1) was that few-label relevance is too noisy
to beat full-data. Hypothesis: estimating relevance on the AE's DENOISED reconstruction (clutter
removed) picks the winning bands from FEWER labels than raw relevance — and can beat full-data at a
smaller label budget. Sweep the label budget on the cluttered L4/L5 regimes.

Selectors (k=24): full-564, PCA(blind), AE(blind), fewF-raw (ANOVA-F on raw few labels),
fewF-denoised (ANOVA-F on AE-reconstructed few labels), AE+den-fuse (z(AE influence)+z(denoised F)),
oracle* (mutual_info on ALL roi labels). Selection labels = classifier-training labels (no test leak);
classifier always trains/tests on the RAW selected bands (you denoise to SELECT, deploy on raw).

Run:  python reports/semisup_denoise.py [--smoke]
"""
from __future__ import annotations

import contextlib
import os
import sys

import numpy as np
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

import method_zoo as mz
from classification_experiment import feature_matrix
from mixnoise_experiment import (BUDGET, EM_STEP, LEVELS, NBANDS, SIZE, _AE_CONV, _clfs, roi_mask)
from realistic_benchmark import build_dataset
from sweep_common import topn_diverse
from swarm_zoo import FlexSpectralAE

REPEATS = 5
BUDGETS = [6, 10, 15, 25, 40]   # map the label-budget crossover: where does full-data overtake selection?
_z = lambda s: (s - np.mean(s)) / (np.std(s) + 1e-9)


def fit_ae(spectra, seed):
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        m = FlexSpectralAE(seed=seed, **_AE_CONV).fit(spectra)
    return np.asarray(m._band_influence(), float), np.asarray(m._recon.cpu().numpy(), float)


def evalc(X, y, cols, L, T, seed):
    sc = StandardScaler().fit(X[L][:, cols])
    Xtr, Xte = sc.transform(X[L][:, cols]), sc.transform(X[T][:, cols])
    fs = []
    for n, c in _clfs(seed).items():
        with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
            c.fit(Xtr, y[L]); fs.append(f1_score(y[T], c.predict(Xte), average="macro"))
    return max(fs)


def main():
    smoke = "--smoke" in sys.argv
    levels = (["L5-severe"] if smoke else ["L4-high", "L5-severe"])
    seeds = ([1] if smoke else [1, 2])
    budgets = ([25, 100] if smoke else BUDGETS)
    methods = ["full", "PCA", "AE", "fewF-raw", "fewF-den", "AE+den", "oracle*"]
    print("=" * 100)
    print("H1b AE-DENOISED RELEVANCE / LABEL EFFICIENCY — high-res 564b, ROI, k=24 (best-NL F1)")
    print("question: does denoised few-label relevance beat raw relevance AND full at a smaller budget?")
    print("=" * 100)
    for lvl in levels:
        params = LEVELS[lvl]
        print(f"\n### {lvl} (clutter={params['clutter_amp']}x{params['clutter_modes']})")
        per_budget = {}
        # precompute per-seed dataset + AE + blind selections + oracle
        data = []
        for seed in seeds:
            sp, gt, y, acq = build_dataset(seed, size=SIZE, em_step=EM_STEP, **params)
            X, colmap = feature_matrix(sp)
            roi = roi_mask(seed, SIZE); Xr, yr = X[roi], y[roi]
            infl, R = fit_ae(sp, seed)
            assert R.shape == X.shape, (R.shape, X.shape)
            Rr = R[roi]
            pca = mz.pca_load(X, colmap, BUDGET, seed, np.random.default_rng(seed), sp, k=6)
            ae = topn_diverse(infl, colmap, BUDGET)
            ora = topn_diverse(np.nan_to_num(mutual_info_classif(Xr, yr, random_state=seed)), colmap, BUDGET)
            data.append((seed, Xr, yr, colmap, infl, Rr, pca, ae, ora))
        for pc in budgets:
            acc = {m: [] for m in methods}
            for seed, Xr, yr, colmap, infl, Rr, pca, ae, ora in data:
                for r in range(REPEATS):
                    rng = np.random.default_rng(seed * 100 + r)
                    L = np.concatenate([rng.choice(np.where(yr == c)[0], pc, replace=False) for c in np.unique(yr)])
                    T = np.setdiff1d(np.arange(len(yr)), L)
                    Fr = np.nan_to_num(f_classif(Xr[L], yr[L])[0])
                    Fd = np.nan_to_num(f_classif(Rr[L], yr[L])[0])
                    sels = {"full": list(range(NBANDS)), "PCA": pca, "AE": ae,
                            "fewF-raw": topn_diverse(Fr, colmap, BUDGET),
                            "fewF-den": topn_diverse(Fd, colmap, BUDGET),
                            "AE+den": topn_diverse(_z(infl) + _z(Fd), colmap, BUDGET),
                            "oracle*": ora}
                    for mn, cols in sels.items():
                        acc[mn].append(evalc(Xr, yr, cols, L, T, seed))
            mean = {m: float(np.mean(acc[m])) for m in methods}
            per_budget[pc] = mean
            f = mean["full"]
            flags = []
            for m in ["fewF-raw", "fewF-den", "AE+den", "oracle*"]:
                if mean[m] > f:
                    flags.append(m)
            print(f"  labels/class={pc:>3}: " + "  ".join(f"{m}={mean[m]:.3f}" for m in methods))
            print(f"            Δvs full: fewF-raw={mean['fewF-raw']-f:+.3f} fewF-den={mean['fewF-den']-f:+.3f} "
                  f"AE+den={mean['AE+den']-f:+.3f} | beats full: {flags or 'none'}")
    print("\nGoal: fewF-den (AE-denoised) beats full at a SMALLER label budget than fewF-raw => the AE")
    print("buys label efficiency. AE+den should be >= fewF-raw at every budget.")


if __name__ == "__main__":
    main()
