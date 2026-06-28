"""Round-2 H13 — selection STABILITY: do selectors pick consistent bands across scenes/label-draws?
A trustworthy selector is stable. Compare blind (PCA, AE) vs supervised (mutInfo, RFimp) at the
cluttered L4 regime via mean pairwise Jaccard of the selected 24-band sets.

Run:  python reports/stability_check.py [--smoke]
"""
from __future__ import annotations

import contextlib
import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

import method_zoo as mz
from classification_experiment import cols_for_bands, feature_matrix
from mixnoise_experiment import BUDGET, EM_STEP, LEVELS, SIZE, _AE_CONV, roi_mask
from realistic_benchmark import build_dataset
from sweep_common import topn_diverse
from swarm_zoo import FlexSpectralAE


def jaccard(sets):
    s = [set(x) for x in sets]
    ps = [len(a & b) / len(a | b) for i, a in enumerate(s) for b in s[i + 1:]]
    return float(np.mean(ps)) if ps else 1.0


def main():
    smoke = "--smoke" in sys.argv
    seeds = ([1, 2] if smoke else [1, 2, 3])
    params = LEVELS["L4-high"]
    sets = {m: [] for m in ["PCA", "AE", "mutInfo*", "RFimp*"]}
    for seed in seeds:
        sp, gt, y, acq = build_dataset(seed, size=SIZE, em_step=EM_STEP, **params)
        X, colmap = feature_matrix(sp)
        roi = roi_mask(seed, SIZE); Xr, yr = X[roi], y[roi]
        sets["PCA"].append(mz.pca_load(X, colmap, BUDGET, seed, np.random.default_rng(seed), sp, k=6))
        with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
            sets["AE"].append(cols_for_bands(colmap, FlexSpectralAE(seed=seed, **_AE_CONV).fit(sp).select(BUDGET)))
        sets["mutInfo*"].append(topn_diverse(np.nan_to_num(mutual_info_classif(Xr, yr, random_state=seed)), colmap, BUDGET))
        rf = RandomForestClassifier(200, random_state=seed, n_jobs=-1)
        with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
            rf.fit(StandardScaler().fit_transform(Xr), yr)
        sets["RFimp*"].append(topn_diverse(rf.feature_importances_, colmap, BUDGET))
    print("=" * 70)
    print(f"H13 SELECTION STABILITY (L4-high, {len(seeds)} seeds) — mean pairwise Jaccard of 24-band sets")
    print("=" * 70)
    for m in ["PCA", "AE", "mutInfo*", "RFimp*"]:
        print(f"  {m:<11} Jaccard = {jaccard(sets[m]):.3f}")
    print("\nHigher = more consistent band choice across scenes (more trustworthy).")


if __name__ == "__main__":
    main()
