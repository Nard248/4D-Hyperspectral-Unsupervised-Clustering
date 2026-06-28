"""Round-2 H10 — on NONLINEAR data (reabsorption / FRET), with the fair CV-panel metric (doc 18, not
few-shot), which selector is best, and does a nonlinear/AE method beat marginal mutInfo? Adds the
supervised baselines (mutInfo, mRMR, RF-importance; * = uses labels) to the doc-18 comparison.

Run:  python reports/supervised_nonlinear.py [--smoke]
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
from metric_suite import mrmr_select
from mixnoise_experiment import BUDGET, SIZE, _AE_CONV, roi_mask
from reabsorption_eval import panel_scores, summarize
from realistic_benchmark import build_dataset
from fret_regime import build_fret_dataset
from sweep_common import topn_diverse
from swarm_zoo import FlexSpectralAE


def ae_cols(sp, colmap, seed):
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        m = FlexSpectralAE(seed=seed, **_AE_CONV).fit(sp)
    return cols_for_bands(colmap, m.select(BUDGET))


def rfimp(X, y, colmap, seed):
    rf = RandomForestClassifier(200, random_state=seed, n_jobs=-1)
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        rf.fit(StandardScaler().fit_transform(X), y)
    return topn_diverse(rf.feature_importances_, colmap, BUDGET)


def regimes(seed, smoke):
    out = {}
    sp, gt, y, acq = build_dataset(seed, size=SIZE, em_step=5, nuisance_amp=1.0, turbidity_amp=0.5,
                                   rayleigh=0.3, raman=0.25, photon_scale=2000, read_sigma=0.005,
                                   reabsorption=True, reabsorption_strength=3.0)
    out["reabsorb"] = (sp, y)
    if not smoke:
        sp2, y2 = build_fret_dataset(seed, size=SIZE)
        out["fret"] = (sp2, y2)
    return out


def main():
    smoke = "--smoke" in sys.argv
    seeds = ([1] if smoke else [1, 2, 3])
    methods = ["random", "variance", "pca_load", "AE", "mutInfo*", "mRMR*", "RFimp*"]
    print("=" * 96)
    print("H10 SUPERVISED/NONLINEAR SELECTORS ON NONLINEAR DATA — fair CV-panel best-NL F1, k=24")
    print("=" * 96)
    agg = {}
    for seed in seeds:
        for reg, (sp, y) in regimes(seed, smoke).items():
            X, colmap = feature_matrix(sp)
            roi = roi_mask(seed, SIZE); Xr, yr = X[roi], y[roi]
            rng = np.random.default_rng(seed)
            MI = np.nan_to_num(mutual_info_classif(Xr, yr, random_state=seed))
            sels = {"random": list(rng.choice(X.shape[1], BUDGET, replace=False)),
                    "variance": topn_diverse(Xr.var(0), colmap, BUDGET),
                    "pca_load": mz.pca_load(X, colmap, BUDGET, seed, rng, sp, k=6),
                    "AE": ae_cols(sp, colmap, seed),
                    "mutInfo*": topn_diverse(MI, colmap, BUDGET),
                    "mRMR*": mrmr_select(Xr, yr, colmap, BUDGET, seed),
                    "RFimp*": rfimp(Xr, yr, colmap, seed)}
            for m, cols in sels.items():
                _, nl, gap = summarize(panel_scores(Xr, yr, cols, seed))
                agg.setdefault((reg, m), []).append((nl, gap))
    for reg in (["reabsorb"] if smoke else ["reabsorb", "fret"]):
        print(f"\n### {reg}")
        print(f"  {'method':<11}{'best-NL':>9}{'gap':>8}")
        base = np.mean([v[0] for v in agg[(reg, "pca_load")]])
        rows = []
        for m in methods:
            nl = np.mean([v[0] for v in agg[(reg, m)]]); gp = np.mean([v[1] for v in agg[(reg, m)]])
            rows.append((m, nl, gp))
        for m, nl, gp in rows:
            tag = "  <-- best" if nl == max(r[1] for r in rows) else (f" (+{nl-base:.3f} vs pca)" if m == "AE" else "")
            print(f"  {m:<11}{nl:>9.3f}{gp:>+8.3f}{tag}")
    print("\nQuestion: does AE or a supervised-nonlinear selector clearly top mutInfo/pca on nonlinear data?")


if __name__ == "__main__":
    main()
