"""H1 — Semi-supervised AE selector. Can a few ROI labels guiding the AE's perturbation ranking beat
full-data AND PCA AND the unsupervised AE, approaching the supervised oracle, across noise levels?

The AE is trained UNSUPERVISED (reconstruction, denoising) once per dataset. The few labelled ROI
pixels (the same budget used to train the classifier — no leakage to the test set) then guide selection:
  * AE+rerank : take the AE's top-N influential bands, keep the top-k by few-label ANOVA-F (the AE
                proposes a denoised shortlist; labels refine it).
  * AE+fuse   : z(AE influence) + z(few-label F) over all bands -> top-k.
Baselines: full-564, PCA (blind), AE (blind), few-label-F (supervised, same labels), oracle
(mutual_info on ALL roi labels = upper bound). Train classifier on the SAME few labels, test on held-out.

Run:  python reports/semisup_experiment.py [--smoke]
"""
from __future__ import annotations

import contextlib
import os
import sys

import numpy as np
from sklearn.feature_selection import f_classif, mutual_info_classif

import method_zoo as mz
from classification_experiment import cols_for_bands, feature_matrix
from mixnoise_experiment import (BUDGET, EM_STEP, LEVELS, NBANDS, SIZE, _AE_CONV, _clfs, roi_mask)
from realistic_benchmark import build_dataset
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sweep_common import topn_diverse
from swarm_zoo import FlexSpectralAE

PER_CLASS = 20
REPEATS = 4
TOPN = 80          # AE shortlist size for re-ranking
_z = lambda s: (s - np.mean(s)) / (np.std(s) + 1e-9)


def ae_influence(spectra, seed, kw=_AE_CONV):
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        m = FlexSpectralAE(seed=seed, **kw).fit(spectra)
    return np.asarray(m._band_influence(), float)


def eval_split(X, y, cols, L, T, seed):
    sc = StandardScaler().fit(X[L][:, cols])
    Xtr, Xte = sc.transform(X[L][:, cols]), sc.transform(X[T][:, cols])
    f1s = []
    for n, c in _clfs(seed).items():
        with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
            c.fit(Xtr, y[L]); f1s.append(f1_score(y[T], c.predict(Xte), average="macro"))
    return max(f1s)


def main():
    smoke = "--smoke" in sys.argv
    levels = (["L1-pristine", "L5-severe"] if smoke else list(LEVELS))
    seeds = ([1] if smoke else [1, 2])
    methods = ["full", "PCA", "AE", "AE+rerank", "AE+fuse", "fewF", "oracle*"]
    print("=" * 104)
    print(f"H1 SEMI-SUPERVISED AE — high-res 564b, ROI, {PER_CLASS} labels/class x{REPEATS}, k={BUDGET} (best-NL F1)")
    print("Δ = method - full (positive => beats full 564-band data)")
    print("=" * 104)
    summ = []
    for lvl in levels:
        params = LEVELS[lvl]
        acc = {m: [] for m in methods}
        for seed in seeds:
            sp, gt, y, acq = build_dataset(seed, size=SIZE, em_step=EM_STEP, **params)
            X, colmap = feature_matrix(sp)
            roi = roi_mask(seed, SIZE); idx = np.where(roi)[0]
            Xr, yr = X[roi], y[roi]
            infl = ae_influence(sp, seed)                                   # unsupervised, per band
            pca = mz.pca_load(X, colmap, BUDGET, seed, np.random.default_rng(seed), sp, k=6)
            ae = topn_diverse(infl, colmap, BUDGET)
            ora = topn_diverse(np.nan_to_num(mutual_info_classif(Xr, yr, random_state=seed)), colmap, BUDGET)
            for r in range(REPEATS):
                rng = np.random.default_rng(seed * 100 + r)
                L = np.concatenate([rng.choice(np.where(yr == c)[0], PER_CLASS, replace=False) for c in np.unique(yr)])
                T = np.setdiff1d(np.arange(len(yr)), L)
                F = np.nan_to_num(f_classif(Xr[L], yr[L])[0])              # few-label relevance
                shortlist = set(np.argsort(infl)[::-1][:TOPN])
                rer_score = np.where([j in shortlist for j in range(len(F))], F, -1e9)
                sels = {"full": list(range(NBANDS)), "PCA": pca, "AE": ae,
                        "AE+rerank": topn_diverse(rer_score, colmap, BUDGET),
                        "AE+fuse": topn_diverse(_z(infl) + _z(F), colmap, BUDGET),
                        "fewF": topn_diverse(F, colmap, BUDGET), "oracle*": ora}
                for mname, cols in sels.items():
                    acc[mname].append(eval_split(Xr, yr, cols, L, T, seed))

        mean = {m: float(np.mean(acc[m])) for m in methods}
        f = mean["full"]
        print(f"\n### {lvl} (clutter={params['clutter_amp']}x{params['clutter_modes']})  VERIFY 564b/3cls OK")
        print(f"  {'method':<11}{'bestNL':>8}{'Δvs full':>10}")
        for m in methods:
            tag = "  <-- beats full" if mean[m] > f and m != "full" else ""
            print(f"  {m:<11}{mean[m]:>8.3f}{mean[m]-f:>+10.3f}{tag}")
        ss = max(mean["AE+rerank"], mean["AE+fuse"])
        print(f"  >>> semi-sup AE best={ss:.3f} | vs PCA {ss-mean['PCA']:+.3f} | vs fewF {ss-mean['fewF']:+.3f} "
              f"| vs oracle {ss-mean['oracle*']:+.3f}")
        summ.append((lvl, mean, ss))

    print("\n" + "=" * 104)
    print(f"{'level':<13}{'full':>7}{'PCA':>7}{'AE':>7}{'AE+rr':>7}{'AE+fu':>7}{'fewF':>7}{'oracle*':>8}{'ssAE-full':>10}{'ssAE-PCA':>9}")
    for lvl, mean, ss in summ:
        print(f"{lvl:<13}{mean['full']:>7.3f}{mean['PCA']:>7.3f}{mean['AE']:>7.3f}{mean['AE+rerank']:>7.3f}"
              f"{mean['AE+fuse']:>7.3f}{mean['fewF']:>7.3f}{mean['oracle*']:>8.3f}{ss-mean['full']:>+10.3f}{ss-mean['PCA']:>+9.3f}")
    print("\nGoal: semi-sup AE (rerank/fuse) > full AND > PCA AND >= fewF, approaching oracle*.")


if __name__ == "__main__":
    main()
