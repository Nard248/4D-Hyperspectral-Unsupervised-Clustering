"""C0..C4 anti-cheat ladder vs the baselines, on labelled balanced 4D ME-HSI.

The "pluggable options + comparison" deliverable for ``docs/spectraforge/03-architecture-plan.md`` /
``04-training-runbook.md``. Over >=3 scene seeds x 2 noise levels, prints one table comparing every
candidate (C0..C4) and the baselines (random, variance-ranking, peak-neighbourhood oracle, all-bands)
on four metrics:

  * KNN macro-F1   — the downstream task (the real criterion)
  * peak_recovery  — did the selection hit each fluorophore's true emission peak (sanity)
  * recon R        — mean per-band corr(input, reconstruction); >0 == the AE learned structure
  * infl-corr      — corr(per-band perturbation influence, raw per-band signal variance); CAE < 0

then evaluates the synthetic gate from doc 04 §4 for each candidate.

Run:  python reports/architecture_comparison.py
"""
from __future__ import annotations

import contextlib
import os

import numpy as np

from spectraforge import ArtifactConfig
from spectraforge.validation import validate_selection

from classification_experiment import (   # reuse the labelled-dataset + classification harness
    BUDGET, build_dataset, cols_for_bands, feature_matrix, knn_macro_f1, peak_neighbourhood_bands,
)
from spectral_select.architectures import CANDIDATES

SEEDS = [1, 2, 3]
NOISE = {
    "low noise": ArtifactConfig(rayleigh_strength=0.1, photon_scale=800, read_sigma=0.01),
    "high noise": ArtifactConfig(rayleigh_strength=0.3, photon_scale=150, read_sigma=0.05),
}
# per-candidate construction (C1 is the expensive spatial one -> fewer epochs; early-stopping helps)
EPOCHS = {"C0": dict(training_epochs=30), "C1": dict(training_epochs=12),
          "C2": dict(epochs=300), "C3": dict(epochs=300), "C4": dict(epochs=300)}


def _candidate_row(cid, spectra, X, colmap, gt, y, seed):
    Cls = CANDIDATES[cid]
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        model = Cls(seed=seed, **EPOCHS[cid]).fit(spectra)
    bands = model.select(BUDGET)
    cols = cols_for_bands(colmap, bands)
    return {
        "f1": knn_macro_f1(X[:, cols], y, seed),
        "peak": validate_selection(gt, bands, tol_nm=10)["peak_recovery"],
        "R": model.reconstruction_r(),
        "corr": model.influence_signal_corr(),
    }


def _baseline_rows(X, colmap, gt, y, acq, seed, rng):
    out = {}
    out["all bands"] = {"f1": knn_macro_f1(X, y, seed)}
    vcols = list(np.argsort(X.var(0))[::-1][:BUDGET])
    out["variance-ranking"] = {"f1": knn_macro_f1(X[:, vcols], y, seed),
                               "peak": validate_selection(gt, [colmap[c] for c in vcols], tol_nm=10)["peak_recovery"]}
    pk = peak_neighbourhood_bands(gt, acq)
    out["peak-neighbourhood"] = {"f1": knn_macro_f1(X[:, cols_for_bands(colmap, pk)], y, seed),
                                 "peak": validate_selection(gt, pk, tol_nm=10)["peak_recovery"]}
    rb = [(float(rng.choice(acq.excitations)), float(rng.choice(acq.emission_grid()))) for _ in range(BUDGET)]
    out["random"] = {"f1": knn_macro_f1(X[:, cols_for_bands(colmap, rb)], y, seed),
                     "peak": validate_selection(gt, rb, tol_nm=10)["peak_recovery"]}
    return out


def _fmt(v, plus=False):
    if v is None:
        return f"{'-':>9}"
    return f"{v:>+9.3f}" if plus else f"{v:>9.3f}"


def main():
    order = ["all bands", "C0", "C1", "C2", "C3", "C4",
             "variance-ranking", "peak-neighbourhood", "random"]
    label = {"C0": "C0 standard-CAE", "C1": "C1 deep-spectral-CAE", "C2": "C2 spectral-AE",
             "C3": "C3 masked-spectral-AE", "C4": "C4 variational-spec-AE"}
    agg = {noise: {k: {"f1": [], "peak": [], "R": [], "corr": []} for k in order} for noise in NOISE}

    print("=" * 104)
    print(f"Architecture ladder C0..C4 vs baselines  |  KNN macro-F1, peak_recovery, recon R, "
          f"infl-corr  |  {BUDGET}-band budget")
    print(f"Dyes: EBFP2, EGFP, mCherry | 3 classes | mean over {len(SEEDS)} scenes per noise level")
    print("=" * 104)

    for noise_label, noise in NOISE.items():
        rng = np.random.default_rng(0)
        for seed in SEEDS:
            spectra, gt, y, acq = build_dataset(seed, noise)
            X, colmap = feature_matrix(spectra)
            for cid in CANDIDATES:
                r = _candidate_row(cid, spectra, X, colmap, gt, y, seed)
                for m in ("f1", "peak", "R", "corr"):
                    agg[noise_label][cid][m].append(r[m])
            for name, r in _baseline_rows(X, colmap, gt, y, acq, seed, rng).items():
                for m, v in r.items():
                    agg[noise_label][name][m].append(v)

        print(f"\n[{noise_label}]")
        print(f"{'selection':<24}{'KNN-F1':>9}{'peak_rec':>9}{'recon R':>9}{'infl-corr':>10}")
        for k in order:
            d = agg[noise_label][k]
            name = label.get(k, k)
            f1 = _fmt(np.mean(d["f1"])) if d["f1"] else _fmt(None)
            peak = _fmt(np.mean(d["peak"])) if d["peak"] else _fmt(None)
            R = _fmt(np.mean(d["R"]), plus=True) if d["R"] else _fmt(None)
            corr = _fmt(np.mean(d["corr"]), plus=True) if d["corr"] else _fmt(None)
            print(f"{name:<24}{f1}{peak}{R}{corr[1:]:>10}")

    # ---- synthetic gate (doc 04 §4): evaluate each candidate, averaged over ALL scenes ----
    print("\n" + "=" * 104)
    print("SYNTHETIC GATE (doc 04 §4) — averaged over all noise levels & seeds")
    print("=" * 104)

    def allvals(k, m):
        return [v for noise in NOISE for v in agg[noise][k][m]]

    var_f1 = float(np.mean(allvals("variance-ranking", "f1")))
    rnd_f1 = float(np.mean(allvals("random", "f1")))
    c0_f1 = float(np.mean(allvals("C0", "f1")))
    print(f"reference F1:  variance-ranking={var_f1:.3f}  random={rnd_f1:.3f}  C0(CAE)={c0_f1:.3f}\n")
    print(f"{'candidate':<24}{'R>0':>7}{'corr>0':>8}{'F1>=var':>9}{'F1>rand':>9}{'F1>CAE':>8}"
          f"{'peak>.33':>9}{'GATE':>7}")
    for cid in CANDIDATES:
        f1 = float(np.mean(allvals(cid, "f1")))
        peak = float(np.mean(allvals(cid, "peak")))
        R = float(np.mean(allvals(cid, "R")))
        corr = float(np.mean(allvals(cid, "corr")))
        checks = [R > 0, corr > 0, f1 >= var_f1 - 0.01, f1 > rnd_f1, f1 > c0_f1, peak > 0.33]
        passed = all(checks)
        marks = "".join(f"{'Y' if c else 'n':>{w}}" for c, w in
                        zip(checks, (7, 8, 9, 9, 8, 9)))
        print(f"{label[cid]:<24}{marks}{('PASS' if passed else 'FAIL'):>7}")
    print("-" * 104)
    print("Gate = R>0 AND corr>0 AND F1>=variance-ranking AND F1>random AND F1>CAE AND peak_recovery>0.33")


if __name__ == "__main__":
    main()
