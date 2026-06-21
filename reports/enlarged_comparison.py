"""Enlarged spectral autoencoders (C5 deep-MLP, C6 1D-conv, C7 masked-conv) vs the small C2/C3 and
the baselines, on BOTH benchmarks:

  * the CLEAN gate (reports/classification_experiment) — must still recover peaks/classify;
  * the REALISTIC confounded regime (reports/realistic_benchmark) — the discriminating test, where
    variance-ranking fails and we want the enlarged nets to close the gap to the oracle.

Run:  python reports/enlarged_comparison.py
"""
from __future__ import annotations

import contextlib
import os

import numpy as np
from sklearn.feature_selection import f_classif

from spectraforge.validation import validate_selection

import classification_experiment as clean
import realistic_benchmark as real
from classification_experiment import cols_for_bands, feature_matrix, knn_macro_f1
from spectraforge import ArtifactConfig
from spectral_select.architectures import (
    ConvSpectralAE, DeepSpectralAE, MaskedConvSpectralAE, MaskedSpectralAE, SpectralAE,
)

BUDGET = 12
LEARNED = [("C2 spectral-AE", SpectralAE), ("C3 masked-AE", MaskedSpectralAE),
           ("C5 deep-AE", DeepSpectralAE), ("C6 conv-AE", ConvSpectralAE),
           ("C7 masked-conv", MaskedConvSpectralAE)]


def _fit_bands(Cls, spectra, seed):
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        return Cls(seed=seed).fit(spectra).select(BUDGET)


def run_clean(seeds):
    noise = ArtifactConfig(rayleigh_strength=0.1, photon_scale=800, read_sigma=0.01)
    rows = {}
    rng = np.random.default_rng(0)
    for seed in seeds:
        spectra, gt, y, acq = clean.build_dataset(seed, noise)
        X, colmap = feature_matrix(spectra)
        F, _ = f_classif(X, y); F = np.nan_to_num(F)

        def rec(name, cols, bands=None):
            r = rows.setdefault(name, {"f1": [], "peak": []})
            r["f1"].append(knn_macro_f1(X[:, cols], y, seed))
            if bands is not None:
                r["peak"].append(validate_selection(gt, bands, tol_nm=10)["peak_recovery"])
        rec("all bands", list(range(X.shape[1])))
        rec("variance-ranking", list(np.argsort(X.var(0))[::-1][:BUDGET]),
            [colmap[c] for c in np.argsort(X.var(0))[::-1][:BUDGET]])
        rec("discriminability-oracle", list(np.argsort(F)[::-1][:BUDGET]),
            [colmap[c] for c in np.argsort(F)[::-1][:BUDGET]])
        rb = [colmap[c] for c in rng.choice(X.shape[1], BUDGET, replace=False)]
        rec("random", cols_for_bands(colmap, rb), rb)
        for name, Cls in LEARNED:
            bands = _fit_bands(Cls, spectra, seed)
            rec(name, cols_for_bands(colmap, bands), bands)
    return rows


def run_real(seeds):
    rows = {}
    for seed in seeds:
        spectra, gt, y, acq = real.build_dataset(seed)
        X, colmap = feature_matrix(spectra)
        disc = real.band_is_discriminative(colmap)
        F, _ = f_classif(X, y); F = np.nan_to_num(F)
        rng = np.random.default_rng(seed)

        def rec(name, cols):
            r = rows.setdefault(name, {"f1": [], "disc": []})
            r["f1"].append(knn_macro_f1(X[:, cols], y, seed))
            r["disc"].append(float(disc[cols].mean()) if len(cols) else 0.0)
        rec("all bands", list(range(X.shape[1])))
        rec("variance-ranking", list(np.argsort(X.var(0))[::-1][:BUDGET]))
        rec("discriminability-oracle", list(np.argsort(F)[::-1][:BUDGET]))
        rec("random", list(rng.choice(X.shape[1], BUDGET, replace=False)))
        for name, Cls in LEARNED:
            bands = _fit_bands(Cls, spectra, seed)
            rec(name, cols_for_bands(colmap, bands))
    return rows


def main():
    seeds = [1, 2, 3]
    order = ["all bands", "variance-ranking", "discriminability-oracle", "random",
             "C2 spectral-AE", "C3 masked-AE", "C5 deep-AE", "C6 conv-AE", "C7 masked-conv"]

    print("=" * 80)
    print("CLEAN gate (low noise, 3 scenes) — KNN macro-F1 / peak_recovery")
    print("=" * 80)
    rc = run_clean(seeds)
    print(f"{'method':<26}{'KNN-F1':>10}{'peak_rec':>10}")
    for m in order:
        d = rc.get(m, {})
        peak = f"{np.mean(d['peak']):.2f}" if d.get("peak") else "   -"
        print(f"{m:<26}{np.mean(d['f1']):>10.3f}{peak:>10}")

    print("\n" + "=" * 80)
    print("REALISTIC confounded regime (3 scenes) — KNN macro-F1 / % bands on discriminative window")
    print("=" * 80)
    rr = run_real(seeds)
    print(f"{'method':<26}{'KNN-F1':>10}{'disc-band %':>13}")
    for m in order:
        d = rr.get(m, {})
        print(f"{m:<26}{np.mean(d['f1']):>10.3f}{100*np.mean(d['disc']):>11.0f}%")
    print("-" * 80)
    print("Goal: the enlarged nets (C5/C6/C7) match the small AEs on the clean gate AND push the")
    print("realistic F1 toward the discriminability oracle (the headroom C2/C3 left on the table).")


if __name__ == "__main__":
    main()
