"""Shared contract + evaluation for the band-selection method search.

A *method* is a function:  ``select(X, colmap, n, seed, rng, spectra=None) -> list[int]``
returning ``n`` column indices into the (pixels x bands) feature matrix ``X`` (``colmap[j]`` is the
``(excitation_nm, emission_nm)`` of column j). Methods are **blind/unsupervised** — they never see the
labels ``y``; labels are used only to *score* the chosen subset (downstream KNN macro-F1) and to
compute the discriminability **oracle** ceiling.

Data regimes span clean -> realistic -> heavy-confound so a winner must be robust, not regime-tuned.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_selection import f_classif

import classification_experiment as ce
import realistic_benchmark as rb
from classification_experiment import feature_matrix, knn_macro_f1
from spectraforge import ArtifactConfig

# name -> kwargs builder; clean uses the FPbase 3-dye scene, the rest use the confounded generator.
REGIMES = ["clean", "mild", "realistic", "dense"]


def make_regime(name: str, seed: int):
    """Return (X, colmap, y, spectra) for a named data regime."""
    if name == "clean":
        noise = ArtifactConfig(rayleigh_strength=0.1, photon_scale=800, read_sigma=0.01)
        spectra, gt, y, acq = ce.build_dataset(seed, noise)
    elif name == "mild":
        spectra, gt, y, acq = rb.build_dataset(seed, nuisance_amp=0.5, turbidity_amp=0.5)
    elif name == "realistic":
        spectra, gt, y, acq = rb.build_dataset(seed)
    elif name == "dense":
        spectra, gt, y, acq = rb.build_dataset(seed, nuisance_amp=4.0, turbidity_amp=1.5, raman=0.6)
    else:
        raise ValueError(f"unknown regime {name}")
    X, colmap = feature_matrix(spectra)
    return X, colmap, np.asarray(y), spectra


def topn_diverse(score, colmap, n, dedup_nm=10.0):
    """Top-n columns by descending score, skipping bands within dedup_nm of a chosen band at the
    same excitation (the standard diversity rule)."""
    chosen: list[int] = []
    for j in np.argsort(np.asarray(score))[::-1]:
        ex, em = colmap[int(j)]
        if all(not (abs(ex - colmap[c][0]) < 1 and abs(em - colmap[c][1]) < dedup_nm) for c in chosen):
            chosen.append(int(j))
        if len(chosen) == n:
            break
    return chosen


def oracle_cols(X, y, n):
    F, _ = f_classif(X, y)
    return list(np.argsort(np.nan_to_num(F))[::-1][:n])


def build_cache(regimes=REGIMES, seeds=(1, 2, 3)):
    """Build every (regime, seed) dataset once: {(regime, seed): (X, colmap, y, spectra, oracle_f1)}."""
    cache = {}
    for r in regimes:
        for s in seeds:
            X, colmap, y, spectra = make_regime(r, s)
            ocols = oracle_cols(X, y, 12)
            cache[(r, s)] = (X, colmap, y, spectra, knn_macro_f1(X[:, ocols], y, s))
    return cache


def score_method(select_fn, cache, n=12, regimes=REGIMES, seeds=(1, 2, 3)):
    """Run a method over the cached datasets. Returns per-regime mean F1 + overall MEAN_F1 / VS_ORACLE.
    A method that raises on any fold is recorded as NaN there (so a broken method doesn't kill a sweep)."""
    out = {}
    all_f1, all_or = [], []
    for r in regimes:
        f1s, ors = [], []
        for s in seeds:
            X, colmap, y, spectra, ocl = cache[(r, s)]
            try:
                cols = list(select_fn(X, colmap, n, s, np.random.default_rng(s), spectra))
                f1 = knn_macro_f1(X[:, cols], y, s) if cols else float("nan")
            except Exception:
                f1 = float("nan")
            f1s.append(f1); ors.append(ocl)
        out[r] = float(np.nanmean(f1s)); all_f1 += f1s; all_or += ors
    out["MEAN_F1"] = float(np.nanmean(all_f1))
    out["MEAN_ORACLE"] = float(np.nanmean(all_or))
    out["VS_ORACLE"] = float(out["MEAN_F1"] / out["MEAN_ORACLE"]) if out["MEAN_ORACLE"] else float("nan")
    return out


def evaluate(select_fn, regimes=REGIMES, seeds=(1, 2, 3), n=12):
    """Convenience for a single method (used by exploration agents): builds the cache and scores."""
    return score_method(select_fn, build_cache(regimes, seeds), n=n, regimes=regimes, seeds=seeds)
