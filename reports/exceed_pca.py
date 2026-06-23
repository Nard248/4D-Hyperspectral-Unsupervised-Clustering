"""Approaches designed to EXCEED pca_load on realistic data by using the AE for what PCA cannot do —
suppress the nuisances. All blind/unsupervised, all built on the masked AE (denoising):

  ae_denoise_var   - variance of the masked-AE *reconstruction* (nuisances it can't reconstruct from
                     masked input are suppressed -> signal-band variance dominates).
  ae_latent_clusterF - KMeans on the AE *latent* (which encodes discriminative structure, not the
                     nuisances) -> per-band f_classif vs the cluster labels (unsupervised oracle on a
                     denoised embedding).
  hybrid_infl_pca  - z-fusion of the AE perturbation influence and the PCA-loading score.
  ensemble_all     - z-fusion of pca-loading + ae-denoise-var + ae-latent-clusterF.

Run:  python reports/exceed_pca.py
"""
from __future__ import annotations

import contextlib
import functools
import os

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif

import method_zoo as mz
import sweep_common as sc
from classification_experiment import cols_for_bands, knn_macro_f1
from spectral_select.architectures import MaskedSpectralAE
from sweep_common import topn_diverse


def _fit_ae(spectra, seed, epochs=300):
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        return MaskedSpectralAE(seed=seed, epochs=epochs, mask_ratio=0.5).fit(spectra)


def ae_denoise_var(X, colmap, n, seed, rng, spectra):
    m = _fit_ae(spectra, seed)
    R = m._recon.cpu().numpy()
    return topn_diverse(R.var(0), colmap, n)


def ae_latent_clusterF(X, colmap, n, seed, rng, spectra, k=6):
    m = _fit_ae(spectra, seed)
    Z = m._latent.cpu().numpy()
    lab = KMeans(k, random_state=seed, n_init=4).fit_predict(Z)
    if len(np.unique(lab)) < 2:
        return topn_diverse(X.var(0), colmap, n)
    F, _ = f_classif(X, lab)
    return topn_diverse(np.nan_to_num(F), colmap, n)


def _z(s):
    s = np.asarray(s, float)
    return (s - s.mean()) / (s.std() + 1e-9)


def _pca_score(X, seed, k=6):
    return np.sum(np.abs(PCA(k, random_state=seed).fit(mz._std(X)).components_), axis=0)


def hybrid_infl_pca(X, colmap, n, seed, rng, spectra):
    m = _fit_ae(spectra, seed)
    return topn_diverse(_z(m._band_influence()) + _z(_pca_score(X, seed)), colmap, n)


def ensemble_all(X, colmap, n, seed, rng, spectra):
    m = _fit_ae(spectra, seed)
    R = m._recon.cpu().numpy()
    Z = m._latent.cpu().numpy()
    lab = KMeans(6, random_state=seed, n_init=4).fit_predict(Z)
    F = np.nan_to_num(f_classif(X, lab)[0]) if len(np.unique(lab)) > 1 else X.var(0)
    return topn_diverse(_z(_pca_score(X, seed)) + _z(R.var(0)) + _z(F), colmap, n)


def main():
    seeds = (1, 2, 3)
    cache = sc.build_cache(["clean", "realistic"], seeds)
    ref = sc.score_method(functools.partial(mz.pca_load, k=6), cache, regimes=["clean", "realistic"], seeds=seeds)
    oracle = float(np.mean([cache[(g, s)][4] for g in ["clean", "realistic"] for s in seeds]))
    print("=" * 80)
    print(f"EXCEED pca_load — target realistic={ref['realistic']:.3f} clean={ref['clean']:.3f} | oracle={oracle:.3f}")
    print("=" * 80)
    print(f"{'method':<22}{'clean':>9}{'realistic':>11}{'beats pca?':>12}")
    print(f"{'pca_load[k6] (ref)':<22}{ref['clean']:>9.3f}{ref['realistic']:>11.3f}")
    for name, fn in [("ae_denoise_var", ae_denoise_var), ("ae_latent_clusterF", ae_latent_clusterF),
                     ("hybrid_infl_pca", hybrid_infl_pca), ("ensemble_all", ensemble_all)]:
        r = sc.score_method(fn, cache, regimes=["clean", "realistic"], seeds=seeds)
        beat = "YES" if r["realistic"] > ref["realistic"] else ""
        print(f"{name:<22}{r['clean']:>9.3f}{r['realistic']:>11.3f}{beat:>12}")


if __name__ == "__main__":
    main()
