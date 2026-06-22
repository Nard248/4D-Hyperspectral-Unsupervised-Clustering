"""A zoo of blind/unsupervised band-selection methods (the breadth of the search).

Every method implements the sweep_common contract: ``select(X, colmap, n, seed, rng, spectra=None)
-> list[int]``. Graph/correlation methods subsample pixels for speed; all are robust (fall back to
variance ranking on numerical failure). HYPERPARAMS lists the configs the grand sweep expands.
"""
from __future__ import annotations

import contextlib
import os

import numpy as np

from sweep_common import topn_diverse


def _sub(X, rng, k=800):
    if X.shape[0] <= k:
        return X
    return X[rng.choice(X.shape[0], k, replace=False)]


def _std(X):
    return (X - X.mean(0)) / (X.std(0) + 1e-9)


def _prep(X, colmap, mode):
    """Per-pixel preprocessing to suppress nuisances before selection (indices preserved):
    l2 = remove brightness magnitude (shape only); snv = standard-normal-variate; deriv = per-excitation
    first derivative (kills broad smooth backgrounds)."""
    if mode in (None, "none"):
        return X
    if mode == "l2":
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    if mode == "snv":
        return (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-9)
    if mode == "deriv":
        ex = np.array([e for e, _ in colmap])
        out = np.zeros_like(X, dtype=float)
        for e in np.unique(ex):
            idx = np.where(ex == e)[0]
            out[:, idx] = np.gradient(X[:, idx], axis=1) if len(idx) > 1 else X[:, idx]
        return out
    raise ValueError(f"unknown prep {mode}")


# ---- statistical -------------------------------------------------------------------------------
def variance(X, colmap, n, seed, rng, spectra=None):
    return topn_diverse(X.var(0), colmap, n)


def mean_intensity(X, colmap, n, seed, rng, spectra=None):
    return topn_diverse(X.mean(0), colmap, n)


def energy(X, colmap, n, seed, rng, spectra=None):
    return topn_diverse((X ** 2).mean(0), colmap, n)


def cv(X, colmap, n, seed, rng, spectra=None):
    return topn_diverse(X.std(0) / (np.abs(X.mean(0)) + 1e-9), colmap, n)


def band_entropy(X, colmap, n, seed, rng, spectra=None, bins=24):
    Xs = _sub(X, rng)
    sc = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        h, _ = np.histogram(Xs[:, j], bins=bins)
        p = h / (h.sum() + 1e-12)
        sc[j] = -(p[p > 0] * np.log(p[p > 0])).sum()
    return topn_diverse(sc, colmap, n)


def derivative_energy(X, colmap, n, seed, rng, spectra=None):
    d = np.diff(X, axis=1, prepend=X[:, :1])
    return topn_diverse(d.var(0), colmap, n)


# ---- decompositions ----------------------------------------------------------------------------
def pca_load(X, colmap, n, seed, rng, spectra=None, k=10, weighted=False, prep="none"):
    from sklearn.decomposition import PCA
    Xp = _prep(X, colmap, prep)
    k = int(min(k, Xp.shape[1], Xp.shape[0] - 1))
    p = PCA(n_components=k, random_state=seed).fit(_std(Xp))
    w = p.explained_variance_ratio_[:, None] if weighted else 1.0
    return topn_diverse(np.sum(np.abs(p.components_) * w, axis=0), colmap, n)


def sparsepca_load(X, colmap, n, seed, rng, spectra=None, k=8):
    from sklearn.decomposition import SparsePCA
    k = int(min(k, X.shape[1]))
    p = SparsePCA(n_components=k, random_state=seed, max_iter=100).fit(_sub(_std(X), rng, 600))
    return topn_diverse(np.sum(np.abs(p.components_), axis=0), colmap, n)


def ica_load(X, colmap, n, seed, rng, spectra=None, k=10):
    from sklearn.decomposition import FastICA
    k = int(min(k, X.shape[1]))
    ica = FastICA(n_components=k, random_state=seed, max_iter=400, whiten="unit-variance").fit(_sub(X, rng))
    return topn_diverse(np.sum(np.abs(ica.mixing_), axis=1), colmap, n)


def nmf_basis(X, colmap, n, seed, rng, spectra=None, k=8):
    from sklearn.decomposition import NMF
    k = int(min(k, X.shape[1]))
    H = NMF(n_components=k, init="nndsvda", random_state=seed, max_iter=400).fit(np.clip(X, 0, None)).components_
    return topn_diverse(H.max(axis=0), colmap, n)


def svd_load(X, colmap, n, seed, rng, spectra=None, k=10):
    from sklearn.decomposition import TruncatedSVD
    k = int(min(k, X.shape[1] - 1))
    s = TruncatedSVD(n_components=k, random_state=seed).fit(X)
    return topn_diverse(np.sum(np.abs(s.components_), axis=0), colmap, n)


def parafac_peaks(X, colmap, n, seed, rng, spectra=None, rank=4):
    """CP/PARAFAC decomposition of the (pixels x excitation x emission) tensor — exploits the trilinear
    EEM structure: each component is a (pixel-load, excitation-profile, emission-profile)."""
    import tensorly as tl
    from tensorly.decomposition import parafac
    ex_list = sorted({ex for ex, _ in colmap})
    n_ex = len(ex_list); n_band = X.shape[1] // n_ex
    T = tl.tensor(_sub(X, rng, 1000).reshape(-1, n_ex, n_band).astype("float64"))
    _, factors = parafac(T, rank=int(rank), n_iter_max=150, init="svd", random_state=seed)
    exf, emf = np.abs(factors[1]), np.abs(factors[2])               # (n_ex,r), (n_band,r)
    score = (exf[:, None, :] * emf[None, :, :]).sum(axis=2).reshape(-1)   # (n_ex*n_band,)
    return topn_diverse(score, colmap, n)


# ---- redundancy-aware --------------------------------------------------------------------------
def mrmr_unsup(X, colmap, n, seed, rng, spectra=None, alpha=1.0):
    Xs = _std(_sub(X, rng, 600))
    rel = X.var(0); rel = rel / (rel.max() + 1e-12)
    chosen, cand = [], list(range(X.shape[1]))
    corr = np.abs(np.corrcoef(Xs.T))
    np.fill_diagonal(corr, 0.0)
    while len(chosen) < n and cand:
        if chosen:
            red = corr[np.ix_(cand, chosen)].mean(axis=1)
        else:
            red = np.zeros(len(cand))
        j = cand[int(np.argmax(rel[cand] - alpha * red))]
        chosen.append(j); cand.remove(j)
    return chosen


def band_cluster(X, colmap, n, seed, rng, spectra=None, method="average"):
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    Xs = _std(_sub(X, rng))
    D = 1 - np.abs(np.corrcoef(Xs.T))
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(np.clip(D, 0, 2), checks=False), method=method)
    lab = fcluster(Z, t=n, criterion="maxclust")
    var = X.var(0); chosen = []
    for cl in np.unique(lab):
        idx = np.where(lab == cl)[0]
        chosen.append(int(idx[np.argmax(var[idx])]))
    if len(chosen) < n:
        chosen += [j for j in np.argsort(var)[::-1] if j not in chosen][:n - len(chosen)]
    return chosen[:n]


def maxvar_decorrelate(X, colmap, n, seed, rng, spectra=None, thresh=0.9):
    Xs = _std(_sub(X, rng))
    corr = np.abs(np.corrcoef(Xs.T))
    order = list(np.argsort(X.var(0))[::-1])
    chosen = []
    for j in order:
        if all(corr[j, c] < thresh for c in chosen):
            chosen.append(int(j))
        if len(chosen) == n:
            break
    if len(chosen) < n:
        chosen += [j for j in order if j not in chosen][:n - len(chosen)]
    return chosen[:n]


# ---- manifold ----------------------------------------------------------------------------------
def laplacian_score(X, colmap, n, seed, rng, spectra=None, k_nn=5):
    from sklearn.neighbors import kneighbors_graph
    Xs = _std(_sub(X, rng, 600))
    W = kneighbors_graph(Xs, n_neighbors=k_nn, mode="connectivity", include_self=False)
    W = 0.5 * (W + W.T)
    D = np.asarray(W.sum(1)).ravel()
    Wd = W.toarray()
    one = np.ones(Xs.shape[0])
    sc = np.zeros(Xs.shape[1])
    for f in range(Xs.shape[1]):
        fr = Xs[:, f]
        fr = fr - (fr @ D) / (one @ D)
        num = fr @ (D * fr) - fr @ (Wd @ fr)
        den = fr @ (D * fr) + 1e-12
        sc[f] = num / den
    return topn_diverse(-sc, colmap, n)            # low Laplacian score = good -> negate


# ---- unsupervised discriminability (the blind analog of the oracle) ----------------------------
def cluster_fratio(X, colmap, n, seed, rng, spectra=None, k=6, method="kmeans", pca=0):
    """Cluster pixels BLINDLY, then rank bands by between-cluster F-ratio using the *cluster* labels —
    an unsupervised proxy for the supervised discriminability oracle. ``pca>0`` clusters on a PCA
    embedding (denoise) so clusters form on structure rather than the loudest nuisance band."""
    from sklearn.cluster import KMeans
    from sklearn.feature_selection import f_classif
    from sklearn.mixture import GaussianMixture
    Xs = _std(X)
    Z = Xs
    if pca:
        from sklearn.decomposition import PCA
        Z = PCA(n_components=int(min(pca, Xs.shape[1])), random_state=seed).fit_transform(Xs)
    if method == "gmm":
        lab = GaussianMixture(int(k), random_state=seed, max_iter=100).fit_predict(Z)
    else:
        lab = KMeans(int(k), random_state=seed, n_init=4).fit_predict(Z)
    if len(np.unique(lab)) < 2:
        return topn_diverse(X.var(0), colmap, n)
    F, _ = f_classif(X, lab)
    return topn_diverse(np.nan_to_num(F), colmap, n)


# ---- learned: AE perturbation family (need spectra) --------------------------------------------
def _ae(Cls, spectra, colmap, n, seed, **kw):
    from classification_experiment import cols_for_bands
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        from spectral_select.architectures import CANDIDATES, LARGE_CANDIDATES
        m = Cls(seed=seed, **kw).fit(spectra)
    return cols_for_bands(colmap, m.select(n))


def ae_spectral(X, colmap, n, seed, rng, spectra=None):
    from spectral_select.architectures import SpectralAE
    return _ae(SpectralAE, spectra, colmap, n, seed)


def ae_masked(X, colmap, n, seed, rng, spectra=None):
    from spectral_select.architectures import MaskedSpectralAE
    return _ae(MaskedSpectralAE, spectra, colmap, n, seed)


def ae_masked_conv(X, colmap, n, seed, rng, spectra=None):
    from spectral_select.architectures import MaskedConvSpectralAE
    return _ae(MaskedConvSpectralAE, spectra, colmap, n, seed)


# ---- registry + hyperparameter grid ------------------------------------------------------------
from gate_zoo import bsnet_attention, concrete_ae, stochastic_gates       # end-to-end gate selectors

SELECTORS = {
    "variance": variance, "mean": mean_intensity, "energy": energy, "cv": cv,
    "entropy": band_entropy, "derivative": derivative_energy,
    "pca_load": pca_load, "sparsepca": sparsepca_load, "ica": ica_load, "nmf": nmf_basis,
    "svd": svd_load, "parafac": parafac_peaks,
    "mrmr": mrmr_unsup, "band_cluster": band_cluster, "maxvar_decorr": maxvar_decorrelate,
    "laplacian": laplacian_score, "cluster_fratio": cluster_fratio,
    "ae_spectral": ae_spectral, "ae_masked": ae_masked, "ae_masked_conv": ae_masked_conv,
    "concrete_ae": concrete_ae, "stochastic_gates": stochastic_gates, "bsnet": bsnet_attention,
}

# method -> list of kwarg dicts to sweep (label suffix derived from kwargs)
HYPERPARAMS = {
    "pca_load": [{"k": k, "weighted": w} for k in (4, 8, 16, 32) for w in (False, True)],
    "sparsepca": [{"k": k} for k in (4, 8, 16)],
    "ica": [{"k": k} for k in (4, 8, 16)],
    "nmf": [{"k": k} for k in (3, 5, 8, 12)],
    "svd": [{"k": k} for k in (8, 16, 32)],
    "parafac": [{"rank": r} for r in (2, 3, 4, 6, 8)],
    "mrmr": [{"alpha": a} for a in (0.3, 0.6, 1.0, 2.0)],
    "band_cluster": [{"method": m} for m in ("average", "ward", "complete")],
    "maxvar_decorr": [{"thresh": t} for t in (0.8, 0.9, 0.95)],
    "laplacian": [{"k_nn": k} for k in (5, 10)],
    "entropy": [{"bins": b} for b in (16, 32)],
    "concrete_ae": [{"epochs": 250}, {"epochs": 250, "hidden": 256}],
    "stochastic_gates": [{"lam": l} for l in (0.02, 0.05, 0.1)],
    "cluster_fratio": [{"k": k, "pca": p, "method": m}
                       for k in (4, 6, 10) for p in (0, 8) for m in ("kmeans", "gmm")],
}


def expand():
    """Yield (label, callable) for every method x hyperparameter config."""
    import functools
    for name, fn in SELECTORS.items():
        grid = HYPERPARAMS.get(name, [{}])
        for kw in grid:
            label = name + ("" if not kw else "[" + ",".join(f"{k}={v}" for k, v in kw.items()) + "]")
            yield label, functools.partial(fn, **kw)
