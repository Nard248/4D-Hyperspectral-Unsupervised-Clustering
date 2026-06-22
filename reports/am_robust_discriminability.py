"""#robust-discriminability band selection.

Strip the few loud (high-variance) PCA directions (scatter/turbidity/Raman nuisances),
KMeans-cluster the residual subspace, then rank bands by between-cluster f_classif F-ratio
using the CLUSTER labels (never real labels). How many directions to strip is decided from
the PCA explained-variance profile: flat profile (clean dye scene) => strip ~nothing
(loud == signal); steep head (realistic confounds) => strip exactly those few.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler

from sweep_common import topn_diverse


def _subsample(X, rng, m=2000):
    if X.shape[0] <= m:
        return X
    idx = rng.choice(X.shape[0], size=m, replace=False)
    return X[idx]


def _cluster_fratio(Xs, Z, k_strip, n_clusters, seed):
    """Cluster the residual subspace (PCA scores with the top-k_strip directions dropped)
    and return per-band f_classif F-scores vs the cluster labels."""
    tail = Z[:, k_strip:]
    if tail.shape[1] < 2:
        return None, 0.0
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(tail)
    if len(np.unique(labels)) < 2:
        return None, 0.0
    F, _ = f_classif(Xs, labels)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    centers = km.cluster_centers_
    overall = tail.mean(0, keepdims=True)
    counts = np.bincount(labels, minlength=n_clusters).astype(float)
    between = (counts[:, None] * (centers - overall) ** 2).sum()
    within = km.inertia_ + 1e-9
    weight = float(between / (between + within))
    return F, weight


def _loud_directions(evr):
    """How many leading PCA directions are loud nuisances: those whose explained-variance
    ratio sticks out far above the bulk. Flat profile => 0 (strip nothing); steep head =>
    the few huge PCs."""
    evr = np.asarray(evr, dtype=float)
    if len(evr) < 4:
        return 0
    body = np.median(evr[2:])
    thr = max(body * 4.0, evr.mean())
    k = 0
    for v in evr:
        if v > thr and k < 8:
            k += 1
        else:
            break
    return k


def select(X, colmap, n, seed, rng, spectra=None):
    X = np.asarray(X, dtype=float)
    n_bands = X.shape[1]
    if n >= n_bands:
        return list(range(n_bands))[:n]

    Xs = StandardScaler().fit_transform(X)

    Xsub = _subsample(Xs, rng, 2000)
    kmax = int(min(20, Xsub.shape[0] - 1, n_bands))
    pca = PCA(n_components=kmax, random_state=seed).fit(Xsub)
    Z_full = pca.transform(Xs)

    n_clusters = int(min(8, max(3, n // 2)))

    sub_idx = rng.choice(X.shape[0], size=min(2000, X.shape[0]), replace=False)
    Xs_sub = Xs[sub_idx]
    Z_sub = Z_full[sub_idx]

    # Strip-count COMMITTED from the EVR profile (the robust signal):
    #   k_loud == 0 -> flat spectrum, loud dirs ARE the dye signal -> window {0, 1}
    #   k_loud >= 1 -> a few PCs are nuisance variance -> window {k_loud, k_loud+1}, never 0.
    # Equal-average rank-normalized F over the tight committed window (no decisiveness
    # weight: the loudest/nuisance clustering is most "decisive" and would mislead).
    k_loud = _loud_directions(pca.explained_variance_ratio_)
    if k_loud == 0:
        cand = [0, 1]
    else:
        cand = [k_loud, k_loud + 1]
    cand = [k for k in cand if 0 <= k < Z_sub.shape[1] - 1 and k <= 8]
    if not cand:
        cand = [0]

    scores = []
    for k_strip in cand:
        F, _ = _cluster_fratio(Xs_sub, Z_sub, k_strip, n_clusters, seed)
        if F is None:
            continue
        order = np.argsort(np.argsort(F))
        rn = order / (len(order) - 1)
        scores.append(rn)

    if not scores:
        return topn_diverse(X.var(0), colmap, n)

    agg = np.mean(np.asarray(scores), axis=0)
    return topn_diverse(agg, colmap, n)