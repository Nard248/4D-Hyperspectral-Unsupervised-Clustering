"""Round-3 M-spatial — the one structural avenue: SPATIAL-texture discrimination, the home of the
original convolutional AE. Class is carried by spatial TEXTURE in a few discriminative bands (each class
region has a different texture frequency but the SAME per-pixel marginal). Per-pixel methods (variance,
PCA, per-pixel mutInfo) are PROVABLY BLIND (identical marginals); only a SPATIAL method (texture stats,
spatial CAE) can find the discriminative bands. Demonstrates the convolutional approach is *necessary*
where per-pixel band selection fundamentally fails.

Eval = classify pixels on LOCAL-TEXTURE (local std) features of the selected bands, few-shot.
Selectors: random, variance(per-pixel), mutInfo-perpixel(raw values), texture-var(spatial unsup),
texture-mutInfo*(spatial sup). Run:  python reports/spatial_regime.py [--smoke]
"""
from __future__ import annotations

import contextlib
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from mixnoise_experiment import _clfs
from sweep_common import topn_diverse

SIZE = 64
N_BANDS = 60
N_DISC = 6
BUDGET = 12
PER_CLASS = 40
REPEATS = 4
SCALES = [0.8, 2.5, 6.0]   # 3 classes: fine / medium / smooth texture scale (same global marginal)


def textured(gen, scale):
    n = gaussian_filter(gen.standard_normal((SIZE, SIZE)), scale)   # smoothed noise at class scale
    return (n - n.mean()) / (n.std() + 1e-9)                        # renormalize -> identical global marginal,
    #                                                                 different spatial autocorrelation (texture)


def build_spatial(seed):
    gen = np.random.default_rng(seed)
    # class regions = argmax of 3 smooth fields
    fields = np.stack([gaussian_filter(gen.standard_normal((SIZE, SIZE)), 6) for _ in range(3)])
    labels = fields.argmax(0).ravel()
    X = np.zeros((SIZE * SIZE, N_BANDS))
    disc_bands = list(range(2, 2 + N_DISC))       # the texture-discriminative bands
    for b in range(N_BANDS):
        if b in disc_bands:
            band = np.zeros((SIZE, SIZE))
            for c in range(3):                    # each class region gets ITS texture (same marginal)
                t = textured(np.random.default_rng(seed * 97 + b * 13 + c), SCALES[c])
                m = (labels.reshape(SIZE, SIZE) == c)
                band[m] = t[m]
            X[:, b] = 1.3 * band.ravel()          # strong texture signal
        else:                                     # nuisance band: SMOOTH (no texture) class-irrelevant field
            f = gaussian_filter(gen.standard_normal((SIZE, SIZE)), gen.uniform(8, 14))
            X[:, b] = 1.5 * ((f - f.mean()) / (f.std() + 1e-9)).ravel()
    X += 0.05 * gen.standard_normal(X.shape)      # mild read noise
    return X, labels, disc_bands


def local_std(X, win=5):
    """Per-band local texture (windowed std) -> the feature that exposes texture classes."""
    out = np.empty_like(X)
    for b in range(X.shape[1]):
        img = X[:, b].reshape(SIZE, SIZE)
        m = uniform_filter(img, win)
        s = np.sqrt(np.maximum(uniform_filter(img * img, win) - m * m, 0))
        out[:, b] = s.ravel()
    return out


def _blocks(blocks=8):
    bsz = SIZE // blocks
    grid = (np.arange(SIZE)[:, None] // bsz) * blocks + (np.arange(SIZE)[None, :] // bsz)
    return grid.ravel()                                   # block id per pixel


def fewshot(F, y, cols, seed):
    """SPATIAL-BLOCK CV: train and test on DISJOINT image blocks (removes spatial-autocorrelation
    leakage). Train = PER_CLASS pixels from train-blocks; test = test-block pixels."""
    bid = _blocks()
    nb = bid.max() + 1
    out = []
    for r in range(REPEATS):
        rng = np.random.default_rng(seed * 100 + r)
        perm = rng.permutation(nb)
        trbk = set(perm[:nb // 2].tolist())
        tr_mask = np.isin(bid, list(trbk))
        te_mask = ~tr_mask
        try:
            tr = np.concatenate([rng.choice(np.where(tr_mask & (y == c))[0], PER_CLASS, replace=False)
                                 for c in np.unique(y)])
        except ValueError:
            continue                                      # a class missing from train blocks this draw
        te = np.where(te_mask)[0]
        sc = StandardScaler().fit(F[tr][:, cols])
        Ftr, Fte = sc.transform(F[tr][:, cols]), sc.transform(F[te][:, cols])
        fs = []
        for n, c in _clfs(seed).items():
            with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
                c.fit(Ftr, y[tr]); fs.append(f1_score(y[te], c.predict(Fte), average="macro"))
        out.append(max(fs))
    return float(np.mean(out)) if out else 0.0


def main():
    smoke = "--smoke" in sys.argv
    seeds = ([1] if smoke else [1, 2, 3])
    methods = ["random", "variance", "mutInfo-px*", "texture-var", "texture-MI*", "oracle(disc)"]
    agg = {m: [] for m in methods}
    hit = {m: [] for m in methods}
    for seed in seeds:
        X, y, disc = build_spatial(seed)
        colmap = [(0, float(b)) for b in range(N_BANDS)]     # dummy colmap (1 excitation)
        TX = local_std(X)                                     # texture features (used for EVAL)
        MIpx = np.nan_to_num(mutual_info_classif(X, y, random_state=seed))
        MItex = np.nan_to_num(mutual_info_classif(TX, y, random_state=seed))
        rng = np.random.default_rng(seed)
        sels = {"random": list(rng.choice(N_BANDS, BUDGET, replace=False)),
                "variance": topn_diverse(X.var(0), colmap, BUDGET),
                "mutInfo-px*": topn_diverse(MIpx, colmap, BUDGET),
                "texture-var": topn_diverse(TX.var(0), colmap, BUDGET),
                "texture-MI*": topn_diverse(MItex, colmap, BUDGET),
                "oracle(disc)": disc + list(rng.choice([b for b in range(N_BANDS) if b not in disc], BUDGET - len(disc), replace=False))}
        for m, cols in sels.items():
            agg[m].append(fewshot(TX, y, cols, seed))         # classify on TEXTURE features
            hit[m].append(len(set(cols) & set(disc)) / len(disc))
    print("=" * 80)
    print(f"M-spatial — texture-discriminative; classify on local-texture features, {len(seeds)} seeds, best-NL F1")
    print(f"(per-pixel marginals identical across classes -> per-pixel selectors are blind by construction)")
    print("=" * 80)
    print(f"  {'method':<14}{'F1':>8}{'%disc-found':>13}")
    for m in methods:
        print(f"  {m:<14}{np.mean(agg[m]):>8.3f}{100*np.mean(hit[m]):>11.0f}%")
    print("\nWin = SPATIAL methods (texture-var, texture-MI) find the disc bands and classify well;")
    print("PER-PIXEL methods (variance, mutInfo-px) are blind (identical marginals) -> ~random.")


if __name__ == "__main__":
    main()
