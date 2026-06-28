"""Round-3 M-repr — the AE's genuine theoretical edge: a NONLINEAR embedding should beat a LINEAR (PCA)
embedding for classification on a nonlinear manifold. Compare at MATCHED dimension (16): PCA-projection
(linear) vs AE-latent (nonlinear), plus their reconstructions, on strongly nonlinear regimes
(reabsorption, FRET) where H16b's linear-denoising win should NOT hold.

Run:  python reports/nonlinear_representation.py [--smoke]
"""
from __future__ import annotations

import contextlib
import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from classification_experiment import feature_matrix
from mixnoise_experiment import SIZE, _clfs, roi_mask
from realistic_benchmark import build_dataset
from fret_regime import build_fret_dataset
from swarm_zoo import FlexSpectralAE

PER_CLASS = 30
REPEATS = 4
Z = 16
AE_KW = dict(backbone="mlp", act="gelu", mask_ratio=0.4, latent_dim=Z, depth=4, width=256, epochs=400)


def fewshot(F, y, seed):
    out = []
    for r in range(REPEATS):
        rng = np.random.default_rng(seed * 100 + r)
        tr = np.concatenate([rng.choice(np.where(y == c)[0], PER_CLASS, replace=False) for c in np.unique(y)])
        te = np.setdiff1d(np.arange(len(y)), tr)
        sc = StandardScaler().fit(F[tr])
        Ftr, Fte = sc.transform(F[tr]), sc.transform(F[te])
        fs = []
        for n, c in _clfs(seed).items():
            with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
                c.fit(Ftr, y[tr]); fs.append(f1_score(y[te], c.predict(Fte), average="macro"))
        out.append(max(fs))
    return float(np.mean(out))


def regimes(seed, smoke):
    out = {}
    sp, gt, y, acq = build_dataset(seed, size=SIZE, em_step=5, nuisance_amp=1.0, turbidity_amp=0.5, rayleigh=0.3,
                                   raman=0.25, photon_scale=2000, read_sigma=0.005, reabsorption=True,
                                   reabsorption_strength=5.0)
    out["reabsorb"] = (sp, y)
    if not smoke:
        sp2, y2 = build_fret_dataset(seed, size=SIZE, k=5.0)
        out["fret"] = (sp2, y2)
    return out


def main():
    smoke = "--smoke" in sys.argv
    seeds = ([1] if smoke else [1, 2, 3])
    feats = ["raw", "PCA-proj16", "PCA-rec16", "AE-latent16", "AE-rec"]
    agg = {}
    for seed in seeds:
        for reg, (sp, y) in regimes(seed, smoke).items():
            X, colmap = feature_matrix(sp)
            roi = roi_mask(seed, SIZE); Xr, yr = X[roi], y[roi]
            Xs = StandardScaler().fit_transform(Xr)
            p = PCA(Z, random_state=0).fit(Xs)
            proj = p.transform(Xs); rec = p.inverse_transform(proj)
            with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
                m = FlexSpectralAE(seed=seed, **AE_KW).fit(sp)
            lat = np.asarray(m._latent.cpu().numpy(), float)[roi]
            aerec = np.asarray(m._recon.cpu().numpy(), float)[roi]
            F = {"raw": Xr, "PCA-proj16": proj, "PCA-rec16": rec, "AE-latent16": lat, "AE-rec": aerec}
            for k, v in F.items():
                agg.setdefault((reg, k), []).append(fewshot(v, yr, seed))
    print("=" * 80)
    print(f"M-repr — nonlinear (AE) vs linear (PCA) embedding, matched dim={Z}, best-NL F1")
    print("=" * 80)
    for reg in (["reabsorb"] if smoke else ["reabsorb", "fret"]):
        print(f"\n### {reg}")
        base = np.mean(agg[(reg, "PCA-proj16")])
        for k in feats:
            v = np.mean(agg[(reg, k)])
            tag = f"  ({v-base:+.3f} vs PCA-proj)" if k.startswith("AE") else ""
            print(f"  {k:<13}{v:>8.3f}{tag}")
    print("\nWin = AE-latent/AE-rec > PCA-proj/PCA-rec on nonlinear data (nonlinear embedding helps).")


if __name__ == "__main__":
    main()
