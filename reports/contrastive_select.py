"""Round-3 M1 (flagship) — fix the CORE failure (blind selection ≈ random under clutter) at its cause.

WHY blind selection fails: reconstruction/variance objectives rank bands by VARIANCE, and clutter is
high-variance-but-class-irrelevant, so they pick clutter ≈ randomly. The objective is misaligned with
discriminability.

MITIGATION: a self-supervised CONTRASTIVE encoder that is INVARIANT to the (known) nuisance model.
For each pixel spectrum we make two augmented views by adding synthetic clutter (random low-rank
band-gain perturbations, like the renderer's clutter) + photon/read noise + illumination gain. NT-Xent
pulls the two views together and pushes different pixels apart → the encoder learns features invariant
to clutter/noise but sensitive to the underlying spectrum (signal). Band selection = occlusion influence
on this invariant representation. This should beat random under clutter where reconstruction/PCA fail.

Eval (honesty controls kept): vs random, variance, pca_load, AE(recon), mutInfo* on the cluttered L4
regime, few-shot best-NL F1. Run:  python reports/contrastive_select.py [--smoke]
"""
from __future__ import annotations

import contextlib
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

import method_zoo as mz
from classification_experiment import cols_for_bands, feature_matrix
from mixnoise_experiment import BUDGET, EM_STEP, LEVELS, NBANDS, SIZE, _AE_CONV, _clfs, roi_mask
from realistic_benchmark import build_dataset
from sweep_common import topn_diverse
from swarm_zoo import FlexSpectralAE

PER_CLASS = 30
REPEATS = 4
torch.manual_seed(0)


class Encoder(nn.Module):
    def __init__(self, d, h=256, z=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, z))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def make_clutter_basis(d, n_modes, gen):
    """Random low-rank band-gain vectors mimicking renderer clutter (gain over a random band subset)."""
    G = torch.zeros(n_modes, d)
    for k in range(n_modes):
        lo, hi = d // 8, d // 3
        m = int(torch.randint(lo, hi, (1,), generator=gen))
        idx = torch.randperm(d, generator=gen)[:m]
        G[k, idx] = torch.randn(m, generator=gen)
    return G


def augment(x, basis, gen, clutter=1.0, noise=0.15, gain=0.1):
    B, d = x.shape
    a = torch.randn(B, basis.shape[0], generator=gen) * clutter         # clutter mode amplitudes
    x = x + a @ basis                                                   # additive low-rank clutter
    x = x * (1 + gain * torch.randn(B, 1, generator=gen))              # illumination gain
    x = x + noise * torch.randn(B, d, generator=gen)                  # photon/read noise
    return x


def nt_xent(z1, z2, temp=0.5):
    B = z1.shape[0]
    z = torch.cat([z1, z2], 0)
    sim = (z @ z.t()) / temp
    sim.fill_diagonal_(-1e9)
    targets = torch.cat([torch.arange(B) + B, torch.arange(B)])
    return F.cross_entropy(sim, targets)


def train_contrastive(Xstd, seed, epochs=300, batch=256, n_modes=24):
    gen = torch.Generator().manual_seed(seed)
    Xt = torch.tensor(Xstd, dtype=torch.float32)
    d = Xt.shape[1]
    basis = make_clutter_basis(d, n_modes, gen)
    enc = Encoder(d)
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3, weight_decay=1e-5)
    n = Xt.shape[0]
    for ep in range(epochs):
        idx = torch.randperm(n, generator=gen)[:batch]
        xb = Xt[idx]
        v1 = augment(xb, basis, gen); v2 = augment(xb, basis, gen)
        loss = nt_xent(enc(v1), enc(v2))
        opt.zero_grad(); loss.backward(); opt.step()
    return enc, Xt


def contrastive_influence(enc, Xt, sample=1024, seed=0):
    """Occlusion influence: zero each band (=mean in std space), measure mean embedding shift."""
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(Xt.shape[0], generator=gen)[:sample]
    x = Xt[idx]
    with torch.no_grad():
        z0 = enc(x)
        infl = np.zeros(Xt.shape[1])
        for j in range(Xt.shape[1]):
            xj = x.clone(); xj[:, j] = 0.0
            infl[j] = (enc(xj) - z0).pow(2).sum(1).mean().item()
    return infl


def fewshot(X, y, cols, seed):
    out = []
    for r in range(REPEATS):
        rng = np.random.default_rng(seed * 100 + r)
        tr = np.concatenate([rng.choice(np.where(y == c)[0], PER_CLASS, replace=False) for c in np.unique(y)])
        te = np.setdiff1d(np.arange(len(y)), tr)
        sc = StandardScaler().fit(X[tr][:, cols])
        Xtr, Xte = sc.transform(X[tr][:, cols]), sc.transform(X[te][:, cols])
        fs = []
        for n, c in _clfs(seed).items():
            with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
                c.fit(Xtr, y[tr]); fs.append(f1_score(y[te], c.predict(Xte), average="macro"))
        out.append(max(fs))
    return float(np.mean(out))


def main():
    smoke = "--smoke" in sys.argv
    seeds = ([1] if smoke else [1, 2, 3])
    params = LEVELS["L4-high"]
    acc = {m: [] for m in ["random", "variance", "pca_load", "AE", "mutInfo*", "contrastive"]}
    rnd_all = []
    for seed in seeds:
        sp, gt, y, acq = build_dataset(seed, size=SIZE, em_step=EM_STEP, **params)
        X, colmap = feature_matrix(sp)
        roi = roi_mask(seed, SIZE); Xr, yr = X[roi], y[roi]
        Xstd = StandardScaler().fit_transform(Xr)
        enc, Xt = train_contrastive(Xstd, seed, epochs=(60 if smoke else 300))
        infl = contrastive_influence(enc, Xt, seed=seed)
        MI = np.nan_to_num(mutual_info_classif(Xr, yr, random_state=seed))
        with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
            ae = FlexSpectralAE(seed=seed, **_AE_CONV).fit(sp)
        sels = {"variance": topn_diverse(Xr.var(0), colmap, BUDGET),
                "pca_load": mz.pca_load(X, colmap, BUDGET, seed, np.random.default_rng(seed), sp, k=6),
                "AE": cols_for_bands(colmap, ae.select(BUDGET)),
                "mutInfo*": topn_diverse(MI, colmap, BUDGET),
                "contrastive": topn_diverse(infl, colmap, BUDGET)}
        for m, cols in sels.items():
            acc[m].append(fewshot(Xr, yr, cols, seed))
        for j in range(12):
            rng = np.random.default_rng(seed * 1000 + j)
            rnd_all.append(fewshot(Xr, yr, list(rng.choice(NBANDS, BUDGET, replace=False)), seed))
    rnd = np.array(rnd_all); r95 = np.percentile(rnd, 95)
    print("=" * 78)
    print(f"M1 CONTRASTIVE clutter-invariant selection — L4-high, {len(seeds)} seeds, best-NL F1")
    print(f"random µ={rnd.mean():.3f}  95th={r95:.3f}")
    print("=" * 78)
    for m in ["variance", "pca_load", "AE", "contrastive", "mutInfo*"]:
        v = float(np.mean(acc[m]))
        tag = "  <-- BEATS random" if v > r95 else ""
        print(f"  {m:<12}{v:>8.3f}{tag}")
    print("\nWin = contrastive beats random's 95th where variance/pca/AE (reconstruction) did not.")


if __name__ == "__main__":
    main()
