"""Round-3 M-sup — the WORKING version of the AE+perturbation idea: make it supervised. An AE with a
classification head; latent trained jointly to (a) reconstruct (all ROI pixels, unsupervised) and
(b) predict the class (few labels, supervised). Band selection = occlusion influence on the CLASSIFIER
output → class-relevant bands (can capture nonlinear/joint structure marginal mutInfo misses).

Tests both hard regimes: clutter (L4) and nonlinear FRET. Compare vs random, full, mutInfo*, RFimp*
(same label budget). Selection labels = downstream-classifier labels (no test leak).

Run:  python reports/supervised_ae.py [--smoke]
"""
from __future__ import annotations

import contextlib
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from classification_experiment import feature_matrix
from fret_regime import build_fret_dataset
from mixnoise_experiment import BUDGET, EM_STEP, LEVELS, NBANDS, SIZE, _clfs, roi_mask
from realistic_benchmark import build_dataset
from sweep_common import topn_diverse

PER_CLASS = 30
REPEATS = 4
torch.manual_seed(0)


class SupAE(nn.Module):
    def __init__(self, d, z=16, nc=3, h=256):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, z))
        self.dec = nn.Sequential(nn.Linear(z, h), nn.GELU(), nn.Linear(h, d))
        self.clf = nn.Sequential(nn.Linear(z, h // 2), nn.GELU(), nn.Linear(h // 2, nc))

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), self.clf(z), z


def train_supae(Xstd, L, yL, seed, nc, epochs=400, lam=3.0, batch=256):
    gen = torch.Generator().manual_seed(seed)
    Xt = torch.tensor(Xstd, dtype=torch.float32)
    XL = torch.tensor(Xstd[L], dtype=torch.float32)
    yt = torch.tensor(yL, dtype=torch.long)
    model = SupAE(Xt.shape[1], nc=nc)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    n = Xt.shape[0]
    for ep in range(epochs):
        idx = torch.randperm(n, generator=gen)[:batch]
        rec, _, _ = model(Xt[idx])
        loss_rec = F.mse_loss(rec, Xt[idx])
        _, logit, _ = model(XL)
        loss_ce = F.cross_entropy(logit, yt)
        loss = loss_rec + lam * loss_ce
        opt.zero_grad(); loss.backward(); opt.step()
    return model, Xt


def supae_influence(model, Xt, seed, sample=1024):
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(Xt.shape[0], generator=gen)[:sample]
    x = Xt[idx]
    with torch.no_grad():
        p0 = F.softmax(model(x)[1], dim=1)
        infl = np.zeros(Xt.shape[1])
        for j in range(Xt.shape[1]):
            xj = x.clone(); xj[:, j] = 0.0
            infl[j] = (F.softmax(model(xj)[1], dim=1) - p0).abs().sum(1).mean().item()
    return infl


def evalc(X, y, cols, L, T, seed):
    sc = StandardScaler().fit(X[L][:, cols])
    Xtr, Xte = sc.transform(X[L][:, cols]), sc.transform(X[T][:, cols])
    fs = []
    for n, c in _clfs(seed).items():
        with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
            c.fit(Xtr, y[L]); fs.append(f1_score(y[T], c.predict(Xte), average="macro"))
    return max(fs)


def regimes(seed, smoke):
    out = {}
    sp, gt, y, acq = build_dataset(seed, size=SIZE, em_step=EM_STEP, **LEVELS["L4-high"])
    out["clutter"] = (sp, y)
    if not smoke:
        sp2, y2 = build_fret_dataset(seed, size=SIZE, k=5.0)
        out["fret"] = (sp2, y2)
    return out


def main():
    smoke = "--smoke" in sys.argv
    seeds = ([1] if smoke else [1, 2, 3])
    methods = ["random", "full", "mutInfo*", "RFimp*", "supAE*"]
    agg = {}
    for seed in seeds:
        for reg, (sp, y) in regimes(seed, smoke).items():
            X, colmap = feature_matrix(sp)
            nb = X.shape[1]
            roi = roi_mask(seed, SIZE); Xr, yr = X[roi], y[roi]
            Xstd = StandardScaler().fit_transform(Xr)
            for r in range(REPEATS):
                rng = np.random.default_rng(seed * 100 + r)
                L = np.concatenate([rng.choice(np.where(yr == c)[0], PER_CLASS, replace=False) for c in np.unique(yr)])
                T = np.setdiff1d(np.arange(len(yr)), L)
                model, Xt = train_supae(Xstd, L, yr[L], seed * 10 + r, nc=len(np.unique(yr)),
                                        epochs=(80 if smoke else 400))
                infl = supae_influence(model, Xt, seed * 10 + r)
                MI = np.nan_to_num(mutual_info_classif(Xr[L], yr[L], random_state=seed))
                rf = RandomForestClassifier(200, random_state=seed, n_jobs=-1)
                with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
                    rf.fit(Xstd[L], yr[L])
                sels = {"full": list(range(nb)),
                        "mutInfo*": topn_diverse(MI, colmap, BUDGET),
                        "RFimp*": topn_diverse(rf.feature_importances_, colmap, BUDGET),
                        "supAE*": topn_diverse(infl, colmap, BUDGET),
                        "random": list(rng.choice(nb, BUDGET, replace=False))}
                for m, cols in sels.items():
                    agg.setdefault((reg, m), []).append(evalc(Xr, yr, cols, L, T, seed))
    print("=" * 78)
    print(f"M-sup SUPERVISED AE + perturbation — {len(seeds)} seeds, few-shot {PER_CLASS}/class best-NL F1")
    print("=" * 78)
    for reg in (["clutter"] if smoke else ["clutter", "fret"]):
        print(f"\n### {reg}")
        rnd = np.mean(agg[(reg, "random")]); full = np.mean(agg[(reg, "full")])
        for m in methods:
            v = np.mean(agg[(reg, m)])
            tag = ""
            if m == "supAE*":
                tag = f"  vs mutInfo {v-np.mean(agg[(reg,'mutInfo*')]):+.3f}, vs random {v-rnd:+.3f}"
            print(f"  {m:<10}{v:>8.3f}{tag}")
    print("\nWin = supAE* beats random AND >= best supervised baseline (mutInfo/RFimp) in BOTH regimes.")


if __name__ == "__main__":
    main()
