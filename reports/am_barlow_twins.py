"""Barlow-Twins / VICReg style blind band selection.

Idea: each pixel spectrum is a sample. Train a small MLP encoder with a
Barlow-Twins objective: two band-dropout augmentations of the same pixel are
pushed to embeddings whose batch cross-correlation matrix -> identity (diagonal
1 = invariance, off-diagonal 0 = redundancy reduction / decorrelated features).
After training, rank bands by their contribution to the de-correlated features:
first-layer input weight-column L2 norm combined with input-gradient saliency of
the de-correlated embedding. Diverse top-n via sweep_common.topn_diverse.
Unsupervised: labels never used.
"""
from __future__ import annotations

import os
import sys

import numpy as np

# Harness modules live in the primary checkout (this worktree's HEAD predates
# them); make them importable without modifying any existing file.
_MAIN = r"C:/Users/meloy/PycharmProjects/spectral-select"
for _p in (os.path.join(_MAIN, "reports"), os.path.join(_MAIN, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sweep_common import topn_diverse


def _standardize(X):
    mu = X.mean(0)
    sd = X.std(0) + 1e-9
    return (X - mu) / sd, mu, sd


def _torch_barlow(Xstd, n_bands, seed, rng):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device("cpu")

    Xt = torch.tensor(Xstd, dtype=torch.float32, device=dev)
    N = Xt.shape[0]

    proj = 64
    hidden = 128
    encoder = nn.Sequential(
        nn.Linear(n_bands, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(),
        nn.Linear(hidden, proj),
    ).to(dev)
    bn = nn.BatchNorm1d(proj, affine=False).to(dev)  # for cross-correlation

    opt = torch.optim.Adam(encoder.parameters(), lr=1e-3, weight_decay=1e-5)

    drop_p = 0.3
    lam = 5e-3
    batch = min(512, N)
    epochs = 30
    steps = max(1, N // batch)

    encoder.train()
    for _ in range(epochs):
        perm = torch.randperm(N, device=dev)
        for s in range(steps):
            idx = perm[s * batch:(s + 1) * batch]
            if idx.numel() < 4:
                continue
            xb = Xt[idx]
            m1 = (torch.rand_like(xb) > drop_p).float()
            m2 = (torch.rand_like(xb) > drop_p).float()
            v1 = xb * m1 / (1.0 - drop_p)
            v2 = xb * m2 / (1.0 - drop_p)
            z1 = bn(encoder(v1))
            z2 = bn(encoder(v2))
            b = z1.shape[0]
            c = (z1.T @ z2) / b
            on_diag = ((torch.diagonal(c) - 1.0) ** 2).sum()
            off = c - torch.diag(torch.diagonal(c))
            off_diag = (off ** 2).sum()
            loss = on_diag + lam * off_diag
            opt.zero_grad()
            loss.backward()
            opt.step()

    encoder.eval()
    W = encoder[0].weight.detach().cpu().numpy()  # [hidden, n_bands]
    w_norm = np.linalg.norm(W, axis=0)

    sub = Xt
    if N > 1024:
        ridx = torch.randperm(N)[:1024]
        sub = Xt[ridx]
    sub = sub.clone().requires_grad_(True)
    z = bn(encoder(sub))
    out = (z ** 2).sum()
    grad = torch.autograd.grad(out, sub)[0].detach().cpu().numpy()
    sal = np.sqrt((grad ** 2).mean(0))

    def _nz(a):
        a = np.asarray(a, dtype=float)
        r = a.max() - a.min()
        return (a - a.min()) / (r + 1e-12)

    return _nz(w_norm) + _nz(sal)


def select(X, colmap, n, seed, rng, spectra=None):
    X = np.asarray(X, dtype=np.float64)
    n_bands = X.shape[1]
    Xstd, _, _ = _standardize(X)
    try:
        score = _torch_barlow(Xstd, n_bands, int(seed), rng)
        if not np.all(np.isfinite(score)) or np.ptp(score) == 0:
            raise ValueError("degenerate score")
    except Exception:
        score = X.var(0)  # robust fallback: variance ranking
    return topn_diverse(score, colmap, n)