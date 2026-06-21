"""End-to-end *gate* selectors — they learn WHICH bands to keep directly, instead of probing a
trained AE. Research-SOTA for unsupervised band selection:

- concrete_ae       : Concrete Autoencoder (Balin et al. 2019) — n Gumbel-softmax gates over the
                      bands, each anneals to a one-hot pick; decoder reconstructs the FULL spectrum,
                      so the chosen bands are optimized to be the most reconstructive subset.
- stochastic_gates  : STG (Yamada et al. 2020) — a learnable Bernoulli gate per band + an L0/L1
                      sparsity penalty; reconstruct from gated input; keep the highest-probability gates.
- bsnet_attention   : BS-Net-lite — a band-attention vector (softmax over bands) reweights the input,
                      a decoder reconstructs it; attention magnitude ranks the bands.

All implement the sweep_common contract and operate on the standardized feature matrix.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from sweep_common import topn_diverse


def _xt(X, seed):
    torch.manual_seed(int(seed))
    return torch.tensor(StandardScaler().fit_transform(X).astype("float32"))


def _dedup_pad(idx, rank_score, n):
    chosen = []
    for j in idx:
        if j not in chosen:
            chosen.append(int(j))
    if len(chosen) < n:
        for j in np.argsort(rank_score)[::-1]:
            if int(j) not in chosen:
                chosen.append(int(j))
            if len(chosen) == n:
                break
    return chosen[:n]


def concrete_ae(X, colmap, n, seed, rng, spectra=None, epochs=300, start_temp=10.0, end_temp=0.1,
                hidden=128, lr=1e-3, bs=256):
    Xt = _xt(X, seed)
    n_pix, d = Xt.shape
    gen = torch.Generator().manual_seed(int(seed) + 1)
    logits = nn.Parameter(torch.randn(n, d, generator=gen) * 0.01)
    dec = nn.Sequential(nn.Linear(n, hidden), nn.LeakyReLU(), nn.Linear(hidden, hidden), nn.LeakyReLU(),
                        nn.Linear(hidden, d))
    opt = torch.optim.Adam([logits, *dec.parameters()], lr)
    for ep in range(epochs):
        temp = start_temp * (end_temp / start_temp) ** (ep / max(1, epochs - 1))
        perm = torch.randperm(n_pix, generator=gen)
        for i in range(0, n_pix, bs):
            xb = Xt[perm[i:i + bs]]
            u = torch.rand(n, d, generator=gen).clamp_(1e-20, 1.0)
            g = -torch.log(-torch.log(u))
            sel = torch.softmax((logits + g) / temp, dim=1)         # (n, d) soft one-hots
            recon = dec(xb @ sel.t())
            loss = ((recon - xb) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        idx = logits.argmax(1).tolist()
        rank = logits.max(0).values.numpy()
    return _dedup_pad(idx, rank, n)


def stochastic_gates(X, colmap, n, seed, rng, spectra=None, epochs=300, lam=0.05, hidden=128, lr=2e-3,
                     bs=256):
    Xt = _xt(X, seed)
    n_pix, d = Xt.shape
    gen = torch.Generator().manual_seed(int(seed) + 2)
    mu = nn.Parameter(torch.zeros(d))                                # gate location; gate = clamp(mu+eps*sigma)
    sigma = 0.5
    enc = nn.Sequential(nn.Linear(d, hidden), nn.LeakyReLU(), nn.Linear(hidden, hidden), nn.LeakyReLU(),
                        nn.Linear(hidden, d))
    opt = torch.optim.Adam([mu, *enc.parameters()], lr)
    for _ in range(epochs):
        perm = torch.randperm(n_pix, generator=gen)
        for i in range(0, n_pix, bs):
            xb = Xt[perm[i:i + bs]]
            z = (mu + sigma * torch.randn(d, generator=gen)).clamp(0, 1)   # hard-concrete-ish gate
            recon = enc(xb * z)
            # reconstruct full from gated input + L1 on gate-open probability (0.5*(1+erf(mu/sigma/sqrt2)))
            reg = torch.sigmoid(mu / sigma).sum()
            loss = ((recon - xb) ** 2).mean() + lam * reg / d
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        prob = torch.sigmoid(mu / sigma).numpy()
    return topn_diverse(prob, colmap, n)


def bsnet_attention(X, colmap, n, seed, rng, spectra=None, epochs=300, hidden=128, lr=2e-3, bs=256):
    Xt = _xt(X, seed)
    n_pix, d = Xt.shape
    gen = torch.Generator().manual_seed(int(seed) + 3)
    attn_logits = nn.Parameter(torch.zeros(d))                       # band-attention (BAM)
    net = nn.Sequential(nn.Linear(d, hidden), nn.LeakyReLU(), nn.Linear(hidden, d))   # reconstruct (RecNet)
    opt = torch.optim.Adam([attn_logits, *net.parameters()], lr)
    for _ in range(epochs):
        perm = torch.randperm(n_pix, generator=gen)
        for i in range(0, n_pix, bs):
            xb = Xt[perm[i:i + bs]]
            a = torch.sigmoid(attn_logits)                           # per-band gate in [0,1]
            recon = net(xb * a)
            loss = ((recon - xb) ** 2).mean() + 1e-3 * a.mean()      # mild sparsity
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        a = torch.sigmoid(attn_logits).numpy()
    return topn_diverse(a, colmap, n)
