"""Round-3 M-spatial-CAE — the definitive test of the ORIGINAL idea in its proper domain. A supervised
spatial CNN (convolutions model texture) predicts per-pixel class from spatial context; band selection =
perturbation (occlude a band-channel) influence on the per-pixel predictions. On texture-discriminative
data (per-pixel methods provably blind), does the spatial-CAE selector FIND the texture bands and beat
the per-pixel selectors + random, approaching the oracle?

Run:  python reports/spatial_cae.py [--smoke]
"""
from __future__ import annotations

import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from spatial_regime import (BUDGET, N_BANDS, SIZE, _blocks, build_spatial, fewshot, local_std)
from sweep_common import topn_diverse
from sklearn.feature_selection import mutual_info_classif
from mixnoise_experiment import _clfs  # noqa: F401  (fewshot uses it)

torch.manual_seed(0)


class SpatialFCN(nn.Module):
    def __init__(self, C, nc, h=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(C, h, 3, padding=1), nn.GELU(),
            nn.Conv2d(h, h, 3, padding=1), nn.GELU(),
            nn.Conv2d(h, h, 3, padding=1), nn.GELU(),
            nn.Conv2d(h, nc, 1))

    def forward(self, x):
        return self.net(x)


def train_fcn(Xstd, y, train_mask, seed, epochs=300):
    gen = torch.Generator().manual_seed(seed)
    C = Xstd.shape[1]
    cube = torch.tensor(Xstd.T.reshape(1, C, SIZE, SIZE), dtype=torch.float32)
    yt = torch.tensor(y.reshape(SIZE, SIZE), dtype=torch.long)
    mask = torch.tensor(train_mask.reshape(SIZE, SIZE), dtype=torch.bool)
    nc = int(y.max() + 1)
    net = SpatialFCN(C, nc)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    for ep in range(epochs):
        logit = net(cube)[0]                                   # (nc, H, W)
        loss = F.cross_entropy(logit.permute(1, 2, 0)[mask], yt[mask])
        opt.zero_grad(); loss.backward(); opt.step()
    return net, cube


def fcn_influence(net, cube):
    with torch.no_grad():
        p0 = F.softmax(net(cube)[0], dim=0)
        infl = np.zeros(cube.shape[1])
        for b in range(cube.shape[1]):
            cj = cube.clone(); cj[0, b] = 0.0
            infl[b] = (F.softmax(net(cj)[0], dim=0) - p0).abs().sum(0).mean().item()
    return infl


def main():
    smoke = "--smoke" in sys.argv
    seeds = ([1] if smoke else [1, 2, 3])
    from sklearn.preprocessing import StandardScaler
    agg = {m: [] for m in ["per-pixel mutInfo*", "texture-MI*", "spatial-CAE*", "random", "oracle"]}
    hitcae = []
    for seed in seeds:
        X, y, disc = build_spatial(seed)
        Xstd = StandardScaler().fit_transform(X)
        colmap = [(0, float(b)) for b in range(N_BANDS)]
        TX = local_std(X)
        bid = _blocks(); nb = bid.max() + 1
        trbk = set(np.random.default_rng(seed).permutation(nb)[:nb // 2].tolist())
        train_mask = np.isin(bid, list(trbk))
        net, cube = train_fcn(Xstd, y, train_mask, seed, epochs=(60 if smoke else 300))
        infl = fcn_influence(net, cube)
        MIpx = np.nan_to_num(mutual_info_classif(X, y, random_state=seed))
        MItex = np.nan_to_num(mutual_info_classif(TX, y, random_state=seed))
        rng = np.random.default_rng(seed)
        sels = {"per-pixel mutInfo*": topn_diverse(MIpx, colmap, BUDGET),
                "texture-MI*": topn_diverse(MItex, colmap, BUDGET),
                "spatial-CAE*": topn_diverse(infl, colmap, BUDGET),
                "random": list(rng.choice(N_BANDS, BUDGET, replace=False)),
                "oracle": disc + list(rng.choice([b for b in range(N_BANDS) if b not in disc], BUDGET - len(disc), replace=False))}
        for m, cols in sels.items():
            agg[m].append(fewshot(TX, y, cols, seed))
        hitcae.append(len(set(topn_diverse(infl, colmap, BUDGET)) & set(disc)) / len(disc))
    print("=" * 78)
    print(f"M-spatial-CAE — supervised spatial CNN selector, {len(seeds)} seeds, block-CV best-NL F1")
    print("=" * 78)
    for m in ["per-pixel mutInfo*", "texture-MI*", "spatial-CAE*", "random", "oracle"]:
        print(f"  {m:<20}{np.mean(agg[m]):>8.3f}")
    print(f"\n  spatial-CAE %disc-found = {100*np.mean(hitcae):.0f}%")
    print("Win = spatial-CAE* finds the disc bands, beats per-pixel + random, approaches oracle.")


if __name__ == "__main__":
    main()
