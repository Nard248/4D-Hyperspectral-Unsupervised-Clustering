"""Conv-architecture experiments on the CAE's design DNA, but PER-PIXEL (no spatial convolution):
keep the per-excitation **parallel branches** + band handling, drop the spatial dims (each pixel is
one sample). Sweep the conv setup of the branches: band-kernel size, depth, channels, whether to
**collapse the band axis** (the CAE's adaptive_avg_pool), and how branches merge (mean vs concat).

This isolates which part of the spatial CAE matters: if per-pixel parallel-branch 1D-conv WITHOUT
band-collapse reconstructs (R2>0) and WITH collapse does not, the band-collapse is the culprit.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from spectral_select.architectures.base import SpectralSelector


class _ParallelBranchAE(nn.Module):
    """Per-pixel autoencoder: one 1D-conv branch per excitation over its emission bands; merge to a
    latent; per-excitation decoder branches. ``collapse`` mimics the CAE's band-axis average-pool."""

    def __init__(self, band_counts, latent=12, ch=16, kb=5, depth=2, collapse=False, merge="concat"):
        super().__init__()
        assert len(set(band_counts)) == 1, "branches assume equal band counts per excitation"
        self.band_counts = list(band_counts)
        self.offsets = np.cumsum([0] + self.band_counts).tolist()
        self.collapse, self.merge, self.nb = collapse, merge, self.band_counts[0]
        nex = len(self.band_counts)
        self.enc = nn.ModuleList()
        for _ in self.band_counts:
            layers, cin = [], 1
            for _d in range(depth):
                layers += [nn.Conv1d(cin, ch, kb, padding=kb // 2), nn.GELU()]
                cin = ch
            self.enc.append(nn.Sequential(*layers))
        self.branch_dim = ch if collapse else ch * self.nb
        merged = self.branch_dim * (nex if merge == "concat" else 1)
        self.to_latent = nn.Sequential(nn.Linear(merged, 64), nn.GELU(), nn.Linear(64, latent))
        self.from_latent = nn.Sequential(nn.Linear(latent, 64), nn.GELU(), nn.Linear(64, merged))
        self.dec = nn.ModuleList(
            nn.Sequential(nn.Linear(self.branch_dim, 64), nn.GELU(), nn.Linear(64, nb))
            for nb in self.band_counts)

    def _branch_feat(self, x):
        feats = []
        for i in range(len(self.band_counts)):
            seg = x[:, self.offsets[i]:self.offsets[i + 1]].unsqueeze(1)   # (B,1,nb)
            h = self.enc[i](seg)                                           # (B,ch,nb)
            feats.append(h.mean(dim=2) if self.collapse else h.reshape(h.shape[0], -1))
        return feats

    def encode(self, x):
        feats = self._branch_feat(x)
        m = torch.cat(feats, dim=1) if self.merge == "concat" else torch.stack(feats, 0).mean(0)
        return self.to_latent(m)

    def decode(self, z):
        m = self.from_latent(z)
        parts = m.split(self.branch_dim, dim=1) if self.merge == "concat" else [m] * len(self.band_counts)
        return torch.cat([self.dec[i](parts[i]) for i in range(len(self.band_counts))], dim=1)


class ParallelBranchSpectralAE(SpectralSelector):
    """Per-pixel parallel-branch conv AE; reuses the SpectralSelector perturbation-selection +
    diagnostics. Knobs expose the conv setup to sweep."""

    latent_dim = 12
    epochs = 300
    lr = 1e-3

    def __init__(self, *, kb=5, depth=2, collapse=False, merge="concat", ch=16, **kw):
        super().__init__(**kw)
        self.kb, self.depth, self.collapse, self.merge, self.ch = kb, depth, collapse, merge, ch

    def _band_counts(self):
        seen = []
        for ex, _ in self.colmap:
            if ex not in seen:
                seen.append(ex)
        from collections import Counter
        c = Counter(ex for ex, _ in self.colmap)
        return [c[ex] for ex in seen]

    def _make_module(self, d):
        return _ParallelBranchAE(self._band_counts(), latent=self.latent_dim, ch=self.ch,
                                 kb=self.kb, depth=self.depth, collapse=self.collapse, merge=self.merge)

    def _loss(self, model, Xt, gen):
        return ((model.decode(model.encode(Xt)) - Xt) ** 2).mean()
