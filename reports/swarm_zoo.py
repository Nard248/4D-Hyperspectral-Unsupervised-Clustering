"""Flexible per-pixel AE for the optimization swarm: configurable depth, width, latent (bottleneck),
activation, dropout, and a masking/denoising objective. Reuses SpectralSelector's perturbation
selection + diagnostics. The swarm varies these to try to EXCEED pca_load on realistic data.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from spectral_select.architectures.base import SpectralSelector

ACTS = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU, "leaky": nn.LeakyReLU,
        "elu": nn.ELU, "tanh": nn.Tanh}


class _FlexMLP(nn.Module):
    def __init__(self, d, latent=8, depth=3, width=128, act="gelu", p=0.0, residual=False):
        super().__init__()
        A = ACTS[act]
        ws = [width] * max(1, depth - 1)
        self.residual = residual and len(set(ws)) == 1

        def stack(dims):
            layers = []
            for i in range(len(dims) - 2):
                layers += [nn.Linear(dims[i], dims[i + 1]), A()]
                if p > 0:
                    layers += [nn.Dropout(p)]
            layers += [nn.Linear(dims[-2], dims[-1])]
            return nn.Sequential(*layers)

        self.enc = stack([d] + ws + [latent])
        self.dec = stack([latent] + ws[::-1] + [d])

    def encode(self, x):
        return self.enc(x)

    def decode(self, z):
        return self.dec(z)


class FlexSpectralAE(SpectralSelector):
    latent_dim = 8
    epochs = 400
    lr = 1e-3

    def __init__(self, *, depth=3, width=128, act="gelu", mask_ratio=0.0, masked_weight=3.0,
                 dropout=0.0, noise=0.0, **kw):
        super().__init__(**kw)
        self.depth, self.width, self.act = depth, width, act
        self.mask_ratio, self.masked_weight, self.dropout, self.noise = mask_ratio, masked_weight, dropout, noise

    def _make_module(self, d):
        return _FlexMLP(d, self.latent_dim, self.depth, self.width, self.act, self.dropout)

    def _loss(self, model, Xt, gen):
        x = Xt
        if self.noise > 0:
            x = x + self.noise * torch.randn(x.shape, generator=gen, device=x.device)
        if self.mask_ratio > 0:
            mask = torch.rand(Xt.shape, generator=gen, device=Xt.device) < self.mask_ratio
            x = x.masked_fill(mask, 0.0)
            rec = model.decode(model.encode(x))
            se = (rec - Xt) ** 2
            w = torch.ones_like(se) + (self.masked_weight - 1.0) * mask.float()
            return (se * w).mean()
        return ((model.decode(model.encode(x)) - Xt) ** 2).mean()
