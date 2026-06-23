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


class _FlexConv(nn.Module):
    """1D-conv-over-bands backbone: reshape (B,d) -> (B, n_ex, n_band), conv along the band axis."""

    def __init__(self, n_ex, n_band, latent=8, depth=3, width=64, act="gelu"):
        super().__init__()
        A = ACTS[act]
        self.n_ex, self.n_band = n_ex, n_band
        enc = [nn.Conv1d(n_ex, width, 7, padding=3), A()]
        for _ in range(max(0, depth - 1)):
            enc += [nn.Conv1d(width, width, 3, padding=1), A()]
        self.enc_conv = nn.Sequential(*enc)
        self.to_latent = nn.Linear(width * n_band, latent)
        self.from_latent = nn.Linear(latent, width * n_band)
        dec = [A()]
        for _ in range(max(0, depth - 1)):
            dec += [nn.Conv1d(width, width, 3, padding=1), A()]
        dec += [nn.Conv1d(width, n_ex, 7, padding=3)]
        self.dec_conv = nn.Sequential(*dec)
        self.width = width

    def encode(self, x):
        b = x.shape[0]
        return self.to_latent(self.enc_conv(x.view(b, self.n_ex, self.n_band)).reshape(b, -1))

    def decode(self, z):
        b = z.shape[0]
        return self.dec_conv(self.from_latent(z).view(b, self.width, self.n_band)).reshape(b, self.n_ex * self.n_band)


class FlexSpectralAE(SpectralSelector):
    latent_dim = 8
    epochs = 400
    lr = 1e-3

    def __init__(self, *, depth=3, width=128, act="gelu", mask_ratio=0.0, masked_weight=3.0,
                 dropout=0.0, noise=0.0, backbone="mlp", **kw):
        super().__init__(**kw)
        self.depth, self.width, self.act, self.backbone = depth, width, act, backbone
        self.mask_ratio, self.masked_weight, self.dropout, self.noise = mask_ratio, masked_weight, dropout, noise

    def _make_module(self, d):
        if self.backbone == "conv":
            ex = sorted({e for e, _ in self.colmap})
            n_ex = len(ex); n_band = d // n_ex
            return _FlexConv(n_ex, n_band, self.latent_dim, self.depth, min(self.width, 64), self.act)
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
