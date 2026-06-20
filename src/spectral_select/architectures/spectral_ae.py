"""C2 — per-pixel spectral autoencoder (the simplest fix; already proven in
``reports/cae_vs_spectral_ae.py::_SpectralAE``).

Every pixel's concatenated multi-excitation spectrum is one training sample, so the batch is
thousands of dense vectors and there is no spatial pooling / band collapse to throw away the
spectral structure. Trained with plain MSE. This is the C2 rung of the anti-cheat ladder.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import SpectralSelector


class _SpectralMLP(nn.Module):
    """The exact topology proven in ``reports/cae_vs_spectral_ae.py`` (D->64->16->latent->16->64->D)."""

    def __init__(self, d: int, latent: int = 8):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, latent))
        self.dec = nn.Sequential(
            nn.Linear(latent, 16), nn.ReLU(), nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, d))

    def encode(self, x):
        return self.enc(x)

    def decode(self, z):
        return self.dec(z)


class SpectralAE(SpectralSelector):
    """C2: per-pixel spectral MLP-AE, plain MSE objective."""

    name = "C2 spectral-AE"

    def _make_module(self, d: int) -> nn.Module:
        return _SpectralMLP(d, self.latent_dim)

    def _loss(self, model: nn.Module, Xt: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        recon = model.decode(model.encode(Xt))
        return ((recon - Xt) ** 2).mean()
