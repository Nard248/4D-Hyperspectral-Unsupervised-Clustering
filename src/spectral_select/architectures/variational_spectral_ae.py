"""C4 — variational spectral autoencoder with an anti-posterior-collapse guard.

A VAE is **not** automatically a fix: minimizing the ELBO can drive the decoder to ignore the
latent (posterior collapse), which would make the perturbation-influence meaningless. We guard it
two ways (``docs/spectraforge/03-architecture-plan.md``):

* **free-bits** — clamp each latent dim's KL at a floor ``free_bits`` (nats) so the loss never
  pushes a dimension's KL below the floor, keeping dimensions alive;
* **KL annealing** — ramp ``beta`` from 0 to its target over the first ``anneal_frac`` of training.

Perturbation is applied to the posterior mean ``mu``. ``active_units()`` reports how many latent
dimensions stayed above the free-bits floor (a posterior-collapse check).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import SpectralSelector


class _SpectralVAE(nn.Module):
    def __init__(self, d: int, latent: int = 8):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 16), nn.ReLU())
        self.fc_mu = nn.Linear(16, latent)
        self.fc_logvar = nn.Linear(16, latent)
        self.dec = nn.Sequential(
            nn.Linear(latent, 16), nn.ReLU(), nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, d))

    def encode_dist(self, x):
        h = self.body(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def encode(self, x):                       # baseline latent for perturbation == posterior mean
        return self.fc_mu(self.body(x))

    def decode(self, z):
        return self.dec(z)


class VariationalSpectralAE(SpectralSelector):
    """C4: variational spectral AE with free-bits + KL annealing."""

    name = "C4 variational-spectral-AE"
    beta = 1.0
    free_bits = 0.5

    def __init__(self, *, beta=None, free_bits=None, anneal_frac=0.33, **kw):
        super().__init__(**kw)
        if beta is not None:
            self.beta = float(beta)
        if free_bits is not None:
            self.free_bits = float(free_bits)
        self.anneal_frac = float(anneal_frac)
        self._epoch = 0

    def _make_module(self, d: int) -> nn.Module:
        self._epoch = 0                        # reset the annealing clock for a fresh fit
        return _SpectralVAE(d, self.latent_dim)

    def _loss(self, model: nn.Module, Xt: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        mu, logvar = model.encode_dist(Xt)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(std.shape, generator=gen, device=Xt.device)
        recon = model.decode(mu + eps * std)
        recon_loss = ((recon - Xt) ** 2).mean()

        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(dim=0)   # (latent,)
        kl = torch.clamp(kl_per_dim, min=self.free_bits).sum()                    # free-bits floor

        warmup = max(1, int(self.anneal_frac * self.epochs))
        beta_t = self.beta * min(1.0, self._epoch / warmup)
        self._epoch += 1
        return recon_loss + beta_t * kl

    def active_units(self) -> int:
        with torch.no_grad():
            mu, logvar = self.model.encode_dist(self._Xt)
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(dim=0)
        return int((kl > self.free_bits + 1e-6).sum())
