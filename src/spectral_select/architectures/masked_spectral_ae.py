"""C3 — masked (denoising) spectral autoencoder (the most principled anti-cheat lever).

Each training step randomly masks a fraction of the input spectrum (zeroes those features) and the
loss rewards reconstructing the **full** spectrum, with the masked positions up-weighted. A model
that predicts the per-band mean cannot satisfy this: the masked bands must be inferred from the
unmasked ones, so the latent is forced to carry inter-band structure. This is the SS-MAE / SMAE
idea (``docs/spectraforge/03-architecture-plan.md``). At selection time the baseline latent is the
encoding of the *unmasked* spectrum.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import SpectralSelector
from .spectral_ae import _SpectralMLP


class MaskedSpectralAE(SpectralSelector):
    """C3: denoising spectral AE. ``mask_ratio`` is the per-feature drop probability; the masked
    positions get ``masked_loss_weight`` x the reconstruction weight."""

    name = "C3 masked-spectral-AE"
    mask_ratio = 0.5

    def __init__(self, *, mask_ratio=None, masked_loss_weight=3.0, **kw):
        super().__init__(**kw)
        if mask_ratio is not None:
            self.mask_ratio = float(mask_ratio)
        self.masked_loss_weight = float(masked_loss_weight)

    def _make_module(self, d: int) -> nn.Module:
        return _SpectralMLP(d, self.latent_dim)

    def _loss(self, model: nn.Module, Xt: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        mask = torch.rand(Xt.shape, generator=gen, device=Xt.device) < self.mask_ratio
        corrupted = Xt.masked_fill(mask, 0.0)
        recon = model.decode(model.encode(corrupted))
        se = (recon - Xt) ** 2
        weight = torch.ones_like(se) + (self.masked_loss_weight - 1.0) * mask.float()
        return (se * weight).mean()
