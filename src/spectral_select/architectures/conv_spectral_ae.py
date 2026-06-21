"""C6/C7 — 1D-convolutional spectral autoencoders (exploit the band axis).

Per-pixel like C2–C5, but the feature vector is reshaped to ``(n_excitations, n_bands)`` and a **1D
convolution runs along the emission-band axis** (excitations as input channels). Convolution encodes
the *local smoothness* of a spectrum and the ordering of the bands — the structure a plain MLP must
learn from scratch — which is the established design for HSI / Raman band selection (e.g. SRL-SOA,
1D-operational autoencoders). GroupNorm keeps it batch-size independent.

- **C6 ``ConvSpectralAE``** — plain MSE.
- **C7 ``MaskedConvSpectralAE``** — denoising: randomly mask input bands, reconstruct the full
  spectrum (the SS-MAE/SMAE idea on a conv backbone — masking + locality together).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import SpectralSelector


class _ConvAE(nn.Module):
    """Length-preserving 1D-conv AE over the band axis; a Linear bottleneck to/from the latent."""

    def __init__(self, n_ex: int, n_band: int, latent: int = 16, width: int = 64, p: float = 0.1):
        super().__init__()
        self.n_ex, self.n_band = n_ex, n_band
        g = 8
        self.enc_conv = nn.Sequential(
            nn.Conv1d(n_ex, 32, 7, padding=3), nn.GroupNorm(g, 32), nn.GELU(), nn.Dropout(p),
            nn.Conv1d(32, width, 5, padding=2), nn.GroupNorm(g, width), nn.GELU())
        self.enc_fc = nn.Linear(width * n_band, latent)
        self.dec_fc = nn.Linear(latent, width * n_band)
        self.dec_conv = nn.Sequential(
            nn.GELU(),
            nn.Conv1d(width, 32, 5, padding=2), nn.GroupNorm(g, 32), nn.GELU(), nn.Dropout(p),
            nn.Conv1d(32, n_ex, 7, padding=3))
        self.width = width

    def encode(self, x):
        b = x.shape[0]
        h = self.enc_conv(x.view(b, self.n_ex, self.n_band))
        return self.enc_fc(h.reshape(b, -1))

    def decode(self, z):
        b = z.shape[0]
        h = self.dec_fc(z).view(b, self.width, self.n_band)
        return self.dec_conv(h).reshape(b, self.n_ex * self.n_band)


class _ConvSpectralBase(SpectralSelector):
    latent_dim = 16
    epochs = 200
    lr = 2e-3
    batch_size = 512
    weight_decay = 1e-4
    scheduler = "cosine"
    patience = 25
    width = 48

    def _make_module(self, d: int) -> nn.Module:
        ex_list = sorted({ex for ex, _ in self.colmap})
        n_ex = len(ex_list)
        n_band, rem = divmod(d, n_ex)
        if rem != 0:
            raise ValueError(f"ConvSpectralAE needs a shared band count per excitation (d={d}, n_ex={n_ex})")
        return _ConvAE(n_ex, n_band, latent=self.latent_dim, width=self.width)


class ConvSpectralAE(_ConvSpectralBase):
    """C6: 1D-conv spectral AE, plain MSE."""

    name = "C6 conv-spectral-AE"

    def _loss(self, model, Xt, gen):
        return ((model.decode(model.encode(Xt)) - Xt) ** 2).mean()


class MaskedConvSpectralAE(_ConvSpectralBase):
    """C7: denoising 1D-conv spectral AE — mask input bands, reconstruct the full spectrum."""

    name = "C7 masked-conv-spectral-AE"
    mask_ratio = 0.5
    masked_loss_weight = 3.0

    def __init__(self, *, mask_ratio=None, masked_loss_weight=None, **kw):
        super().__init__(**kw)
        if mask_ratio is not None:
            self.mask_ratio = float(mask_ratio)
        if masked_loss_weight is not None:
            self.masked_loss_weight = float(masked_loss_weight)

    def _loss(self, model, Xt, gen):
        mask = torch.rand(Xt.shape, generator=gen, device=Xt.device) < self.mask_ratio
        recon = model.decode(model.encode(Xt.masked_fill(mask, 0.0)))
        se = (recon - Xt) ** 2
        weight = torch.ones_like(se) + (self.masked_loss_weight - 1.0) * mask.float()
        return (se * weight).mean()
