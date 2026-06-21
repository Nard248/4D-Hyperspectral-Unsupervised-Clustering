"""C5 — an enlarged per-pixel spectral autoencoder (the deep-net "bag of tricks").

Same per-pixel principle as C2, scaled up with standard practices: a wider trunk (256), pre-activation
**residual blocks**, **LayerNorm**, **GELU**, **dropout**, a larger **latent (16)**, and trained with
**minibatch SGD + AdamW (weight decay) + cosine LR + early stopping** (the opt-in training path in
:class:`SpectralSelector`). LayerNorm (not BatchNorm) keeps it batch-size robust for the final
full-data encode used by the perturbation step.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import SpectralSelector


class _ResBlock(nn.Module):
    def __init__(self, width: int, p: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(),
            nn.Dropout(p), nn.Linear(width, width))

    def forward(self, x):
        return x + self.net(x)


class _DeepMLP(nn.Module):
    def __init__(self, d: int, latent: int = 16, width: int = 256, blocks: int = 2, p: float = 0.1):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(d, width), nn.GELU(), *[_ResBlock(width, p) for _ in range(blocks)],
            nn.LayerNorm(width), nn.Linear(width, latent))
        self.dec = nn.Sequential(
            nn.Linear(latent, width), nn.GELU(), *[_ResBlock(width, p) for _ in range(blocks)],
            nn.LayerNorm(width), nn.Linear(width, d))

    def encode(self, x):
        return self.enc(x)

    def decode(self, z):
        return self.dec(z)


class DeepSpectralAE(SpectralSelector):
    """C5: wide residual MLP spectral AE + modern training recipe."""

    name = "C5 deep-spectral-AE"
    latent_dim = 16
    epochs = 200
    lr = 2e-3
    batch_size = 512
    weight_decay = 1e-4
    scheduler = "cosine"
    patience = 25

    width = 256
    blocks = 2
    dropout = 0.1

    def _make_module(self, d: int) -> nn.Module:
        return _DeepMLP(d, latent=self.latent_dim, width=self.width, blocks=self.blocks, p=self.dropout)

    def _loss(self, model, Xt, gen):
        return ((model.decode(model.encode(Xt)) - Xt) ** 2).mean()
