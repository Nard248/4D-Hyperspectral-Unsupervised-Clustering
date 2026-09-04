"""Compressive-bottleneck Conv1d group-structured autoencoder.

Unlike ``TemporalGroupedAutoencoder`` (whose latent keeps the full time axis, giving it
enough capacity to memorise per-channel noise and producing an *inverted* perturbation
influence), this variant pools over time to a length-``pool`` latent (default 1). The
bottleneck forces the encoder to keep only the low-rank, class-relevant structure shared
across informative channels, so the perturbation influence tracks channel informativeness
(loud informative channels high, noise low) instead of inverting.
"""
from __future__ import annotations
from typing import Hashable

import torch
import torch.nn as nn
import torch.nn.functional as F


def _key(g: Hashable) -> str:
    return str(g).replace(".", "_").replace(" ", "_")


class BottleneckTemporalAutoencoder(nn.Module):
    def __init__(self, channels_per_group: dict[Hashable, int], time_len: int,
                 latent_dim: int = 8, hidden: int = 32, pool: int = 1,
                 latent_act: bool = True, fusion: str = "mean") -> None:
        super().__init__()
        if fusion not in ("mean", "max"):
            raise ValueError("fusion must be 'mean' or 'max'")
        self.groups = list(channels_per_group.keys())
        self.channels_per_group = dict(channels_per_group)
        self.time_len = time_len
        self.latent_dim = latent_dim
        self.pool = pool
        self.fusion = fusion

        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        for g in self.groups:
            cin = channels_per_group[g]
            enc = [nn.Conv1d(cin, hidden, kernel_size=5, padding=2), nn.ReLU(),
                   nn.Conv1d(hidden, latent_dim, kernel_size=5, padding=2)]
            if latent_act:                                # optional: signed latent when False
                enc.append(nn.ReLU())
            enc.append(nn.AdaptiveAvgPool1d(pool))        # <-- compress the time axis
            self.encoders[_key(g)] = nn.Sequential(*enc)
            self.decoders[_key(g)] = nn.Sequential(
                nn.Conv1d(latent_dim, hidden, kernel_size=5, padding=2), nn.ReLU(),
                nn.Conv1d(hidden, cin, kernel_size=5, padding=2),
            )

    def encode(self, batch: dict[Hashable, torch.Tensor]) -> torch.Tensor:
        feats = []
        for g in self.groups:
            x = batch[g].permute(0, 2, 1)                 # (B, ch, time)
            feats.append(self.encoders[_key(g)](x))       # (B, latent_dim, pool)
        stacked = torch.stack(feats, dim=1)               # (B, n_groups, latent_dim, pool)
        return stacked.amax(dim=1) if self.fusion == "max" else stacked.mean(dim=1)

    def decode(self, latent: torch.Tensor) -> dict[Hashable, torch.Tensor]:
        # broadcast the pooled latent back to full time before the per-group decoders
        z = latent if latent.shape[-1] == self.time_len else \
            F.interpolate(latent, size=self.time_len, mode="nearest")
        out = {}
        for g in self.groups:
            out[g] = self.decoders[_key(g)](z).permute(0, 2, 1)   # (B, time, ch)
        return out

    def forward(self, batch):
        return self.decode(self.encode(batch))
