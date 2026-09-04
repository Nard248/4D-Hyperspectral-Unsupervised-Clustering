"""Spatial (Conv2D) compressive-bottleneck group-structured autoencoder.

The imaging-domain encoder: channels are spectral bands, the regular axis is the 2-D
image plane. Identical selection engine as the temporal path -- only the encoder changes
(per-group Conv2D + spatial pooling to a per-patch bottleneck). Pooling the spatial axes
keeps the latent from absorbing per-band noise, so the perturbation influence remains a
reliable band-importance signal (the same fix as the temporal bottleneck).
"""
from __future__ import annotations
from typing import Hashable

import torch
import torch.nn as nn
import torch.nn.functional as F


def _key(g: Hashable) -> str:
    return str(g).replace(".", "_").replace(" ", "_")


class SpatialBottleneckAutoencoder(nn.Module):
    def __init__(self, channels_per_group: dict[Hashable, int], spatial_size: tuple[int, int],
                 latent_dim: int = 8, hidden: int = 32, latent_act: bool = False,
                 fusion: str = "mean") -> None:
        super().__init__()
        if fusion not in ("mean", "max"):
            raise ValueError("fusion must be 'mean' or 'max'")
        self.groups = list(channels_per_group.keys())
        self.channels_per_group = dict(channels_per_group)
        self.spatial_size = tuple(spatial_size)
        self.latent_dim = latent_dim
        self.fusion = fusion

        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        for g in self.groups:
            cin = channels_per_group[g]
            enc = [nn.Conv2d(cin, hidden, kernel_size=3, padding=1), nn.ReLU(),
                   nn.Conv2d(hidden, latent_dim, kernel_size=3, padding=1)]
            if latent_act:
                enc.append(nn.ReLU())
            enc.append(nn.AdaptiveAvgPool2d(1))            # <-- compress the image plane
            self.encoders[_key(g)] = nn.Sequential(*enc)
            self.decoders[_key(g)] = nn.Sequential(
                nn.Conv2d(latent_dim, hidden, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv2d(hidden, cin, kernel_size=3, padding=1),
            )

    def encode(self, batch: dict[Hashable, torch.Tensor]) -> torch.Tensor:
        feats = []
        for g in self.groups:
            x = batch[g].permute(0, 3, 1, 2)               # (B, H, W, ch) -> (B, ch, H, W)
            feats.append(self.encoders[_key(g)](x))        # (B, latent_dim, 1, 1)
        stacked = torch.stack(feats, dim=1)
        return stacked.amax(dim=1) if self.fusion == "max" else stacked.mean(dim=1)

    def decode(self, latent: torch.Tensor) -> dict[Hashable, torch.Tensor]:
        z = F.interpolate(latent, size=self.spatial_size, mode="nearest")   # (B, latent, H, W)
        out = {}
        for g in self.groups:
            out[g] = self.decoders[_key(g)](z).permute(0, 2, 3, 1)          # (B, H, W, ch)
        return out

    def forward(self, batch):
        return self.decode(self.encode(batch))
