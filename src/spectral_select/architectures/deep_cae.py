"""C1 — deeper, spectral-preserving convolutional autoencoder.

Two changes from the published ``HyperspectralCAEWithMasking``, isolating "more capacity + keep the
spectral axis" while holding the objective (MSE) fixed:

1. **No band-axis collapse.** The published encoder does ``adaptive_avg_pool3d(x, (1, H, W))``
   (``models/autoencoder.py:159``), squashing the emission axis to length 1 — the latent throws away
   spectral resolution. Here the emission-band axis is preserved all the way through the latent.
2. **More depth** — an extra conv block in the encoder and decoder, ReLU hidden activations.

This is Analyzer-interface compatible (``encode``/``decode`` + ``excitation_wavelengths`` /
``emission_bands``), so it is driven through the real :class:`spectral_select.Analyzer` via
``config.autoencoder_architecture`` — same data normalization, training, and selection as C0; only
the architecture differs (a clean ablation).

Assumes every excitation shares the emission-band count (true for SpectraForge ME-HSI and for the
processed real cubes, which share one emission grid). Raises otherwise.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepSpectralCAE(nn.Module):
    """Deeper spatial CAE that preserves the emission-band axis in the latent."""

    def __init__(self, excitations_data, k1: int = 20, k3: int = 20, filter_size: int = 5, **kw):
        super().__init__()
        self.excitation_wavelengths = sorted(excitations_data)
        self.emission_bands = {ex: int(excitations_data[ex].shape[2]) for ex in self.excitation_wavelengths}
        band_counts = set(self.emission_bands.values())
        if len(band_counts) != 1:
            raise ValueError(
                f"DeepSpectralCAE requires a shared emission-band count across excitations, got "
                f"{self.emission_bands}. Use the 'standard' architecture for ragged band counts.")
        self.n_bands = band_counts.pop()
        self.ex_to_key = {ex: str(ex).replace(".", "_") for ex in self.excitation_wavelengths}

        fs = filter_size
        pad = fs // 2
        kb = min(5, self.n_bands)               # depth (band-axis) kernel
        padb = kb // 2

        # per-excitation input stem (depth kernel over the band axis, NO pooling)
        self.enc_conv1 = nn.ModuleDict()
        for ex in self.excitation_wavelengths:
            self.enc_conv1[self.ex_to_key[ex]] = nn.Conv3d(1, k1, (kb, fs, fs), padding=(padb, pad, pad))
        # extra shared depth
        self.enc_conv2 = nn.Conv3d(k1, k1, (3, fs, fs), padding=(1, pad, pad))
        self.enc_conv3 = nn.Conv3d(k1, k3, (3, fs, fs), padding=(1, pad, pad))

        # decoder: shared body then per-excitation reconstruction head
        self.dec_conv1 = nn.Conv3d(k3, k1, (3, fs, fs), padding=(1, pad, pad))
        self.dec_conv2 = nn.Conv3d(k1, k1, (3, fs, fs), padding=(1, pad, pad))
        self.dec_out = nn.ModuleDict()
        for ex in self.excitation_wavelengths:
            self.dec_out[self.ex_to_key[ex]] = nn.Conv3d(k1, 1, (kb, fs, fs), padding=(padb, pad, pad))

        # C1 holds the objective fixed at plain MSE: no sparsity term (the published CAE's sparsity
        # KL assumes sigmoid-bounded activations; our ReLU latent is unbounded). The Analyzer trainer
        # reads ``sparsity_weight`` and calls ``compute_sparsity_loss`` — we disable both.
        self.sparsity_weight = 0.0

    def encode(self, data_dict):
        feats = []
        for ex in self.excitation_wavelengths:
            if ex not in data_dict:
                continue
            x = data_dict[ex].permute(0, 3, 1, 2).unsqueeze(1)   # (B, 1, bands, H, W)
            x = F.relu(self.enc_conv1[self.ex_to_key[ex]](x))
            x = F.relu(self.enc_conv2(x))
            feats.append(x)
        if not feats:
            raise ValueError("No valid excitation wavelengths found in input data")
        shared = torch.mean(torch.stack(feats, dim=1), dim=1)    # (B, k1, bands, H, W)
        return F.relu(self.enc_conv3(shared))                    # (B, k3, bands, H, W) — band axis kept

    def decode(self, latent):
        h = F.relu(self.dec_conv1(latent))
        h = F.relu(self.dec_conv2(h))
        out = {}
        for ex in self.excitation_wavelengths:
            y = torch.sigmoid(self.dec_out[self.ex_to_key[ex]](h))  # (B, 1, bands, H, W), data in [0,1]
            out[ex] = y.squeeze(1).permute(0, 2, 3, 1)              # (B, H, W, bands)
        return out

    def forward(self, data_dict):
        return self.decode(self.encode(data_dict))

    def compute_masked_loss(self, output_dict, target_dict, valid_mask_dict=None, spatial_mask=None):
        """Plain masked MSE (matches the Analyzer trainer's call signature)."""
        total, n = 0.0, 0
        for ex in target_dict:
            if ex not in output_dict:
                continue
            out, tgt = output_dict[ex], target_dict[ex]
            if out.shape != tgt.shape:
                continue
            se = (out - tgt) ** 2
            if spatial_mask is not None:
                m = spatial_mask
                while m.dim() < tgt.dim():
                    m = m.unsqueeze(-1)
                m = m.expand_as(tgt)
                denom = m.sum()
                total = total + (se * m).sum() / denom if denom > 0 else total + se.mean()
            else:
                total = total + se.mean()
            n += 1
        return total / n if n > 0 else total

    def compute_sparsity_loss(self, encoded):
        return encoded.sum() * 0.0                                  # disabled (see __init__)
