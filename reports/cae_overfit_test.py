"""Isolate the bug: can the CAE architecture OVERFIT a single rank-1 image with a plain Adam/MSE loop
(no train_with_masking, no chunking, no sparsity, no dropout)? A working AE must drive R2->1 here.

If some config overfits -> the architecture is fine and train_with_masking is the bug.
If nothing overfits -> the architecture itself cannot represent the data. A tiny per-pixel MLP control
confirms the data is trivially fittable.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from cae_data_search import build_simple
from spectral_select.models.autoencoder import HyperspectralCAEWithMasking


def r2(x, r):
    x = x.reshape(-1, x.shape[-1]); r = r.reshape(-1, r.shape[-1])
    out = []
    for b in range(x.shape[1]):
        m = ((x[:, b] - r[:, b]) ** 2).mean(); m0 = ((x[:, b] - x[:, b].mean()) ** 2).mean()
        out.append(1 - m / m0 if m0 > 1e-12 else 0.0)
    return float(np.nanmean(out))


def main():
    spectra, _ = build_simple(1, 1)                      # single dye, dense, ~rank-1
    ex = spectra.excitation_wavelengths[0]
    cube = spectra.get_excitation(ex).cube.astype(np.float32)
    gmin, gmax = cube.min(), cube.max()
    cube = (cube - gmin) / (gmax - gmin + 1e-9)          # global min-max like the dataset
    X = torch.tensor(cube)[None]                          # (1,H,W,bands)
    inp = {float(ex): X}
    exdata = {float(ex): cube}
    print(f"rank-1 cube {cube.shape}; effective rank check: ", end="")
    flat = cube.reshape(-1, cube.shape[-1]); sv = np.linalg.svd(flat - flat.mean(0), compute_uv=False)[:3]
    print(f"top-3 singular values {sv.round(2)} (rank-1 => one dominates)")

    print(f"\n{'model':<34}{'lr':>7}{'final MSE':>12}{'reconR2':>10}")
    for hid, out in [("sigmoid", "sigmoid"), ("relu", "sigmoid"), ("relu", "identity"), ("gelu", "identity")]:
        for lr in (1e-3, 1e-2):
            torch.manual_seed(0)
            m = HyperspectralCAEWithMasking(exdata, k1=20, k3=20, filter_size=5, sparsity_weight=0.0,
                                            dropout_rate=0.0, hidden_activation=hid, output_activation=out)
            opt = torch.optim.Adam(m.parameters(), lr)
            m.train()
            for _ in range(1500):
                opt.zero_grad()
                rec = m(inp)[float(ex)]
                loss = ((rec - X) ** 2).mean()
                loss.backward(); opt.step()
            m.eval()
            with torch.no_grad():
                rec = m(inp)[float(ex)].numpy()
            print(f"CAE {hid}/{out:<10}{'':<8}{lr:>7.0e}{float(loss):>12.5f}{r2(cube, rec):>10.3f}")

    # control: tiny per-pixel MLP (must overfit -> R2 ~ 1, proving the data is trivially fittable)
    torch.manual_seed(0)
    d = cube.shape[-1]
    mlp = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 8), nn.ReLU(), nn.Linear(8, 32),
                        nn.ReLU(), nn.Linear(32, d))
    Xpix = torch.tensor(flat)
    opt = torch.optim.Adam(mlp.parameters(), 1e-3)
    for _ in range(1500):
        opt.zero_grad(); loss = ((mlp(Xpix) - Xpix) ** 2).mean(); loss.backward(); opt.step()
    with torch.no_grad():
        recp = mlp(Xpix).numpy().reshape(cube.shape)
    print(f"\nCONTROL per-pixel MLP{'':<13}{'1e-3':>7}{float(loss):>12.5f}{r2(cube, recp):>10.3f}  "
          f"(should be ~1.0 -> data is trivially fittable)")


if __name__ == "__main__":
    main()
