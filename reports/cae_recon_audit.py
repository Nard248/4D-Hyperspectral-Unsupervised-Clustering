"""Reconstruction audit: does the published CAE actually reconstruct, or literally predict the
per-band mean? The loss alone is deceiving — inspect numerically AND visually, and verify the data
pipeline did not corrupt pixels.

For each excitation we report, per band: corr(input,recon), and R^2 = 1 - MSE_model/MSE_meanpredictor
(the decisive number: R^2<=0 means the model is no better than predicting the per-band spatial mean;
R^2->1 means faithful reconstruction). We also (1) verify dataset normalization round-trips to the
original spectra (no corruption), and (2) save side-by-side images input | recon | residual | mean.

Run:  python reports/cae_recon_audit.py
"""
from __future__ import annotations

import contextlib
import os
import pathlib
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import realistic_benchmark as rb
from spectraforge import ArtifactConfig
from spectral_select import Analyzer, Config

OUT = pathlib.Path(__file__).parent / "exp_records" / "recon_audit"
OUT.mkdir(parents=True, exist_ok=True)


def fit_cae(spectra, **cfg_over):
    kw = dict(sample_name="audit", n_bands_to_select=12, training_epochs=120, device="cpu",
              random_seed=0, output_dir=pathlib.Path(tempfile.mkdtemp()))
    kw.update(cfg_over)
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        a = Analyzer(Config(**kw))
        a.fit(spectra)
    return a


def audit(tag, spectra, **cfg_over):
    a = fit_cae(spectra, **cfg_over)
    model, data = a._model, a._dataset.get_all_data()
    exes = list(model.excitation_wavelengths)

    # (1) pipeline integrity: does the normalized data round-trip to the original cube?
    p = a._dataset.normalization_params
    ex0 = exes[0]
    orig = spectra.get_excitation(ex0).cube
    recon_orig = data[ex0].numpy() * (p["max"] - p["min"]) + p["min"]
    rt_err = float(np.max(np.abs(orig - recon_orig)))

    with torch.no_grad():
        inp = {ex: data[ex].unsqueeze(0) for ex in exes}
        rec = model.decode(model.encode(inp))

    print(f"\n[{tag}] cfg_over={cfg_over}")
    print(f"  normalization round-trip max|orig-denorm| = {rt_err:.2e}  (≈0 => pipeline preserves pixels)")
    print(f"  {'ex':>6}{'meanR':>9}{'meanR2':>9}{'model/meanMSE':>15}{'fracR2>0.1':>12}")
    allr, allr2 = [], []
    for ex in exes:
        X = inp[ex][0].numpy(); R = rec[ex][0].numpy()
        nb = X.shape[2]
        rs, r2s = [], []
        m_mse, mean_mse = 0.0, 0.0
        for b in range(nb):
            xb, rb_ = X[:, :, b].ravel(), R[:, :, b].ravel()
            sx, sr = xb.std(), rb_.std()
            rs.append(float(np.corrcoef(xb, rb_)[0, 1]) if sx > 1e-9 and sr > 1e-9 else 0.0)
            mse_m = float(np.mean((xb - rb_) ** 2))
            mse_mean = float(np.mean((xb - xb.mean()) ** 2))           # predict-the-band-mean
            r2s.append(1 - mse_m / mse_mean if mse_mean > 1e-12 else 0.0)
            m_mse += mse_m; mean_mse += mse_mean
        rs, r2s = np.array(rs), np.array(r2s)
        allr += list(rs); allr2 += list(r2s)
        print(f"  {ex:>6.0f}{np.nanmean(rs):>9.3f}{np.nanmean(r2s):>9.3f}"
              f"{m_mse / (mean_mse + 1e-12):>15.3f}{np.mean(r2s > 0.1):>12.2f}")
    allr, allr2 = np.array(allr), np.array(allr2)
    print(f"  OVERALL meanR={np.nanmean(allr):+.3f}  meanR2={np.nanmean(allr2):+.3f}  "
          f"(R2<=0 => CAE is a per-band-mean predictor; R2->1 => faithful)")

    # (2) visual: the 3 highest-variance bands of ex0 — input | recon | residual | per-band mean
    ex = ex0
    X = inp[ex][0].numpy(); R = rec[ex][0].numpy()
    var = X.reshape(-1, X.shape[2]).var(0)
    bands = list(np.argsort(var)[::-1][:3])
    fig, axes = plt.subplots(len(bands), 4, figsize=(11, 3 * len(bands)))
    if len(bands) == 1:
        axes = axes[None, :]
    for i, b in enumerate(bands):
        meanimg = np.full_like(X[:, :, b], X[:, :, b].mean())
        for j, (img, ttl) in enumerate([(X[:, :, b], "input"), (R[:, :, b], "recon"),
                                        (R[:, :, b] - X[:, :, b], "residual"), (meanimg, "band-mean")]):
            ax = axes[i, j]
            im = ax.imshow(img, cmap="viridis")
            ax.set_title(f"b{b} {ttl}", fontsize=8); ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"{tag} ex{ex:.0f}: input vs CAE recon (does recon show structure or a flat mean?)")
    fig.tight_layout()
    fig.savefig(OUT / f"recon_{tag}.png", dpi=90)
    plt.close(fig)
    print(f"  -> saved {OUT / f'recon_{tag}.png'}")
    return float(np.nanmean(allr2))


def main():
    print("=" * 90)
    print("CAE RECONSTRUCTION AUDIT — is R~0 a real mean-predictor failure or an artifact?")
    print("=" * 90)
    real, _, _, _ = rb.build_dataset(1)
    audit("realistic_default", real)
    # a clean bright scene (the easy case) for contrast
    from classification_experiment import build_dataset as clean_build
    clean, _, _, _ = clean_build(1, ArtifactConfig(rayleigh_strength=0.1, photon_scale=800, read_sigma=0.01))
    audit("clean_default", clean)


if __name__ == "__main__":
    main()
