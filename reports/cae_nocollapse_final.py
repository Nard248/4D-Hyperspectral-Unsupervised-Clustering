"""Decisive test: is the band-collapse (adaptive_avg_pool3d) the reason the CAE can't reconstruct
multi-fluorophore data? Compare, on clean 3-dye data with the HONEST signal-band metric:
  - MLP control (reference; must reconstruct -> high signalCorr)
  - standard CAE (band-collapse), gelu, no-reg, no-LR-collapse
  - band-preserving CAE (DeepSpectralCAE, no collapse), no-reg, no-LR-collapse
signalCorr = mean corr(input,recon) on the top-15 variance (signal) bands; selF1 = perturbation
selection KNN-F1. Keeps the spatial-CAE + perturbation idea; only the band-collapse changes.

Run:  python reports/cae_nocollapse_final.py
"""
from __future__ import annotations

import contextlib
import os
import pathlib
import tempfile

import numpy as np
import torch
import torch.nn as nn

from classification_experiment import build_dataset as clean_build, cols_for_bands, feature_matrix, knn_macro_f1
from spectraforge import ArtifactConfig
from spectral_select import Analyzer, Config
from spectral_select.architectures.deep_cae import DeepSpectralCAE
from sweep_common import oracle_cols


def signal_corr(inp, rec, topk=15):
    cs = []
    for ex in inp:
        X = inp[ex].reshape(-1, inp[ex].shape[-1]); R = rec[ex].reshape(-1, rec[ex].shape[-1])
        for b in np.argsort(X.var(0))[::-1][:topk]:
            if X[:, b].std() > 1e-9 and R[:, b].std() > 1e-9:
                cs.append(np.corrcoef(X[:, b], R[:, b])[0, 1])
    return float(np.nanmean(cs)) if cs else 0.0


def fit(spectra, arch, **over):
    kw = dict(sample_name="f", n_bands_to_select=12, n_important_dimensions=18,
              perturbation_method="percentile", normalization_method="none", use_diversity_constraint=True,
              training_epochs=250, training_scheduler_patience=10000, model_dropout_rate=0.0,
              model_sparsity_weight=0.0, autoencoder_architecture=arch,
              device="cpu", random_seed=0, output_dir=pathlib.Path(tempfile.mkdtemp()))
    kw.update(over)
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        a = Analyzer(Config(**kw)); a.fit(spectra)
    return a


def cae_eval(a, X, colmap, y):
    model, data = a._model, a._dataset.get_all_data()
    with torch.no_grad():
        inp = {ex: data[ex].unsqueeze(0) for ex in model.excitation_wavelengths}
        rec = model.decode(model.encode(inp))
    sc = signal_corr({e: inp[e][0].numpy() for e in inp}, {e: rec[e][0].numpy() for e in rec})
    cols = cols_for_bands(colmap, [(b.excitation_nm, b.emission_nm) for b in a.get_wavelengths()])
    f1 = knn_macro_f1(X[:, cols], y, 1) if cols else float("nan")
    return sc, f1


def mlp_signalcorr(spectra):
    X, _ = feature_matrix(spectra)
    sc_ = (X - X.mean(0)) / (X.std(0) + 1e-9)
    Xt = torch.tensor(sc_.astype(np.float32)); d = Xt.shape[1]
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 8), nn.ReLU(),
                      nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, d))
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    for _ in range(800):
        opt.zero_grad(); loss = ((m(Xt) - Xt) ** 2).mean(); loss.backward(); opt.step()
    with torch.no_grad():
        R = m(Xt).numpy()
    nb = X.shape[1]
    cs = [np.corrcoef(sc_[:, b], R[:, b])[0, 1] for b in np.argsort(X.var(0))[::-1][:15]]
    return float(np.nanmean(cs))


def main():
    clean, _, y, _ = clean_build(1, ArtifactConfig(rayleigh_strength=0.1, photon_scale=800, read_sigma=0.01))
    y = np.asarray(y); X, colmap = feature_matrix(clean)
    orc = knn_macro_f1(X[:, oracle_cols(X, y, 12)], y, 1)
    print("=" * 80)
    print("BAND-COLLAPSE vs BAND-PRESERVING CAE on clean 3-dye (signal-band metric)")
    print(f"oracle selF1={orc:.3f}; MLP-control signalCorr={mlp_signalcorr(clean):.3f} (reference)")
    print("=" * 80)
    print(f"{'architecture':<34}{'signalCorr':>12}{'selF1':>8}")
    a1 = fit(clean, "standard", model_hidden_activation="gelu", model_output_activation="sigmoid")
    sc1, f11 = cae_eval(a1, X, colmap, y)
    print(f"{'standard CAE (band-collapse, gelu)':<34}{sc1:>12.3f}{f11:>8.3f}")
    a2 = fit(clean, DeepSpectralCAE)
    sc2, f12 = cae_eval(a2, X, colmap, y)
    print(f"{'band-preserving CAE (no collapse)':<34}{sc2:>12.3f}{f12:>8.3f}")
    print("-" * 80)
    print("If no-collapse signalCorr >> collapse, the band-collapse was the reconstruction bottleneck.")


if __name__ == "__main__":
    main()
