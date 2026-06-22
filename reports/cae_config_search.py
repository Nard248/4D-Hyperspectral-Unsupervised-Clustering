"""Search for a CAE config where reconstruction is genuine (R2>0) AND the perturbation selection
works. The published defaults use dropout=0.5 + sparsity_weight=1.0 + only 30 epochs on a single
64x64 image (1 chunk) — all of which fight reconstruction. We flip these knobs (and try removing the
band-collapse via the no-pool architecture) and measure, per regime, both reconstruction quality
(R2 vs a per-band-mean predictor) and downstream KNN-F1 of the perturbation selection.

Goal: scientific evidence for whether the CAE + perturbation mechanism can be made to work on
synthetic data, and under what configuration.

Run:  python reports/cae_config_search.py
"""
from __future__ import annotations

import contextlib
import os
import pathlib
import tempfile

import numpy as np
import torch

import realistic_benchmark as rb
from classification_experiment import build_dataset as clean_build, cols_for_bands, feature_matrix, knn_macro_f1
from spectraforge import ArtifactConfig
from spectral_select import Analyzer, Config
from spectral_select.architectures.deep_cae import DeepSpectralCAE
from sweep_common import oracle_cols

OUT = pathlib.Path(__file__).parent / "exp_records"


def regimes():
    clean, _, yc, _ = clean_build(1, ArtifactConfig(rayleigh_strength=0.1, photon_scale=800, read_sigma=0.01))
    real, _, yr, _ = rb.build_dataset(1)
    return {"clean": (clean, np.asarray(yc)), "realistic": (real, np.asarray(yr))}


def recon_r2(model, data):
    exes = list(model.excitation_wavelengths)
    with torch.no_grad():
        inp = {ex: data[ex].unsqueeze(0) for ex in exes}
        rec = model.decode(model.encode(inp))
    r2 = []
    for ex in exes:
        X = inp[ex][0].numpy(); R = rec[ex][0].numpy()
        for b in range(X.shape[2]):
            xb, rb_ = X[:, :, b].ravel(), R[:, :, b].ravel()
            mse_m = np.mean((xb - rb_) ** 2); mse_mean = np.mean((xb - xb.mean()) ** 2)
            r2.append(1 - mse_m / mse_mean if mse_mean > 1e-12 else 0.0)
    return float(np.nanmean(r2))


def fit_eval(spectra, y, cfg_over):
    kw = dict(sample_name="s", n_bands_to_select=12, n_important_dimensions=18,
              perturbation_method="percentile", normalization_method="none",
              use_diversity_constraint=True, training_epochs=30, device="cpu", random_seed=0,
              output_dir=pathlib.Path(tempfile.mkdtemp()))
    kw.update(cfg_over)
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        a = Analyzer(Config(**kw)); a.fit(spectra)
    r2 = recon_r2(a._model, a._dataset.get_all_data())
    X, colmap = feature_matrix(spectra)
    cols = cols_for_bands(colmap, [(b.excitation_nm, b.emission_nm) for b in a.get_wavelengths()])
    f1 = knn_macro_f1(X[:, cols], y, 1) if cols else float("nan")
    return r2, f1


VARIANTS = {
    "default(pub)": {},
    "no_dropout": {"model_dropout_rate": 0.0},
    "no_sparsity": {"model_sparsity_weight": 0.0},
    "no_reg": {"model_dropout_rate": 0.0, "model_sparsity_weight": 0.0},
    "no_reg_long": {"model_dropout_rate": 0.0, "model_sparsity_weight": 0.0, "training_epochs": 300},
    "no_reg_long_k40": {"model_dropout_rate": 0.0, "model_sparsity_weight": 0.0, "training_epochs": 300,
                        "model_k1": 40, "model_k3": 40},
    "nocollapse_no_reg_long": {"model_dropout_rate": 0.0, "model_sparsity_weight": 0.0,
                               "training_epochs": 300, "autoencoder_architecture": DeepSpectralCAE},
}


def main():
    regs = regimes()
    oracle = {r: knn_macro_f1(feature_matrix(s)[0][:, oracle_cols(feature_matrix(s)[0], y, 12)], y, 1)
              for r, (s, y) in regs.items()}
    print("=" * 92)
    print("CAE CONFIG SEARCH — reconstruction R2 and selection F1 per variant per regime")
    print(f"oracle F1: clean={oracle['clean']:.3f} realistic={oracle['realistic']:.3f}")
    print("=" * 92)
    print(f"{'variant':<26}{'clean R2':>10}{'clean F1':>10}{'real R2':>10}{'real F1':>10}")
    rows = []
    for name, cov in VARIANTS.items():
        out = {}
        for r, (s, y) in regs.items():
            out[r] = fit_eval(s, y, cov)
        print(f"{name:<26}{out['clean'][0]:>+10.3f}{out['clean'][1]:>10.3f}"
              f"{out['realistic'][0]:>+10.3f}{out['realistic'][1]:>10.3f}")
        rows.append((name, out))
    print("-" * 92)
    print("Read: R2>0 => genuine reconstruction (not a mean predictor); F1 vs oracle => selection works.")


if __name__ == "__main__":
    main()
