"""Hyperparameter ablations for C3 (mask ratio) and C4 (beta, free-bits) — doc 04 §6 asks which
settings work. Light + fast (per-pixel AEs only; no spatial-CAE training). Mean over a few scenes.

Run:  python reports/architecture_ablation.py
"""
from __future__ import annotations

import numpy as np

from spectraforge import ArtifactConfig
from spectraforge.validation import validate_selection

from classification_experiment import BUDGET, build_dataset, cols_for_bands, feature_matrix, knn_macro_f1
from spectral_select.architectures import MaskedSpectralAE, VariationalSpectralAE

SEEDS = [1, 2, 3]
NOISE = {"low": ArtifactConfig(rayleigh_strength=0.1, photon_scale=800, read_sigma=0.01),
         "high": ArtifactConfig(rayleigh_strength=0.3, photon_scale=150, read_sigma=0.05)}


def _scenes():
    out = []
    for nl, noise in NOISE.items():
        for seed in SEEDS:
            spectra, gt, y, acq = build_dataset(seed, noise)
            X, colmap = feature_matrix(spectra)
            out.append((nl, seed, spectra, gt, y, X, colmap))
    return out


def _score(model, spectra, gt, y, X, colmap, seed):
    model.fit(spectra)
    bands = model.select(BUDGET)
    cols = cols_for_bands(colmap, bands)
    return (knn_macro_f1(X[:, cols], y, seed),
            validate_selection(gt, bands, tol_nm=10)["peak_recovery"],
            model.reconstruction_r(), model.influence_signal_corr())


def main():
    scenes = _scenes()

    print("=" * 78)
    print("C3 masked-spectral-AE — mask-ratio sweep (mean over all scenes)")
    print("=" * 78)
    print(f"{'mask_ratio':>11}{'KNN-F1':>9}{'peak':>7}{'recon R':>9}{'infl-corr':>11}")
    for mr in (0.2, 0.4, 0.5, 0.6, 0.75, 0.9):
        rows = [_score(MaskedSpectralAE(epochs=300, mask_ratio=mr, seed=s), sp, gt, y, X, cm, s)
                for (_, s, sp, gt, y, X, cm) in scenes]
        f1, pk, R, co = np.mean(rows, axis=0)
        print(f"{mr:>11.2f}{f1:>9.3f}{pk:>7.2f}{R:>+9.3f}{co:>+11.3f}")

    print("\n" + "=" * 78)
    print("C4 variational-spectral-AE — beta x free-bits sweep (mean over all scenes)")
    print("=" * 78)
    print(f"{'beta':>6}{'free_bits':>10}{'KNN-F1':>9}{'peak':>7}{'recon R':>9}{'infl-corr':>11}{'active':>8}")
    for beta in (0.1, 1.0, 4.0):
        for fb in (0.0, 0.5, 1.0):
            rows, au = [], []
            for (_, s, sp, gt, y, X, cm) in scenes:
                m = VariationalSpectralAE(epochs=300, beta=beta, free_bits=fb, seed=s)
                rows.append(_score(m, sp, gt, y, X, cm, s))
                au.append(m.active_units())
            f1, pk, R, co = np.mean(rows, axis=0)
            print(f"{beta:>6.1f}{fb:>10.2f}{f1:>9.3f}{pk:>7.2f}{R:>+9.3f}{co:>+11.3f}{np.mean(au):>8.1f}")
    print("-" * 78)
    print("active = mean # latent dims with per-dim KL above the free-bits floor (posterior-collapse check)")


if __name__ == "__main__":
    main()
