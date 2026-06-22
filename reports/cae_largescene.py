"""Does the CAE reconstruct when the scene is at REAL-DATA resolution (many chunks, real batched
training) instead of a single 64x64 chunk? Tests the hypothesis that single-chunk/batch=1 was the
cause, and exercises the chunk split path. Reconstruction R2 is measured chunk-wise (memory-safe);
selection F1 on subsampled pixels.

Run:  python reports/cae_largescene.py
"""
from __future__ import annotations

import contextlib
import os
import pathlib
import tempfile

import numpy as np
import torch

import realistic_benchmark as rb
from classification_experiment import cols_for_bands, feature_matrix, knn_macro_f1
from spectral_select import Analyzer, Config
from spectral_select.models.training import create_spatial_chunks
from sweep_common import oracle_cols

CFG = {"default": {}, "no_reg": {"model_dropout_rate": 0.0, "model_sparsity_weight": 0.0, "training_epochs": 60}}


def chunkwise_r2(model, data):
    exes = list(model.excitation_wavelengths)
    chunks = {ex: create_spatial_chunks(data[ex].numpy(), chunk_size=64, chunk_overlap=8)[0] for ex in exes}
    nch = len(chunks[exes[0]])
    r2 = []
    with torch.no_grad():
        for ci in range(nch):
            inp = {ex: torch.tensor(chunks[ex][ci][None], dtype=torch.float32) for ex in exes}
            rec = model.decode(model.encode(inp))
            for ex in exes:
                X = inp[ex][0].numpy(); R = rec[ex][0].numpy()
                for b in range(X.shape[2]):
                    xb, rb_ = X[:, :, b].ravel(), R[:, :, b].ravel()
                    m, m0 = np.mean((xb - rb_) ** 2), np.mean((xb - xb.mean()) ** 2)
                    r2.append(1 - m / m0 if m0 > 1e-12 else 0.0)
    return float(np.nanmean(r2)), nch


def run(size, regime, cfgname, seed=1):
    nuis = 0.0 if regime == "clean" else 2.0
    turb = 0.0 if regime == "clean" else 1.0
    spectra, gt, y, acq = rb.build_dataset(seed, size=size, nuisance_amp=nuis, turbidity_amp=turb,
                                           raman=0.0 if regime == "clean" else 0.4)
    y = np.asarray(y)
    kw = dict(sample_name="L", n_bands_to_select=12, n_important_dimensions=18,
              perturbation_method="percentile", normalization_method="none", use_diversity_constraint=True,
              training_epochs=30, device="cpu", random_seed=0, output_dir=pathlib.Path(tempfile.mkdtemp()))
    kw.update(CFG[cfgname])
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        a = Analyzer(Config(**kw)); a.fit(spectra)
    r2, nch = chunkwise_r2(a._model, a._dataset.get_all_data())
    X, colmap = feature_matrix(spectra)
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], min(8000, X.shape[0]), replace=False)
    Xs, ys = X[idx], y[idx]
    cols = cols_for_bands(colmap, [(b.excitation_nm, b.emission_nm) for b in a.get_wavelengths()])
    f1 = knn_macro_f1(Xs[:, cols], ys, seed) if cols else float("nan")
    orc = knn_macro_f1(Xs[:, oracle_cols(Xs, ys, 12)], ys, seed)
    print(f"  {size}px {regime:<10}{cfgname:<9} chunks={nch:>3}  reconR2={r2:>+8.3f}  CAE_F1={f1:.3f}  oracle={orc:.3f}")
    return r2, f1


def main():
    print("=" * 86)
    print("CAE at REAL-DATA RESOLUTION (many chunks => real batched training)")
    print("=" * 86)
    print(f"  {'':5}{'regime':<10}{'config':<9}{'chunks':>9}{'reconR2':>13}{'CAE_F1':>9}{'oracle':>9}")
    run(256, "clean", "default")
    run(256, "clean", "no_reg")
    run(256, "realistic", "no_reg")
    run(512, "clean", "no_reg")
    print("-" * 86)
    print("Compare reconR2 to the 64x64 single-chunk result (~ -0.1 clean / <0 realistic). If many")
    print("chunks lift R2>0, single-chunk/batch=1 was the cause; if not, the architecture is.")


if __name__ == "__main__":
    main()
