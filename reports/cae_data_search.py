"""Final search: is there ANY synthetic-data regime where the spatial CAE reconstructs (R2>0) and
its perturbation selection works? Model side is exhausted (no config gives R2>0); here we vary the
DATA from trivial to realistic using the BEST CAE config (no regularization, 500 epochs, k=40):

  1dye      - a single fluorophore, smooth field, dense, near-noiseless (the easiest possible target)
  2dye/3dye - more components (clean, dense, low noise)
  3dye+ppn  - 3 dyes but per-pixel L2-normalized input (removes the brightness/sparsity that the
              global min-max normalization introduces)

If the CAE can't even overfit a single dense smooth field, the spatial architecture is the cause.

Run:  python reports/cae_data_search.py
"""
from __future__ import annotations

import contextlib
import os
import pathlib
import tempfile

import numpy as np
import torch

from classification_experiment import cols_for_bands, feature_matrix, knn_macro_f1
from spectraforge import AcquisitionConfig, ArtifactConfig, Material, PhysicsConfig
from spectraforge.fluorophore import Fluorophore
from spectraforge.forward import render
from spectraforge.scenegen import random_field
from spectral_select import Analyzer, Config
from spectral_select.types import ExcitationData, SpectraData
from sweep_common import oracle_cols

BEST = dict(model_dropout_rate=0.0, model_sparsity_weight=0.0, training_epochs=500,
            model_k1=40, model_k3=40)


def build_simple(seed, n_dye, perpixel=False, photon=3000):
    dyes = [Fluorophore(f"D{i}", ex_peak_nm=470 + 30 * i, ex_fwhm_nm=40, em_peak_nm=500 + 45 * i,
                        em_fwhm_nm=45, extinction=0.6, quantum_yield=0.6) for i in range(n_dye)]
    lib = {d.name: d for d in dyes}
    ex = [470.0, 500.0, 530.0][:max(1, n_dye)]
    acq = AcquisitionConfig(excitations=ex, em_min=420, em_max=700, em_step=5)
    from spectraforge.scene import Scene
    scene = Scene(64, 64)
    fields = [random_field(64, 64, seed * 7 + 11 * i) for i in range(n_dye)]
    for i, d in enumerate(dyes):
        scene.paint_map(Material(d.name, {d.name: 1.0}), fields[i])
    labels = np.stack(fields).argmax(0) if n_dye > 1 else np.zeros((64, 64), int)
    spectra, gt = render(scene, lib, acq, artifacts=ArtifactConfig(photon_scale=photon, read_sigma=0.002),
                         physics=PhysicsConfig(psf_sigma_px=1.0), seed=seed)
    if perpixel:
        new = {}
        for e in spectra.excitation_wavelengths:
            exd = spectra.get_excitation(e)
            cube = exd.cube.astype(float)
            nrm = np.linalg.norm(cube, axis=2, keepdims=True) + 1e-9
            new[e] = ExcitationData(excitation_nm=e, cube=(cube / nrm).astype(np.float32),
                                    emission_wavelengths=exd.emission_wavelengths)
        spectra = SpectraData(excitations=new, sample_name="ppn")
    return spectra, labels.ravel()


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
            mse_m = np.mean((xb - rb_) ** 2); mse0 = np.mean((xb - xb.mean()) ** 2)
            r2.append(1 - mse_m / mse0 if mse0 > 1e-12 else 0.0)
    return float(np.nanmean(r2))


def run(tag, spectra, y):
    kw = dict(sample_name="d", n_bands_to_select=12, n_important_dimensions=18,
              perturbation_method="percentile", normalization_method="none", use_diversity_constraint=True,
              device="cpu", random_seed=0, output_dir=pathlib.Path(tempfile.mkdtemp()))
    kw.update(BEST)
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        a = Analyzer(Config(**kw)); a.fit(spectra)
    r2 = recon_r2(a._model, a._dataset.get_all_data())
    X, colmap = feature_matrix(spectra)
    nclass = len(set(y.tolist()))
    if nclass > 1:
        cols = cols_for_bands(colmap, [(b.excitation_nm, b.emission_nm) for b in a.get_wavelengths()])
        f1 = knn_macro_f1(X[:, cols], y, 1) if cols else float("nan")
        orc = knn_macro_f1(X[:, oracle_cols(X, y, 12)], y, 1)
    else:
        f1 = orc = float("nan")
    print(f"  {tag:<14} reconR2={r2:>+8.3f}   CAE_F1={f1 if np.isnan(f1) else round(f1,3)!s:>6}  "
          f"oracle={orc if np.isnan(orc) else round(orc,3)!s:>6}")
    return r2


def main():
    print("=" * 78)
    print("CAE DATA-COMPLEXITY LADDER (best config: no-reg, 500 epochs, k=40)")
    print("=" * 78)
    run("1dye", *build_simple(1, 1))
    run("2dye", *build_simple(1, 2))
    run("3dye", *build_simple(1, 3))
    run("3dye+perpix", *build_simple(1, 3, perpixel=True))
    print("-" * 78)
    print("If even 1dye (a single dense smooth field) has reconR2<=0, the spatial CAE architecture")
    print("cannot reconstruct this data class at all — model-config-independent.")


if __name__ == "__main__":
    main()
