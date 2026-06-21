"""Does DENSE data rescue the published CAE? — the experiment that decides whether the CAE's
synthetic failure is a sparsity artifact (doc 02's hypothesis: "real data is dense, so the CAE works
there") or a genuine method flaw.

Setup: 3 BRIGHT, WELL-SEPARATED discriminative dyes (the easy, clean-informative case the CAE should
handle) + one controllable BROADBAND autofluorescence background painted with a spatially-varying
field. Sweeping the background amplitude takes the cube from SPARSE (only the 3 dye peaks carry
signal -> mostly zero after normalization) to DENSE (every band carries varying signal). At each
density we train the real CAE (C0) and measure reconstruction R, influence-signal corr, KNN-F1, and
the fraction of bands it puts on the informative dye peaks; a per-pixel spectral AE (C2) and the
variance/oracle baselines are the references.

Question 1 (density -> R): does the CAE stop predicting-the-mean (R rises) once the data is dense?
Question 2 (R -> selection): even if R rises, does the CAE then SELECT the informative dye bands?

Run:  python reports/cae_density_study.py
"""
from __future__ import annotations

import contextlib
import os

import numpy as np

from classification_experiment import cols_for_bands, feature_matrix, knn_macro_f1
from spectraforge import AcquisitionConfig, ArtifactConfig, Material, PhysicsConfig
from spectraforge.fluorophore import Fluorophore
from spectraforge.forward import render
from spectraforge.scenegen import make_confounded_scene
from spectral_select.architectures import SpectralAE, StandardCAE

BUDGET = 12
EXCITATIONS = [400.0, 488.0, 560.0]
# bright, well-separated dyes -> the informative bands are their 3 emission peaks (the easy case)
PEAKS = [(400.0, 460.0), (488.0, 540.0), (560.0, 620.0)]
DISC = {
    "B1": Fluorophore("B1", ex_peak_nm=400, ex_fwhm_nm=45, em_peak_nm=460, em_fwhm_nm=40,
                      extinction=0.6, quantum_yield=0.6),
    "G1": Fluorophore("G1", ex_peak_nm=488, ex_fwhm_nm=40, em_peak_nm=540, em_fwhm_nm=40,
                      extinction=0.6, quantum_yield=0.6),
    "R1": Fluorophore("R1", ex_peak_nm=560, ex_fwhm_nm=40, em_peak_nm=620, em_fwhm_nm=40,
                      extinction=0.6, quantum_yield=0.6),
}
# a broad autofluorescence background spanning the whole emission grid (the density knob)
BG = {"autofl": Fluorophore("autofl", ex_peak_nm=440, ex_fwhm_nm=220, em_peak_nm=560, em_fwhm_nm=260,
                            extinction=1.0, quantum_yield=0.8)}
LIB = {**DISC, **BG}


def build(seed, bg_amp):
    acq = AcquisitionConfig(excitations=EXCITATIONS, em_min=420, em_max=700, em_step=5)
    disc_mats = [Material(n, {n: 1.0}) for n in DISC]
    bg_mats = [Material(n, {n: 1.0}) for n in BG]
    scene, labels, _ = make_confounded_scene(disc_mats, bg_mats, 64, 64, seed,
                                             disc_amp=1.0, nuisance_amp=bg_amp, turbidity_amp=0.0)
    artifacts = ArtifactConfig(photon_scale=600, read_sigma=0.005)        # shot noise, no scatter
    spectra, gt = render(scene, LIB, acq, artifacts=artifacts, physics=PhysicsConfig(psf_sigma_px=1.0),
                         seed=seed)
    return spectra, labels.ravel()


def disc_mask(colmap):
    return np.array([any(abs(ex - px) <= 1 and abs(em - pe) <= 25 for px, pe in PEAKS)
                     for ex, em in colmap])


def _fit(Cls, spectra, seed, **kw):
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        return Cls(seed=seed, **kw).fit(spectra)


def main():
    seeds = [1, 2, 3]
    levels = [0.0, 0.5, 1.0, 2.0, 4.0]
    print("=" * 100)
    print("Does DENSE data rescue the CAE?  (bright separated dyes + broadband autofluorescence; "
          "12-band budget)")
    print("=" * 100)
    print(f"{'bg_amp':>7}{'density':>9}{'corr(var,?)':>0}"
          f"{'| CAE R':>10}{'CAE corr':>10}{'CAE F1':>9}{'CAE disc%':>10}"
          f"{'| C2 R':>9}{'C2 F1':>8}{'| var F1':>9}{'oracle':>8}{'all':>7}")
    for L in levels:
        d, cR, cC, cF, cD, sR, sF, vF, oF, aF = ([] for _ in range(10))
        for seed in seeds:
            spectra, y = build(seed, L)
            X, colmap = feature_matrix(spectra)
            dm = disc_mask(colmap)
            var = X.var(0)
            from sklearn.feature_selection import f_classif
            F, _ = f_classif(X, y); F = np.nan_to_num(F)
            d.append(float((X > 0.05 * X.max()).mean()))             # density = fraction of "lit" entries
            cae = _fit(StandardCAE, spectra, seed, n_bands=BUDGET, training_epochs=30)
            cae_cols = cols_for_bands(colmap, cae.select(BUDGET))
            cR.append(cae.reconstruction_r()); cC.append(cae.influence_signal_corr())
            cF.append(knn_macro_f1(X[:, cae_cols], y, seed)); cD.append(float(dm[cae_cols].mean()))
            sae = _fit(SpectralAE, spectra, seed)
            sR.append(sae.reconstruction_r())
            sF.append(knn_macro_f1(X[:, cols_for_bands(colmap, sae.select(BUDGET))], y, seed))
            vF.append(knn_macro_f1(X[:, np.argsort(var)[::-1][:BUDGET]], y, seed))
            oF.append(knn_macro_f1(X[:, np.argsort(F)[::-1][:BUDGET]], y, seed))
            aF.append(knn_macro_f1(X, y, seed))
        print(f"{L:>7.1f}{np.mean(d):>9.2f}{'':>0}"
              f"{np.mean(cR):>+10.3f}{np.mean(cC):>+10.3f}{np.mean(cF):>9.3f}{100*np.mean(cD):>9.0f}%"
              f"{np.mean(sR):>+9.3f}{np.mean(sF):>8.3f}{np.mean(vF):>9.3f}{np.mean(oF):>8.3f}{np.mean(aF):>7.3f}")
    print("-" * 100)
    print("Q1 density->R: does CAE R climb above ~0 as the data gets dense?")
    print("Q2 R->selection: does CAE F1 / disc%% recover, or does it reconstruct well yet still miss the dyes?")


if __name__ == "__main__":
    main()
