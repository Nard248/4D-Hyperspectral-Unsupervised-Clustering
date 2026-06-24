"""Realistic ME-HSI benchmark: does a regime where variance != informativeness finally separate a
learned band selector from the trivial variance baseline?

Built from the photophysics in docs/spectraforge/06: classes are set by DIM, spectrally-OVERLAPPING
discriminative dyes (low-variance signal, discrimination in shape); on top sit BRIGHT nuisance
autofluorophores (high variance, class-irrelevant) and spatially-varying Rayleigh+Raman scatter
(high variance, zero info) plus signal-dependent shot noise. The highest-variance bands therefore
carry NO class information.

Reports, per method: KNN macro-F1, and how many selected bands land on the discriminative window vs
the nuisance/scatter bands. Also the key diagnostic: corr(per-band variance, per-band discriminability).

Run:  python reports/realistic_benchmark.py
"""
from __future__ import annotations

import contextlib
import os

import numpy as np
from sklearn.feature_selection import f_classif

from spectraforge import AcquisitionConfig, ArtifactConfig, Material, PhysicsConfig
from spectraforge.fluorophore import Fluorophore
from spectraforge.forward import render
from spectraforge.scenegen import make_confounded_scene

from classification_experiment import cols_for_bands, feature_matrix, knn_macro_f1
from spectral_select.architectures import CANDIDATES

BUDGET = 12
EXCITATIONS = [405.0, 470.0, 488.0, 506.0]
DISC_WINDOW = (500.0, 575.0)        # where the overlapping, excitation-differentiated dyes emit

# Discriminative dyes: overlapping emission (515/535/555) but DIFFERENT excitation peaks, so they
# are classifiable from the right bands; moderate brightness (eps*Phi ~ 0.22). Differ in shape too.
DISC = {
    "D1": Fluorophore("D1", ex_peak_nm=470, ex_fwhm_nm=35, em_peak_nm=515, em_fwhm_nm=45,
                      extinction=0.5, quantum_yield=0.45, em_skew=0.4),
    "D2": Fluorophore("D2", ex_peak_nm=488, ex_fwhm_nm=35, em_peak_nm=535, em_fwhm_nm=45,
                      extinction=0.5, quantum_yield=0.45, vibronic=0.5, em_skew=0.2),
    "D3": Fluorophore("D3", ex_peak_nm=506, ex_fwhm_nm=35, em_peak_nm=555, em_fwhm_nm=45,
                      extinction=0.5, quantum_yield=0.45, em_skew=0.6),
}
# Bright nuisance autofluorophores emitting OUTSIDE the discriminative window (eps*Phi ~ 0.8),
# class-irrelevant; their large spatial concentration variation dominates band variance.
NUIS = {
    "NADH":       Fluorophore("NADH", ex_peak_nm=405, ex_fwhm_nm=60, em_peak_nm=445, em_fwhm_nm=70,
                              extinction=1.0, quantum_yield=0.8, em_skew=0.5),
    "lipofuscin": Fluorophore("lipofuscin", ex_peak_nm=488, ex_fwhm_nm=80, em_peak_nm=665, em_fwhm_nm=85,
                              extinction=1.0, quantum_yield=0.8, em_skew=0.4),
}
LIB = {**DISC, **NUIS}


def build_dataset(seed, *, disc_amp=1.0, nuisance_amp=2.0, turbidity_amp=1.0,
                  rayleigh=0.5, raman=0.4, photon_scale=600, read_sigma=0.005, size=64,
                  reabsorption=False, reabsorption_strength=2.5):
    """Render one confounded scene. All confound strengths are overridable for sweeps; the defaults
    reproduce the headline realistic regime (doc 06). ``nuisance_amp=0, turbidity_amp=0, rayleigh=0,
    raman=0`` recovers a clean (variance≈informativeness) regime for the phase-diagram endpoints.

    ``reabsorption=True`` turns on the secondary inner-filter (self-absorption) in the renderer — a
    concentration-dependent RESHAPING of each emission band that relocates discriminative information
    into band shape/ratios (the honest nonlinear regime; see docs 15-16)."""
    acq = AcquisitionConfig(excitations=EXCITATIONS, em_min=420, em_max=700, em_step=5)
    disc_mats = [Material(n, {n: 1.0}) for n in DISC]
    nuis_mats = [Material(n, {n: 1.0}) for n in NUIS]
    scene, labels, scatter = make_confounded_scene(
        disc_mats, nuis_mats, size, size, seed,
        disc_amp=disc_amp, nuisance_amp=nuisance_amp, turbidity_amp=turbidity_amp)
    artifacts = ArtifactConfig(rayleigh_strength=rayleigh, raman_strength=raman, second_order=True,
                               photon_scale=photon_scale, read_sigma=read_sigma)
    physics = PhysicsConfig(psf_sigma_px=1.0, reabsorption=reabsorption,
                            reabsorption_strength=reabsorption_strength)
    spectra, gt = render(scene, LIB, acq, artifacts=artifacts, physics=physics,
                         seed=seed, scatter_field=scatter)
    return spectra, gt, labels.ravel(), acq


def band_is_discriminative(colmap):
    """True for feature columns whose emission falls in the discriminative window."""
    return np.array([DISC_WINDOW[0] <= em <= DISC_WINDOW[1] for _, em in colmap])


def _candidate_bands(cid, spectra, seed):
    Cls = CANDIDATES[cid]
    kw = dict(training_epochs=12) if cid in ("C0", "C1") else dict(epochs=300)
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        model = Cls(seed=seed, **(dict(n_bands=BUDGET, **kw) if cid in ("C0", "C1") else kw)).fit(spectra)
    return model.select(BUDGET)


def main():
    seeds = [1, 2, 3]
    methods = ["all bands", "variance-ranking", "discriminability-oracle", "random",
               "C0 standard-CAE", "C2 spectral-AE", "C3 masked-spectral-AE", "C4 variational"]
    agg = {m: {"f1": [], "disc_frac": []} for m in methods}
    var_info_corr = []

    print("=" * 92)
    print("REALISTIC confounded ME-HSI: variance != informativeness  (KNN macro-F1, 12-band budget)")
    print("Dim overlapping discriminative dyes + bright nuisance autofluorescence + Rayleigh/Raman scatter")
    print("=" * 92)

    for seed in seeds:
        spectra, gt, y, acq = build_dataset(seed)
        X, colmap = feature_matrix(spectra)
        disc_mask = band_is_discriminative(colmap)
        var = X.var(0)
        F, _ = f_classif(X, y)
        F = np.nan_to_num(F)
        var_info_corr.append(np.corrcoef(var, F)[0, 1])

        def record(name, cols):
            agg[name]["f1"].append(knn_macro_f1(X[:, cols], y, seed))
            agg[name]["disc_frac"].append(float(disc_mask[cols].mean()) if len(cols) else 0.0)

        rng = np.random.default_rng(seed)
        record("all bands", list(range(X.shape[1])))
        record("variance-ranking", list(np.argsort(var)[::-1][:BUDGET]))
        record("discriminability-oracle", list(np.argsort(F)[::-1][:BUDGET]))
        record("random", list(rng.choice(X.shape[1], BUDGET, replace=False)))
        for cid, name in [("C0", "C0 standard-CAE"), ("C2", "C2 spectral-AE"),
                          ("C3", "C3 masked-spectral-AE"), ("C4", "C4 variational")]:
            bands = _candidate_bands(cid, spectra, seed)
            record(name, cols_for_bands(colmap, bands))

    print(f"\ncorr(per-band variance, per-band discriminability F) = {np.mean(var_info_corr):+.3f}")
    print("(near 0 / negative == the high-variance bands are NOT the informative ones)\n")
    print(f"{'method':<26}{'KNN macro-F1':>14}{'% bands on discriminative window':>34}")
    for m in methods:
        f1 = np.mean(agg[m]["f1"])
        frac = 100 * np.mean(agg[m]["disc_frac"])
        print(f"{m:<26}{f1:>14.3f}{frac:>32.0f}%")
    print("-" * 92)
    print("Expectation: variance-ranking selects nuisance/scatter bands (low discriminative %) and its")
    print("F1 drops toward random; the learned spectral AEs (C2/C3/C4) should land on the discriminative")
    print("window and beat variance-ranking — the advantage the clean benchmark could not show.")


if __name__ == "__main__":
    main()
