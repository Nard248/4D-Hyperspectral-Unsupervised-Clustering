"""Strongly-nonlinear regime via FRET: the class lives in dye CO-LOCALIZATION (XOR), and Förster
transfer relocates it into a low-variance donor/acceptor band ratio that a linear selector (PCA)
fundamentally cannot represent. The honest big test of "the AE beats PCA by far more on nonlinear data."

Donor D1 (ex 470 / em 515) -> Acceptor D2 (ex 510 / em 580). Excite at 470 (donor band): the acceptor's
580 nm emission then appears ONLY by FRET, so its intensity ~ co-localization c_D1·E(c_D2) — the XOR
signal. Direct acceptor excitation (ex 506) gives a class-INDEPENDENT acceptor amplitude (a foil).

Evaluation = the fair panel (reabsorption_eval.panel_scores): linear, best-NL, nonlinear-only gap.
Fairness precondition: all-bands gap must be LARGE (the regime is genuinely, strongly nonlinear).

Run:  python reports/fret_regime.py
"""
from __future__ import annotations

import sys

import numpy as np
from sklearn.feature_selection import mutual_info_classif

from spectraforge import AcquisitionConfig, ArtifactConfig, Material, PhysicsConfig
from spectraforge.fluorophore import Fluorophore
from spectraforge.forward import render
from spectraforge.scenegen import make_interaction_scene

import method_zoo as mz
from classification_experiment import feature_matrix
from reabsorption_eval import _ae_select, panel_scores, summarize
from sweep_common import topn_diverse

BUDGET = 12
EXC = [405.0, 470.0, 488.0, 506.0]
D1 = Fluorophore("D1", ex_peak_nm=470, ex_fwhm_nm=35, em_peak_nm=515, em_fwhm_nm=40, extinction=0.6, quantum_yield=0.5)
D2 = Fluorophore("D2", ex_peak_nm=510, ex_fwhm_nm=35, em_peak_nm=580, em_fwhm_nm=45, extinction=0.6, quantum_yield=0.5)
NADH = Fluorophore("NADH", ex_peak_nm=405, ex_fwhm_nm=60, em_peak_nm=445, em_fwhm_nm=70, extinction=1.0, quantum_yield=0.8, em_skew=0.5)
LIPO = Fluorophore("lipofuscin", ex_peak_nm=488, ex_fwhm_nm=80, em_peak_nm=665, em_fwhm_nm=85, extinction=1.0, quantum_yield=0.8, em_skew=0.4)
LIB = {"D1": D1, "D2": D2, "NADH": NADH, "lipofuscin": LIPO}


def build_fret_dataset(seed, *, k=4.0, nuisance_amp=2.0, size=48):
    acq = AcquisitionConfig(excitations=EXC, em_min=420, em_max=700, em_step=5)
    donor, acceptor = Material("D1", {"D1": 1.0}), Material("D2", {"D2": 1.0})
    nuis = [Material("NADH", {"NADH": 1.0}), Material("lipofuscin", {"lipofuscin": 1.0})]
    scene, labels, scatter = make_interaction_scene(donor, acceptor, nuis, size, size, seed,
                                                    disc_amp=1.0, nuisance_amp=nuisance_amp, turbidity_amp=1.0)
    artifacts = ArtifactConfig(rayleigh_strength=0.5, raman_strength=0.4, second_order=True,
                               photon_scale=600, read_sigma=0.005)
    spectra, gt = render(scene, LIB, acq, artifacts=artifacts, physics=PhysicsConfig(psf_sigma_px=1.0),
                         seed=seed, scatter_field=scatter, fret_pairs=[("D1", "D2", k)])
    return spectra, labels.ravel()


def main():
    smoke = "--smoke" in sys.argv
    seeds = [1] if smoke else [1, 2, 3, 4]
    size = 32 if smoke else 48
    print("=" * 86)
    print(f"FRET regime (XOR co-localization, k=4) — fair panel CV macro-F1, {len(seeds)} seed(s)")
    print("fairness precondition: all-bands gap must be LARGE (strongly nonlinear)")
    print("=" * 86)
    rec = {m: [] for m in ["all-bands", "variance", "pca_load", "AE", "oracle_MI"]}
    for seed in seeds:
        spectra, y = build_fret_dataset(seed, size=size)
        X, colmap = feature_matrix(spectra)
        rng = np.random.default_rng(seed)
        MI = mutual_info_classif(X, y, random_state=seed)
        sels = {
            "all-bands": list(range(X.shape[1])),
            "variance": topn_diverse(X.var(0), colmap, BUDGET),
            "pca_load": mz.pca_load(X, colmap, BUDGET, seed, rng, spectra, k=6),
            "AE": _ae_select(spectra, colmap, seed),
            "oracle_MI": topn_diverse(MI, colmap, BUDGET),
        }
        for name, cols in sels.items():
            rec[name].append(summarize(panel_scores(X, y, cols, seed)))

    print(f"{'method':<12}{'linear':>9}{'best-NL':>9}{'gap':>8}{'(std NL)':>10}")
    pca_nl = np.mean([b for _, b, _ in rec['pca_load']])
    for name, vals in rec.items():
        lin = np.mean([a for a, _, _ in vals]); nl = np.mean([b for _, b, _ in vals])
        gap = np.mean([g for _, _, g in vals]); sd = np.std([b for _, b, _ in vals])
        mark = "  <-- AE beats pca" if name == "AE" and nl > pca_nl else ""
        print(f"{name:<12}{lin:>9.3f}{nl:>9.3f}{gap:>+8.3f}{sd:>10.3f}{mark}")
    ae_nl = np.mean([b for _, b, _ in rec['AE']])
    ab_gap = np.mean([g for _, _, g in rec['all-bands']])
    print("-" * 86)
    print(f"all-bands nonlinear gap = {ab_gap:+.3f} (precondition: large => fair nonlinear test)")
    print(f"AE - pca margin (best-NL) = {ae_nl - pca_nl:+.3f}")


if __name__ == "__main__":
    main()
