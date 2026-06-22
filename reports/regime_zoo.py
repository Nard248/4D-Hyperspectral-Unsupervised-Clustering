"""Parameterized synthetic-regime generator for GENERALIZATION testing.

The fixed realistic benchmark is one point in a large space of generation assumptions; a trustworthy
blind selector should win across *many* plausible regimes, not just one. This builds diverse
confounded ME-HSI datasets varying: number of discriminative dyes (classes), their spectral overlap,
the discriminative/nuisance brightness ratio, the number/breadth of nuisances, and noise. Returns the
sweep_common-compatible (X, colmap, y, spectra).
"""
from __future__ import annotations

import numpy as np

from classification_experiment import feature_matrix
from spectraforge import AcquisitionConfig, ArtifactConfig, Material, PhysicsConfig
from spectraforge.fluorophore import Fluorophore
from spectraforge.forward import render
from spectraforge.scenegen import make_confounded_scene


def build_variant(seed, *, n_disc=3, overlap="med", disc_bright=0.22, nuis_bright=0.8, n_nuis=2,
                  nuisance_amp=2.0, turbidity=1.0, raman=0.4, photon_scale=600, size=64,
                  em=(420, 700, 5), excit=(405, 470, 488, 506, 540)):
    """A confounded labelled scene with configurable structure. ``overlap`` in {low,med,high} sets how
    close the discriminative emission peaks sit; brightness ratio = disc_bright/nuis_bright."""
    spread = {"low": 60, "med": 40, "high": 22}[overlap]            # nm between adjacent disc peaks
    fwhm = {"low": 40, "med": 50, "high": 60}[overlap]
    em_lo = 505
    excit = list(excit)[: max(3, n_disc)]
    disc = {}
    for i in range(n_disc):
        empk = em_lo + i * spread
        expk = excit[i % len(excit)]
        disc[f"D{i}"] = Fluorophore(f"D{i}", ex_peak_nm=expk, ex_fwhm_nm=35, em_peak_nm=empk,
                                    em_fwhm_nm=fwhm, extinction=0.5,
                                    quantum_yield=disc_bright / 0.5, em_skew=0.3 + 0.1 * i,
                                    vibronic=0.4 if i % 2 else 0.0)
    nuis = {}
    nuis_specs = [(445, 70, 405), (665, 85, 488), (590, 140, 540), (700, 120, 470)][:n_nuis]
    for j, (empk, fw, expk) in enumerate(nuis_specs):
        nuis[f"N{j}"] = Fluorophore(f"N{j}", ex_peak_nm=expk, ex_fwhm_nm=80, em_peak_nm=empk,
                                    em_fwhm_nm=fw, extinction=1.0, quantum_yield=nuis_bright,
                                    em_skew=0.4)
    lib = {**disc, **nuis}
    acq = AcquisitionConfig(excitations=[float(e) for e in excit], em_min=em[0], em_max=em[1], em_step=em[2])
    scene, labels, scatter = make_confounded_scene(
        [Material(k, {k: 1.0}) for k in disc], [Material(k, {k: 1.0}) for k in nuis],
        size, size, seed, disc_amp=1.0, nuisance_amp=nuisance_amp, turbidity_amp=turbidity)
    spectra, gt = render(scene, lib, acq,
                         artifacts=ArtifactConfig(rayleigh_strength=0.5, raman_strength=raman,
                                                  photon_scale=photon_scale, read_sigma=0.005),
                         physics=PhysicsConfig(psf_sigma_px=1.0), seed=seed, scatter_field=scatter)
    X, colmap = feature_matrix(spectra)
    return X, colmap, np.asarray(labels).ravel(), spectra


# named generalization regimes (held-out from the round-1/2 tuning regimes)
VARIANTS = {
    "g_2class": dict(n_disc=2, overlap="med"),
    "g_5class": dict(n_disc=5, overlap="med"),
    "g_highoverlap": dict(n_disc=3, overlap="high"),
    "g_lowoverlap": dict(n_disc=3, overlap="low"),
    "g_faint": dict(n_disc=3, disc_bright=0.10, nuisance_amp=3.0),     # very dim signal vs bright nuisance
    "g_heavynuis": dict(n_disc=3, n_nuis=4, nuisance_amp=3.0, turbidity=1.5),
    "g_lownoise": dict(n_disc=3, photon_scale=3000, raman=0.2, nuisance_amp=1.0),
}
