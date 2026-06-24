"""The forward model: scene + fluorophores + acquisition -> SpectraData + GroundTruth."""
from __future__ import annotations

import numpy as np

from spectral_select.types import ExcitationData, SpectraData

from spectraforge.groundtruth import GroundTruth


def render(scene, library, acquisition, artifacts=None, physics=None, seed=None, sample_name="synthetic",
           scatter_field=None, fret_pairs=None):
    """Render a synthetic ME-HSI dataset.

    Returns ``(SpectraData, GroundTruth)``. With ``artifacts=None`` and ``physics`` off the result
    is the clean, exactly-linear forward model: ``render(A+B) == render(A)+render(B)``. Pass a
    ``PhysicsConfig`` to add optical PSF blur, Beer-Lambert inner-filter (nonlinear), or
    autofluorescence — see :mod:`spectraforge.physics`.

    ``scatter_field`` is an optional (H, W) turbidity/reflectance map; Rayleigh/Raman scatter scales
    with it per pixel (spatially-varying, high-variance, no chemical information).
    """
    conc = scene.resolve()                       # {fname: (H, W)}
    h, w = scene.height, scene.width
    em = acquisition.emission_grid()
    rng = np.random.default_rng(seed)

    # Per-pixel, per-emission-wavelength absorbance Σ_k ε_k c_k · absorption_k(λ_em), used for
    # reabsorption (secondary inner-filter). Absorption cross-section ≈ the excitation profile.
    em_absorbance = None
    if physics is not None and getattr(physics, "reabsorption", False):
        em_absorbance = np.zeros((h, w, len(em)), dtype=float)
        for fname, cmap in conc.items():
            f = library[fname]
            em_absorbance += f.extinction * cmap[:, :, None] * f.excitation(em)[None, None, :]

    excitations = {}
    clean_cubes = {}
    per_fluorophore = {}                          # fname -> {ex -> (n_em,) per-pixel-max spectrum}
    for ex in acquisition.excitations:
        scale = (
            acquisition.lamp_for(ex)
            * acquisition.exposure_for(ex)
            * acquisition.power_for(ex)
        )
        absorbance = np.zeros((h, w), dtype=float)        # excitation absorbance (inner-filter)
        contribs = {}                                     # fname -> (H, W, n_em) per-fluorophore signal
        absorbed = {}                                     # fname -> (H, W) absorbed excitation energy
        for fname, cmap in conc.items():
            f = library[fname]
            exc = float(f.excitation(ex))
            amp = f.extinction * f.quantum_yield * exc                      # scalar
            em_profile = f.emission(em)                                     # (n_em,)
            contribs[fname] = (cmap * amp)[:, :, None] * em_profile[None, None, :]
            absorbed[fname] = f.extinction * exc * cmap
            absorbance += absorbed[fname]
        if fret_pairs:
            from spectraforge.physics import apply_fret

            apply_fret(contribs, absorbed, conc, library, em, fret_pairs)
        cube = np.zeros((h, w, len(em)), dtype=float)
        for fname, contrib in contribs.items():
            cube += contrib
            band_max = contrib.reshape(-1, len(em)).max(axis=0) * scale     # (n_em,) post-FRET
            per_fluorophore.setdefault(fname, {})[float(ex)] = band_max
        cube *= scale
        if physics is not None:
            from spectraforge.physics import apply_physics

            cube = apply_physics(cube, physics, em, scale, absorbance, em_absorbance=em_absorbance)
        clean_cubes[float(ex)] = cube.copy()
        if artifacts is not None:
            from spectraforge.artifacts import add_noise, add_scatter_lines

            add_scatter_lines(cube, ex, em, artifacts, scale, reflectance=scatter_field)
            cube = add_noise(cube, artifacts, rng)
        excitations[float(ex)] = ExcitationData(
            cube=cube,
            excitation_nm=float(ex),
            emission_wavelengths=[float(x) for x in em],
            exposure_time=acquisition.exposure_for(ex),
            laser_power=acquisition.power_for(ex),
        )

    spectra = SpectraData(excitations=excitations, sample_name=sample_name)
    gt = GroundTruth(
        concentration_maps=conc,
        clean_cubes=clean_cubes,
        emission_grid=em,
        excitations=[float(e) for e in acquisition.excitations],
        per_fluorophore_spectra=per_fluorophore,
        seed=seed,
    )
    return spectra, gt
