"""Tests for the realistic-simulation additions (doc 06): water-Raman scatter, spatially-varying
scatter, vibronic/asymmetric emission, and the confounded scene generator."""
from __future__ import annotations

import numpy as np
import pytest

from spectraforge import AcquisitionConfig, ArtifactConfig, Material, PhysicsConfig
from spectraforge.artifacts import add_scatter_lines, raman_center_nm
from spectraforge.fluorophore import Fluorophore
from spectraforge.forward import render
from spectraforge.scenegen import make_confounded_scene


def test_raman_center_is_fixed_wavenumber_offset():
    # water Raman ~3400 cm^-1: 350 nm -> ~397 nm, 488 nm -> ~585 nm
    assert raman_center_nm(350.0, 3400.0) == pytest.approx(397.2, abs=1.5)
    assert raman_center_nm(488.0, 3400.0) == pytest.approx(585.0, abs=2.0)
    # the offset is constant in wavenumber, so it grows in nm with excitation wavelength
    assert raman_center_nm(488.0) - 488.0 > raman_center_nm(350.0) - 350.0


def test_add_scatter_lines_adds_raman_band():
    em = np.arange(420.0, 700.0 + 1e-9, 5.0)
    cube = np.zeros((4, 4, len(em)))
    cfg = ArtifactConfig(rayleigh_strength=0.0, raman_strength=0.5)
    add_scatter_lines(cube, 488.0, em, cfg, scale=1.0)
    peak_band = int(np.argmax(cube[0, 0]))
    assert em[peak_band] == pytest.approx(raman_center_nm(488.0), abs=5.0)


def test_scatter_field_scales_scatter_spatially():
    em = np.arange(420.0, 700.0 + 1e-9, 5.0)
    cube = np.zeros((2, 2, len(em)))
    refl = np.array([[0.0, 1.0], [2.0, 0.5]])
    cfg = ArtifactConfig(rayleigh_strength=1.0, second_order=False, raman_strength=0.0)
    add_scatter_lines(cube, 500.0, em, cfg, scale=1.0, reflectance=refl)
    line = cube.max(axis=2)                          # peak scatter per pixel
    assert line[0, 0] == 0.0                          # zero turbidity -> no scatter
    assert line[1, 0] > line[0, 1] > line[1, 1] > 0   # scales with reflectance (2.0 > 1.0 > 0.5)


def test_vibronic_emission_is_red_tailed_and_normalized():
    grid = np.arange(400.0, 750.0, 2.0)
    sym = Fluorophore("s", 480, 40, 540, 50)
    asy = Fluorophore("a", 480, 40, 540, 50, em_skew=0.6, vibronic=0.5)
    es, ea = sym.emission(grid), asy.emission(grid)
    assert es.sum() == pytest.approx(1.0, abs=1e-6)
    assert ea.sum() == pytest.approx(1.0, abs=1e-6)         # still area-normalized
    peak = 540.0
    red = grid > peak + 20
    blue = grid < peak - 20
    assert ea[red].sum() > ea[blue].sum()                  # heavier red tail than the symmetric one
    assert ea[red].sum() > es[red].sum()


def test_make_confounded_scene_labels_from_discriminative_only():
    disc = [Material("D1", {"D1": 1.0}), Material("D2", {"D2": 1.0})]
    nuis = [Material("N1", {"N1": 1.0})]
    scene, labels, scatter = make_confounded_scene(disc, nuis, 32, 32, seed=1, nuisance_amp=3.0)
    assert labels.shape == (32, 32)
    assert set(np.unique(labels)) <= {0, 1}
    assert scatter.shape == (32, 32) and scatter.min() >= 0.0
    maps = scene.resolve()
    assert {"D1", "D2", "N1"} <= set(maps)                 # all painted
    # both classes present (i.i.d. fields -> roughly balanced, definitely non-degenerate)
    assert 0.2 < labels.mean() < 0.8


def test_confounded_render_runs_and_is_finite():
    disc = [Material("D1", {"D1": 1.0}), Material("D2", {"D2": 1.0})]
    nuis = [Material("N1", {"N1": 1.0})]
    lib = {
        "D1": Fluorophore("D1", 470, 35, 515, 45, extinction=0.5, quantum_yield=0.45),
        "D2": Fluorophore("D2", 506, 35, 555, 45, extinction=0.5, quantum_yield=0.45),
        "N1": Fluorophore("N1", 405, 60, 445, 70, extinction=1.0, quantum_yield=0.8),
    }
    acq = AcquisitionConfig(excitations=[405.0, 488.0], em_min=420, em_max=700, em_step=10)
    scene, labels, scatter = make_confounded_scene(disc, nuis, 24, 24, seed=2)
    spectra, gt = render(scene, lib, acq,
                         artifacts=ArtifactConfig(rayleigh_strength=0.5, raman_strength=0.4, photon_scale=400),
                         physics=PhysicsConfig(psf_sigma_px=1.0), seed=2, scatter_field=scatter)
    for ex in spectra.excitation_wavelengths:
        assert np.all(np.isfinite(spectra.get_excitation(ex).cube))
