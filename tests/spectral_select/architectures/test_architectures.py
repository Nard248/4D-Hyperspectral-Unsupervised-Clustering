"""Unit + smoke tests for the C0..C4 anti-cheat autoencoder ladder.

Per ``docs/spectraforge/04-training-runbook.md`` §2: shapes round-trip (encode->decode), ``select``
returns ``n_bands`` valid ``(ex, em)`` pairs, and a smoke ``fit`` at 2-3 epochs runs without error.
The per-pixel candidates (C2/C3/C4) are fast; the spatial ones (C0/C1) run through the real Analyzer
and are marked ``slow``.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from spectral_select.architectures import (
    CANDIDATES, DeepSpectralCAE, MaskedSpectralAE, SpectralAE, VariationalSpectralAE, diverse_topk,
)
from spectral_select.architectures.spectral_ae import _SpectralMLP
from spectral_select.architectures.variational_spectral_ae import _SpectralVAE
from spectral_select.types import ExcitationData, SpectraData


# --------------------------------------------------------------------------------------------------
# fixtures: small structured ME-HSI cubes (real structure so reconstruction is learnable)
# --------------------------------------------------------------------------------------------------
def _make_spectra(h, w, excitations, n_bands, seed=0):
    rng = np.random.default_rng(seed)
    grid = np.linspace(420.0, 660.0, n_bands)
    peaks = [470.0, 600.0]                                   # two "materials"
    fields = [rng.random((h, w)) for _ in peaks]            # smooth-ish concentration fields
    excit = {ex: [0.9, 0.4] if i == 0 else [0.4, 0.9] for i, ex in enumerate(excitations)}
    ex_data = {}
    for ex in excitations:
        cube = np.zeros((h, w, n_bands), dtype=np.float32)
        for m, pk in enumerate(peaks):
            shape = np.exp(-0.5 * ((grid - pk) / 18.0) ** 2)
            cube += (fields[m][:, :, None] * excit[ex][m]) * shape[None, None, :]
        cube += 0.01 * rng.standard_normal(cube.shape).astype(np.float32)
        cube = np.clip(cube / cube.max(), 0.0, 1.0)
        ex_data[ex] = ExcitationData(excitation_nm=ex, cube=cube,
                                     emission_wavelengths=[float(g) for g in grid])
    return SpectraData(excitations=ex_data, sample_name="tiny")


@pytest.fixture(scope="module")
def tiny_spectra():
    return _make_spectra(16, 16, [488.0, 560.0], 10, seed=1)


@pytest.fixture(scope="module")
def spatial_spectra():
    # >= default patch_size (32) so the Analyzer's patch baseline has valid patches
    return _make_spectra(40, 40, [488.0, 560.0], 8, seed=2)


def _valid_bands(spectra, bands, n):
    assert len(bands) == n
    grid = list(spectra.get_excitation(spectra.excitation_wavelengths[0]).emission_wavelengths)
    for ex, em in bands:
        assert ex in spectra.excitation_wavelengths
        assert min(grid) - 1 <= em <= max(grid) + 1


# --------------------------------------------------------------------------------------------------
# module-level shape round-trips
# --------------------------------------------------------------------------------------------------
def test_spectral_mlp_roundtrip_shapes():
    m = _SpectralMLP(d=12, latent=4)
    x = torch.randn(5, 12)
    z = m.encode(x)
    assert z.shape == (5, 4)
    assert m.decode(z).shape == (5, 12)


def test_spectral_vae_roundtrip_shapes():
    m = _SpectralVAE(d=12, latent=4)
    x = torch.randn(5, 12)
    mu, logvar = m.encode_dist(x)
    assert mu.shape == logvar.shape == (5, 4)
    assert m.encode(x).shape == (5, 4)            # baseline latent == posterior mean
    assert m.decode(mu).shape == (5, 12)


def test_deep_cae_preserves_band_axis_and_roundtrips():
    h, w, nb = 8, 8, 6
    exdata = {488.0: np.zeros((h, w, nb), np.float32), 560.0: np.zeros((h, w, nb), np.float32)}
    model = DeepSpectralCAE(exdata, k1=6, k3=6, filter_size=3)
    batch = {ex: torch.randn(2, h, w, nb) for ex in exdata}
    latent = model.encode(batch)
    assert latent.shape[0] == 2 and latent.shape[2] == nb     # emission-band axis NOT collapsed
    out = model.decode(latent)
    for ex in exdata:
        assert out[ex].shape == (2, h, w, nb)
    fwd = model.forward(batch)
    assert set(fwd) == set(exdata)


def test_deep_cae_rejects_ragged_band_counts():
    exdata = {488.0: np.zeros((8, 8, 6), np.float32), 560.0: np.zeros((8, 8, 5), np.float32)}
    with pytest.raises(ValueError):
        DeepSpectralCAE(exdata)


# --------------------------------------------------------------------------------------------------
# diverse_topk diversity rule
# --------------------------------------------------------------------------------------------------
def test_diverse_topk_dedups_within_excitation():
    colmap = [(488.0, 500.0), (488.0, 505.0), (488.0, 560.0), (560.0, 500.0)]
    infl = np.array([1.0, 0.9, 0.8, 0.7])         # cols 0 and 1 are 5 nm apart -> 1 is skipped
    picked = diverse_topk(infl, colmap, 3, dedup_nm=10.0)
    assert (488.0, 505.0) not in picked
    assert (488.0, 500.0) in picked and (488.0, 560.0) in picked and (560.0, 500.0) in picked


# --------------------------------------------------------------------------------------------------
# per-pixel candidates (C2/C3/C4): fast fit + select smoke
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("Cls", [SpectralAE, MaskedSpectralAE, VariationalSpectralAE])
def test_spectral_candidate_fit_select(tiny_spectra, Cls):
    model = Cls(epochs=3, latent_dim=4, seed=0).fit(tiny_spectra)
    bands = model.select(4)
    _valid_bands(tiny_spectra, bands, 4)
    assert np.isfinite(model.reconstruction_r())
    assert np.isfinite(model.influence_signal_corr())


def test_masked_ae_select_count(tiny_spectra):
    model = MaskedSpectralAE(epochs=3, latent_dim=4, mask_ratio=0.6, seed=0).fit(tiny_spectra)
    assert len(model.select(5)) == 5


def test_variational_active_units_is_int(tiny_spectra):
    model = VariationalSpectralAE(epochs=3, latent_dim=4, seed=0).fit(tiny_spectra)
    au = model.active_units()
    assert isinstance(au, int) and 0 <= au <= 4


def test_candidate_registry_has_ladder():
    assert list(CANDIDATES) == ["C0", "C1", "C2", "C3", "C4"]


# --------------------------------------------------------------------------------------------------
# spatial candidates (C0/C1): smoke through the real Analyzer (slow)
# --------------------------------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("cid", ["C0", "C1"])
def test_spatial_candidate_fit_select(spatial_spectra, cid):
    model = CANDIDATES[cid](n_bands=4, training_epochs=2, n_important=6, seed=0).fit(spatial_spectra)
    bands = model.select(4)
    _valid_bands(spatial_spectra, bands, 4)
    assert np.isfinite(model.reconstruction_r())
    assert np.isfinite(model.influence_signal_corr())
