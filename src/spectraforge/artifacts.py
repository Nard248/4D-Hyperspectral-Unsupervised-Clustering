"""Instrument artifacts: Rayleigh / 2nd-order scatter lines and detector noise."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_FWHM_TO_SIGMA = 1.0 / 2.3548200450309493


@dataclass
class ArtifactConfig:
    rayleigh_strength: float = 0.0   # 0 disables scatter
    rayleigh_fwhm: float = 10.0
    second_order: bool = True
    raman_strength: float = 0.0      # 0 disables the water-Raman line
    raman_shift_cm: float = 3400.0   # O-H stretch wavenumber offset (water), ~3300-3600 cm^-1
    raman_fwhm: float = 15.0
    photon_scale: float = 0.0        # 0 disables shot noise; else Poisson(signal*scale)/scale
    read_sigma: float = 0.0          # additive Gaussian read noise


def _line(em_grid, center, fwhm):
    sigma = fwhm * _FWHM_TO_SIGMA
    return np.exp(-0.5 * ((em_grid - center) / sigma) ** 2)


def raman_center_nm(ex: float, shift_cm: float = 3400.0) -> float:
    """Emission wavelength (nm) of the inelastic water-Raman line for excitation ``ex`` (nm):
    a *fixed wavenumber* offset, 1/lambda_R = 1/ex - shift_cm * 1e-7."""
    return 1.0 / (1.0 / ex - shift_cm * 1e-7)


def add_scatter_lines(cube, ex, em_grid, cfg, scale, reflectance=None) -> None:
    """Add Rayleigh (em=ex), optional 2nd-order (em=2*ex), and water-Raman lines, in place.

    ``reflectance`` is an optional (H, W) turbidity/reflectance field: scatter intensity scales with
    it per pixel, so a spatially-varying medium gives the scatter bands large spatial variance while
    carrying no fluorophore-discrimination information (the canonical variance != informativeness trap)."""
    if cfg.rayleigh_strength <= 0 and cfg.raman_strength <= 0:
        return
    h, w, _ = cube.shape
    refl = np.ones((h, w)) if reflectance is None else np.asarray(reflectance, dtype=float)
    if cfg.rayleigh_strength > 0:
        bump = cfg.rayleigh_strength * scale * _line(em_grid, ex, cfg.rayleigh_fwhm)
        cube += refl[:, :, None] * bump[None, None, :]
        if cfg.second_order and (2 * ex) <= em_grid[-1]:
            bump2 = cfg.rayleigh_strength * scale * _line(em_grid, 2 * ex, cfg.rayleigh_fwhm)
            cube += refl[:, :, None] * bump2[None, None, :]
    if cfg.raman_strength > 0:
        center = raman_center_nm(ex, cfg.raman_shift_cm)
        if em_grid[0] <= center <= em_grid[-1]:
            rbump = cfg.raman_strength * scale * _line(em_grid, center, cfg.raman_fwhm)
            cube += refl[:, :, None] * rbump[None, None, :]


def add_noise(cube, cfg, rng):
    """Return cube with Poisson shot noise + Gaussian read noise (seeded by ``rng``)."""
    out = cube
    if cfg.photon_scale > 0:
        out = rng.poisson(np.clip(out, 0, None) * cfg.photon_scale) / cfg.photon_scale
    if cfg.read_sigma > 0:
        out = out + rng.normal(0.0, cfg.read_sigma, size=out.shape)
    return out
