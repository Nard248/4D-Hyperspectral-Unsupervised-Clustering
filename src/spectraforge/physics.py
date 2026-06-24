"""Optional richer sample/optics physics for the forward model.

All effects default OFF, so ``PhysicsConfig()`` (or ``physics=None``) leaves the exactly-linear
dilute model untouched — ``render(A+B) == render(A)+render(B)``. Turn effects on to add realism:

- ``psf_sigma_px``        — spatial Gaussian point-spread blur (optical resolution), per band.
- ``inner_filter``        — Beer-Lambert primary inner-filter: excitation light is absorbed as it
                            penetrates, so local signal is attenuated by ``exp(-strength * A_ex)``
                            where ``A_ex`` is the per-pixel absorbance at the excitation wavelength.
                            This is the model's first deliberate NONLINEARITY in concentration.
- ``reabsorption``        — Beer-Lambert SECONDARY inner-filter (self-absorption): emitted light is
                            reabsorbed on its way out, by ``exp(-strength * A_em(λ))`` where
                            ``A_em(λ) = Σ_k ε_k c_k(x,y) · absorption_k(λ)`` is the per-pixel,
                            per-EMISSION-wavelength absorbance. Because absorption overlaps the blue
                            edge of emission, this suppresses the blue edge and red-shifts the apparent
                            peak *as a function of concentration* — a concentration-dependent
                            RESHAPING of the band (nonlinear; relocates information into band shape /
                            ratios rather than amplitude). The honest nonlinear regime an autoencoder
                            could exploit where a linear method cannot.
- ``autofluorescence``    — a spatially-uniform broadband fluorescent background.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_FWHM_TO_SIGMA = 1.0 / 2.3548200450309493


@dataclass
class PhysicsConfig:
    psf_sigma_px: float = 0.0
    inner_filter: bool = False
    inner_filter_strength: float = 1.0
    reabsorption: bool = False           # secondary inner-filter (self-absorption); reshapes the band
    reabsorption_strength: float = 1.0
    autofluorescence: float = 0.0        # 0 disables
    autofluor_peak_nm: float = 480.0
    autofluor_fwhm_nm: float = 150.0


def _broadband(em_grid, peak, fwhm):
    sigma = fwhm * _FWHM_TO_SIGMA
    return np.exp(-0.5 * ((em_grid - peak) / sigma) ** 2)


def apply_fret(contribs, absorbed, conc, library, em_grid, pairs):
    """In-place Förster resonance energy transfer between fluorophore pairs (a strong nonlinearity).

    For each ``(donor, acceptor, k)``: a *saturating* fraction ``E = k·c_A/(1+k·c_A)`` of the donor's
    absorbed energy is transferred to the acceptor — the donor emission is **quenched** by ``(1-E)`` and
    the acceptor gains a **sensitized** emission ``E·absorbed_donor·Φ_A``. Because ``E`` saturates in the
    acceptor concentration and the sensitized term ``~ c_D·E(c_A)`` is a *product*, this relocates
    information that lives in dye **co-localization** into the donor/acceptor band ratio — structure a
    linear basis cannot represent. ``contribs``/``absorbed`` are per-fluorophore (H,W,n_em)/(H,W) maps.
    """
    for donor, acceptor, k in pairs:
        if donor not in contribs or acceptor not in contribs:
            continue
        cA = conc[acceptor]
        E = (k * cA) / (1.0 + k * cA)                            # (H, W) saturating transfer fraction
        contribs[donor] = contribs[donor] * (1.0 - E)[:, :, None]
        fA = library[acceptor]
        sensitized = E * absorbed[donor] * fA.quantum_yield      # (H, W) energy emitted by acceptor
        contribs[acceptor] = contribs[acceptor] + sensitized[:, :, None] * fA.emission(em_grid)[None, None, :]
    return contribs


def apply_physics(cube, cfg: PhysicsConfig, em_grid, scale, absorbance, em_absorbance=None):
    """Apply (in order) primary inner-filter, reabsorption, autofluorescence, then PSF blur.

    ``absorbance`` is the per-pixel (H, W) excitation absorbance accumulated by the caller
    (``Σ_k ε_k c_k(x,y) · excitation_k(λ_ex)``); only used when ``inner_filter`` is on.
    ``em_absorbance`` is the per-pixel, per-emission-band (H, W, n_em) absorbance
    (``Σ_k ε_k c_k(x,y) · absorption_k(λ_em)``); only used when ``reabsorption`` is on. Because it is
    wavelength-dependent (and concentration-dependent), the resulting ``exp(-s·A_em(λ))`` factor
    RESHAPES the per-pixel emission band — the nonlinear, shape-encoding effect.
    """
    out = cube
    if cfg.inner_filter:
        factor = np.exp(-cfg.inner_filter_strength * absorbance)     # (H, W) scalar per pixel
        out = out * factor[:, :, None]
    if cfg.reabsorption and em_absorbance is not None:
        out = out * np.exp(-cfg.reabsorption_strength * em_absorbance)   # (H, W, n_em) reshapes band
    if cfg.autofluorescence > 0:
        bg = cfg.autofluorescence * scale * _broadband(em_grid, cfg.autofluor_peak_nm, cfg.autofluor_fwhm_nm)
        out = out + bg[None, None, :]
    if cfg.psf_sigma_px > 0:
        from scipy.ndimage import gaussian_filter
        out = gaussian_filter(out, sigma=(cfg.psf_sigma_px, cfg.psf_sigma_px, 0.0), mode="reflect")
    return out
