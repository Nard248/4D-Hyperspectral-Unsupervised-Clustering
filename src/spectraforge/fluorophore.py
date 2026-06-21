"""Fluorophore: parametric Gaussian excitation/emission spectra (dilute-regime model)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_FWHM_TO_SIGMA = 1.0 / 2.3548200450309493  # 1 / (2*sqrt(2*ln2))


@dataclass(frozen=True)
class Fluorophore:
    """A single fluorophore defined by Gaussian excitation/emission bands.

    The dilute-regime amplitude of this fluorophore's contribution is
    ``extinction * quantum_yield * excitation(λex) * emission(λem) * concentration``.
    """

    name: str
    ex_peak_nm: float
    ex_fwhm_nm: float
    em_peak_nm: float
    em_fwhm_nm: float
    quantum_yield: float = 0.5      # Phi
    extinction: float = 1.0         # relative epsilon (unitless in v1)
    em_skew: float = 0.0            # 0 = symmetric Gaussian; >0 = red-tailed (emission asymmetry)
    vibronic: float = 0.0           # 0 = none; >0 = a Franck-Condon vibronic shoulder on the red side

    def excitation(self, wl):
        """Relative absorption probability at ``wl``, normalized to peak 1."""
        wl = np.asarray(wl, dtype=float)
        sigma = self.ex_fwhm_nm * _FWHM_TO_SIGMA
        return np.exp(-0.5 * ((wl - self.ex_peak_nm) / sigma) ** 2)

    def emission(self, wl):
        """Emission shape over grid ``wl``, normalized to unit sum (area) on that grid.

        With ``em_skew``/``vibronic`` at 0 this is a symmetric Gaussian (unchanged). Otherwise it is a
        split-Gaussian with a longer red tail (real emission is asymmetric) plus an optional vibronic
        shoulder ~1300 cm^-1 to the red — the Franck-Condon progression that gives a fluorophore its
        characteristic *shape* (what discriminates two overlapping dyes beyond their peak position)."""
        wl = np.asarray(wl, dtype=float)
        sigma = self.em_fwhm_nm * _FWHM_TO_SIGMA
        if self.em_skew == 0.0 and self.vibronic == 0.0:
            g = np.exp(-0.5 * ((wl - self.em_peak_nm) / sigma) ** 2)
        else:
            d = wl - self.em_peak_nm
            s_red = sigma * (1.0 + self.em_skew)
            s_blue = sigma * max(0.2, 1.0 - 0.4 * self.em_skew)
            s = np.where(d >= 0, s_red, s_blue)
            g = np.exp(-0.5 * (d / s) ** 2)
            if self.vibronic > 0:
                spacing_nm = self.em_peak_nm ** 2 * 1300e-7   # 1300 cm^-1 expressed in nm at this peak
                g = g + self.vibronic * np.exp(-0.5 * ((wl - (self.em_peak_nm + spacing_nm)) / s_red) ** 2)
        total = g.sum()
        return g / total if total > 0 else g
