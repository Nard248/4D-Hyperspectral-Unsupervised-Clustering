"""C0 — the published spatial CAE, as a :class:`BandSelectorModel` reference.

A thin wrapper over the real :class:`spectral_select.Analyzer` with the untouched ``"standard"``
architecture (``HyperspectralCAEWithMasking``). This is the reference rung: on synthetic ME-HSI it
is expected to be degenerate (reconstruction R approx 0, influence anti-correlated with signal).
Nothing about the published CAE or the selection math is modified here.

C1 reuses the same wrapper, only swapping in :class:`DeepSpectralCAE` via the
``autoencoder_architecture`` seam.
"""
from __future__ import annotations

from .base import AnalyzerBackedSelector
from .deep_cae import DeepSpectralCAE


class StandardCAE(AnalyzerBackedSelector):
    """C0: the published spatial CAE (``autoencoder_architecture='standard'``)."""

    name = "C0 standard-CAE"
    architecture = "standard"


class DeepCAE(AnalyzerBackedSelector):
    """C1: deeper, spectral-preserving spatial CAE driven through the Analyzer seam."""

    name = "C1 deep-spectral-CAE"
    architecture = DeepSpectralCAE
