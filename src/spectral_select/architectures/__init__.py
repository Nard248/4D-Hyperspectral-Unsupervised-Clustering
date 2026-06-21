"""Anti-cheat autoencoder ladder (C0..C4) — pluggable, comparable band-selection architectures.

See ``docs/spectraforge/03-architecture-plan.md`` (the ladder + rationale) and
``04-training-runbook.md`` (the contract). Each candidate is a :class:`BandSelectorModel`; the
per-pixel spectral family (C2/C3/C4) and the spatial-CAE family (C0/C1) share the perturbation ->
influence math via :mod:`selection_core`.

    from spectral_select.architectures import CANDIDATES, StandardCAE, SpectralAE
"""
from __future__ import annotations

from .base import AnalyzerBackedSelector, BandSelectorModel, SpectralSelector, diverse_topk, feature_matrix
from .cae_baseline import DeepCAE, StandardCAE
from .conv_spectral_ae import ConvSpectralAE, MaskedConvSpectralAE
from .deep_cae import DeepSpectralCAE
from .deep_spectral_ae import DeepMaskedSpectralAE, DeepSpectralAE
from .masked_spectral_ae import MaskedSpectralAE
from .spectral_ae import SpectralAE
from .variational_spectral_ae import VariationalSpectralAE

# id -> factory, in ladder order (C0..C4). Each factory takes (**kwargs) forwarded to the candidate.
CANDIDATES = {
    "C0": StandardCAE,
    "C1": DeepCAE,
    "C2": SpectralAE,
    "C3": MaskedSpectralAE,
    "C4": VariationalSpectralAE,
}

# Enlarged, research-practice candidates (deeper/wider MLP, 1D-conv, masked conv). Kept in a separate
# registry so the C0..C4 ladder semantics are unchanged.
LARGE_CANDIDATES = {
    "C5": DeepSpectralAE,
    "C5b": DeepMaskedSpectralAE,
    "C6": ConvSpectralAE,
    "C7": MaskedConvSpectralAE,
}

__all__ = [
    "BandSelectorModel", "SpectralSelector", "AnalyzerBackedSelector",
    "StandardCAE", "DeepCAE", "DeepSpectralCAE",
    "SpectralAE", "MaskedSpectralAE", "VariationalSpectralAE",
    "DeepSpectralAE", "DeepMaskedSpectralAE", "ConvSpectralAE", "MaskedConvSpectralAE",
    "CANDIDATES", "LARGE_CANDIDATES", "feature_matrix", "diverse_topk",
]
