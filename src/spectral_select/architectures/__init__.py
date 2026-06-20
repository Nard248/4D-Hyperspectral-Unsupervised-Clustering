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
from .deep_cae import DeepSpectralCAE
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

__all__ = [
    "BandSelectorModel", "SpectralSelector", "AnalyzerBackedSelector",
    "StandardCAE", "DeepCAE", "DeepSpectralCAE",
    "SpectralAE", "MaskedSpectralAE", "VariationalSpectralAE",
    "CANDIDATES", "feature_matrix", "diverse_topk",
]
