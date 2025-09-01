"""
Wavelength Analysis Core Package

This package provides comprehensive tools for hyperspectral wavelength selection
and analysis using latent space perturbation techniques.
"""

from .analyzer import WavelengthAnalyzer
from .selector import WavelengthSelector
from .experiments import ExperimentFramework
from .visualization import WavelengthVisualizer
from .config import AnalysisConfig

__version__ = "1.0.0"
__author__ = "Hyperspectral Analysis Team"

__all__ = [
    "WavelengthAnalyzer",
    "WavelengthSelector", 
    "ExperimentFramework",
    "WavelengthVisualizer",
    "AnalysisConfig"
]