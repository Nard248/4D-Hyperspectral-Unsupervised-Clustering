"""
Compatibility layer for the old WavelengthSelector interface

This module provides backwards compatibility with the original WavelengthSelector
while redirecting to the new WavelengthAnalyzer architecture.
"""

from .analyzer import WavelengthAnalyzer
from .config import AnalysisConfig

# Re-export for backwards compatibility
WavelengthSelector = WavelengthAnalyzer