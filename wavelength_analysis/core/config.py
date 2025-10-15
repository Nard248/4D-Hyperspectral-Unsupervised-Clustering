"""
Configuration management for wavelength analysis
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import json


@dataclass
class AnalysisConfig:
    """Configuration class for wavelength analysis experiments"""
    
    # Data configuration
    sample_name: str = "Lime"
    data_path: str = ""
    mask_path: str = ""
    model_path: str = ""
    
    # Analysis parameters
    dimension_selection_method: str = "activation"  # "variance", "activation", "pca"
    n_important_dimensions: int = 15
    perturbation_method: str = "percentile"  # "percentile", "standard_deviation", "absolute_range"
    perturbation_magnitudes: List[float] = field(default_factory=lambda: [10, 20, 30])
    perturbation_directions: List[str] = field(default_factory=lambda: ["bidirectional"])
    normalization_method: str = "variance"  # "variance", "max_per_excitation", "none"
    
    # Selection parameters
    n_bands_to_select: int = 30
    n_layers_to_extract: int = 10

    # Diversity constraint parameters (NEW)
    use_diversity_constraint: bool = False  # Enable MMR-based diversity selection
    diversity_method: str = "mmr"  # "mmr", "min_distance", "none"
    lambda_diversity: float = 0.5  # MMR diversity parameter (0.0 = no diversity, 1.0 = max diversity)
    min_distance_nm: float = 15.0  # Minimum spectral distance for "min_distance" method
    
    # Output configuration
    output_dir: str = ""
    save_tiff_layers: bool = True
    save_visualizations: bool = True
    save_detailed_results: bool = True
    
    # Technical parameters
    device: str = "cuda"
    n_baseline_patches: int = 50
    patch_size: int = 32
    patch_stride: int = 16
    
    def __post_init__(self):
        """Set default paths based on sample name if not provided"""
        if not self.data_path:
            self.data_path = f"../data/processed/{self.sample_name}/{self.sample_name.lower()}_data_masked.pkl"
        if not self.mask_path:
            self.mask_path = f"../data/processed/{self.sample_name}/{self.sample_name.lower()}_mask.npy"
        if not self.model_path:
            self.model_path = "../results/wavelength_selection/best_model.pth"
        if not self.output_dir:
            self.output_dir = f"./results/{self.sample_name}"
    
    def save(self, filepath: str):
        """Save configuration to JSON file"""
        config_dict = {
            # Convert Path objects to strings for JSON serialization
            key: str(value) if isinstance(value, Path) else value
            for key, value in self.__dict__.items()
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'AnalysisConfig':
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in self.__dict__.items()
        }


# Predefined configuration templates
LIME_CONFIG = AnalysisConfig(
    sample_name="Lime",
    dimension_selection_method="activation",
    perturbation_method="percentile",
    perturbation_magnitudes=[10, 20, 30],
    n_bands_to_select=30,
    n_layers_to_extract=10
)

KIWI_CONFIG = AnalysisConfig(
    sample_name="Kiwi", 
    dimension_selection_method="activation",
    perturbation_method="percentile",
    perturbation_magnitudes=[10, 20, 30],
    n_bands_to_select=30,
    n_layers_to_extract=10
)

LICHENS_CONFIG = AnalysisConfig(
    sample_name="Lichens",
    dimension_selection_method="activation", 
    perturbation_method="percentile",
    perturbation_magnitudes=[10, 20, 30],
    n_bands_to_select=30,
    n_layers_to_extract=10
)

# Experimental configurations for testing different approaches
EXPERIMENTAL_CONFIGS = {
    "aggressive_std": AnalysisConfig(
        sample_name="Lime",
        dimension_selection_method="variance",
        perturbation_method="standard_deviation",
        perturbation_magnitudes=[25, 50, 75, 100],
        n_important_dimensions=20
    ),
    "high_resolution": AnalysisConfig(
        sample_name="Lime",
        dimension_selection_method="activation", 
        perturbation_method="percentile",
        perturbation_magnitudes=[1, 2, 5, 7, 10, 15, 20, 25],
        n_important_dimensions=40
    ),
    "pca_based": AnalysisConfig(
        sample_name="Lime",
        dimension_selection_method="pca",
        perturbation_method="absolute_range", 
        perturbation_magnitudes=[20, 40, 60, 80],
        n_important_dimensions=25
    )
}