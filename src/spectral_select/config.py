"""Configuration system for spectral analysis.

This module provides the unified Config dataclass that controls the entire
wavelength selection pipeline. It supports pluggable components through
protocol-based interfaces.

Example:
    # Using built-in components via string identifiers
    config = Config(
        sample_name="Lichens_2",
        classifier="knn",
        clustering="kmeans",
    )

    # Using custom implementations
    config = Config(
        sample_name="Lichens_2",
        classifier=MyCustomClassifier,
    )

    # Loading from file
    config = Config.from_yaml("config.yaml")
    config = Config.from_json("config.json")

    # Saving to file
    config.to_yaml("config.yaml")
    config.to_json("config.json")
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Type, Union

try:
    import yaml

    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

if TYPE_CHECKING:
    from .protocols import (
        AutoencoderProtocol,
        ClassifierProtocol,
        ClusteringProtocol,
        WavelengthRankerProtocol,
    )

# Built-in component registries
# Actual implementations will be added in later phases
BUILT_IN_CLASSIFIERS: dict[str, Any] = {"knn": None}
BUILT_IN_CLUSTERING: dict[str, Any] = {"kmeans": None}
# Built-in autoencoder: "standard" maps to None (placeholder), resolved lazily at runtime
# to avoid circular imports. The actual HyperspectralCAEWithMasking is imported
# in Analyzer._create_model() when the string identifier "standard" is received.
BUILT_IN_AUTOENCODERS: dict[str, Any] = {"standard": None}
BUILT_IN_WAVELENGTH_RANKERS: dict[str, Any] = {"perturbation": None}

# Valid options for validation
_VALID_DIMENSION_METHODS = frozenset({"variance", "activation", "pca"})
_VALID_PERTURBATION_METHODS = frozenset(
    {"percentile", "standard_deviation", "absolute_range"}
)
_VALID_NORMALIZATION_METHODS = frozenset({"variance", "max_per_excitation", "none"})
_VALID_DIVERSITY_METHODS = frozenset({"mmr", "min_distance", "none"})
_VALID_DEVICES = frozenset({"cuda", "cpu", "mps"})


@dataclass
class Config:
    """Unified configuration for the wavelength selection pipeline.

    This dataclass serves as the control center for the entire analysis.
    Configure analysis parameters and pluggable components here.

    Attributes:
        sample_name: Identifier for the sample being analyzed.
        data_path: Path to input spectral data file.
        mask_path: Path to mask file for valid pixels.
        model_path: Path to trained autoencoder model.
        output_dir: Directory for saving results.
        dimension_selection_method: Method for selecting important latent dimensions.
            One of "variance", "activation", or "pca".
        n_important_dimensions: Number of latent dimensions to analyze.
        perturbation_method: Method for perturbing latent dimensions.
            One of "percentile", "standard_deviation", or "absolute_range".
        perturbation_magnitudes: List of perturbation strengths to apply.
        perturbation_directions: Direction(s) of perturbation.
        normalization_method: How to normalize sensitivity scores.
            One of "variance", "max_per_excitation", or "none".
        n_bands_to_select: Target number of wavelength bands to select.
        n_layers_to_extract: Number of layers to extract from model.
        use_diversity_constraint: Whether to enforce spectral diversity.
        diversity_method: Diversity enforcement method.
            One of "mmr", "min_distance", or "none".
        lambda_diversity: Trade-off parameter for MMR (0=relevance, 1=diversity).
        min_distance_nm: Minimum spectral distance in nanometers.
        save_tiff_layers: Whether to save intermediate TIFF layers.
        save_visualizations: Whether to generate and save plots.
        save_detailed_results: Whether to save detailed analysis results.
        device: Computation device ("cuda", "cpu", or "mps").
        n_baseline_patches: Number of patches for baseline computation.
        patch_size: Size of image patches in pixels.
        patch_stride: Stride between patches.
        random_seed: Seed for reproducibility.
        classifier: Classification component (string ID or class).
        clustering: Clustering component (string ID or class).
        autoencoder_architecture: Autoencoder architecture (string ID or class).
        wavelength_ranker: Wavelength ranking component (string ID or class).
    """

    # Data configuration
    sample_name: str = "sample"
    data_path: Optional[Path] = None
    mask_path: Optional[Path] = None
    model_path: Optional[Path] = None
    output_dir: Optional[Path] = None

    # Analysis parameters
    dimension_selection_method: str = "activation"
    n_important_dimensions: int = 15
    perturbation_method: str = "percentile"
    perturbation_magnitudes: List[float] = field(default_factory=lambda: [10, 20, 30])
    perturbation_directions: List[str] = field(
        default_factory=lambda: ["bidirectional"]
    )
    normalization_method: str = "variance"

    # Selection parameters
    n_bands_to_select: int = 30
    n_layers_to_extract: int = 10

    # Diversity constraint parameters
    use_diversity_constraint: bool = False
    diversity_method: str = "mmr"
    lambda_diversity: float = 0.5
    min_distance_nm: float = 15.0

    # Output configuration
    save_tiff_layers: bool = True
    save_visualizations: bool = True
    save_detailed_results: bool = True

    # Technical parameters
    device: str = "cuda"
    n_baseline_patches: int = 50
    patch_size: int = 32
    patch_stride: int = 16
    random_seed: int = 42

    # Model architecture parameters
    model_k1: int = 20  # Number of filters in first encoder layer
    model_k3: int = 20  # Number of filters in third encoder layer
    model_filter_size: int = 5  # Convolutional kernel size (must be odd)
    model_sparsity_target: float = 0.1  # Target sparsity for regularization
    model_sparsity_weight: float = 1.0  # Weight of sparsity loss term
    model_dropout_rate: float = 0.5  # Dropout probability during training
    model_hidden_activation: str = "sigmoid"  # hidden activation; "relu" fixes vanishing-gradient stall
    model_output_activation: str = "sigmoid"  # output activation ("identity"=linear)

    # Training parameters
    training_epochs: int = 30  # Number of training epochs for autoencoder
    training_lr: float = 0.001  # Learning rate for optimizer
    training_chunk_size: int = 64  # Spatial chunk size for training
    training_chunk_overlap: int = 8  # Overlap between adjacent chunks
    training_early_stopping_patience: Optional[int] = None  # Early stopping (None=disabled)
    training_scheduler_patience: int = 5  # LR scheduler patience

    # Pluggable components
    # Type hint uses Union for flexibility with string identifiers or class references
    classifier: Union[str, Type[ClassifierProtocol], Callable[..., Any]] = "knn"
    clustering: Union[str, Type[ClusteringProtocol], Callable[..., Any]] = "kmeans"
    autoencoder_architecture: Union[
        str, Type[AutoencoderProtocol], Callable[..., Any]
    ] = "standard"
    wavelength_ranker: Union[
        str, Type[WavelengthRankerProtocol], Callable[..., Any]
    ] = "perturbation"

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate_string_options()
        self._validate_numeric_ranges()
        self._convert_paths()

    def _validate_string_options(self) -> None:
        """Validate string-based configuration options."""
        # Validate sample_name is not empty or whitespace-only
        if not self.sample_name or not self.sample_name.strip():
            raise ValueError("sample_name cannot be empty or whitespace-only")

        if self.dimension_selection_method not in _VALID_DIMENSION_METHODS:
            raise ValueError(
                f"dimension_selection_method must be one of {sorted(_VALID_DIMENSION_METHODS)}, "
                f"got '{self.dimension_selection_method}'"
            )

        if self.perturbation_method not in _VALID_PERTURBATION_METHODS:
            raise ValueError(
                f"perturbation_method must be one of {sorted(_VALID_PERTURBATION_METHODS)}, "
                f"got '{self.perturbation_method}'"
            )

        if self.normalization_method not in _VALID_NORMALIZATION_METHODS:
            raise ValueError(
                f"normalization_method must be one of {sorted(_VALID_NORMALIZATION_METHODS)}, "
                f"got '{self.normalization_method}'"
            )

        if self.diversity_method not in _VALID_DIVERSITY_METHODS:
            raise ValueError(
                f"diversity_method must be one of {sorted(_VALID_DIVERSITY_METHODS)}, "
                f"got '{self.diversity_method}'"
            )

        if self.device not in _VALID_DEVICES:
            raise ValueError(
                f"device must be one of {sorted(_VALID_DEVICES)}, "
                f"got '{self.device}'"
            )

    def _validate_numeric_ranges(self) -> None:
        """Validate numeric parameter ranges."""
        if self.n_important_dimensions <= 0:
            raise ValueError(
                f"n_important_dimensions must be positive, got {self.n_important_dimensions}"
            )

        if self.n_bands_to_select <= 0:
            raise ValueError(
                f"n_bands_to_select must be positive, got {self.n_bands_to_select}"
            )

        if self.n_layers_to_extract <= 0:
            raise ValueError(
                f"n_layers_to_extract must be positive, got {self.n_layers_to_extract}"
            )

        if not 0.0 <= self.lambda_diversity <= 1.0:
            raise ValueError(
                f"lambda_diversity must be between 0 and 1, got {self.lambda_diversity}"
            )

        if self.min_distance_nm < 0:
            raise ValueError(
                f"min_distance_nm must be non-negative, got {self.min_distance_nm}"
            )

        if self.n_baseline_patches <= 0:
            raise ValueError(
                f"n_baseline_patches must be positive, got {self.n_baseline_patches}"
            )

        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {self.patch_size}")

        if self.patch_stride <= 0:
            raise ValueError(f"patch_stride must be positive, got {self.patch_stride}")

        if not self.perturbation_magnitudes:
            raise ValueError("perturbation_magnitudes cannot be empty")

        if any(m < 0 for m in self.perturbation_magnitudes):
            raise ValueError(
                f"perturbation_magnitudes must all be non-negative, "
                f"got {self.perturbation_magnitudes}"
            )

        # Model architecture validation
        if self.model_k1 <= 0:
            raise ValueError(f"model_k1 must be positive, got {self.model_k1}")

        if self.model_k3 <= 0:
            raise ValueError(f"model_k3 must be positive, got {self.model_k3}")

        if self.model_filter_size <= 0:
            raise ValueError(
                f"model_filter_size must be positive, got {self.model_filter_size}"
            )

        if self.model_filter_size % 2 == 0:
            raise ValueError(
                f"model_filter_size must be odd for symmetric padding, "
                f"got {self.model_filter_size}"
            )

        if not 0.0 <= self.model_sparsity_target <= 1.0:
            raise ValueError(
                f"model_sparsity_target must be between 0 and 1, "
                f"got {self.model_sparsity_target}"
            )

        if self.model_sparsity_weight < 0:
            raise ValueError(
                f"model_sparsity_weight must be non-negative, "
                f"got {self.model_sparsity_weight}"
            )

        if not 0.0 <= self.model_dropout_rate <= 1.0:
            raise ValueError(
                f"model_dropout_rate must be between 0 and 1, "
                f"got {self.model_dropout_rate}"
            )

        # Training parameters validation
        if self.training_epochs <= 0:
            raise ValueError(
                f"training_epochs must be positive, got {self.training_epochs}"
            )

        if self.training_lr <= 0:
            raise ValueError(f"training_lr must be positive, got {self.training_lr}")

        if self.training_chunk_size <= 0:
            raise ValueError(
                f"training_chunk_size must be positive, got {self.training_chunk_size}"
            )

        if self.training_chunk_overlap < 0:
            raise ValueError(
                f"training_chunk_overlap must be non-negative, "
                f"got {self.training_chunk_overlap}"
            )

        if self.training_chunk_overlap >= self.training_chunk_size:
            raise ValueError(
                f"training_chunk_overlap ({self.training_chunk_overlap}) must be less "
                f"than training_chunk_size ({self.training_chunk_size})"
            )

        if (
            self.training_early_stopping_patience is not None
            and self.training_early_stopping_patience <= 0
        ):
            raise ValueError(
                f"training_early_stopping_patience must be positive or None, "
                f"got {self.training_early_stopping_patience}"
            )

        if self.training_scheduler_patience <= 0:
            raise ValueError(
                f"training_scheduler_patience must be positive, "
                f"got {self.training_scheduler_patience}"
            )

    def _convert_paths(self) -> None:
        """Convert string paths to Path objects and auto-resolve defaults."""
        if self.data_path is not None and not isinstance(self.data_path, Path):
            object.__setattr__(self, "data_path", Path(self.data_path))
        if self.mask_path is not None and not isinstance(self.mask_path, Path):
            object.__setattr__(self, "mask_path", Path(self.mask_path))
        if self.model_path is not None and not isinstance(self.model_path, Path):
            object.__setattr__(self, "model_path", Path(self.model_path))
        if self.output_dir is not None and not isinstance(self.output_dir, Path):
            object.__setattr__(self, "output_dir", Path(self.output_dir))

        # Auto-resolve model_path based on sample_name if not provided
        # This enables automatic model caching without explicit path specification
        if self.model_path is None and self.sample_name:
            default_model_path = Path("model_output") / self.sample_name / "model.pth"
            object.__setattr__(self, "model_path", default_model_path)

    def resolve_classifier(self) -> Any:
        """Resolve classifier to actual implementation.

        Returns:
            The classifier class or callable.

        Raises:
            ValueError: If classifier string is not a known built-in.
        """
        return self._resolve_component(
            self.classifier, BUILT_IN_CLASSIFIERS, "classifier"
        )

    def resolve_clustering(self) -> Any:
        """Resolve clustering to actual implementation.

        Returns:
            The clustering class or callable.

        Raises:
            ValueError: If clustering string is not a known built-in.
        """
        return self._resolve_component(
            self.clustering, BUILT_IN_CLUSTERING, "clustering"
        )

    def resolve_autoencoder(self) -> Any:
        """Resolve autoencoder architecture to actual implementation.

        Returns:
            The autoencoder class or callable.

        Raises:
            ValueError: If autoencoder_architecture string is not a known built-in.
        """
        return self._resolve_component(
            self.autoencoder_architecture, BUILT_IN_AUTOENCODERS, "autoencoder_architecture"
        )

    def resolve_wavelength_ranker(self) -> Any:
        """Resolve wavelength ranker to actual implementation.

        Returns:
            The wavelength ranker class or callable.

        Raises:
            ValueError: If wavelength_ranker string is not a known built-in.
        """
        return self._resolve_component(
            self.wavelength_ranker, BUILT_IN_WAVELENGTH_RANKERS, "wavelength_ranker"
        )

    def _resolve_component(
        self,
        component: Union[str, Type[Any], Callable[..., Any]],
        registry: dict[str, Any],
        component_name: str,
    ) -> Any:
        """Resolve a component from string identifier or return as-is.

        Args:
            component: String identifier or class/callable.
            registry: Built-in component registry.
            component_name: Name for error messages.

        Returns:
            The resolved component.

        Raises:
            ValueError: If string is not in registry or component is invalid.
        """
        if isinstance(component, str):
            if component not in registry:
                raise ValueError(
                    f"Unknown {component_name}: '{component}'. "
                    f"Available: {sorted(registry.keys())}"
                )
            impl = registry[component]
            if impl is None:
                # Placeholder - actual implementation not yet available
                return component  # Return string for now
            return impl

        # If not a string, assume it's a class or callable
        if not callable(component):
            raise ValueError(
                f"{component_name} must be a string identifier or callable, "
                f"got {type(component).__name__}"
            )
        return component

    # -------------------------------------------------------------------------
    # Serialization: Loading from files/dicts
    # -------------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create a Config instance from a dictionary.

        Args:
            data: Dictionary containing configuration values.

        Returns:
            A new Config instance.

        Note:
            Unknown keys in the dictionary are skipped with a warning
            for forward compatibility.
        """
        # Get valid field names from dataclass
        valid_fields = {f.name for f in fields(cls)}

        # Prepare kwargs, converting and filtering as needed
        kwargs: Dict[str, Any] = {}

        for key, value in data.items():
            if key not in valid_fields:
                warnings.warn(
                    f"Unknown configuration key '{key}' - skipping",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            # Handle Path conversion for path fields
            if key in ("data_path", "mask_path", "model_path", "output_dir"):
                if value is not None and isinstance(value, str):
                    value = Path(value)

            # Handle component fields with dict format
            if key in (
                "classifier",
                "clustering",
                "autoencoder_architecture",
                "wavelength_ranker",
            ):
                if isinstance(value, dict) and "type" in value:
                    # If it's a custom component serialized as dict, keep the type string
                    # Custom class restoration would need additional machinery
                    value = value.get("type", value)

            kwargs[key] = value

        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML file.

        Returns:
            A new Config instance.

        Raises:
            ImportError: If PyYAML is not installed.
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the YAML content is invalid.
        """
        if not _YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required for YAML support. "
                "Install it with: pip install pyyaml"
            )

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {path}: {e}") from e

        if data is None:
            data = {}

        if not isinstance(data, dict):
            raise ValueError(f"YAML file must contain a mapping, got {type(data).__name__}")

        return cls.from_dict(data)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from a JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            A new Config instance.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the JSON content is invalid.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {path}: {e}") from e

        if not isinstance(data, dict):
            raise ValueError(f"JSON file must contain an object, got {type(data).__name__}")

        return cls.from_dict(data)

    # -------------------------------------------------------------------------
    # Serialization: Saving to files/dicts
    # -------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary suitable for serialization.

        Returns:
            Dictionary with all configuration values. Path objects are converted
            to strings, and pluggable components are serialized appropriately.
        """
        result: Dict[str, Any] = {}

        for f in fields(self):
            value = getattr(self, f.name)

            # Convert Path objects to strings
            if isinstance(value, Path):
                value = str(value)

            # Handle pluggable components
            if f.name in (
                "classifier",
                "clustering",
                "autoencoder_architecture",
                "wavelength_ranker",
            ):
                if isinstance(value, str):
                    # Built-in identifier, keep as-is
                    pass
                elif callable(value):
                    # Custom class or callable - store qualified name
                    if hasattr(value, "__module__") and hasattr(value, "__qualname__"):
                        value = {
                            "type": "custom",
                            "class": f"{value.__module__}.{value.__qualname__}",
                        }
                    else:
                        value = {"type": "custom", "class": str(value)}

            result[f.name] = value

        return result

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to a YAML file.

        Args:
            path: Path to save the YAML file.

        Raises:
            ImportError: If PyYAML is not installed.
        """
        if not _YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required for YAML support. "
                "Install it with: pip install pyyaml"
            )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def to_json(self, path: Union[str, Path]) -> None:
        """Save configuration to a JSON file.

        Args:
            path: Path to save the JSON file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    # -------------------------------------------------------------------------
    # Special methods
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a readable string representation of the configuration."""
        return (
            f"Config(\n"
            f"  sample_name={self.sample_name!r},\n"
            f"  n_bands_to_select={self.n_bands_to_select},\n"
            f"  n_important_dimensions={self.n_important_dimensions},\n"
            f"  dimension_selection_method={self.dimension_selection_method!r},\n"
            f"  perturbation_method={self.perturbation_method!r},\n"
            f"  device={self.device!r},\n"
            f"  ...  # {len(fields(self)) - 6} more fields\n"
            f")"
        )

    def __eq__(self, other: object) -> bool:
        """Compare two configurations for equality."""
        if not isinstance(other, Config):
            return NotImplemented

        for f in fields(self):
            if getattr(self, f.name) != getattr(other, f.name):
                return False
        return True
