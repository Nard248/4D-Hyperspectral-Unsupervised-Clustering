"""Core wavelength selection analysis engine.

This module provides the Analyzer class, the main interface for running
wavelength selection analysis on hyperspectral data.

Example:
    from spectral_select import Analyzer, Config, SpectraData

    config = Config(sample_name="Lichens_2")
    analyzer = Analyzer(config)

    # Load data
    data = SpectraData.from_pickle("processed_data.pkl")

    # Run analysis
    analyzer.fit(data)

    # Get selected wavelengths
    bands = analyzer.get_wavelengths()
    print(f"Selected {len(bands)} wavelength combinations")
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import tifffile
import torch

import selection_core

from .config import Config
from .types import AnalysisMetrics, SpectraData, WavelengthBand, WavelengthResult

if TYPE_CHECKING:
    from .results import ResultsManager

logger = logging.getLogger(__name__)


class Analyzer:
    """Main analyzer for wavelength selection using autoencoder perturbation.

    This class provides a scikit-learn-style interface for running wavelength
    selection analysis on hyperspectral data. It identifies the most informative
    wavelength combinations by analyzing how perturbations in the autoencoder's
    latent space affect the reconstruction.

    Attributes:
        config: Configuration object controlling analysis parameters.
        device: PyTorch device used for computation.

    Example:
        config = Config(sample_name="sample", n_bands_to_select=30)
        analyzer = Analyzer(config)
        analyzer.fit(data)
        wavelengths = analyzer.get_wavelengths()
    """

    def __init__(
        self,
        config: Config,
        results_manager: Optional["ResultsManager"] = None,
    ) -> None:
        """Initialize analyzer with configuration.

        Args:
            config: Configuration object with analysis parameters.
            results_manager: Optional ResultsManager for structured output paths.
                If not provided, one will be created lazily when save_results()
                is called using config.output_dir and config.sample_name.
        """
        self._config = config
        self._results_manager = results_manager

        # Setup compute device
        if config.device == "cuda" and torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif config.device == "mps" and torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")

        # Internal state (populated during fit)
        self._data: Optional[SpectraData] = None
        self._dataset: Optional[Any] = None  # MaskedHyperspectralDataset
        self._model: Optional[Any] = None
        self._baseline_latent: Optional[Any] = None
        self._baseline_reconstruction: Optional[Dict[float, Any]] = None
        self._patch_coords: Optional[List[Tuple[int, int]]] = None
        self._influence_matrix: Optional[Dict[str, Any]] = None
        self._result: Optional[WavelengthResult] = None
        self._is_prepared: bool = False

        # Create output directory if specified (backward compatibility)
        if config.output_dir is not None:
            config.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Analyzer initialized for '{config.sample_name}' "
            f"on device '{self._device}'"
        )

    @property
    def config(self) -> Config:
        """The configuration object."""
        return self._config

    @property
    def device(self) -> torch.device:
        """The PyTorch device used for computation."""
        return self._device

    @property
    def is_fitted(self) -> bool:
        """Whether the analyzer has been fitted to data."""
        return self._result is not None

    @property
    def is_prepared(self) -> bool:
        """Whether the model + baseline are ready (so select() can run)."""
        return self._is_prepared

    @property
    def result(self) -> Optional[WavelengthResult]:
        """The analysis result, or None if not yet fitted."""
        return self._result

    @property
    def influence_matrix(self) -> Optional[Dict[str, Any]]:
        """The computed influence matrix, or None if not yet fitted."""
        return self._influence_matrix

    @property
    def results_manager(self) -> "ResultsManager":
        """ResultsManager for structured output paths.

        Lazily creates a ResultsManager if not provided at initialization,
        using config.output_dir (or "results/") as base_dir and config.sample_name.

        Returns:
            ResultsManager instance for this analyzer.
        """
        if self._results_manager is None:
            from .results import ResultsManager

            base_dir = self._config.output_dir or Path("results")
            self._results_manager = ResultsManager(
                base_dir=base_dir,
                sample_name=self._config.sample_name,
            )
        return self._results_manager

    def prepare(self, data: SpectraData, *, progress_callback=None) -> Analyzer:
        """Slow, one-time setup: load data, load-or-train the model, set up baseline.

        After this, :meth:`select` can be called repeatedly with different selection
        parameters without retraining. Separated from selection so a GUI can train once
        and let the user tune band selection interactively.

        Args:
            data: The hyperspectral data to analyze.
            progress_callback: Optional ``(epoch, total_epochs, loss)`` callback forwarded
                to training (used by the GUI for a live progress bar). Ignored when a
                pretrained model is loaded.

        Returns:
            self, for method chaining.
        """
        logger.info(f"Preparing analyzer for {self._config.sample_name}")
        self._load_data(data)
        self._load_or_train_model(progress_callback=progress_callback)
        self._setup_baseline()
        self._is_prepared = True
        return self

    def select(self, config: Optional[Config] = None) -> WavelengthResult:
        """Run band selection on the prepared model + baseline (fast, re-runnable).

        Uses the analysis / perturbation / normalization / selection fields of
        ``config`` (defaults to the analyzer's current config). Does NOT retrain — call
        :meth:`prepare` (or :meth:`fit`) first.

        Args:
            config: Optional config whose selection parameters override the current one.

        Returns:
            The WavelengthResult (also stored on the analyzer).
        """
        if not self._is_prepared:
            raise RuntimeError(
                "Analyzer must be prepared before select(); call prepare(data) or fit(data)."
            )
        if config is not None:
            self._config = config

        self._select_important_dimensions()
        self._compute_influence_scores()
        if self._config.normalization_method != "none":
            self._normalize_influences()
        selected_bands = self._select_top_bands()

        total_available = sum(
            self._model.emission_bands[ex]
            for ex in self._model.excitation_wavelengths
        )
        self._result = WavelengthResult(
            sample_name=self._config.sample_name,
            selected_bands=selected_bands,
            metrics=AnalysisMetrics.from_bands(selected_bands, total_available),
            config_snapshot=self._config.to_dict(),
            method_summary={
                "dimension_selection": self._config.dimension_selection_method,
                "perturbation": self._config.perturbation_method,
                "normalization": self._config.normalization_method,
            },
        )
        logger.info(f"Selection complete: {len(selected_bands)} bands selected")
        return self._result

    def fit(self, data: SpectraData) -> Analyzer:
        """Run the full pipeline: ``prepare(data)`` then ``select()``.

        Backward-compatible single-call entry point.

        Args:
            data: The hyperspectral data to analyze.

        Returns:
            self, for method chaining.
        """
        logger.info(f"Starting analysis for {self._config.sample_name}")
        self.prepare(data)
        self.select(self._config)
        return self

    def transform(self, data: SpectraData) -> SpectraData:
        """Apply fitted selection to extract only selected wavelengths.

        Uses the wavelength selection from a previous fit() call to
        extract a reduced dataset containing only the selected bands.

        Args:
            data: The hyperspectral data to transform.

        Returns:
            A new SpectraData object with only the selected wavelengths.

        Raises:
            RuntimeError: If the analyzer has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Analyzer must be fitted before transform()")

        from .types import ExcitationData

        logger.info(f"Transforming data to {len(self._result.selected_bands)} bands")

        # Group selected bands by excitation wavelength
        bands_by_excitation: Dict[float, List[WavelengthBand]] = {}
        for band in self._result.selected_bands:
            if band.excitation_nm not in bands_by_excitation:
                bands_by_excitation[band.excitation_nm] = []
            bands_by_excitation[band.excitation_nm].append(band)

        # Create new excitations with only selected bands
        new_excitations: Dict[float, ExcitationData] = {}

        for ex_nm, bands in bands_by_excitation.items():
            # Get original excitation data
            orig_ex = data.get_excitation(ex_nm)

            # Extract only the selected emission bands
            band_indices = [b.emission_band_index for b in bands]
            band_indices_sorted = sorted(band_indices)

            # Extract cube with only selected bands
            new_cube = orig_ex.cube[:, :, band_indices_sorted]

            # Extract corresponding emission wavelengths
            new_emission_wls = [orig_ex.emission_wavelengths[i] for i in band_indices_sorted]

            new_excitations[ex_nm] = ExcitationData(
                excitation_nm=ex_nm,
                cube=new_cube,
                emission_wavelengths=new_emission_wls,
                exposure_time=orig_ex.exposure_time,
                laser_power=orig_ex.laser_power,
            )

        # Create new SpectraData with reduced bands
        transformed = SpectraData(
            excitations=new_excitations,
            mask=data.mask,
            sample_name=f"{data.sample_name}_selected",
            loading_options=data.loading_options,
            metadata={
                **data.metadata,
                "transform_source": data.sample_name,
                "n_selected_bands": len(self._result.selected_bands),
            },
        )

        logger.info(
            f"Transformed: {data.n_excitations} excitations -> "
            f"{transformed.n_excitations} excitations with selected bands"
        )

        return transformed

    def fit_transform(self, data: SpectraData) -> SpectraData:
        """Convenience method: fit the analyzer and transform in one step.

        Equivalent to calling fit(data).transform(data).

        Args:
            data: The hyperspectral data to analyze and transform.

        Returns:
            A new SpectraData object with only the selected wavelengths.
        """
        return self.fit(data).transform(data)

    def get_wavelengths(self) -> List[WavelengthBand]:
        """Return the list of selected wavelength combinations.

        Returns the selected wavelength bands from the analysis, ordered
        by their importance/influence score (highest first).

        Returns:
            List of WavelengthBand objects representing the selection.

        Raises:
            RuntimeError: If the analyzer has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Analyzer must be fitted before get_wavelengths(). "
                "Call fit(data) first."
            )
        return self._result.selected_bands

    def save_results(self, output_dir: Optional[Path] = None) -> Path:
        """Save analysis results to disk.

        Saves the wavelength selection results to JSON format, and
        optionally saves TIFF layers for visualization.

        Args:
            output_dir: Directory to save results. If None, uses the
                output_dir from config.

        Returns:
            Path to the saved results directory.

        Raises:
            RuntimeError: If the analyzer has not been fitted.
            ValueError: If no output directory is available.
        """
        if not self.is_fitted:
            raise RuntimeError("Analyzer must be fitted before save_results()")

        # Determine output directory
        out_dir = output_dir if output_dir is not None else self._config.output_dir
        if out_dir is None:
            raise ValueError(
                "No output_dir specified and config.output_dir is None. "
                "Provide output_dir or set config.output_dir."
            )

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving results to {out_dir}")

        # Save primary result JSON
        result_path = out_dir / "wavelength_result.json"
        self._result.to_json(result_path)
        logger.info(f"Saved result JSON: {result_path}")

        # Optionally save TIFF layers
        if self._config.save_tiff_layers:
            self._extract_wavelength_layers(out_dir)

        # Optionally save detailed results
        if self._config.save_detailed_results:
            # Save config
            config_path = out_dir / "analysis_config.json"
            with open(config_path, "w") as f:
                json.dump(self._config.to_dict(), f, indent=2)
            logger.info(f"Saved config: {config_path}")

            # Save human-readable text summary
            txt_path = out_dir / "selected_bands.txt"
            self._write_text_summary(txt_path)
            logger.info(f"Saved text summary: {txt_path}")

        logger.info(f"Results saved to {out_dir}")
        return out_dir

    def save_model(
        self,
        checkpoint_type: str = "final",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save trained model to results directory using ResultsManager.

        Delegates to ResultsManager.save_model_checkpoint() for consistent
        checkpoint naming and metadata management.

        Args:
            checkpoint_type: Type of checkpoint - "best" or "final".
            metadata: Optional metadata to save alongside the checkpoint.
                If not provided, basic metadata (sample_name, timestamp) is added.

        Returns:
            Path to the saved model checkpoint file.

        Raises:
            RuntimeError: If no model has been trained/loaded.
        """
        if self._model is None:
            raise RuntimeError(
                "No model available. Call fit() first or load a model."
            )

        # Build default metadata if not provided
        if metadata is None:
            metadata = {}

        metadata.setdefault("sample_name", self._config.sample_name)
        metadata.setdefault("checkpoint_type", checkpoint_type)
        metadata.setdefault("timestamp", datetime.now().isoformat())
        metadata.setdefault("config_snapshot", self._config.to_dict())

        path = self.results_manager.save_model_checkpoint(
            self._model, checkpoint_type, metadata
        )
        logger.info(f"Saved {checkpoint_type} model to {path}")
        return path

    def _extract_wavelength_layers(self, output_dir: Path) -> List[Dict[str, Any]]:
        """Extract and save wavelength layers as TIFF files.

        Args:
            output_dir: Base output directory.

        Returns:
            List of layer metadata dictionaries.
        """
        logger.info(
            f"Extracting top {self._config.n_layers_to_extract} wavelength layers..."
        )

        # Create layers subdirectory
        layers_dir = output_dir / "layers"
        layers_dir.mkdir(exist_ok=True)

        # Get full spatial data
        all_data = self._dataset.get_all_data()
        layer_info: List[Dict[str, Any]] = []

        for i, band in enumerate(
            self._result.selected_bands[: self._config.n_layers_to_extract]
        ):
            ex = band.excitation_nm
            band_idx = band.emission_band_index
            em_wavelength = band.emission_nm
            influence = band.influence_score

            logger.info(
                f"Layer {i + 1}: Ex {ex}nm, Em {em_wavelength}nm, "
                f"Influence: {influence:.6f}"
            )

            if ex in all_data:
                # Extract the specific emission band for this excitation
                layer_data = all_data[ex][:, :, band_idx].numpy()

                # Handle NaN values
                layer_data = np.nan_to_num(layer_data, nan=0.0)

                # Normalize to 0-1 range
                layer_min = float(np.min(layer_data))
                layer_max = float(np.max(layer_data))
                if layer_max > layer_min:
                    layer_normalized = (layer_data - layer_min) / (layer_max - layer_min)
                else:
                    layer_normalized = np.zeros_like(layer_data)

                # Convert to 16-bit for TIFF
                layer_16bit = (layer_normalized * 65535).astype(np.uint16)

                # Create filename
                filename = (
                    f"layer_{i + 1:02d}_ex{ex:.0f}nm_em{em_wavelength:.0f}nm_"
                    f"inf{influence:.6f}.tiff"
                )
                filepath = layers_dir / filename

                # Save as TIFF
                tifffile.imwrite(str(filepath), layer_16bit)

                # Store metadata
                layer_info.append({
                    "rank": i + 1,
                    "excitation_nm": float(ex),
                    "emission_nm": float(em_wavelength),
                    "emission_band_index": int(band_idx),
                    "influence_score": float(influence),
                    "data_range_original": [layer_min, layer_max],
                    "filename": filename,
                })

        # Save layer metadata
        metadata_path = layers_dir / "layer_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(
                {
                    "layers": layer_info,
                    "extraction_date": datetime.now().isoformat(),
                    "config": self._config.to_dict(),
                },
                f,
                indent=2,
            )

        logger.info(f"Extracted {len(layer_info)} layers to {layers_dir}")
        return layer_info

    def _write_text_summary(self, path: Path) -> None:
        """Write human-readable text summary of results.

        Args:
            path: Path to output text file.
        """
        metrics = self._result.metrics

        def format_score(score: float) -> str:
            """Format influence score with adaptive precision."""
            if abs(score) < 0.001 and score != 0:
                return f"{score:<15.2e}"
            return f"{score:<15.6f}"

        with open(path, "w") as f:
            f.write(f"Wavelength Analysis Results: {self._config.sample_name}\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(
                f"Method: {self._config.dimension_selection_method} + "
                f"{self._config.perturbation_method}\n"
            )
            f.write(
                f"Total bands selected: {metrics.bands_selected} out of "
                f"{metrics.total_bands_available}\n"
            )
            f.write(f"Compression ratio: {metrics.compression_ratio:.2f}x\n\n")
            f.write(
                f"{'Rank':<5} {'Excitation(nm)':<15} {'Emission(nm)':<15} "
                f"{'Influence':<15}\n"
            )
            f.write("-" * 60 + "\n")

            for band in self._result.selected_bands:
                f.write(
                    f"{band.rank:<5} {band.excitation_nm:<15.1f} "
                    f"{band.emission_nm:<15.1f} {format_score(band.influence_score)}\n"
                )

    def __repr__(self) -> str:
        """Return a readable string representation."""
        status = "fitted" if self.is_fitted else "not fitted"
        n_bands = self._result.n_bands if self.is_fitted else "-"
        return (
            f"Analyzer("
            f"sample='{self._config.sample_name}', "
            f"device='{self._device}', "
            f"status={status}, "
            f"n_bands={n_bands})"
        )

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _load_data(self, data: SpectraData) -> None:
        """Load and prepare data for analysis.

        Converts SpectraData into the format expected by MaskedHyperspectralDataset,
        which is used by the autoencoder model.

        Args:
            data: The hyperspectral SpectraData object.
        """
        # Store reference to original data
        self._data = data

        logger.info(
            f"Loading data: {data.n_excitations} excitations, "
            f"spatial shape {data.spatial_shape}, "
            f"mask {'present' if data.mask is not None else 'absent'}"
        )

        from spectral_select.models.dataset import MaskedHyperspectralDataset

        # Convert SpectraData to format expected by MaskedHyperspectralDataset
        # Expected format: {'data': {ex_str: {'cube': array, 'wavelengths': list}},
        #                   'excitation_wavelengths': [...]}
        data_dict: Dict[str, Any] = {
            "data": {},
            "excitation_wavelengths": data.excitation_wavelengths,
        }

        for ex_nm in data.excitation_wavelengths:
            ex_data = data.get_excitation(ex_nm)
            ex_str = str(ex_nm)
            data_dict["data"][ex_str] = {
                "cube": ex_data.cube,
                "wavelengths": ex_data.emission_wavelengths,
            }

        # Extract mask from SpectraData
        mask = data.mask

        # Create dataset with normalization
        self._dataset = MaskedHyperspectralDataset(
            data_dict=data_dict,
            mask=mask,
            normalize=True,
        )

        # Store spatial dimensions for later use
        self._spatial_shape = self._dataset.get_spatial_dimensions()

        logger.info(
            f"Dataset created: {len(self._dataset.excitation_wavelengths)} excitations, "
            f"spatial {self._spatial_shape[0]}x{self._spatial_shape[1]}"
        )

    def _create_model(self, excitations_data: Dict[float, np.ndarray]) -> Any:
        """Create model instance based on config.autoencoder_architecture.

        Supports built-in 'standard' architecture or custom classes implementing
        AutoencoderProtocol.

        Args:
            excitations_data: Dictionary mapping excitation wavelengths to numpy arrays.

        Returns:
            Instantiated model ready for training or inference.

        Raises:
            ValueError: If autoencoder_architecture is an unknown string identifier.
        """
        architecture = self._config.resolve_autoencoder()

        # Handle built-in string identifiers
        if isinstance(architecture, str):
            if architecture == "standard":
                from spectral_select.models.autoencoder import HyperspectralCAEWithMasking

                logger.info("Using built-in 'standard' autoencoder architecture")
                return HyperspectralCAEWithMasking(
                    excitations_data=excitations_data,
                    k1=self._config.model_k1,
                    k3=self._config.model_k3,
                    filter_size=self._config.model_filter_size,
                    sparsity_target=self._config.model_sparsity_target,
                    sparsity_weight=self._config.model_sparsity_weight,
                    dropout_rate=self._config.model_dropout_rate,
                    hidden_activation=self._config.model_hidden_activation,
                    output_activation=self._config.model_output_activation,
                )
            else:
                raise ValueError(f"Unknown built-in autoencoder: {architecture}")

        # Handle custom class/callable
        logger.info(f"Using custom autoencoder: {architecture}")
        return architecture(
            excitations_data=excitations_data,
            k1=self._config.model_k1,
            k3=self._config.model_k3,
            filter_size=self._config.model_filter_size,
        )

    def _load_or_train_model(self, progress_callback=None) -> None:
        """Load pretrained model or train a new one if needed.

        Attempts to load model weights from config.model_path. If the file
        is missing or architecture doesn't match, trains a new model.
        ``progress_callback`` is forwarded to training (no-op when loading).
        """
        if self._dataset is None:
            raise RuntimeError("_load_data must be called before _load_or_train_model")

        # Get data from dataset to initialize model
        all_data = self._dataset.get_all_data()

        # Create model using factory method (uses config architecture)
        excitations_data = {ex: data.numpy() for ex, data in all_data.items()}
        self._model = self._create_model(excitations_data)

        # Attempt to load pretrained weights
        model_path = self._config.model_path
        if model_path is not None and model_path.exists():
            try:
                state_dict = torch.load(model_path, map_location=self._device)
                self._model.load_state_dict(state_dict)
                self._model = self._model.to(self._device)
                # Set model to evaluation mode (not inference mode)
                self._model.train(False)
                logger.info(f"Model loaded from {model_path}")
                return
            except RuntimeError as e:
                error_msg = str(e)
                if "Missing key(s)" in error_msg or "size mismatch" in error_msg:
                    logger.warning(
                        f"Model architecture mismatch: {error_msg[:100]}... "
                        "Training new model."
                    )
                else:
                    raise
        else:
            if model_path is not None:
                logger.info(f"Model file not found at {model_path}")
            else:
                logger.info("No model path specified")

        # Train new model if loading failed or no file exists
        self._train_new_model(progress_callback=progress_callback)

    def _train_new_model(self, progress_callback=None) -> None:
        """Train a new autoencoder model from scratch.

        Uses the training parameters from spectral_select.models.training with
        reasonable defaults for wavelength selection analysis. ``progress_callback``
        (epoch, total_epochs, loss) is forwarded to the training loop.
        """
        logger.warning(
            "Training new autoencoder model. "
            "This may take several minutes..."
        )

        from spectral_select.models.training import train_with_masking

        # Determine output directory for training - use same directory as model_path
        if self._config.model_path is not None:
            train_output_dir = self._config.model_path.parent
        elif self._config.output_dir is not None:
            train_output_dir = self._config.output_dir / "model_training"
        else:
            train_output_dir = Path("model_output") / self._config.sample_name

        # Get mask from dataset
        mask = self._dataset.processed_mask

        # Train with config-driven parameters
        self._model, losses = train_with_masking(
            model=self._model,
            dataset=self._dataset,
            num_epochs=self._config.training_epochs,
            learning_rate=self._config.training_lr,
            chunk_size=self._config.training_chunk_size,
            chunk_overlap=self._config.training_chunk_overlap,
            device=str(self._device),
            early_stopping_patience=self._config.training_early_stopping_patience,
            scheduler_patience=self._config.training_scheduler_patience,
            mask=mask,
            output_dir=str(train_output_dir),
            verbose=True,
            progress_callback=progress_callback,
        )

        # Move to device and set to evaluation mode
        self._model = self._model.to(self._device)
        self._model.train(False)

        # Save model if path specified
        model_path = self._config.model_path
        if model_path is not None:
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self._model.state_dict(), model_path)
            logger.info(f"Model saved to {model_path}")

        logger.info(
            f"Model training complete. Final loss: {losses[-1]:.6f}"
        )

    def _setup_baseline(self) -> None:
        """Setup baseline latent representations for perturbation analysis.

        Extracts patches from the spatial domain, validates them against the mask,
        encodes them through the model, and stores baseline latent representations.
        These baselines are used for measuring the effect of wavelength perturbations.
        """
        if self._dataset is None or self._model is None:
            raise RuntimeError(
                "_load_data and _load_or_train_model must be called before _setup_baseline"
            )

        logger.info("Setting up baseline latent representations...")

        # Get all data and spatial dimensions from dataset
        all_data = self._dataset.get_all_data()
        height, width = self._dataset.get_spatial_dimensions()
        mask = self._dataset.processed_mask

        # Get patch parameters from config
        patch_size = self._config.patch_size
        stride = self._config.patch_stride
        n_baseline_patches = self._config.n_baseline_patches

        # Find valid patch coordinates (more than 50% valid pixels in mask)
        patch_coords: List[Tuple[int, int]] = []

        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                if mask is not None:
                    patch_mask = mask[y : y + patch_size, x : x + patch_size]
                    valid_ratio = np.sum(patch_mask) / (patch_size * patch_size)
                    if valid_ratio <= 0.5:
                        continue

                patch_coords.append((y, x))
                if len(patch_coords) >= n_baseline_patches:
                    break
            if len(patch_coords) >= n_baseline_patches:
                break

        # If no valid patches found with current patch_size, try smaller sizes
        if not patch_coords and mask is not None:
            for smaller_patch_size in [16, 8, 4]:
                smaller_stride = smaller_patch_size // 2
                for y in range(0, height - smaller_patch_size + 1, smaller_stride):
                    for x in range(0, width - smaller_patch_size + 1, smaller_stride):
                        patch_mask = mask[y : y + smaller_patch_size, x : x + smaller_patch_size]
                        valid_ratio = np.sum(patch_mask) / (smaller_patch_size * smaller_patch_size)
                        if valid_ratio > 0.5:
                            patch_coords.append((y, x))
                            if len(patch_coords) >= n_baseline_patches:
                                break
                    if len(patch_coords) >= n_baseline_patches:
                        break

                if patch_coords:
                    patch_size = smaller_patch_size
                    stride = smaller_stride
                    logger.info(f"Using smaller patch size {patch_size} due to mask coverage")
                    break

        # Final fallback: use any pixel location within mask
        if not patch_coords and mask is not None:
            # Get coordinates of valid mask pixels
            valid_y, valid_x = np.where(mask)
            if len(valid_y) > 0:
                # Use smallest possible patch (1x1) centered on valid pixels
                # But we need at least patch_size, so pad around valid region
                min_y, max_y = valid_y.min(), valid_y.max()
                min_x, max_x = valid_x.min(), valid_x.max()

                # Find a patch_size that fits within the mask region
                mask_height = max_y - min_y + 1
                mask_width = max_x - min_x + 1
                usable_size = min(mask_height, mask_width, patch_size)

                if usable_size >= 2:
                    patch_size = usable_size
                    stride = max(1, usable_size // 2)
                    # Center the patch on the mask region
                    center_y = (min_y + max_y) // 2
                    center_x = (min_x + max_x) // 2
                    start_y = max(0, min(center_y - patch_size // 2, height - patch_size))
                    start_x = max(0, min(center_x - patch_size // 2, width - patch_size))
                    patch_coords.append((start_y, start_x))
                    logger.info(f"Using single patch of size {patch_size} centered on mask region")

        if not patch_coords:
            raise ValueError(
                f"No valid patches found. Mask may be too small or empty. "
                f"Mask has {np.sum(mask) if mask is not None else 0} valid pixels."
            )

        # Store patch coordinates for reproducibility
        self._patch_coords = patch_coords

        logger.info(
            f"Selected {len(patch_coords)} patches of size {patch_size}x{patch_size}"
        )

        # Extract patches for each excitation wavelength
        patches_data: Dict[float, torch.Tensor] = {}
        for ex in all_data:
            patches = []
            for y, x in patch_coords:
                patch = all_data[ex][y : y + patch_size, x : x + patch_size, :]
                patches.append(patch)
            patches_data[ex] = torch.stack(patches)

        # Encode patches through model to get baseline latent representations
        with torch.no_grad():
            # Move patches to device
            patches_data_device = {
                ex: data.to(self._device) for ex, data in patches_data.items()
            }

            # Encode patches
            self._baseline_latent = self._model.encode(patches_data_device)

            # Decode for baseline reconstruction
            self._baseline_reconstruction = self._model.decode(self._baseline_latent)

        logger.info(f"Baseline latent shape: {self._baseline_latent.shape}")

    def _select_important_dimensions(self) -> None:
        """Select the most important latent dimensions for perturbation.

        Delegates the ranking math to ``selection_core.select_important_dimensions``
        (variance / activation / pca). Stores results in self._important_dims as
        List[(score, (c, l, h, w))] — coords have one entry per non-batch latent axis.
        """
        if self._baseline_latent is None:
            raise RuntimeError("_setup_baseline must be called before _select_important_dimensions")

        method = self._config.dimension_selection_method
        logger.info(f"Selecting important dimensions using {method} method...")

        self._important_dims = selection_core.select_important_dimensions(
            self._baseline_latent, method, self._config.n_important_dimensions
        )

        logger.info(f"Selected top {len(self._important_dims)} dimensions")
        # Log top 5 for visibility
        for i, (score, coords) in enumerate(self._important_dims[:5]):
            logger.info(f"  {i + 1}. {coords}: importance = {score:.6f}")

    def _compute_influence_scores(self) -> None:
        """Compute influence scores through latent space perturbation.

        For each important dimension, applies perturbations at various magnitudes
        and directions, measuring how much each emission band changes in the
        reconstruction. Results are accumulated into self._influence_matrix.
        """
        if self._baseline_latent is None or self._important_dims is None:
            raise RuntimeError(
                "_setup_baseline and _select_important_dimensions must be called first"
            )

        logger.info(
            f"Computing influence scores using {self._config.perturbation_method} method..."
        )
        logger.info(
            f"Analyzing {len(self._important_dims)} dimensions with magnitudes "
            f"{self._config.perturbation_magnitudes}"
        )

        # Per-excitation band counts -> the channel-per-group map the shared engine expects.
        channels_per_group = {
            ex: self._model.emission_bands[ex]
            for ex in self._model.excitation_wavelengths
        }

        # Delegate the perturbation accumulation loop to the shared engine.
        self._influence_matrix = selection_core.accumulate_influence(
            self._model.decode,
            list(self._model.excitation_wavelengths),
            channels_per_group,
            self._baseline_latent,
            self._baseline_reconstruction,
            self._important_dims,
            magnitudes=self._config.perturbation_magnitudes,
            directions=self._config.perturbation_directions,
            perturbation_method=self._config.perturbation_method,
        )

        logger.info("Completed perturbation analysis")

    def _normalize_influences(self) -> None:
        """Normalize influence scores. Delegates to selection_core.normalize_influence.

        The HSI Analyzer computes the variance normalization in the data's native float32
        (``variance_float64=False``) to keep byte-identical historical output. Config's
        "max_per_excitation" maps to the engine's domain-agnostic "max_per_group".
        """
        if self._influence_matrix is None:
            raise RuntimeError("_compute_influence_scores must be called first")

        method = self._config.normalization_method
        logger.info(f"Applying {method} normalization...")

        core_method = "max_per_group" if method == "max_per_excitation" else method
        self._influence_matrix = selection_core.normalize_influence(
            self._influence_matrix,
            self._dataset.get_all_data(),
            core_method,
            variance_float64=False,
        )

    def _select_top_bands(self) -> List[WavelengthBand]:
        """Select top wavelength combinations based on influence scores.

        Applies optional diversity constraints (MMR or min_distance) to
        avoid selecting redundant wavelengths.

        Returns:
            List of WavelengthBand objects ordered by rank.
        """
        if self._influence_matrix is None:
            raise RuntimeError(
                "_compute_influence_scores must be called before _select_top_bands"
            )

        logger.info(f"Selecting top {self._config.n_bands_to_select} wavelength combinations...")

        # Build list of all (excitation, emission_idx, emission_wavelength, influence) tuples
        all_combinations: List[Dict[str, Any]] = []

        for ex in self._influence_matrix:
            # Get emission wavelengths for this excitation
            emission_wavelengths = self._dataset.emission_wavelengths.get(ex, None)

            for band_idx, influence in enumerate(self._influence_matrix[ex]):
                if emission_wavelengths and band_idx < len(emission_wavelengths):
                    em_wavelength = emission_wavelengths[band_idx]
                else:
                    em_wavelength = float(band_idx)  # Fallback to index

                all_combinations.append({
                    "excitation": ex,
                    "emission_idx": band_idx,
                    "emission_wavelength": em_wavelength,
                    "influence": float(influence),
                    "rank": 0,  # Will be set later
                })

        # Sort by influence and assign initial ranks
        all_combinations.sort(key=lambda x: x["influence"], reverse=True)
        for i, combo in enumerate(all_combinations):
            combo["rank"] = i + 1

        # Apply diversity constraint if enabled
        if self._config.use_diversity_constraint:
            logger.info(f"Applying diversity constraint: {self._config.diversity_method}")
            if self._config.diversity_method == "mmr":
                selected = self._select_bands_mmr(all_combinations)
            elif self._config.diversity_method == "min_distance":
                selected = self._select_bands_min_distance(all_combinations)
            else:
                # Fallback to simple top-N
                selected = all_combinations[: self._config.n_bands_to_select]
        else:
            selected = all_combinations[: self._config.n_bands_to_select]

        # Convert to WavelengthBand objects with new ranks
        bands: List[WavelengthBand] = []
        for i, combo in enumerate(selected):
            bands.append(
                WavelengthBand(
                    rank=i + 1,
                    excitation_nm=combo["excitation"],
                    emission_nm=combo["emission_wavelength"],
                    emission_band_index=combo["emission_idx"],
                    influence_score=combo["influence"],
                )
            )

        logger.info(f"Selected {len(bands)} bands")
        if bands:
            logger.info(
                f"Influence range: {bands[-1].influence_score:.2e} to "
                f"{bands[0].influence_score:.2e}"
            )

        return bands

    def _select_bands_mmr(
        self, all_combinations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Maximum Marginal Relevance (MMR) selection for wavelength diversity.

        Balances influence (relevance) with spectral diversity by penalizing
        similarity to already-selected bands.

        Args:
            all_combinations: Sorted list of band dictionaries.

        Returns:
            List of selected band dictionaries.
        """
        from sklearn.preprocessing import normalize

        logger.info(f"Using MMR with lambda={self._config.lambda_diversity}")

        # Get full hyperspectral data for computing spectral profiles
        all_data = self._dataset.get_all_data()

        # Build mapping from combination to spectral profile
        band_profiles: Dict[Tuple[float, int], np.ndarray] = {}
        for ex in all_data:
            ex_data = all_data[ex].numpy()
            for band_idx in range(ex_data.shape[-1]):
                # Mean spectral profile across spatial locations (flatten spatial)
                profile = ex_data[:, :, band_idx].flatten()
                key = (ex, band_idx)
                band_profiles[key] = profile

        # Normalize profiles for cosine similarity
        for key in band_profiles:
            profile = band_profiles[key].reshape(1, -1)
            band_profiles[key] = normalize(profile, axis=1).flatten()

        # Start with highest influence band
        selected_bands = [all_combinations[0]]
        selected_keys = [
            (all_combinations[0]["excitation"], all_combinations[0]["emission_idx"])
        ]

        logger.info(
            f"Initial selection: Ex{all_combinations[0]['excitation']:.0f} "
            f"Em{all_combinations[0]['emission_wavelength']:.1f}nm"
        )

        # Maximum influence for normalization
        max_influence = all_combinations[0]["influence"]
        if max_influence < 1e-10:
            max_influence = 1.0  # Avoid division by zero

        # Iterative MMR selection
        while len(selected_bands) < self._config.n_bands_to_select:
            best_mmr_score = -np.inf
            best_combo = None
            best_key = None

            for combo in all_combinations:
                combo_key = (combo["excitation"], combo["emission_idx"])

                # Skip if already selected
                if combo_key in selected_keys:
                    continue

                # Skip if profile not available
                if combo_key not in band_profiles:
                    continue

                # Relevance: normalized influence score
                relevance = combo["influence"] / max_influence

                # Diversity: maximum similarity to any selected band
                max_similarity = 0.0
                for sel_key in selected_keys:
                    if sel_key in band_profiles:
                        # Cosine similarity between spectral profiles
                        similarity = np.dot(
                            band_profiles[combo_key], band_profiles[sel_key]
                        )
                        max_similarity = max(max_similarity, abs(similarity))

                # MMR score: relevance - λ × max_similarity
                mmr_score = relevance - self._config.lambda_diversity * max_similarity

                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_combo = combo
                    best_key = combo_key

            if best_combo is not None:
                selected_bands.append(best_combo)
                selected_keys.append(best_key)

                if len(selected_bands) % 10 == 0:
                    logger.info(
                        f"Selected {len(selected_bands)}/{self._config.n_bands_to_select}: "
                        f"Ex{best_combo['excitation']:.0f} "
                        f"Em{best_combo['emission_wavelength']:.1f}nm "
                        f"(MMR score: {best_mmr_score:.4f})"
                    )
            else:
                # No more valid candidates
                logger.warning(
                    f"Only found {len(selected_bands)} bands with diversity constraint"
                )
                break

        return selected_bands

    def _select_bands_min_distance(
        self, all_combinations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Minimum distance constraint selection.

        Ensures selected wavelengths within the same excitation are at least
        min_distance_nm apart in emission wavelength.

        Args:
            all_combinations: Sorted list of band dictionaries.

        Returns:
            List of selected band dictionaries.
        """
        logger.info(
            f"Using minimum distance constraint: {self._config.min_distance_nm} nm"
        )

        selected_bands: List[Dict[str, Any]] = []

        for combo in all_combinations:
            # Check if this wavelength is far enough from already-selected ones
            is_valid = True

            for selected in selected_bands:
                # Only check distance for same excitation wavelength
                if combo["excitation"] == selected["excitation"]:
                    distance = abs(
                        combo["emission_wavelength"] - selected["emission_wavelength"]
                    )
                    if distance < self._config.min_distance_nm:
                        is_valid = False
                        break

            if is_valid:
                selected_bands.append(combo)

                if len(selected_bands) % 10 == 0:
                    logger.info(
                        f"Selected {len(selected_bands)}/{self._config.n_bands_to_select}: "
                        f"Ex{combo['excitation']:.0f} "
                        f"Em{combo['emission_wavelength']:.1f}nm"
                    )

            if len(selected_bands) >= self._config.n_bands_to_select:
                break

        if len(selected_bands) < self._config.n_bands_to_select:
            logger.warning(
                f"Only found {len(selected_bands)} bands satisfying distance constraint"
            )

        return selected_bands
