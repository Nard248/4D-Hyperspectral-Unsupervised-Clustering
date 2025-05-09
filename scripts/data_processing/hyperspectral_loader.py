"""
Hyperspectral data loader module for loading and visualizing 4D hyperspectral data.
"""

import numpy as np
import os
import glob
import re
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Optional, Tuple, Dict, List, Union, Any
import pickle
import warnings
from sklearn.decomposition import PCA, FastICA, NMF

from .hyperspectral_utils import load_data_from_pickle, save_data_to_pickle


class HyperspectralDataLoader:
    """
    A comprehensive class for loading, processing, and managing hyperspectral data.
    Handles loading from .im3 files, preprocessing, and provides utilities for
    data manipulation and ML model integration.
    """

    def __init__(self,
                 data_path: Optional[str] = None,
                 metadata_path: Optional[str] = None,
                 cutoff_offset: int = 20,
                 use_fiji: bool = True,
                 verbose: bool = True):
        """
        Initialize the data loader.

        Args:
            data_path: Path to the directory containing .im3 files
            metadata_path: Path to the Excel file with exposure metadata
            cutoff_offset: Offset in nm used to define the region around 2*excitation for cutoff
            use_fiji: Whether to use ImageJ/Fiji for loading
            verbose: Whether to print loading progress
        """
        # Convert paths to Path objects for better cross-platform compatibility
        self.data_path = Path(data_path) if data_path else None
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self.cutoff_offset = cutoff_offset
        self.use_fiji = use_fiji
        self.verbose = verbose

        # Will store data after loading
        self.raw_data = {}  # Store raw hyperspectral data
        self.data = {}  # Store processed data
        self.metadata = {
            'processed_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'cutoff_offset': cutoff_offset
        }
        self.excitation_wavelengths = []

        # Initialize ImageJ if needed
        self._ij = None
        if use_fiji and data_path is not None:
            self._init_imagej()

    def _init_imagej(self):
        """Initialize ImageJ/Fiji for loading .im3 files"""
        try:
            import imagej
            if self.verbose:
                print("Initializing ImageJ (Fiji)...")
            self._ij = imagej.init('sc.fiji:fiji')
        except Exception as e:
            warnings.warn(f"Failed to initialize ImageJ: {str(e)}. Direct loading methods will be used instead.")
            self.use_fiji = False

    def load_data(self, apply_cutoff: bool = True,
                  pattern: str = "*.im3", sheet_name: int = 0) -> Dict:
        """
        Load hyperspectral data from .im3 files.

        Args:
            apply_cutoff: Whether to apply spectral cutoff to remove artifacts
            pattern: File pattern for hyperspectral data files
            sheet_name: Sheet name or index in metadata Excel file

        Returns:
            Dictionary of processed data
        """
        if self.data_path is None:
            raise ValueError("data_path must be set")

        # --- Find all hyperspectral files ----------------------------------------------------
        cube_paths = sorted(self.data_path.glob(pattern))
        if len(cube_paths) == 0:
            raise FileNotFoundError(f"No files matching pattern '{pattern}' found in {self.data_path}")

        # --- Read Excel metadata (Excitation | Exposure) if available ------------------------
        exposure_lookup = {}
        if self.metadata_path:
            try:
                df = pd.read_excel(self.metadata_path, sheet_name=sheet_name)
                df.columns = df.columns.str.strip()  # trim any stray spaces

                # build a dict: {excitation_value: exposure_value}
                exposure_lookup = dict(
                    zip(df["Excitation"].astype(float), df["Exposure"])
                )
            except Exception as e:
                warnings.warn(f"Failed to load metadata: {str(e)}")

        # --- Load each hyperspectral file ----------------------------------------------------
        for cube_path in cube_paths:
            cube_name = cube_path.name

            if self.verbose:
                print(f"Loading {cube_name} ...")

            # Extract excitation wavelength from filename
            excitation = float(cube_name.split('.')[0])

            try:
                # Load the data cube
                if self.use_fiji and self._ij is not None:
                    img = self._ij.io().open(str(cube_path))
                    cube = np.array(self._ij.py.from_java(img), dtype=float)
                else:
                    # Fallback loading method 
                    cube = self._load_im3_directly(str(cube_path))

                # Determine emission wavelength range
                em_start = 420.0 if excitation <= 400.0 else excitation + 20.0
                em_end = 720
                step = 10
                num_bands = cube.shape[2] if len(cube.shape) >= 3 else 1

                # Calculate emission wavelength array
                em_arr = [em_start + i * step for i in range(num_bands)]

                # Get exposure value from metadata if available
                expos_val = None
                if excitation in exposure_lookup:
                    expos_val = float(exposure_lookup[excitation])
                elif self.verbose:
                    print(f"  ⚠ no exposure value in metadata for Ex={excitation}")

                # Store raw data
                self.raw_data[str(excitation)] = {
                    "ex": excitation,
                    "em_start": em_start,
                    "em_end": em_end,
                    "step": step,
                    "num_rows": cube.shape[0],
                    "num_cols": cube.shape[1],
                    "num_bands": num_bands,
                    "expos_val": expos_val,
                    "notes": None,
                    "data": cube,
                    "em_arr": em_arr
                }

                # Track excitation wavelengths for later use
                if excitation not in self.excitation_wavelengths:
                    self.excitation_wavelengths.append(excitation)

            except Exception as e:
                warnings.warn(f"Error loading {cube_name}: {str(e)}")
                continue

        # Sort excitation wavelengths
        self.excitation_wavelengths.sort()
        self.metadata['excitation_wavelengths'] = self.excitation_wavelengths

        # Process the data if requested
        if apply_cutoff:
            self._process_data(apply_cutoff=apply_cutoff)

        return self.data

    def _load_im3_directly(self, file_path: str) -> np.ndarray:
        """
        Load .im3 file directly without using ImageJ.
        This is a placeholder that needs to be implemented based on the .im3 format.

        Args:
            file_path: Path to the .im3 file

        Returns:
            Numpy array containing the data cube
        """
        raise NotImplementedError(
            "Direct loading of .im3 files is not implemented. "
            "Please use ImageJ/Fiji for loading or provide your own loading method."
        )

    def _process_data(self, apply_cutoff: bool = True) -> None:
        """
        Process the raw data to apply spectral cutoff.

        Args:
            apply_cutoff: Whether to apply spectral cutoff
        """
        if self.verbose:
            print(f"Processing data with cutoff offset: {self.cutoff_offset}nm...")

        # Process each excitation wavelength
        for excitation in self.excitation_wavelengths:
            excitation_str = str(excitation)
            if excitation_str not in self.raw_data:
                continue

            # Get the data for this excitation
            raw_data_dict = self.raw_data[excitation_str]
            cube = raw_data_dict["data"]
            wavelengths = raw_data_dict["em_arr"]

            # Apply spectral cutoff if requested
            if apply_cutoff:
                filtered_cube, filtered_wavelengths = self.apply_spectral_cutoff(
                    cube, wavelengths, excitation
                )
            else:
                filtered_cube, filtered_wavelengths = cube, wavelengths

            # Store filtered data for this excitation
            self.data[excitation_str] = {
                'cube': filtered_cube,
                'wavelengths': filtered_wavelengths,
                'excitation': excitation,
                'raw': raw_data_dict  # Store reference to raw data
            }

    def apply_spectral_cutoff(self, data: np.ndarray, wavelengths: List[float],
                              excitation: float) -> Tuple[np.ndarray, List[float]]:
        """
        Apply dual spectral cutoff (both Rayleigh and second-order) to remove artifacts.

        Args:
            data: Hyperspectral data cube (height, width, bands)
            wavelengths: Array of emission wavelengths
            excitation: Excitation wavelength

        Returns:
            filtered_data: Data after applying cutoff
            filtered_wavelengths: Wavelengths after applying cutoff
        """
        # Convert wavelengths to numpy array
        wavelengths_arr = np.array(wavelengths)

        # Create a mask to keep valid wavelengths
        keep_mask = np.ones(len(wavelengths_arr), dtype=bool)

        # 1. Apply Rayleigh cutoff - remove wavelengths below (excitation + cutoff_offset)
        rayleigh_cutoff = excitation + self.cutoff_offset
        rayleigh_mask = wavelengths_arr >= rayleigh_cutoff
        keep_mask = np.logical_and(keep_mask, rayleigh_mask)

        # 2. Apply second-order cutoff - remove wavelengths in (2*excitation ± cutoff_offset)
        second_order_min = 2 * excitation - self.cutoff_offset
        second_order_max = 2 * excitation + self.cutoff_offset
        second_order_mask = np.logical_or(wavelengths_arr < second_order_min, wavelengths_arr > second_order_max)
        keep_mask = np.logical_and(keep_mask, second_order_mask)

        # Apply the combined mask to the third dimension (emission wavelengths)
        filtered_data = data[:, :, keep_mask]
        filtered_wavelengths = wavelengths_arr[keep_mask].tolist()

        if self.verbose:
            print(f"Applied dual cutoff for excitation {excitation}nm")
            print(f"Removed wavelengths below {rayleigh_cutoff}nm (Rayleigh cutoff)")
            print(f"Removed wavelengths between {second_order_min}nm and {second_order_max}nm (second-order cutoff)")
            print(f"Original data shape: {data.shape}, filtered shape: {filtered_data.shape}")

        return filtered_data, filtered_wavelengths

    def get_cube(self, excitation: float, processed: bool = True) -> Tuple[np.ndarray, List[float]]:
        """
        Get data cube for a specific excitation wavelength.

        Args:
            excitation: Excitation wavelength
            processed: Whether to return processed (cutoff applied) or raw data

        Returns:
            Data cube and corresponding emission wavelengths
        """
        excitation_str = str(excitation)

        if processed:
            if excitation_str not in self.data:
                raise ValueError(f"No processed data found for excitation {excitation}")

            return self.data[excitation_str]['cube'], self.data[excitation_str]['wavelengths']
        else:
            if excitation_str not in self.raw_data:
                raise ValueError(f"No raw data found for excitation {excitation}")

            return self.raw_data[excitation_str]['data'], self.raw_data[excitation_str]['em_arr']

    def get_emission_spectrum(self, excitation: float, row: int, col: int,
                              processed: bool = True) -> Tuple[np.ndarray, List[float]]:
        """
        Get emission spectrum for a specific pixel.

        Args:
            excitation: Excitation wavelength
            row, col: Pixel coordinates
            processed: Whether to use processed or raw data

        Returns:
            Emission spectrum and corresponding wavelengths
        """
        cube, wavelengths = self.get_cube(excitation, processed)
        # Access the spectrum for this pixel (bands are in the 3rd dimension)
        spectrum = cube[row, col, :]

        return spectrum, wavelengths

    def get_mean_spectrum(self, excitation: float = None, processed: bool = True,
                          region: Optional[Tuple[int, int, int, int]] = None) -> Tuple[np.ndarray, List[float]]:
        """
        Get mean spectrum averaged over an image or region.

        Args:
            excitation: Excitation wavelength (if None, returns mean over all available excitations)
            processed: Whether to use processed or raw data
            region: Region of interest (tuple of (row_min, row_max, col_min, col_max))

        Returns:
            Mean spectrum and corresponding wavelengths
        """
        if excitation is not None:
            # Get mean spectrum for a specific excitation
            cube, wavelengths = self.get_cube(excitation, processed)

            if region is not None:
                row_min, row_max, col_min, col_max = region
                roi = cube[row_min:row_max, col_min:col_max, :]
                mean_spectrum = np.mean(roi, axis=(0, 1))  # Mean over spatial dimensions
            else:
                mean_spectrum = np.mean(cube, axis=(0, 1))  # Mean over spatial dimensions

            return mean_spectrum, wavelengths
        else:
            # Get mean spectra for all excitations
            mean_spectra = {}
            for ex in self.excitation_wavelengths:
                try:
                    spectrum, wavelengths = self.get_mean_spectrum(ex, processed, region)
                    mean_spectra[ex] = {'spectrum': spectrum, 'wavelengths': wavelengths}
                except Exception:
                    continue

            return mean_spectra

    def visualize_spectrum(self, excitation: float = None, row: Optional[int] = None,
                           col: Optional[int] = None, processed: bool = True,
                           region: Optional[Tuple[int, int, int, int]] = None,
                           ax: Optional[plt.Axes] = None,
                           plot_kwargs: dict = None) -> plt.Figure:
        """
        Visualize emission spectrum for a specific pixel or region.

        Args:
            excitation: Excitation wavelength
            row, col: Pixel coordinates (if None, averages over the image)
            processed: Whether to use processed or raw data
            region: Region of interest (tuple of (row_min, row_max, col_min, col_max))
            ax: Matplotlib axes to plot on
            plot_kwargs: Additional keyword arguments for plotting

        Returns:
            Matplotlib figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure

        # Set default plot kwargs
        if plot_kwargs is None:
            plot_kwargs = {}
        default_kwargs = {'linewidth': 2, 'alpha': 0.8}
        plot_kwargs = {**default_kwargs, **plot_kwargs}

        if excitation is not None:
            # Plot for a specific excitation
            if row is not None and col is not None:
                # Plot spectrum for a specific pixel
                spectrum, wavelengths = self.get_emission_spectrum(excitation, row, col, processed)
                ax.plot(wavelengths, spectrum, label=f'Ex={excitation}nm (Pixel: {row},{col})', **plot_kwargs)
            elif region is not None:
                # Plot mean spectrum for a region
                spectrum, wavelengths = self.get_mean_spectrum(excitation, processed, region)
                row_min, row_max, col_min, col_max = region
                ax.plot(wavelengths, spectrum,
                        label=f'Ex={excitation}nm (Region: {row_min}:{row_max}, {col_min}:{col_max})',
                        **plot_kwargs)
            else:
                # Plot mean spectrum for the whole image
                spectrum, wavelengths = self.get_mean_spectrum(excitation, processed)
                ax.plot(wavelengths, spectrum, label=f'Ex={excitation}nm (Mean)', **plot_kwargs)
        else:
            # Plot for all excitations
            for ex in self.excitation_wavelengths:
                try:
                    if row is not None and col is not None:
                        spectrum, wavelengths = self.get_emission_spectrum(ex, row, col, processed)
                        ax.plot(wavelengths, spectrum, label=f'Ex={ex}nm (Pixel: {row},{col})', **plot_kwargs)
                    elif region is not None:
                        spectrum, wavelengths = self.get_mean_spectrum(ex, processed, region)
                        row_min, row_max, col_min, col_max = region
                        ax.plot(wavelengths, spectrum,
                                label=f'Ex={ex}nm (Region: {row_min}:{row_max}, {col_min}:{col_max})',
                                **plot_kwargs)
                    else:
                        spectrum, wavelengths = self.get_mean_spectrum(ex, processed)
                        ax.plot(wavelengths, spectrum, label=f'Ex={ex}nm (Mean)', **plot_kwargs)
                except Exception:
                    continue

        ax.set_xlabel('Emission Wavelength (nm)')
        ax.set_ylabel('Intensity')
        ax.set_title('Emission Spectrum')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig

    def visualize_cutoff(self, excitation: float) -> plt.Figure:
        """
        Visualize the effect of spectral cutoff for a specific excitation wavelength.

        Args:
            excitation: Excitation wavelength

        Returns:
            Matplotlib figure
        """
        excitation_str = str(excitation)

        # Check if the data is available
        if excitation_str not in self.raw_data:
            raise ValueError(f"No raw data found for excitation {excitation}")

        if excitation_str not in self.data:
            raise ValueError(f"No processed data found for excitation {excitation}")

        # Get raw and processed data
        raw_data = self.raw_data[excitation_str]
        processed_data = self.data[excitation_str]

        # Get full and filtered data
        full_data = raw_data["data"]
        full_wavelengths = raw_data["em_arr"]
        filtered_data = processed_data["cube"]
        filtered_wavelengths = processed_data["wavelengths"]

        # Calculate mean spectra
        full_mean = np.mean(full_data, axis=(0, 1))
        filtered_mean = np.mean(filtered_data, axis=(0, 1))

        # Create figure showing cutoff
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot full spectrum with cutoff region highlighted
        ax.plot(full_wavelengths, full_mean, 'b-', label='Full Spectrum', linewidth=2)

        # Highlight Rayleigh cutoff region
        rayleigh_cutoff = excitation + self.cutoff_offset
        ax.axvspan(excitation - 5, rayleigh_cutoff, color='r', alpha=0.2,
                   label=f'Rayleigh Cutoff Region (Ex: {excitation}nm + {self.cutoff_offset}nm)')
        ax.axvline(x=rayleigh_cutoff, color='r', linestyle='--')

        # Highlight second-order region
        second_order_min = 2 * excitation - self.cutoff_offset
        second_order_max = 2 * excitation + self.cutoff_offset
        ax.axvspan(second_order_min, second_order_max, color='y', alpha=0.1,
                   label=f'Second-Order Region (2×Ex ± {self.cutoff_offset}nm)')

        # Plot filtered spectrum
        ax.plot(filtered_wavelengths, filtered_mean, 'g-', linewidth=2, label='Filtered Spectrum')

        ax.set_xlabel('Emission Wavelength (nm)')
        ax.set_ylabel('Mean Signal')
        ax.set_title(f'Spectral Cutoff Effect - Excitation {excitation}nm')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Highlight excitation wavelength
        ax.axvline(x=excitation, color='orange', linestyle='-', label='Excitation Wavelength')

        plt.tight_layout()
        return fig

    def visualize_image(self, excitation: float, emission_wavelength: Optional[float] = None,
                        processed: bool = True, ax: Optional[plt.Axes] = None,
                        cmap: str = 'viridis', percentile: int = 99,
                        colorbar: bool = True) -> plt.Figure:
        """
        Visualize a single band image.

        Args:
            excitation: Excitation wavelength
            emission_wavelength: Emission wavelength (if None, uses max emission)
            processed: Whether to use processed or raw data
            ax: Matplotlib axes to plot on
            cmap: Colormap to use
            percentile: Percentile for colormap scaling (to avoid outliers)
            colorbar: Whether to show a colorbar

        Returns:
            Matplotlib figure
        """
        # Get the data cube
        cube, wavelengths = self.get_cube(excitation, processed)

        # Determine which emission wavelength to use
        if emission_wavelength is None:
            # Use the wavelength with highest mean signal
            mean_spectrum = np.mean(cube, axis=(0, 1))  # Mean over spatial dimensions
            emission_idx = np.argmax(mean_spectrum)
        else:
            # Find the closest wavelength
            wavelengths_arr = np.array(wavelengths)
            emission_idx = np.argmin(np.abs(wavelengths_arr - emission_wavelength))

        # Extract the image for the selected emission wavelength
        image = cube[:, :, emission_idx]
        used_wavelength = wavelengths[emission_idx]

        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            fig = ax.figure

        # Plot the image with proper scaling
        vmax = np.percentile(image, percentile)
        im = ax.imshow(image, cmap=cmap, vmin=0, vmax=vmax)

        # Add colorbar if requested
        if colorbar:
            fig.colorbar(im, ax=ax, label='Intensity')

        ax.set_title(f'Ex: {excitation}nm, Em: {used_wavelength}nm')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')

        plt.tight_layout()
        return fig

    def visualize_false_color(self, excitation: float, method: str = 'rgb',
                              processed: bool = True, ax: Optional[plt.Axes] = None,
                              percentile: int = 99) -> plt.Figure:
        """
        Create a false color visualization of the data.

        Args:
            excitation: Excitation wavelength
            method: Method for false color ('rgb', 'max', 'mean', 'pca')
            processed: Whether to use processed or raw data
            ax: Matplotlib axes to plot on
            percentile: Percentile for colormap scaling

        Returns:
            Matplotlib figure
        """
        # Get the data cube
        cube, wavelengths = self.get_cube(excitation, processed)

        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            fig = ax.figure

        # Create RGB image based on method
        if method == 'rgb':
            # Use three wavelengths as RGB channels
            # Try to choose wavelengths from different parts of the spectrum
            num_bands = len(wavelengths)
            if num_bands >= 3:
                indices = [int(num_bands * 0.2), int(num_bands * 0.5), int(num_bands * 0.8)]
                r_idx, g_idx, b_idx = indices

                r_band = cube[:, :, r_idx]
                g_band = cube[:, :, g_idx]
                b_band = cube[:, :, b_idx]

                # Scale each channel
                r_scaled = r_band / np.percentile(r_band, percentile)
                g_scaled = g_band / np.percentile(g_band, percentile)
                b_scaled = b_band / np.percentile(b_band, percentile)

                # Clip to [0, 1]
                r_scaled = np.clip(r_scaled, 0, 1)
                g_scaled = np.clip(g_scaled, 0, 1)
                b_scaled = np.clip(b_scaled, 0, 1)

                # Create RGB image
                rgb = np.stack([r_scaled, g_scaled, b_scaled], axis=2)

                ax.imshow(rgb)
                ax.set_title(f'RGB False Color (Ex: {excitation}nm)\n'
                             f'R: {wavelengths[r_idx]}nm, G: {wavelengths[g_idx]}nm, B: {wavelengths[b_idx]}nm')
            else:
                raise ValueError("Not enough bands for RGB visualization. Try 'max' or 'mean' method.")

        elif method == 'max':
            # Create a color image from max projection along wavelength
            max_proj = np.max(cube, axis=2)  # Max along emission wavelength axis
            vmax = np.percentile(max_proj, percentile)
            ax.imshow(max_proj, cmap='viridis', vmin=0, vmax=vmax)
            ax.set_title(f'Max Projection (Ex: {excitation}nm)')
            plt.colorbar(ax.images[0], ax=ax, label='Max Intensity')

        elif method == 'mean':
            # Create a color image from mean projection along wavelength
            mean_proj = np.mean(cube, axis=2)  # Mean along emission wavelength axis
            vmax = np.percentile(mean_proj, percentile)
            ax.imshow(mean_proj, cmap='viridis', vmin=0, vmax=vmax)
            ax.set_title(f'Mean Projection (Ex: {excitation}nm)')
            plt.colorbar(ax.images[0], ax=ax, label='Mean Intensity')

        elif method == 'pca':
            # Apply PCA to the spectral dimension and visualize first 3 components as RGB
            # Reshape the cube for PCA
            height, width, num_bands = cube.shape
            reshaped_cube = cube.reshape(height * width, num_bands)  # (height*width, bands)

            # Apply PCA
            pca = PCA(n_components=3)
            pca_result = pca.fit_transform(reshaped_cube)

            # Reshape back to image format
            pca_result = pca_result.reshape(height, width, 3)

            # Scale the result for display
            for i in range(3):
                channel = pca_result[:, :, i]
                pca_result[:, :, i] = np.clip(
                    (channel - np.min(channel)) / (np.percentile(channel, percentile) - np.min(channel)),
                    0, 1
                )

            ax.imshow(pca_result)
            ax.set_title(f'PCA False Color (Ex: {excitation}nm)')

            # Add a text for variance explained
            var_explained = pca.explained_variance_ratio_
            ax.text(0.02, 0.98,
                    f'Variance explained:\nPC1: {var_explained[0]:.2%}\nPC2: {var_explained[1]:.2%}\nPC3: {var_explained[2]:.2%}',
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        else:
            raise ValueError(f"Unknown method: {method}. Try 'rgb', 'max', 'mean', or 'pca'.")

        ax.set_xlabel('Column')
        ax.set_ylabel('Row')

        plt.tight_layout()
        return fig

    def get_features_for_ml(self, method: str = 'flatten', n_components: Optional[int] = None,
                            excitation: Optional[float] = None, preserve_full_data: bool = False) -> np.ndarray:
        """
        Prepare features for machine learning models.

        Args:
            method: Feature extraction method ('flatten', 'pca', 'mean', 'max', etc.)
            n_components: Number of components to keep (for dimensionality reduction)
            excitation: Specific excitation wavelength to use (if None, uses all)

        Returns:
            Features array ready for ML models
        """
        if not self.data:
            raise ValueError("No data available. Load data first.")

        if preserve_full_data:
            method = 'flatten'

        if excitation is not None:
            excitation_str = str(excitation)
            if excitation_str not in self.data:
                raise ValueError(f"No data found for excitation {excitation}")

            cube = self.data[excitation_str]['cube']

            # Extract features based on method
            if method == 'flatten':
                # Reshape the cube to have pixels as rows and spectrum as features
                height, width, bands = cube.shape
                features = cube.reshape(height * width, bands)  # (height*width, bands)

            elif method == 'mean':
                # Use mean spectrum for each pixel
                height, width, _ = cube.shape
                features = np.mean(cube, axis=2).reshape(-1, 1)  # (height*width, 1)

            elif method == 'max':
                # Use max value for each pixel
                height, width, _ = cube.shape
                features = np.max(cube, axis=2).reshape(-1, 1)  # (height*width, 1)

            elif method == 'pca':
                # Apply PCA to the spectral dimension
                height, width, bands = cube.shape
                reshaped_cube = cube.reshape(height * width, bands)  # (height*width, bands)

                # Determine number of components
                if n_components is None:
                    n_components = min(bands, 10)  # Default to min(bands, 10)

                # Apply PCA
                pca = PCA(n_components=n_components)
                features = pca.fit_transform(reshaped_cube)  # (height*width, n_components)

            else:
                raise ValueError(f"Unknown method: {method}. Try 'flatten', 'mean', 'max', or 'pca'.")

        else:
            all_features = []

            for ex_str in self.excitation_wavelengths:
                ex_str = str(ex_str)
                if ex_str in self.data:
                    ex_features = self.get_features_for_ml(method, n_components, float(ex_str))
                    all_features.append(ex_features)

            features = np.hstack(all_features)

        return features

    def apply_dimensionality_reduction(self, method: str = 'pca', n_components: int = 10,
                                       excitation: Optional[float] = None,
                                       preserve_full_data: bool = False) -> Dict:
        """
        Apply dimensionality reduction to the hyperspectral data.

        Args:
            method: Dimensionality reduction method ('pca', 'ica', 'nmf')
            n_components: Number of components to keep
            excitation: Specific excitation wavelength to use (if None, uses all)
            preserve_full_data: If True, returns original data without reduction

        Returns:
            Dictionary with reduced data and components (or original data if preserve_full_data=True)
        """
        if not self.data:
            raise ValueError("No data available. Load data first.")

        # If specific excitation is provided, use only that
        if excitation is not None:
            excitation_str = str(excitation)
            if excitation_str not in self.data:
                raise ValueError(f"No data found for excitation {excitation}")

            cube = self.data[excitation_str]['cube']

            # If preserve_full_data is True, return the original data without reduction
            if preserve_full_data:
                return {
                    'reduced_cube': cube,  # This is actually the full cube
                    'components': None,  # No components when preserving full data
                    'model': None,  # No model when preserving full data
                    'original_data': True  # Flag to indicate this is original data
                }

            # Reshape the cube for dimensionality reduction
            height, width, num_bands = cube.shape
            reshaped_cube = cube.reshape(height * width, num_bands)  # (pixels, bands)

            # Apply dimensionality reduction
            if method == 'pca':
                model = PCA(n_components=n_components)
            elif method == 'ica':
                model = FastICA(n_components=n_components, random_state=42)
            elif method == 'nmf':
                model = NMF(n_components=n_components, random_state=42)
            else:
                raise ValueError(f"Unknown method: {method}. Try 'pca', 'ica', or 'nmf'.")

            # Fit and transform the data
            reduced_data = model.fit_transform(reshaped_cube)  # (height*width, n_components)

            # Reshape reduced data back to image format for each component
            reduced_cube = []
            for i in range(n_components):
                component_image = reduced_data[:, i].reshape(height, width)
                reduced_cube.append(component_image)

            reduced_cube = np.array(reduced_cube)  # (n_components, height, width)

            # Get the components/basis vectors
            if method == 'pca':
                components = model.components_  # (n_components, bands)
                explained_variance = model.explained_variance_ratio_
                return {
                    'reduced_cube': reduced_cube,
                    'components': components,
                    'explained_variance': explained_variance,
                    'model': model,
                    'original_data': False
                }
            else:
                components = model.components_  # (n_components, bands)
                return {
                    'reduced_cube': reduced_cube,
                    'components': components,
                    'model': model,
                    'original_data': False
                }

        else:
            # Use all excitations
            # Apply dim reduction to each excitation separately and return a dictionary
            results = {}

            if preserve_full_data:
                # If preserving full data, just collect original data for all excitations
                for ex in self.excitation_wavelengths:
                    ex_str = str(ex)
                    if ex_str in self.data:
                        results[ex_str] = {
                            'reduced_cube': self.data[ex_str]['cube'],  # Full cube
                            'components': None,
                            'model': None,
                            'original_data': True
                        }
                return results
            else:
                # Otherwise apply dimensionality reduction to each excitation
                for ex in self.excitation_wavelengths:
                    try:
                        ex_result = self.apply_dimensionality_reduction(method, n_components, ex, preserve_full_data)
                        results[str(ex)] = ex_result
                    except Exception as e:
                        warnings.warn(f"Error reducing dimensionality for excitation {ex}: {str(e)}")

                return results

    def save_to_pkl(self, output_file: str = 'hyperspectral_data.pkl') -> None:
        """
        Save the processed data to a pickle file.

        Args:
            output_file: Path to the output file
        """
        output_data = {
            'data': self.data,
            'raw_data': self.raw_data,
            'metadata': self.metadata,
            'excitation_wavelengths': self.excitation_wavelengths,
            'cutoff_offset': self.cutoff_offset
        }

        save_data_to_pickle(output_data, output_file)

        if self.verbose:
            print(f"Data saved to {output_file}")

    def load_from_pkl(self, input_file: str) -> Dict:
        """
        Load processed data from a pickle file.

        Args:
            input_file: Path to the input file

        Returns:
            Loaded data
        """
        try:
            data = load_data_from_pickle(input_file)

            # Update class attributes
            self.data = data.get('data', {})
            self.raw_data = data.get('raw_data', {})
            self.metadata = data.get('metadata', {})
            self.excitation_wavelengths = data.get('excitation_wavelengths', [])
            self.cutoff_offset = data.get('cutoff_offset', self.cutoff_offset)

            if self.verbose:
                print(f"Data loaded from {input_file}")
                print(f"Number of excitation wavelengths: {len(self.excitation_wavelengths)}")
                print(f"Excitation wavelengths: {self.excitation_wavelengths}")

            return data

        except Exception as e:
            raise IOError(f"Error loading data from {input_file}: {str(e)}")

    def get_summary(self) -> Dict:
        """
        Get a summary of the loaded data.

        Returns:
            Dictionary with summary information
        """
        summary = {
            'excitation_wavelengths': self.excitation_wavelengths,
            'num_excitations': len(self.excitation_wavelengths),
            'processed_date': self.metadata.get('processed_date', None),
            'cutoff_offset': self.cutoff_offset,
            'excitation_details': {}
        }

        # Add details for each excitation wavelength
        for ex in self.excitation_wavelengths:
            ex_str = str(ex)
            if ex_str in self.data:
                cube = self.data[ex_str]['cube']
                wavelengths = self.data[ex_str]['wavelengths']

                summary['excitation_details'][ex_str] = {
                    'cube_shape': cube.shape,
                    'emission_wavelengths': wavelengths,
                    'emission_range': (min(wavelengths), max(wavelengths)),
                    'num_emission_bands': len(wavelengths)
                }

        return summary

    def print_summary(self) -> None:
        """Print a summary of the loaded data."""
        summary = self.get_summary()

        print("\nHyperspectral Data Summary:")
        print(f"Number of excitation wavelengths: {summary['num_excitations']}")
        print(f"Excitation wavelengths: {summary['excitation_wavelengths']}")
        print(f"Processed date: {summary['processed_date']}")
        print(f"Cutoff offset: {summary['cutoff_offset']} nm")

        print("\nExcitation Details:")
        for ex, details in summary['excitation_details'].items():
            print(f"\n  Excitation {ex} nm:")
            print(f"    Cube shape: {details['cube_shape']}")
            print(f"    Emission range: {details['emission_range']} nm")
            print(f"    Number of emission bands: {details['num_emission_bands']}")