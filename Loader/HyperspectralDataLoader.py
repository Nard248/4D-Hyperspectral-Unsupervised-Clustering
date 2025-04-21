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
from scipy import ndimage
from sklearn.decomposition import PCA, FastICA, NMF
import copy
import warnings

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
        self.data_path = data_path
        self.metadata_path = metadata_path
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
            apply_cutoff: Whether to apply spectral cutoff to remove second-order scattering artifacts
            pattern: File pattern for hyperspectral data files
            sheet_name: Sheet name or index in metadata Excel file

        Returns:
            Dictionary of processed data
        """
        if self.data_path is None:
            raise ValueError("data_path must be set")

        data_path = Path(self.data_path)

        # --- Find all hyperspectral files ----------------------------------------------------
        cube_paths = sorted(data_path.glob(pattern))
        if len(cube_paths) == 0:
            raise FileNotFoundError(f"No files matching pattern '{pattern}' found in {data_path}")

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
                    # Fallback loading method using numpy
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
        Attempt to parse the binary format.

        Args:
            file_path: Path to the .im3 file

        Returns:
            Numpy array containing the data cube
        """
        # This is a placeholder - you would need to implement the actual
        # binary file parsing based on the .im3 format specification
        raise NotImplementedError(
            "Direct loading of .im3 files is not implemented. "
            "Please use ImageJ/Fiji for loading or provide your own loading method."
        )

    def _process_data(self, apply_cutoff: bool = True) -> None:
        """
        Process the raw data to apply spectral cutoff to remove second-order scattering artifacts.

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
        Apply spectral cutoff to remove second-order scattering artifacts.
        Filters emission spectra to remove wavelengths in the second-order scattering region
        (2*excitation ± cutoff_offset).

        Args:
            data: Hyperspectral data cube (height, width, bands)
            wavelengths: Array of emission wavelengths
            excitation: Excitation wavelength

        Returns:
            filtered_data: Data after applying cutoff
            filtered_wavelengths: Wavelengths after applying cutoff
        """
        # Convert wavelengths to numpy array if it's not already
        wavelengths_arr = np.array(wavelengths)

        # Create a mask to keep valid wavelengths
        keep_mask = np.ones(len(wavelengths_arr), dtype=bool)

        # Remove wavelengths in the second-order zone (2*excitation ± cutoff_offset)
        second_order_min = 2 * excitation - self.cutoff_offset
        second_order_max = 2 * excitation + self.cutoff_offset
        second_order_mask = np.logical_or(wavelengths_arr < second_order_min, wavelengths_arr > second_order_max)
        keep_mask = np.logical_and(keep_mask, second_order_mask)

        # Apply the mask to the third dimension (emission wavelengths)
        filtered_data = data[:, :, keep_mask]
        filtered_wavelengths = wavelengths_arr[keep_mask].tolist()

        if self.verbose:
            print(f"Applied cutoff for excitation {excitation}nm")
            print(f"Removed wavelengths between {second_order_min}nm and {second_order_max}nm")
            print(f"Original data shape: {data.shape}, filtered shape: {filtered_data.shape}")

        return filtered_data, filtered_wavelengths

    # Method removed as reflectance cube concept is no longer needed

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

    def get_excitation_spectrum(self, emission_idx: int, row: int, col: int,
                                processed: bool = True) -> Tuple[np.ndarray, List[float]]:
        """
        Get excitation spectrum for a specific pixel and emission wavelength.

        Args:
            emission_idx: Index of emission wavelength relative to the filtered wavelengths
            row, col: Pixel coordinates
            processed: Whether to use processed or raw data

        Returns:
            Excitation spectrum and corresponding wavelengths
        """
        # Initialize arrays to store the spectrum
        spectrum = []
        ex_wavelengths = []

        # Loop through all excitation wavelengths
        for ex in self.excitation_wavelengths:
            try:
                cube, wavelengths = self.get_cube(ex, processed)

                # Check if this emission index is valid for this cube
                if emission_idx < cube.shape[0]:
                    spectrum.append(cube[emission_idx, row, col])
                    ex_wavelengths.append(ex)
            except Exception:
                continue

        return np.array(spectrum), ex_wavelengths

    def get_emission_wavelength_index(self, excitation: float,
                                      emission_wavelength: float,
                                      processed: bool = True) -> int:
        """
        Get the index of a specific emission wavelength for a given excitation.

        Args:
            excitation: Excitation wavelength
            emission_wavelength: Target emission wavelength
            processed: Whether to use processed or raw data

        Returns:
            Index of the closest emission wavelength
        """
        _, wavelengths = self.get_cube(excitation, processed)
        wavelengths_arr = np.array(wavelengths)

        # Find the closest wavelength
        closest_idx = np.argmin(np.abs(wavelengths_arr - emission_wavelength))

        return closest_idx

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
        full_mean = np.mean(full_data, axis=(1, 2))
        filtered_mean = np.mean(filtered_data, axis=(1, 2))

        # Create figure showing cutoff
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot full spectrum with cutoff region highlighted
        ax.plot(full_wavelengths, full_mean, 'b-', label='Full Spectrum', linewidth=2)

        # Highlight cutoff region
        cutoff_wavelength = excitation + self.cutoff_offset
        ax.axvspan(excitation - 5, cutoff_wavelength, color='r', alpha=0.2,
                   label=f'Cutoff Region (Ex: {excitation}nm + {self.cutoff_offset}nm)')
        ax.axvline(x=cutoff_wavelength, color='r', linestyle='--')

        # Plot filtered spectrum
        ax.plot(filtered_wavelengths, filtered_mean, 'g-', linewidth=2, label='Filtered Spectrum')

        ax.set_xlabel('Emission Wavelength (nm)')
        ax.set_ylabel('Mean Signal')
        ax.set_title(f'Spectral Cutoff Effect - Excitation {excitation}nm')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Highlight excitation wavelength
        ax.axvline(x=excitation, color='orange', linestyle='-', label='Excitation Wavelength')

        # Highlight second-order region
        second_order_min = 2 * excitation - self.cutoff_offset
        second_order_max = 2 * excitation + self.cutoff_offset
        ax.axvspan(second_order_min, second_order_max, color='y', alpha=0.1,
                   label=f'Second-Order Region (2×Ex ± {self.cutoff_offset}nm)')

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
            emission_idx = self.get_emission_wavelength_index(excitation, emission_wavelength, processed)

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

    def get_data_as_dataframe(self) -> pd.DataFrame:
        """
        Convert the data structure to a pandas DataFrame for easier analysis.

        Returns:
            DataFrame representation of the data
        """
        # Create a dictionary that will be converted to a DataFrame
        df_data = {}

        # Add each excitation wavelength data
        for excitation_key, data_dict in self.data.items():
            df_data[excitation_key] = data_dict['cube']

        # Create a DataFrame
        return pd.DataFrame(df_data)

    def get_features_for_ml(self, method: str = 'flatten', n_components: Optional[int] = None,
                            excitation: Optional[float] = None) -> np.ndarray:
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

        # If specific excitation is provided, use only that
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
            # Use all excitations - stack features from each excitation
            all_features = []

            for ex_str in self.excitation_wavelengths:
                ex_str = str(ex_str)
                if ex_str in self.data:
                    # Get features for this excitation
                    ex_features = self.get_features_for_ml(method, n_components, float(ex_str))
                    all_features.append(ex_features)

            # Stack all features side by side (assuming same spatial dimensions)
            features = np.hstack(all_features)

        return features

    def apply_dimensionality_reduction(self, method: str = 'pca', n_components: int = 10,
                                       excitation: Optional[float] = None) -> Dict:
        """
        Apply dimensionality reduction to the hyperspectral data.

        Args:
            method: Dimensionality reduction method ('pca', 'ica', 'nmf')
            n_components: Number of components to keep
            excitation: Specific excitation wavelength to use (if None, uses all)

        Returns:
            Dictionary with reduced data and components
        """
        if not self.data:
            raise ValueError("No data available. Load data first.")

        # If specific excitation is provided, use only that
        if excitation is not None:
            excitation_str = str(excitation)
            if excitation_str not in self.data:
                raise ValueError(f"No data found for excitation {excitation}")

            cube = self.data[excitation_str]['cube']

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
                    'model': model
                }
            else:
                components = model.components_  # (n_components, bands)
                return {
                    'reduced_cube': reduced_cube,
                    'components': components,
                    'model': model
                }

        else:
            # Use all excitations
            # We'll apply dim reduction to each excitation separately and return a list
            results = {}

            for ex in self.excitation_wavelengths:
                try:
                    ex_result = self.apply_dimensionality_reduction(method, n_components, ex)
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

        with open(output_file, 'wb') as f:
            pickle.dump(output_data, f)

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
            with open(input_file, 'rb') as f:
                data = pickle.load(f)

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
        print(f"Cutoff offset (for 2nd order scattering): {summary['cutoff_offset']} nm")

        print("\nExcitation Details:")
        for ex, details in summary['excitation_details'].items():
            print(f"\n  Excitation {ex} nm:")
            print(f"    Cube shape: {details['cube_shape']}")
            print(f"    Emission range: {details['emission_range']} nm")
            print(f"    Number of emission bands: {details['num_emission_bands']}")


def create_excitation_emission_dataframe(data_dict: Dict,
                                        sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Transform 4D hyperspectral data into a 2D dataframe.

    Args:
        data_dict: Dictionary containing hyperspectral data
        sample_size: Optional number of random pixels to sample (for large datasets)

    Returns:
        DataFrame with x, y coordinates and intensity values for each valid excitation-emission combination
    """
    # First, collect all valid excitation-emission combinations
    valid_combinations = []
    all_excitations = []

    # Check what excitations we actually have in the data
    for ex_str in data_dict['data'].keys():
        excitation = float(ex_str)
        all_excitations.append(excitation)

        # Get the valid emission wavelengths for this excitation
        emissions = data_dict['data'][ex_str]['wavelengths']

        # Add all valid combinations to our list
        for emission in emissions:
            col_name = f"{int(emission)}-{int(excitation)}"
            valid_combinations.append((excitation, emission, col_name))

    print(f"Found {len(all_excitations)} excitation wavelengths")
    print(f"Generated {len(valid_combinations)} valid excitation-emission combinations")

    # Create an empty dataframe with x, y coordinates
    # First, determine the dimensions of our data
    first_ex = str(all_excitations[0])
    cube_shape = data_dict['data'][first_ex]['cube'].shape
    height, width = cube_shape[0], cube_shape[1]

    print(f"Image dimensions: {height} x {width} pixels")

    # Initialize the dataframe with columns for x and y coordinates
    total_pixels = height * width

    # Create coordinate arrays - this is the correct way to flatten spatial dimensions
    # Create a meshgrid of coordinates
    y_coords, x_coords = np.mgrid[0:height, 0:width]

    # Flatten the coordinates
    x_coords = x_coords.flatten()
    y_coords = y_coords.flatten()

    # Create initial dataframe with coordinates
    df = pd.DataFrame({
        'x': x_coords,
        'y': y_coords
    })

    # If sample_size is provided, take a random sample of pixels
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
        print(f"Sampled {sample_size} pixels out of {total_pixels}")

    print(f"Created initial dataframe with {len(df)} rows")

    # Now, fill in the intensity values for each valid combination
    for excitation, emission, col_name in valid_combinations:
        # Get the data cube for this excitation
        ex_str = str(excitation)
        cube = data_dict['data'][ex_str]['cube']
        wavelengths = data_dict['data'][ex_str]['wavelengths']

        # Find the index of this emission wavelength
        try:
            em_idx = wavelengths.index(emission)

            # Extract the intensity values for this emission wavelength
            # For the sampled rows only
            if sample_size is not None and sample_size < total_pixels:
                # Get the x, y coordinates of the sampled pixels
                sampled_coords = df[['x', 'y']].values
                # Extract intensity values for these coordinates
                intensities = [cube[y, x, em_idx] for x, y in zip(sampled_coords[:, 0], sampled_coords[:, 1])]
                df[col_name] = intensities
            else:
                # Extract for all pixels - flatten in the same order as the coordinates
                intensities = cube[:, :, em_idx].flatten()
                df[col_name] = intensities

        except ValueError:
            # This emission wavelength doesn't exist for this excitation
            # We're skipping it as requested instead of adding NaN values
            continue

    print(f"Final dataframe has {len(df.columns)} columns")
    return df

def load_data_and_create_df(pickle_file: str, sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Load data from pickle file and create the dataframe

    Args:
        pickle_file: Path to the pickle file
        sample_size: Optional number of random pixels to sample

    Returns:
        Transformed dataframe
    """
    # Load the data
    with open(pickle_file, 'rb') as f:
        data_dict = pickle.load(f)

    # Create the dataframe
    return create_excitation_emission_dataframe(data_dict, sample_size)

def save_dataframe(df: pd.DataFrame, output_file: str) -> None:
    """Save the dataframe to a file"""
    print(f"Saving dataframe to {output_file}")

    # Determine file extension and save accordingly
    ext = Path(output_file).suffix
    if ext == '.csv':
        df.to_csv(output_file, index=False)
    elif ext == '.parquet':
        df.to_parquet(output_file, index=False)
    elif ext == '.pkl' or ext == '.pickle':
        df.to_pickle(output_file)
    else:
        print(f"Unrecognized extension {ext}, saving as pickle")
        df.to_pickle(output_file)

    print(f"Saved dataframe with {len(df)} rows and {len(df.columns)} columns")

def normalize_hyperspectral_data(
    data_dict: Dict,
    reference_type: str = 'min',
    output_file: Optional[str] = None
) -> Dict:
    """
    Normalize hyperspectral data based on exposure time.

    Args:
        data_dict: Dictionary containing hyperspectral data with exposure time in metadata
        reference_type: Type of reference exposure time ('min', 'max', or float value)
        output_file: Path to save the normalized data pickle file (optional)

    Returns:
        Dictionary containing normalized hyperspectral data
    """
    print(f"Normalizing hyperspectral data using {reference_type} exposure as reference...")

    # Create a deep copy of the data to avoid modifying the original
    normalized_data = copy.deepcopy(data_dict)

    # Extract exposure times for each excitation wavelength
    exposure_times = {}

    for ex_str in data_dict['data'].keys():
        # Try to get exposure time from different possible locations in the data structure
        if 'raw' in data_dict['data'][ex_str] and 'expos_val' in data_dict['data'][ex_str]['raw']:
            exposure_times[ex_str] = data_dict['data'][ex_str]['raw']['expos_val']
        elif 'expos_val' in data_dict['data'][ex_str]:
            exposure_times[ex_str] = data_dict['data'][ex_str]['expos_val']

    if not exposure_times:
        raise ValueError("Could not find exposure time information in the data")

    print(f"Found exposure times for {len(exposure_times)} excitation wavelengths")

    # Determine the reference exposure time
    if reference_type == 'min':
        reference_exposure = min(exposure_times.values())
        print(f"Using minimum exposure time as reference: {reference_exposure}")
    elif reference_type == 'max':
        reference_exposure = max(exposure_times.values())
        print(f"Using maximum exposure time as reference: {reference_exposure}")
    elif isinstance(reference_type, (int, float)):
        reference_exposure = float(reference_type)
        print(f"Using provided exposure time as reference: {reference_exposure}")
    else:
        raise ValueError("Invalid reference_type. Use 'min', 'max', or a float value.")

    # Store the normalization information in metadata
    if 'metadata' not in normalized_data:
        normalized_data['metadata'] = {}

    normalized_data['metadata']['normalization'] = {
        'reference_type': reference_type,
        'reference_exposure': reference_exposure,
        'original_exposures': exposure_times
    }

    # Normalize each data cube
    print("Normalizing data cubes...")
    for ex_str, exposure in exposure_times.items():
        # Calculate normalization factor: E₁/E₂
        normalization_factor = reference_exposure / exposure

        # Apply normalization to the data cube
        original_cube = data_dict['data'][ex_str]['cube']

        # Normalize: I_ij^norm = I_ij × (E₁/E₂)
        normalized_data['data'][ex_str]['cube'] = original_cube * normalization_factor

        # Store normalization factor in metadata
        normalized_data['data'][ex_str]['normalization_factor'] = normalization_factor

        print(f"  Normalized excitation {ex_str}nm (Exposure: {exposure}, Factor: {normalization_factor:.4f})")

    # Save the normalized data if output file is provided
    if output_file:
        with open(output_file, 'wb') as f:
            pickle.dump(normalized_data, f)
        print(f"Normalized data saved to {output_file}")

    return normalized_data

def print_exposure_info(data_dict: Dict) -> None:
    """
    Print exposure time information from the data dictionary.

    Args:
        data_dict: Dictionary containing hyperspectral data
    """
    print("\nExposure Time Information:")

    exposure_times = {}

    for ex_str in data_dict['data'].keys():
        # Try to get exposure time from different possible locations
        if 'raw' in data_dict['data'][ex_str] and 'expos_val' in data_dict['data'][ex_str]['raw']:
            exposure_times[ex_str] = data_dict['data'][ex_str]['raw']['expos_val']
        elif 'expos_val' in data_dict['data'][ex_str]:
            exposure_times[ex_str] = data_dict['data'][ex_str]['expos_val']

    if not exposure_times:
        print("No exposure time information found in the data")
        return

    # Convert to sorted list of tuples
    sorted_exposures = sorted([(float(ex), exp) for ex, exp in exposure_times.items()])

    print(f"{'Excitation (nm)':<15} {'Exposure Time':<15}")
    print("-" * 30)

    for ex, exp in sorted_exposures:
        print(f"{ex:<15.1f} {exp:<15}")

    print("\nSummary:")
    print(f"Minimum exposure: {min(exposure_times.values())}")
    print(f"Maximum exposure: {max(exposure_times.values())}")
    print(f"Ratio max/min: {max(exposure_times.values()) / min(exposure_times.values()):.2f}")

def normalize_and_save_both_versions(
    input_file: str,
    output_dir: Optional[str] = None
) -> Tuple[Path, Path]:
    """
    Load data, normalize it using both min and max exposure times, and save both versions.

    Args:
        input_file: Path to the input pickle file
        output_dir: Directory to save the output files (default: same as input file)

    Returns:
        Tuple of (up_normalized_data, down_normalized_data)
    """
    # Load the data
    print(f"Loading data from {input_file}...")
    with open(input_file, 'rb') as f:
        data_dict = pickle.load(f)

    # Print exposure information
    print_exposure_info(data_dict)

    # Set up output directory
    input_path = Path(input_file)
    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Create output file names
    base_name = input_path.stem
    up_output_file = output_dir / f"{base_name}_normalized_exposure_up.pkl"
    down_output_file = output_dir / f"{base_name}_normalized_exposure_down.pkl"

    # Normalize up (using max exposure as reference)
    up_normalized_data = normalize_hyperspectral_data(
        data_dict,
        reference_type='max',
        output_file=str(up_output_file)
    )

    # Normalize down (using min exposure as reference)
    down_normalized_data = normalize_hyperspectral_data(
        data_dict,
        reference_type='min',
        output_file=str(down_output_file)
    )

    print("\nNormalization complete!")
    print(f"Up-normalized data (max exposure reference) saved to: {up_output_file}")
    print(f"Down-normalized data (min exposure reference) saved to: {down_output_file}")

    return up_output_file, down_output_file


def load_masked_data_and_create_df(pickle_file: str, sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Load hyperspectral data from pickle file and create a dataframe,
    excluding masked pixels (indicated by NaN values).

    Args:
        pickle_file: Path to the pickle file
        sample_size: Optional number of random pixels to sample from non-masked pixels

    Returns:
        Transformed dataframe with masked pixels excluded
    """
    # Load the data
    with open(pickle_file, 'rb') as f:
        data_dict = pickle.load(f)

    # Create the dataframe excluding masked pixels
    return create_masked_excitation_emission_dataframe(data_dict, sample_size)


def create_masked_excitation_emission_dataframe(data_dict: Dict,
                                                sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Transform 4D hyperspectral data into a 2D dataframe, excluding masked pixels (NaN values).

    Args:
        data_dict: Dictionary containing hyperspectral data
        sample_size: Optional number of random pixels to sample from non-masked pixels

    Returns:
        DataFrame with x, y coordinates and intensity values for non-masked pixels
    """
    # First, collect all valid excitation-emission combinations
    valid_combinations = []
    all_excitations = []

    # Check what excitations we actually have in the data
    for ex_str in data_dict['data'].keys():
        excitation = float(ex_str)
        all_excitations.append(excitation)

        # Get the valid emission wavelengths for this excitation
        emissions = data_dict['data'][ex_str]['wavelengths']

        # Add all valid combinations to our list
        for emission in emissions:
            col_name = f"{int(emission)}-{int(excitation)}"
            valid_combinations.append((excitation, emission, col_name))

    print(f"Found {len(all_excitations)} excitation wavelengths")
    print(f"Generated {len(valid_combinations)} valid excitation-emission combinations")

    # Create an empty dataframe with x, y coordinates
    # First, determine the dimensions of our data
    first_ex = str(all_excitations[0])
    cube_shape = data_dict['data'][first_ex]['cube'].shape
    height, width = cube_shape[0], cube_shape[1]

    print(f"Image dimensions: {height} x {width} pixels")

    # Create coordinate arrays - using the same approach as in your original code
    y_coords, x_coords = np.mgrid[0:height, 0:width]

    # Flatten the coordinates
    flat_x_coords = x_coords.flatten()
    flat_y_coords = y_coords.flatten()

    # Create a mask to identify non-NaN pixels (pixels to keep)
    # We'll use the first wavelength of the first excitation to determine masked pixels
    first_cube = data_dict['data'][first_ex]['cube']
    first_band = first_cube[:, :, 0]  # Use the first band to determine mask
    valid_mask = ~np.isnan(first_band)  # NaN values indicate masked pixels

    # Check if any masking has been applied
    if not np.any(np.isnan(first_band)):
        print("No masked pixels detected (no NaN values found)")
    else:
        # Flatten the mask and filter coordinates
        flat_mask = valid_mask.flatten()
        flat_x_coords = flat_x_coords[flat_mask]
        flat_y_coords = flat_y_coords[flat_mask]

        print(f"Identified {np.sum(~valid_mask)} masked pixels, keeping {np.sum(valid_mask)} pixels")

    # Create initial dataframe with filtered coordinates
    df = pd.DataFrame({
        'x': flat_x_coords,
        'y': flat_y_coords
    })

    # If sample_size is provided, take a random sample of pixels
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
        print(f"Sampled {sample_size} pixels out of {len(df.index)} non-masked pixels")

    print(f"Created initial dataframe with {len(df)} rows")

    # Now, fill in the intensity values for each valid combination
    for excitation, emission, col_name in valid_combinations:
        # Get the data cube for this excitation
        ex_str = str(excitation)
        cube = data_dict['data'][ex_str]['cube']
        wavelengths = data_dict['data'][ex_str]['wavelengths']

        # Find the index of this emission wavelength
        try:
            em_idx = wavelengths.index(emission)

            # Extract the intensity values for these coordinates - vectorized approach
            # Convert coordinates to integers and make sure they're within bounds
            y_indices = df['y'].astype(int).values
            x_indices = df['x'].astype(int).values

            # Get the intensities for these coordinates at the specified emission wavelength
            df[col_name] = cube[y_indices, x_indices, em_idx]

        except ValueError:
            # This emission wavelength doesn't exist for this excitation
            # Skip it instead of adding NaN values
            continue

    print(f"Final dataframe has {len(df.columns)} columns")
    return df


def normalize_and_save_masked_versions(
        input_file: str,
        output_dir: Optional[str] = None,
        sample_size: Optional[int] = None
) -> Tuple[Path, Path]:
    """
    Load masked data, normalize it using both min and max exposure times,
    and save both versions as parquet files, excluding masked pixels.

    Args:
        input_file: Path to the input pickle file
        output_dir: Directory to save the output files (default: same as input file)
        sample_size: Optional number of random pixels to sample from non-masked pixels

    Returns:
        Tuple of (up_normalized_parquet_path, down_normalized_parquet_path)
    """
    # Load the data
    print(f"Loading data from {input_file}...")
    with open(input_file, 'rb') as f:
        data_dict = pickle.load(f)

    # Print exposure information
    print_exposure_info(data_dict)

    # Set up output directory
    input_path = Path(input_file)
    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Create output file names
    base_name = input_path.stem
    up_output_pickle = output_dir / f"{base_name}_normalized_exposure_up.pkl"
    down_output_pickle = output_dir / f"{base_name}_normalized_exposure_down.pkl"
    up_output_parquet = output_dir / f"{base_name}_normalized_exposure_up.parquet"
    down_output_parquet = output_dir / f"{base_name}_normalized_exposure_down.parquet"

    # Normalize up (using max exposure as reference)
    up_normalized_data = normalize_hyperspectral_data(
        data_dict,
        reference_type='max',
        output_file=str(up_output_pickle)
    )

    # Normalize down (using min exposure as reference)
    down_normalized_data = normalize_hyperspectral_data(
        data_dict,
        reference_type='min',
        output_file=str(down_output_pickle)
    )

    print("\nNormalization complete!")
    print(f"Up-normalized data (max exposure reference) saved to: {up_output_pickle}")
    print(f"Down-normalized data (min exposure reference) saved to: {down_output_pickle}")

    # Create and save dataframes (excluding masked pixels)
    print("\nCreating dataframes with masked pixels excluded...")

    # Process up-normalized data
    print("\nProcessing up-normalized data...")
    df_up = create_masked_excitation_emission_dataframe(up_normalized_data, sample_size)
    save_dataframe(df_up, str(up_output_parquet))

    # Process down-normalized data
    print("\nProcessing down-normalized data...")
    df_down = create_masked_excitation_emission_dataframe(down_normalized_data, sample_size)
    save_dataframe(df_down, str(down_output_parquet))

    print(f"\nSaved filtered dataframes to:")
    print(f"Up-normalized: {up_output_parquet}")
    print(f"Down-normalized: {down_output_parquet}")

    return up_output_parquet, down_output_parquet