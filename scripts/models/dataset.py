"""
Dataset handling module for hyperspectral data.

This module provides the MaskedHyperspectralDataset class and functions for loading
hyperspectral data with proper mask support.
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Union, Any


class MaskedHyperspectralDataset(Dataset):
    """
    Enhanced dataset for hyperspectral data with proper mask support.
    Handles NaN values and missing data in a principled way.
    """

    def __init__(self,
                 data_dict: Dict,
                 mask: Optional[np.ndarray] = None,
                 excitation_wavelengths: List[float] = None,
                 normalize: bool = True,
                 normalization_method: str = 'global_percentile',
                 percentile_range: Tuple[float, float] = (1.0, 99.0),
                 downscale_factor: int = 1,
                 roi: Optional[Tuple[int, int, int, int]] = None):
        """
        Initialize the enhanced dataset.

        Args:
            data_dict: Dictionary containing hyperspectral data
            mask: Optional binary mask (1=valid, 0=masked out)
            excitation_wavelengths: List of excitation wavelengths to use
            normalize: Whether to normalize the data
            normalization_method: Method for normalization
            percentile_range: Percentile range for normalization
            downscale_factor: Factor to downscale spatial dimensions
            roi: Region of interest as (row_min, row_max, col_min, col_max)
        """
        self.data_dict = data_dict
        self.mask = mask
        self.normalize = normalize
        self.normalization_method = normalization_method
        self.percentile_range = percentile_range
        self.downscale_factor = downscale_factor
        self.roi = roi

        # Initialize properties
        self.emission_wavelengths = {}
        self.processed_data = {}
        self.normalization_params = {}
        self.processed_mask = None

        # If no excitation wavelengths are specified, use all available
        if excitation_wavelengths is None:
            self.excitation_wavelengths = [
                float(ex) for ex in data_dict['excitation_wavelengths']
            ]
        else:
            self.excitation_wavelengths = excitation_wavelengths

        # Prepare data
        self._prepare_data()

    def _prepare_data(self):
        """
        Prepare the hyperspectral data with mask handling.
        """
        print(f"Preparing data for {len(self.excitation_wavelengths)} excitation wavelengths...")

        # First, determine spatial dimensions and check emission band lengths
        emission_band_lengths = {}
        height = None
        width = None

        for ex in self.excitation_wavelengths:
            ex_str = str(ex)
            if ex_str not in self.data_dict['data']:
                print(f"Warning: Excitation wavelength {ex} not found in data dictionary")
                continue

            cube = self.data_dict['data'][ex_str]['cube']
            h, w, bands = cube.shape

            if height is None:
                height = h
                width = w

            # Store emission band length for this excitation
            emission_band_lengths[ex] = bands

            # Store wavelengths
            if 'wavelengths' in self.data_dict['data'][ex_str]:
                self.emission_wavelengths[ex] = self.data_dict['data'][ex_str]['wavelengths']
            else:
                self.emission_wavelengths[ex] = list(range(bands))  # Fallback

        # Print emission band information
        print("Emission band lengths for each excitation wavelength:")
        for ex, bands in emission_band_lengths.items():
            print(f"  - Excitation {ex} nm: {bands} bands")

        # Apply ROI if specified
        if self.roi:
            row_min, row_max, col_min, col_max = self.roi
            height, width = row_max - row_min, col_max - col_min
            print(f"Using ROI: ({row_min}:{row_max}, {col_min}:{col_max})")
        else:
            row_min, row_max = 0, height
            col_min, col_max = 0, width

        # Apply downscaling if requested
        if self.downscale_factor > 1:
            height = height // self.downscale_factor
            width = width // self.downscale_factor
            print(f"Downscaling by factor of {self.downscale_factor} to {height}x{width}")

        # Process the mask if provided
        if self.mask is not None:
            # Apply ROI to mask
            mask_roi = self.mask[row_min:row_max, col_min:col_max]

            # Downscale mask if needed
            if self.downscale_factor > 1:
                from scipy.ndimage import zoom
                self.processed_mask = zoom(mask_roi, 1 / self.downscale_factor, order=0)
                # Ensure binary mask
                self.processed_mask = (self.processed_mask > 0.5).astype(np.float32)
            else:
                self.processed_mask = mask_roi.astype(np.float32)

            print(f"Mask processed. Valid pixels: {np.sum(self.processed_mask)}/{height * width} " +
                  f"({np.sum(self.processed_mask) / (height * width) * 100:.2f}%)")
        else:
            # Create a default mask where all pixels are valid
            self.processed_mask = np.ones((height, width), dtype=np.float32)
            print("No mask provided. All pixels considered valid.")

        # Prepare data for each excitation wavelength separately
        self.processed_data = {}

        # Track valid values for normalization
        all_valid_values = []

        for ex in self.excitation_wavelengths:
            ex_str = str(ex)
            if ex_str not in self.data_dict['data']:
                continue

            cube = self.data_dict['data'][ex_str]['cube']

            # Apply ROI
            roi_data = cube[row_min:row_max, col_min:col_max, :]

            # Apply downscaling if needed
            if self.downscale_factor > 1:
                num_bands = roi_data.shape[2]
                downscaled_data = np.zeros((height, width, num_bands))

                for h in range(height):
                    for w in range(width):
                        h_start = h * self.downscale_factor
                        h_end = min((h + 1) * self.downscale_factor, roi_data.shape[0])
                        w_start = w * self.downscale_factor
                        w_end = min((w + 1) * self.downscale_factor, roi_data.shape[1])

                        block = roi_data[h_start:h_end, w_start:w_end, :]
                        # Average over spatial dimensions only
                        downscaled_data[h, w, :] = np.mean(block, axis=(0, 1))

                processed = downscaled_data
            else:
                processed = roi_data

            # Count NaN values before processing
            nan_count = np.isnan(processed).sum()
            if nan_count > 0:
                print(f"Warning: {nan_count} NaN values detected in excitation {ex} " +
                      f"({nan_count / processed.size * 100:.4f}% of data)")

            # Create a mask showing which values are valid (not NaN)
            valid_mask = ~np.isnan(processed)

            # Keep the valid mask for each excitation
            self.processed_data[ex] = {
                'data': processed,
                'valid_mask': valid_mask
            }

            # Collect all valid (non-NaN) values for normalization
            all_valid_values.extend(processed[valid_mask].flatten())

        # Normalize if requested
        if self.normalize:
            # Calculate normalization parameters using only valid values
            all_valid_values = np.array(all_valid_values)
            global_min = np.min(all_valid_values)
            global_max = np.max(all_valid_values)

            print(f"Global data range (valid values only): [{global_min:.4f}, {global_max:.4f}]")

            # Check if min and max are the same (constant data)
            if global_min == global_max:
                print("Warning: Valid data has constant value. Normalization will result in zeros.")
                for ex in self.processed_data:
                    self.processed_data[ex]['data'] = np.zeros_like(self.processed_data[ex]['data'])
            else:
                # Normalize each excitation to [0, 1] using global min/max on valid data only
                for ex in self.processed_data:
                    data = self.processed_data[ex]['data']
                    valid_mask = self.processed_data[ex]['valid_mask']

                    # Normalize only valid values, leave NaNs as they are
                    normalized_data = np.copy(data)
                    normalized_data[valid_mask] = (data[valid_mask] - global_min) / (global_max - global_min)

                    self.processed_data[ex]['data'] = normalized_data

            # Store normalization parameters
            self.normalization_params = {
                'min': global_min,
                'max': global_max
            }

            print(f"Data normalized to range [0, 1] using global normalization")

        print(f"Data preparation complete. Spatial dimensions: {height}x{width}")

    def get_mask_tensor(self, device='cpu'):
        """Get the processed mask as a tensor."""
        return torch.tensor(self.processed_mask, dtype=torch.float32, device=device)

    def __len__(self):
        """Return the number of excitation wavelengths."""
        return len(self.processed_data)

    def __getitem__(self, idx):
        """
        Get the data for a specific excitation wavelength.

        Args:
            idx: Index of the excitation wavelength to retrieve

        Returns:
            Dictionary with data tensor and valid mask tensor
        """
        # Convert index to excitation wavelength
        if idx >= len(self.excitation_wavelengths):
            raise IndexError(f"Index {idx} out of range for {len(self.excitation_wavelengths)} excitation wavelengths")

        ex = self.excitation_wavelengths[idx]

        if ex not in self.processed_data:
            raise ValueError(f"No processed data available for excitation {ex}")

        # Get data and valid mask
        data = self.processed_data[ex]['data']
        valid_mask = self.processed_data[ex]['valid_mask']

        # Replace NaNs with zeros for model input but keep track of valid positions
        data_no_nan = np.nan_to_num(data, nan=0.0)

        return {
            'excitation': ex,
            'data': torch.tensor(data_no_nan, dtype=torch.float32),
            'valid_mask': torch.tensor(valid_mask, dtype=torch.float32)
        }

    def get_all_data(self):
        """
        Get all processed data as a dictionary.

        Returns:
            Dictionary mapping excitation wavelengths to processed data tensors
        """
        result = {}
        for ex in self.processed_data:
            # Replace NaNs with zeros for model input
            data_no_nan = np.nan_to_num(self.processed_data[ex]['data'], nan=0.0)
            result[ex] = torch.tensor(data_no_nan, dtype=torch.float32)

        return result

    def get_all_valid_masks(self):
        """
        Get all valid masks as a dictionary.

        Returns:
            Dictionary mapping excitation wavelengths to valid mask tensors
        """
        return {ex: torch.tensor(self.processed_data[ex]['valid_mask'], dtype=torch.float32)
                for ex in self.processed_data}

    def get_spatial_dimensions(self):
        """
        Get the spatial dimensions of the processed data.

        Returns:
            Height and width as a tuple
        """
        # Get first available excitation
        first_ex = list(self.processed_data.keys())[0]
        height, width, _ = self.processed_data[first_ex]['data'].shape
        return height, width


def load_hyperspectral_data(data_path):
    """
    Load hyperspectral data from a pickle file.

    Args:
        data_path: Path to the pickle file containing hyperspectral data

    Returns:
        Loaded data dictionary
    """
    print(f"Loading data from {data_path}...")
    with open(data_path, 'rb') as f:
        data_dict = pickle.load(f)

    # Print a summary of the loaded data
    print("Data Summary:")
    print(f"Number of excitation wavelengths: {len(data_dict['excitation_wavelengths'])}")
    print(f"Excitation wavelengths: {data_dict['excitation_wavelengths']}")

    # Check the first excitation wavelength
    first_ex = str(data_dict['excitation_wavelengths'][0])
    if first_ex in data_dict['data']:
        cube_shape = data_dict['data'][first_ex]['cube'].shape
        print(f"Data shape for first excitation ({first_ex} nm): {cube_shape}")

    return data_dict


def load_mask(mask_path):
    """
    Load a binary mask from a numpy file.

    Args:
        mask_path: Path to the mask file (.npy)

    Returns:
        Binary mask as numpy array
    """
    print(f"Loading mask from {mask_path}...")
    mask = np.load(mask_path)

    # Ensure mask is binary
    if not np.all(np.isin(mask, [0, 1])):
        print("Warning: Mask is not binary. Converting to binary (0 and 1).")
        mask = (mask != 0).astype(np.uint8)

    valid_pixels = np.sum(mask)
    total_pixels = mask.size
    print(f"Mask loaded: {valid_pixels}/{total_pixels} valid pixels ({valid_pixels / total_pixels * 100:.2f}%)")

    return mask
