"""
Hyperspectral Dataset Module

This module provides data handling capabilities for hyperspectral data,
including loading from raw data, handling variable emission band lengths,
replacing NaN values, and global normalization.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union


class HyperspectralDataset(Dataset):
    """
    Dataset for loading hyperspectral data for the autoencoder.
    Handles variable emission band lengths across different excitation wavelengths,
    replaces NaN values, and applies global normalization.
    """

    def __init__(self, data_dict: Dict,
                 excitation_wavelengths: List[float] = None,
                 normalize: bool = True,
                 downscale_factor: int = 1,
                 roi: Optional[Tuple[int, int, int, int]] = None):
        """
        Initialize the dataset from the hyperspectral data dictionary.

        Args:
            data_dict: Dictionary containing hyperspectral data (output from HyperspectralDataLoader)
            excitation_wavelengths: List of excitation wavelengths to use
            normalize: Whether to normalize the data globally to [0,1]
            downscale_factor: Factor to downscale the spatial dimensions (1 = full resolution)
            roi: Region of interest as (row_min, row_max, col_min, col_max)
        """
        self.data_dict = data_dict
        self.normalize = normalize
        self.downscale_factor = downscale_factor
        self.roi = roi

        # Initialize emission_wavelengths dictionary
        self.emission_wavelengths = {}  # This was missing!
        self.processed_data = {}  # Initialize this too for clarity

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
        Prepare the hyperspectral data for training:
        1. Extract the full resolution data or specified ROI for each excitation wavelength
        2. Downscale if requested
        3. Handle NaN values by replacing with zeros
        4. Apply global normalization across all dimensions to [0,1] range
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
                wavelengths = list(range(bands))  # Fallback if wavelengths not provided

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

        # Prepare data for each excitation wavelength separately
        self.processed_data = {}
        self.emission_wavelengths = {}

        # Track statistics for global normalization
        all_values = []
        nan_counts = {}

        for ex in self.excitation_wavelengths:
            ex_str = str(ex)
            if ex_str not in self.data_dict['data']:
                continue

            cube = self.data_dict['data'][ex_str]['cube']

            # Get emission wavelengths for this excitation
            if 'wavelengths' in self.data_dict['data'][ex_str]:
                self.emission_wavelengths[ex] = self.data_dict['data'][ex_str]['wavelengths']

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

            # Count NaN values before replacement
            nan_count = np.isnan(processed).sum()
            nan_counts[ex] = nan_count
            if nan_count > 0:
                print(
                    f"Warning: {nan_count} NaN values detected in excitation {ex} ({nan_count / (processed.size) * 100:.4f}% of data)")

            # Replace NaN values with zeros
            processed = np.nan_to_num(processed, nan=0.0)

            # Store processed data for this excitation
            self.processed_data[ex] = processed

            # Collect all non-NaN values for global normalization
            all_values.append(processed.flatten())

        # Stack all values for global min/max calculation
        all_values = np.concatenate(all_values)

        # Print NaN statistics
        total_nans = sum(nan_counts.values())
        total_values = all_values.size
        print(
            f"Total NaN values replaced: {total_nans} ({total_nans / (total_values + total_nans) * 100:.4f}% of entire dataset)")

        # Normalize if requested
        if self.normalize:
            # Calculate global min and max across all excitations
            global_min = np.min(all_values)
            global_max = np.max(all_values)

            print(f"Global data range: [{global_min:.4f}, {global_max:.4f}]")

            # Check if min and max are the same (constant data)
            if global_min == global_max:
                print("Warning: Data has constant value. Normalization will result in zeros.")
                for ex in self.processed_data:
                    self.processed_data[ex] = np.zeros_like(self.processed_data[ex])
            else:
                # Normalize each excitation to [0, 1] using global min/max
                for ex in self.processed_data:
                    self.processed_data[ex] = (self.processed_data[ex] - global_min) / (global_max - global_min)

            # Store normalization parameters for later use
            self.normalization_params = {
                'min': global_min,
                'max': global_max
            }

            print(f"Data normalized to range [0, 1] using global normalization")

        print(f"Data preparation complete. Spatial dimensions: {height}x{width}")

    def __len__(self):
        """Return the number of excitation wavelengths"""
        return len(self.processed_data)

    def __getitem__(self, idx):
        """
        Get the data for a specific excitation wavelength.

        Args:
            idx: Index of the excitation wavelength to retrieve

        Returns:
            Data tensor for that excitation
        """
        # Convert index to excitation wavelength
        if idx >= len(self.excitation_wavelengths):
            raise IndexError(f"Index {idx} out of range for {len(self.excitation_wavelengths)} excitation wavelengths")

        ex = self.excitation_wavelengths[idx]

        if ex not in self.processed_data:
            raise ValueError(f"No processed data available for excitation {ex}")

        # Return as tensor
        return torch.tensor(self.processed_data[ex], dtype=torch.float32)

    def get_all_data(self):
        """
        Get all processed data as a dictionary.

        Returns:
            Dictionary mapping excitation wavelengths to processed data tensors
        """
        return {ex: torch.tensor(data, dtype=torch.float32)
                for ex, data in self.processed_data.items()}

    def get_spatial_dimensions(self):
        """
        Get the spatial dimensions of the processed data.

        Returns:
            Height and width as a tuple
        """
        # Get first available excitation
        first_ex = list(self.processed_data.keys())[0]
        height, width, _ = self.processed_data[first_ex].shape
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


def create_spatial_chunks(data_tensor, chunk_size=128, overlap=16):
    """
    Split a large spatial hyperspectral tensor into overlapping chunks for memory-efficient processing.
    Handles any input shape by preserving the last dimension (emission bands).

    Args:
        data_tensor: Input tensor of shape [height, width, emission_bands]
                     or [num_excitations, height, width, emission_bands]
        chunk_size: Size of each spatial chunk
        overlap: Overlap between adjacent chunks

    Returns:
        List of chunk tensors and their positions (y_start, y_end, x_start, x_end)
    """
    # Determine input shape
    if len(data_tensor.shape) == 4:  # [num_excitations, height, width, emission_bands]
        height, width = data_tensor.shape[1], data_tensor.shape[2]
    else:  # [height, width, emission_bands]
        height, width = data_tensor.shape[0], data_tensor.shape[1]

    # Calculate stride
    stride = chunk_size - overlap

    # Calculate number of chunks in each dimension
    num_chunks_y = max(1, (height - overlap) // stride)
    num_chunks_x = max(1, (width - overlap) // stride)

    # Adjust to ensure we cover the entire image
    if stride * num_chunks_y + overlap < height:
        num_chunks_y += 1
    if stride * num_chunks_x + overlap < width:
        num_chunks_x += 1

    # Create list to store chunks and their positions
    chunks = []
    positions = []

    # Extract chunks
    for i in range(num_chunks_y):
        for j in range(num_chunks_x):
            # Calculate start and end positions
            y_start = i * stride
            x_start = j * stride
            y_end = min(y_start + chunk_size, height)
            x_end = min(x_start + chunk_size, width)

            # Handle edge cases by adjusting start positions
            if y_end == height:
                y_start = max(0, height - chunk_size)
            if x_end == width:
                x_start = max(0, width - chunk_size)

            # Extract chunk based on input shape
            if len(data_tensor.shape) == 4:  # [num_excitations, height, width, emission_bands]
                chunk = data_tensor[:, y_start:y_end, x_start:x_end, :]
            else:  # [height, width, emission_bands]
                chunk = data_tensor[y_start:y_end, x_start:x_end, :]

            # Add to lists
            chunks.append(chunk)
            positions.append((y_start, y_end, x_start, x_end))

    print(f"Created {len(chunks)} chunks of size up to {chunk_size}x{chunk_size} with {overlap} overlap")
    return chunks, positions


def merge_chunk_reconstructions(chunks, positions, full_height, full_width):
    """
    Merge the reconstructed chunks back into a full image.
    For overlapping regions, take the average of the reconstructions.

    Args:
        chunks: List of reconstructed chunk tensors
        positions: List of positions (y_start, y_end, x_start, x_end) for each chunk
        full_height: Height of the full reconstructed image
        full_width: Width of the full reconstructed image

    Returns:
        Merged full reconstruction
    """
    # Determine shape from the first chunk
    first_chunk = chunks[0]

    if len(first_chunk.shape) == 4:  # [batch, height, width, emission_bands]
        batch_size, _, _, num_bands = first_chunk.shape
        merged = torch.zeros((batch_size, full_height, full_width, num_bands),
                             device=first_chunk.device)
        weights = torch.zeros((batch_size, full_height, full_width, num_bands),
                              device=first_chunk.device)
    elif len(first_chunk.shape) == 5:  # [batch, num_excitations, height, width, emission_bands]
        batch_size, num_excitations, _, _, num_bands = first_chunk.shape
        merged = torch.zeros((batch_size, num_excitations, full_height, full_width, num_bands),
                             device=first_chunk.device)
        weights = torch.zeros((batch_size, num_excitations, full_height, full_width, num_bands),
                              device=first_chunk.device)
    else:
        raise ValueError(f"Unexpected chunk shape: {first_chunk.shape}")

    # Merge chunks
    for chunk, (y_start, y_end, x_start, x_end) in zip(chunks, positions):
        if len(chunk.shape) == 4:  # [batch, height, width, emission_bands]
            merged[:, y_start:y_end, x_start:x_end, :] += chunk
            weights[:, y_start:y_end, x_start:x_end, :] += 1
        else:  # [batch, num_excitations, height, width, emission_bands]
            merged[:, :, y_start:y_end, x_start:x_end, :] += chunk
            weights[:, :, y_start:y_end, x_start:x_end, :] += 1

    # Average overlapping regions
    merged = merged / torch.clamp(weights, min=1.0)

    return merged