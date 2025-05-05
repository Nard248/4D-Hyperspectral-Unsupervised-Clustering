"""
Hyperspectral Data Modeling Framework

This module provides a comprehensive framework for hyperspectral data processing,
including masked autoencoder training and efficient clustering with the following features:

1. Proper handling of masked regions in hyperspectral data
2. Memory-efficient processing with chunking
3. Optimized clustering for pixel-wise segmentation
4. Visualization utilities for model outputs and clustering results

"""

import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset

from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score


# =============================================================================
# 1. DATASET HANDLING WITH PROPER MASK SUPPORT
# =============================================================================

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


# =============================================================================
# 2. IMPROVED AUTOENCODER MODEL
# =============================================================================

class HyperspectralCAEWithMasking(nn.Module):
    """
    An improved convolutional autoencoder for hyperspectral data with mask support.
    """

    def __init__(
            self,
            excitations_data: Dict[float, torch.Tensor],
            k1: int = 20,  # Number of filters in first layer
            k3: int = 20,  # Number of filters in third layer
            filter_size: int = 5,
            sparsity_target: float = 0.1,
            sparsity_weight: float = 1.0,
            dropout_rate: float = 0.5,
            debug: bool = False
    ):
        """
        Initialize the Improved Hyperspectral Convolutional Autoencoder.

        Args:
            excitations_data: Dictionary mapping excitation wavelengths to data tensors
            k1: Number of filters in first layer
            k3: Number of filters in third layer
            filter_size: Size of convolutional filters
            sparsity_target: Target sparsity for regularization
            sparsity_weight: Weight of sparsity regularization
            dropout_rate: Dropout probability
            debug: Whether to print debug information
        """
        super(HyperspectralCAEWithMasking, self).__init__()

        self.excitation_wavelengths = sorted(list(excitations_data.keys()))
        self.num_excitations = len(self.excitation_wavelengths)
        self.debug = debug

        # Store mapping from excitation wavelength to sanitized key
        self.ex_to_key = {ex: self._sanitize_key(ex) for ex in self.excitation_wavelengths}
        self.key_to_ex = {key: ex for ex, key in self.ex_to_key.items()}

        # Get spatial dimensions (assumed same for all excitations)
        first_ex = self.excitation_wavelengths[0]
        height, width, _ = excitations_data[first_ex].shape
        self.input_height = height
        self.input_width = width

        # Store emission band counts for each excitation
        self.emission_bands = {ex: data.shape[2] for ex, data in excitations_data.items()}

        self.filter_size = filter_size
        self.k1 = k1
        self.k3 = k3
        self.sparsity_target = sparsity_target
        self.sparsity_weight = sparsity_weight
        self.dropout_rate = dropout_rate

        padding = filter_size // 2

        # Create separate encoder convolutions for each excitation wavelength
        self.enc_conv1 = nn.ModuleDict()
        for ex in self.excitation_wavelengths:
            key = self.ex_to_key[ex]
            num_bands = self.emission_bands[ex]
            self.enc_conv1[key] = nn.Conv3d(
                in_channels=1,  # One channel per excitation
                out_channels=k1,
                kernel_size=(filter_size, filter_size, min(5, num_bands)),
                padding=(padding, padding, min(5, num_bands) // 2)
            )

        # Third layer: Convolution on shared feature maps
        self.enc_conv3 = nn.Conv3d(
            in_channels=k1,
            out_channels=k3,
            kernel_size=(filter_size, filter_size, 1),
            padding=(padding, padding, 0)
        )

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

        # Decoder
        self.dec_conv1 = nn.Conv3d(
            in_channels=k3,
            out_channels=k1,
            kernel_size=(filter_size, filter_size, 1),
            padding=(padding, padding, 0)
        )

        # Separate decoders for each excitation wavelength
        self.dec_conv2 = nn.ModuleDict()
        for ex in self.excitation_wavelengths:
            key = self.ex_to_key[ex]
            num_bands = self.emission_bands[ex]
            self.dec_conv2[key] = nn.Conv3d(
                in_channels=k1,
                out_channels=num_bands,  # Output one channel per emission band
                kernel_size=(filter_size, filter_size, 1),
                padding=(padding, padding, 0)
            )

    def _sanitize_key(self, key):
        """Convert a key to a valid ModuleDict key by replacing dots with underscores."""
        return str(key).replace('.', '_')

    def encode(self, data_dict):
        """
        Encode the input data to the shared feature representation.

        Args:
            data_dict: Dictionary mapping excitation wavelengths to tensors
                      with shape [batch_size, height, width, emission_bands]

        Returns:
            Shared feature maps
        """
        batch_size = next(iter(data_dict.values())).shape[0]

        # Process each excitation wavelength separately
        feature_maps = []

        for ex in self.excitation_wavelengths:
            if ex not in data_dict:
                continue

            # Get data for this excitation
            x = data_dict[ex]

            if self.debug:
                print(f"Input data for ex={ex}: {x.shape}")

            # Add channel dimension and permute to [batch, channel, emission_bands, height, width]
            x = x.permute(0, 3, 1, 2).unsqueeze(1)

            if self.debug:
                print(f"After permute for ex={ex}: {x.shape}")

            # Apply first convolution (using sanitized key)
            key = self.ex_to_key[ex]
            x = self.enc_conv1[key](x)
            x = F.sigmoid(x)  # Using sigmoid activation for [0,1] data

            if self.debug:
                print(f"After enc_conv1 for ex={ex}: {x.shape}")

            # Standardize the emission bands dimension using adaptive pooling
            x = F.adaptive_avg_pool3d(x, (1, x.shape[3], x.shape[4]))

            if self.debug:
                print(f"After adaptive_pool for ex={ex}: {x.shape}")

            # Add to feature maps
            feature_maps.append(x)

        # Stack feature maps along channel dimension and average
        if not feature_maps:
            raise ValueError("No valid excitation wavelengths found in input data")

        stacked = torch.stack(feature_maps, dim=1)

        if self.debug:
            print(f"Stacked feature maps: {stacked.shape}")

        mean_features = torch.mean(stacked, dim=1)

        if self.debug:
            print(f"Mean features: {mean_features.shape}")

        # Apply third layer convolution
        encoded = self.enc_conv3(mean_features)
        encoded = F.sigmoid(encoded)  # Using sigmoid activation

        if self.debug:
            print(f"Encoded: {encoded.shape}")

        # Apply dropout
        encoded = self.dropout(encoded)

        return encoded

    def decode(self, encoded):
        """
        Decode from the shared feature representation back to reconstructed input.

        Args:
            encoded: Shared feature maps from the encoder

        Returns:
            Dictionary mapping excitation wavelengths to reconstructed data
        """
        if self.debug:
            print(f"Encoded shape in decode: {encoded.shape}")

        # First decoding layer
        x = self.dec_conv1(encoded)
        x = F.sigmoid(x)  # Using sigmoid activation

        if self.debug:
            print(f"After dec_conv1: {x.shape}")

        # Decode separately for each excitation wavelength
        reconstructed = {}

        for ex in self.excitation_wavelengths:
            # Apply final convolution for this excitation (using sanitized key)
            key = self.ex_to_key[ex]
            num_bands = self.emission_bands[ex]

            # Get this excitation's decoder and apply it to generate all emission bands at once
            recon = self.dec_conv2[key](x)

            if self.debug:
                print(f"After dec_conv2 for ex={ex}: {recon.shape}")

            # Apply activation
            recon = F.sigmoid(recon)  # Using sigmoid activation

            # Reshape back to original format [batch, height, width, emission_bands]
            recon = recon.squeeze(2)  # Remove the dimension with size 1
            if self.debug:
                print(f"After squeeze for ex={ex}: {recon.shape}")

            # Now permute the remaining dimensions
            recon = recon.permute(0, 2, 3, 1)  # [batch, height, width, emission_bands]

            if self.debug:
                print(f"After permute for ex={ex}: {recon.shape}")

            reconstructed[ex] = recon

        return reconstructed

    def forward(self, data_dict):
        """
        Forward pass through the autoencoder.

        Args:
            data_dict: Dictionary mapping excitation wavelengths to tensors

        Returns:
            Reconstructed data dictionary
        """
        encoded = self.encode(data_dict)
        decoded = self.decode(encoded)
        return decoded

    def compute_sparsity_loss(self, encoded):
        """
        Compute the sparsity regularization loss (KL divergence) for sigmoid activations.

        Args:
            encoded: Encoded representation (values between 0 and 1)

        Returns:
            KL divergence loss
        """
        # Calculate average activation
        rho_hat = encoded.mean(dim=(0, 2, 3, 4))

        # Compute KL divergence with small epsilon to prevent log(0)
        rho = torch.tensor(self.sparsity_target).to(encoded.device)
        kl_loss = rho * torch.log((rho + 1e-8) / (rho_hat + 1e-8)) + \
                  (1 - rho) * torch.log((1 - rho + 1e-8) / (1 - rho_hat + 1e-8))

        return kl_loss.sum()

    def compute_masked_loss(self, output_dict, target_dict, valid_mask_dict=None, spatial_mask=None):
        """
        Compute the masked MSE loss, ignoring invalid or masked pixels.

        Args:
            output_dict: Dictionary of model outputs
            target_dict: Dictionary of target values
            valid_mask_dict: Dictionary of valid pixel masks (1=valid, 0=invalid)
            spatial_mask: Optional global spatial mask (1=valid, 0=masked)

        Returns:
            MSE loss calculated only on valid pixels
        """
        # Calculate reconstruction loss ignoring invalid values
        recon_loss = 0
        num_valid = 0

        for ex in target_dict:
            if ex in output_dict:
                # Get output and target
                output = output_dict[ex]
                target = target_dict[ex]

                # Make sure shapes match
                if output.shape != target.shape:
                    print(f"Shape mismatch for excitation {ex}. Output: {output.shape}, Target: {target.shape}")
                    continue

                # Create combined mask
                combined_mask = torch.ones_like(target)

                # Apply valid mask if provided (from NaN handling)
                if valid_mask_dict is not None and ex in valid_mask_dict:
                    valid_mask = valid_mask_dict[ex]
                    # Expand mask to match dimensions
                    while len(valid_mask.shape) < len(target.shape):
                        valid_mask = valid_mask.unsqueeze(-1)
                    valid_mask = valid_mask.expand_as(target)
                    combined_mask = combined_mask * valid_mask

                # Apply spatial mask if provided (from ROI/segmentation)
                if spatial_mask is not None:
                    # Expand mask to match dimensions
                    mask_expanded = spatial_mask
                    while len(mask_expanded.shape) < len(target.shape):
                        mask_expanded = mask_expanded.unsqueeze(-1)
                    mask_expanded = mask_expanded.expand_as(target)
                    combined_mask = combined_mask * mask_expanded

                # Calculate squared error
                squared_error = (output - target) ** 2

                # Apply combined mask
                masked_error = squared_error * combined_mask

                # Sum error and count valid pixels
                total_error = masked_error.sum()
                valid_pixels = combined_mask.sum()

                if valid_pixels > 0:
                    recon_loss += total_error / valid_pixels
                    num_valid += 1

        # Return average loss
        if num_valid > 0:
            return recon_loss / num_valid
        else:
            return torch.tensor(0.0, device=output_dict[next(iter(output_dict))].device)


# =============================================================================
# 3. MEMORY-EFFICIENT TRAINING WITH MASKING
# =============================================================================

def create_spatial_chunks(data_tensor, mask=None, chunk_size=128, overlap=16, chunk_overlap=None):
    """
    Split a large spatial hyperspectral tensor into overlapping chunks.

    Args:
        data_tensor: Input tensor of shape [height, width, emission_bands]
        mask: Optional binary mask of shape [height, width]
        chunk_size: Size of each spatial chunk
        overlap: Overlap between adjacent chunks

    Returns:
        List of chunk tensors, their positions, and optionally mask chunks
    """
    # Determine input shape
    if len(data_tensor.shape) == 4:  # [num_excitations, height, width, emission_bands]
        height, width = data_tensor.shape[1], data_tensor.shape[2]
    else:  # [height, width, emission_bands]
        height, width = data_tensor.shape[0], data_tensor.shape[1]

    if chunk_overlap is not None:
        overlap = chunk_overlap
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

    # Create lists to store chunks and their positions
    chunks = []
    positions = []
    mask_chunks = [] if mask is not None else None

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

            # Extract mask chunk if provided
            if mask is not None:
                mask_chunk = mask[y_start:y_end, x_start:x_end]
                mask_chunks.append(mask_chunk)

            # Add to lists
            chunks.append(chunk)
            positions.append((y_start, y_end, x_start, x_end))

    if mask is not None:
        return chunks, positions, mask_chunks
    else:
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


def train_with_masking(
        model,
        dataset,
        num_epochs=50,
        learning_rate=0.001,
        chunk_size=64,
        chunk_overlap=8,
        batch_size=1,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        early_stopping_patience=None,
        scheduler_patience=5,
        mask=None,
        output_dir="model_output",
        verbose=True
):
    """
    Train the hyperspectral autoencoder with masked loss.

    Args:
        model: HyperspectralCAEWithMasking model
        dataset: MaskedHyperspectralDataset with data
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate for the optimizer
        chunk_size: Size of spatial chunks for processing
        chunk_overlap: Overlap between adjacent chunks
        batch_size: Batch size for training
        device: Device to use for training
        early_stopping_patience: Number of epochs with no improvement before stopping
        scheduler_patience: Number of epochs with no improvement before reducing learning rate
        mask: Optional binary mask (1=valid, 0=masked)
        output_dir: Directory to save model outputs
        verbose: Whether to print detailed progress

    Returns:
        Trained model and training losses
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Move model to device
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=scheduler_patience, verbose=True
    )

    # Get all processed data
    all_data = dataset.get_all_data()
    all_valid_masks = dataset.get_all_valid_masks()

    # Get spatial dimensions
    height, width = dataset.get_spatial_dimensions()

    # Process the mask if provided, otherwise use the dataset's processed mask
    if mask is None:
        mask = dataset.processed_mask

    mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device)

    # Track losses
    train_losses = []
    best_loss = float('inf')
    best_epoch = 0

    # Early stopping counter
    no_improvement_count = 0

    # Create spatial chunks for each excitation wavelength
    if verbose:
        print("Creating spatial chunks for each excitation wavelength...")

    chunks_dict = {}
    positions_dict = {}
    mask_chunks = []

    # Create chunks of the spatial mask
    mask_chunks, mask_positions = create_spatial_chunks(
        mask[..., np.newaxis],  # Add dummy dimension for compatibility
        chunk_size=chunk_size,
        overlap=chunk_overlap
    )[0:2]  # Only get chunks and positions

    # Remove dummy dimension
    mask_chunks = [chunk[..., 0] for chunk in mask_chunks]

    # Create chunks for each excitation wavelength
    for ex in all_data:
        # Get data and convert to numpy
        data_np = all_data[ex].numpy()

        # Generate chunks for this excitation
        chunks_result = create_spatial_chunks(
            data_np,
            chunk_size=chunk_size,
            overlap=chunk_overlap
        )
        chunks = chunks_result[0]
        positions = chunks_result[1]

        chunks_dict[ex] = chunks
        positions_dict[ex] = positions

    # Check if we have any valid chunks
    if not chunks_dict or not next(iter(chunks_dict.values())):
        raise ValueError("No valid chunks found in the dataset")

    # Get number of chunks (should be same for all excitations)
    num_chunks = len(next(iter(chunks_dict.values())))

    if verbose:
        print(f"Created {num_chunks} chunks for each excitation")

    # Create batches of chunks
    batches = []
    mask_batches = []

    for i in range(0, num_chunks, batch_size):
        # Data batch
        batch = {}
        for ex in chunks_dict:
            # Get chunks for this batch
            batch_chunks = chunks_dict[ex][i:i + batch_size]
            if batch_chunks:  # Only add if we have chunks for this batch
                # Convert to tensor with batch dimension
                batch[ex] = torch.tensor(np.stack(batch_chunks), dtype=torch.float32).to(device)
        batches.append(batch)

        # Mask batch
        mask_batch_chunks = mask_chunks[i:i + batch_size]
        if mask_batch_chunks:
            mask_batches.append(torch.tensor(np.stack(mask_batch_chunks), dtype=torch.float32).to(device))
        else:
            mask_batches.append(None)

    if verbose:
        print(f"Starting training for {num_epochs} epochs with {len(batches)} batches...")

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_sparsity_loss = 0.0

        # Train on each batch
        for i, (batch, mask_batch) in enumerate(zip(batches, mask_batches)):
            # Forward pass
            output = model(batch)

            # Compute masked reconstruction loss
            recon_loss = model.compute_masked_loss(
                output_dict=output,
                target_dict=batch,
                spatial_mask=mask_batch
            )

            # Compute sparsity loss
            encoded = model.encode(batch)
            sparsity_loss = model.compute_sparsity_loss(encoded)

            # Total loss
            loss = recon_loss + model.sparsity_weight * sparsity_loss

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_sparsity_loss += sparsity_loss.item()

            # Print progress
            if verbose and ((i + 1) % 5 == 0 or i == len(batches) - 1):
                print(f"  Processed batch {i + 1}/{len(batches)}", end="\r")

        # Record average loss for this epoch
        avg_loss = epoch_loss / len(batches)
        avg_recon_loss = epoch_recon_loss / len(batches)
        avg_sparsity_loss = epoch_sparsity_loss / len(batches)
        train_losses.append(avg_loss)

        # Update learning rate scheduler
        scheduler.step(avg_loss)

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f} "
              f"(Recon: {avg_recon_loss:.4f}, Sparsity: {avg_sparsity_loss:.4f}), "
              f"Time: {epoch_time:.2f}s")

        # Check if this is the best epoch so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            no_improvement_count = 0

            # Save best model
            best_model_path = os.path.join(output_dir, "best_hyperspectral_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best model saved to {best_model_path} (loss: {best_loss:.4f})")

            # Save training curves
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, marker='o')
            plt.title('Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
            plt.yscale('log')  # Use log scale to better visualize loss decrease
            curves_path = os.path.join(output_dir, "training_curves.png")
            plt.savefig(curves_path)
            plt.close()
        else:
            no_improvement_count += 1
            print(
                f"  No improvement for {no_improvement_count} epochs (best: {best_loss:.4f} at epoch {best_epoch + 1})")

        # Early stopping
        if early_stopping_patience is not None and no_improvement_count >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    # Save final model
    final_model_path = os.path.join(output_dir, "final_hyperspectral_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

    # Load the best model
    model.load_state_dict(torch.load(best_model_path))

    # Save loss values
    np.save(os.path.join(output_dir, "training_losses.npy"), np.array(train_losses))

    # Save final training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    curves_path = os.path.join(output_dir, "final_training_curves.png")
    plt.savefig(curves_path)
    plt.close()

    print(f"Training completed. Best loss: {best_loss:.4f} at epoch {best_epoch + 1}")
    return model, train_losses


def evaluate_model_with_masking(
        model,
        dataset,
        chunk_size=64,
        chunk_overlap=8,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        output_dir=None
):
    """
    Evaluate the trained model with masking and calculate metrics only on valid pixels.

    Args:
        model: Trained HyperspectralCAEWithMasking model
        dataset: MaskedHyperspectralDataset with test data
        chunk_size: Size of spatial chunks for processing
        chunk_overlap: Overlap between adjacent chunks
        device: Device to use for evaluation
        output_dir: Directory to save evaluation results

    Returns:
        Dictionary with evaluation metrics and reconstructions
    """
    model = model.to(device)
    model.eval()

    # Get all data and masks
    all_data = dataset.get_all_data()
    all_valid_masks = dataset.get_all_valid_masks()
    spatial_mask = dataset.processed_mask

    # Get spatial dimensions
    height, width = dataset.get_spatial_dimensions()

    # Store results
    results = {
        'metrics': {},
        'reconstructions': {}
    }

    print("Evaluating model on test data...")
    with torch.no_grad():
        overall_mse = 0.0
        overall_mae = 0.0
        num_excitations = 0

        for ex in all_data:
            data = all_data[ex]
            valid_mask = all_valid_masks[ex] if ex in all_valid_masks else None

            # Create chunks for this excitation
            chunks_result = create_spatial_chunks(data.numpy(), chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = chunks_result[0]
            positions = chunks_result[1]
            # Process chunks
            reconstructed_chunks = []
            for i, chunk in enumerate(chunks):
                # Convert to tensor and add batch dimension
                chunk_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(device)

                # Create input dictionary for this excitation only
                chunk_dict = {ex: chunk_tensor}

                # Generate reconstruction
                output = model(chunk_dict)

                # Add to reconstructed chunks
                if ex in output:
                    reconstructed_chunks.append(output[ex])

                # Print progress
                if (i + 1) % 10 == 0 or i == len(chunks) - 1:
                    print(f"  Processed chunk {i + 1}/{len(chunks)} for excitation {ex}", end="\r")

            # Skip this excitation if no valid reconstructions
            if not reconstructed_chunks:
                print(f"Warning: No valid reconstructions for excitation {ex}")
                continue

            # Merge chunks
            full_reconstruction = merge_chunk_reconstructions(
                reconstructed_chunks, positions, height, width
            )

            # Remove batch dimension
            full_reconstruction = full_reconstruction[0]

            # Store reconstruction
            results['reconstructions'][ex] = full_reconstruction

            # Apply masks for metric calculation
            # Find this section in the function:
            if valid_mask is not None:
                # Check dimensions of valid_mask
                if len(valid_mask.shape) == len(data.shape):
                    # Valid mask already has same dimension structure as data
                    valid_mask_expanded = valid_mask
                elif len(valid_mask.shape) == len(data.shape) - 1:
                    # Need to add one dimension to match data
                    valid_mask_expanded = valid_mask.unsqueeze(-1).expand_as(data)
                else:
                    # Handle other cases - reshape as needed
                    print(f"Warning: Valid mask shape {valid_mask.shape} doesn't match data shape {data.shape}")
                    # Try to reshape in a best-effort manner
                    if len(valid_mask.shape) == 2:  # Spatial mask [height, width]
                        valid_mask_expanded = valid_mask.unsqueeze(-1).expand(valid_mask.shape[0],
                                                                              valid_mask.shape[1],
                                                                              data.shape[2])
                    else:
                        # Last resort - just use data shape
                        valid_mask_expanded = torch.ones_like(data, device=device)

                # Move valid_mask_expanded to the specified device - ADD THIS LINE
                valid_mask_expanded = valid_mask_expanded.to(device)

                # Apply spatial mask if available
                if spatial_mask is not None:
                    spatial_mask_tensor = torch.tensor(spatial_mask, dtype=torch.float32, device=device)

                    # Similar dimension checking for spatial mask
                    if len(spatial_mask_tensor.shape) == len(data.shape):
                        spatial_mask_expanded = spatial_mask_tensor
                    elif len(spatial_mask_tensor.shape) == len(data.shape) - 1:
                        spatial_mask_expanded = spatial_mask_tensor.unsqueeze(-1).expand_as(data)
                    else:
                        # Handle other cases
                        if len(spatial_mask_tensor.shape) == 2:  # Spatial mask [height, width]
                            spatial_mask_expanded = spatial_mask_tensor.unsqueeze(-1).expand(
                                spatial_mask_tensor.shape[0],
                                spatial_mask_tensor.shape[1],
                                data.shape[2])
                        else:
                            spatial_mask_expanded = torch.ones_like(data, device=device)

                    # Move spatial_mask_expanded to the specified device - ADD THIS LINE IF NEEDED
                    spatial_mask_expanded = spatial_mask_expanded.to(device)

                    combined_mask = valid_mask_expanded * spatial_mask_expanded
                else:
                    combined_mask = valid_mask_expanded

                # Calculate metrics only on valid pixels
                masked_squared_error = ((full_reconstruction - data.to(device)) ** 2) * combined_mask.to(device)
                masked_abs_error = torch.abs(full_reconstruction - data.to(device)) * combined_mask.to(device)

                num_valid_pixels = combined_mask.sum().item()

                if num_valid_pixels > 0:
                    mse = masked_squared_error.sum().item() / num_valid_pixels
                    mae = masked_abs_error.sum().item() / num_valid_pixels

                    # Calculate PSNR
                    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')

                    # Store metrics
                    results['metrics'][ex] = {
                        'mse': mse,
                        'mae': mae,
                        'psnr': psnr,
                        'valid_pixels': num_valid_pixels
                    }

                    # Update overall metrics
                    overall_mse += mse
                    overall_mae += mae
                    num_excitations += 1

                    print(f"Excitation {ex}nm - MSE: {mse:.4f}, MAE: {mae:.4f}, PSNR: {psnr:.2f} dB "
                          f"(on {num_valid_pixels} valid pixels)")
                else:
                    print(f"Warning: No valid pixels for excitation {ex}")
            else:
                # If no valid mask, use all pixels
                mse = F.mse_loss(full_reconstruction, data.to(device)).item()
                mae = torch.mean(torch.abs(full_reconstruction - data.to(device))).item()
                psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')

                results['metrics'][ex] = {
                    'mse': mse,
                    'mae': mae,
                    'psnr': psnr
                }

                overall_mse += mse
                overall_mae += mae
                num_excitations += 1

                print(f"Excitation {ex}nm - MSE: {mse:.4f}, MAE: {mae:.4f}, PSNR: {psnr:.2f} dB")

        # Calculate overall metrics
        if num_excitations > 0:
            results['metrics']['overall'] = {
                'mse': overall_mse / num_excitations,
                'mae': overall_mae / num_excitations,
                'psnr': 10 * np.log10(1.0 / (overall_mse / num_excitations)) if overall_mse > 0 else float('inf')
            }

            print(f"Overall - MSE: {results['metrics']['overall']['mse']:.4f}, "
                  f"MAE: {results['metrics']['overall']['mae']:.4f}, "
                  f"PSNR: {results['metrics']['overall']['psnr']:.2f} dB")

    # Save evaluation results if output directory provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

        # Save metrics to CSV
        import csv
        metrics_file = os.path.join(output_dir, "evaluation_metrics.csv")
        with open(metrics_file, 'w', newline='') as csvfile:
            fieldnames = ['excitation', 'mse', 'mae', 'psnr', 'valid_pixels']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for ex, metrics in results['metrics'].items():
                if ex != 'overall':
                    row = {'excitation': ex}
                    for metric, value in metrics.items():
                        row[metric] = value
                    writer.writerow(row)

            # Write overall metrics
            if 'overall' in results['metrics']:
                row = {'excitation': 'overall'}
                for metric, value in results['metrics']['overall'].items():
                    row[metric] = value
                writer.writerow(row)

        print(f"Evaluation metrics saved to {metrics_file}")

    return results


# =============================================================================
# 4. OPTIMIZED CLUSTERING FOR PIXEL-WISE SEGMENTATION
# =============================================================================

def extract_encoded_features(
        model,
        data_dict,
        mask=None,
        chunk_size=64,
        chunk_overlap=8,
        device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Extract encoded features efficiently using chunking.
    """
    # Ensure model is in eval mode
    model.eval()
    model = model.to(device)

    # Get spatial dimensions from first excitation
    first_ex = next(iter(data_dict.values()))
    height, width = first_ex.shape[0], first_ex.shape[1]

    # Store features and shapes
    encoded_features = {}
    spatial_shapes = {}

    # Extract features for each excitation separately
    with torch.no_grad():
        for ex, data in data_dict.items():
            print(f"Extracting features for excitation {ex}...")

            # Create chunks for this excitation - USE MORE ROBUST APPROACH HERE
            # Instead of unpacking directly, get the full result and extract what we need
            chunks_result = create_spatial_chunks(
                data.numpy(),
                mask=mask,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

            # Extract chunks and positions from the result
            chunks = chunks_result[0]  # First item is always chunks
            positions = chunks_result[1]  # Second item is always positions

            # Create feature map to store the results
            all_features = None
            all_positions = []

            # Process chunks
            for i, chunk in enumerate(chunks):
                # Convert to tensor and add batch dimension
                chunk_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(device)

                # Create input dictionary for this excitation only
                chunk_dict = {ex: chunk_tensor}

                # Extract encoded representation
                encoded = model.encode(chunk_dict)

                # Convert to numpy and store
                features = encoded.cpu().numpy()[0]  # Remove batch dimension

                # If this is the first chunk, initialize the feature array
                if all_features is None:
                    all_features = []
                    for feat_idx in range(features.shape[0]):
                        # Create empty feature map for each feature channel
                        all_features.append(np.zeros((height, width)))

                # Store features and positions
                y_start, y_end, x_start, x_end = positions[i]
                all_positions.append((y_start, y_end, x_start, x_end))

                # Extract spatial features (remove channel and emission dimensions)
                spatial_features = features.squeeze(1)  # Remove emission dimension which is 1

                # Store features in the appropriate positions
                for feat_idx in range(spatial_features.shape[0]):
                    # Handle overlapping regions by taking the mean
                    feature_chunk = spatial_features[feat_idx]
                    current = all_features[feat_idx][y_start:y_end, x_start:x_end]
                    overlap_mask = current != 0  # Areas where we already have features

                    # Set new areas directly
                    new_areas = ~overlap_mask
                    current[new_areas] = feature_chunk[new_areas]

                    # Average overlapping areas
                    if np.any(overlap_mask):
                        current[overlap_mask] = (current[overlap_mask] + feature_chunk[overlap_mask]) / 2

                    all_features[feat_idx][y_start:y_end, x_start:x_end] = current

                # Print progress
                if (i + 1) % 10 == 0 or i == len(chunks) - 1:
                    print(f"  Processed chunk {i + 1}/{len(chunks)}", end="\r")

            # Stack features into a single array and reshape
            features_array = np.stack(all_features)

            # Store features and shapes
            encoded_features[ex] = features_array
            spatial_shapes[ex] = (height, width)

            print(f"Extracted {features_array.shape[0]} features for excitation {ex}")

    return encoded_features, spatial_shapes


def optimize_kmeans_clustering(
        features,
        n_clusters=5,
        max_samples=10000,
        random_state=42
):
    """
    Apply K-means clustering with optimizations for large datasets.

    Args:
        features: Feature array of shape [n_features, height, width]
                or [n_samples, n_features]
        n_clusters: Number of clusters
        max_samples: Maximum number of samples to use for initial clustering
        random_state: Random state for reproducibility

    Returns:
        Cluster labels and clustering model
    """
    print(f"Starting optimized K-means clustering with {n_clusters} clusters...")

    # Check if features need reshaping
    if len(features.shape) == 3:
        # Reshape from [n_features, height, width] to [height*width, n_features]
        n_features, height, width = features.shape
        features_reshaped = features.reshape(n_features, -1).T
    else:
        # Already in shape [n_samples, n_features]
        features_reshaped = features

    n_samples, n_features = features_reshaped.shape
    print(f"Feature matrix shape: {features_reshaped.shape}")

    # Apply PCA if the number of features is large
    if n_features > 50:
        print(f"Applying PCA to reduce dimensions from {n_features} to 50...")
        pca = PCA(n_components=min(50, n_features - 1), random_state=random_state)
        features_reshaped = pca.fit_transform(features_reshaped)
        print(f"PCA reduced feature shape: {features_reshaped.shape}")
        print(f"Explained variance: {sum(pca.explained_variance_ratio_):.2f}")

    # Use MiniBatchKMeans for large datasets
    if n_samples > max_samples:
        print(f"Using MiniBatchKMeans for {n_samples} samples (more than {max_samples})...")
        # Sample a subset for initial clustering
        indices = np.random.choice(n_samples, max_samples, replace=False)
        sample_features = features_reshaped[indices]

        # Fit MiniBatchKMeans on the sample
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=min(1000, max_samples),
            max_iter=300,
            random_state=random_state
        )
        kmeans.fit(sample_features)

        # Now predict cluster labels for all data
        print(f"Predicting cluster labels for all {n_samples} samples...")
        labels = kmeans.predict(features_reshaped)
    else:
        print(f"Using standard KMeans for {n_samples} samples...")
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10
        )
        labels = kmeans.fit_predict(features_reshaped)

    # Reshape labels if needed
    if len(features.shape) == 3:
        _, height, width = features.shape
        labels_reshaped = labels.reshape(height, width)
        print(f"Reshaped labels to {labels_reshaped.shape}")
        return labels_reshaped, kmeans
    else:
        return labels, kmeans


def run_pixel_wise_clustering(
        model,
        dataset,
        n_clusters=5,
        excitation_to_use=None,
        chunk_size=64,
        chunk_overlap=8,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        output_dir=None,
        calculate_metrics=True  # Add this parameter
):
    """
    Run efficient pixel-wise clustering on hyperspectral data.

    Args:
        model: Trained HyperspectralCAEWithMasking model
        dataset: MaskedHyperspectralDataset instance
        n_clusters: Number of clusters for K-means
        excitation_to_use: Specific excitation wavelength to use (if None, uses the first one)
        chunk_size: Size of spatial chunks for processing
        chunk_overlap: Overlap between adjacent chunks
        device: Device to use for computation
        output_dir: Directory to save clustering results
        calculate_metrics: Whether to calculate clustering quality metrics

    Returns:
        Dictionary with clustering results
    """
    print("Starting pixel-wise clustering...")

    # Create output directory if needed
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    # Get data
    all_data = dataset.get_all_data()

    # Use mask if available
    mask = dataset.processed_mask if hasattr(dataset, 'processed_mask') else None

    # Set model to evaluation mode
    model.eval()
    model = model.to(device)

    # Step 1: Extract encoded features
    print("Extracting encoded features...")
    encoded_features, spatial_shapes = extract_encoded_features(
        model=model,
        data_dict=all_data,
        mask=mask,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        device=device
    )

    # Determine which excitation to use for clustering
    if excitation_to_use is None:
        excitation_to_use = next(iter(encoded_features.keys()))

    if excitation_to_use not in encoded_features:
        raise ValueError(f"Excitation {excitation_to_use} not found in encoded features")

    print(f"Using excitation {excitation_to_use} for clustering")

    # Get features for the selected excitation
    features = encoded_features[excitation_to_use]

    # Step 2: Apply optimized K-means clustering
    print(f"Applying K-means clustering with {n_clusters} clusters...")
    cluster_labels, clustering_model = optimize_kmeans_clustering(
        features=features,
        n_clusters=n_clusters
    )

    # Mask the cluster labels if a mask is provided
    if mask is not None:
        # Ensure mask has the same shape as cluster labels
        if mask.shape != cluster_labels.shape:
            from scipy.ndimage import zoom
            mask_resized = zoom(mask,
                                (cluster_labels.shape[0] / mask.shape[0],
                                 cluster_labels.shape[1] / mask.shape[1]),
                                order=0)
            # Ensure binary mask
            mask_resized = (mask_resized > 0.5).astype(np.uint8)
        else:
            mask_resized = mask

        # Apply mask - set masked areas to -1 (typically used for "no cluster")
        cluster_labels[mask_resized == 0] = -1

    # Create cluster map visualization
    plt.figure(figsize=(10, 8))
    plt.imshow(cluster_labels, cmap='tab10', interpolation='nearest')
    plt.colorbar(label='Cluster ID')
    plt.title(f'Pixel-wise Clustering (Ex={excitation_to_use}nm, K={n_clusters})')
    plt.axis('off')

    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, f"cluster_map_ex{excitation_to_use}_k{n_clusters}.png"),
                    dpi=300, bbox_inches='tight')

        # Save cluster labels
        np.save(os.path.join(output_dir, f"cluster_labels_ex{excitation_to_use}_k{n_clusters}.npy"),
                cluster_labels)

    plt.close()

    print(f"Clustering complete. Found {len(np.unique(cluster_labels))} unique clusters")

    # Calculate clustering metrics if requested
    metrics = None
    if calculate_metrics:
        print("\nEvaluating clustering quality...")
        metrics = evaluate_clustering_quality(
            features=features,
            cluster_labels=cluster_labels,
            original_data=all_data,
            mask=mask,
            output_dir=output_dir
        )

        # Save metrics to file
        if output_dir is not None:
            metrics_file = os.path.join(output_dir, f"clustering_metrics_ex{excitation_to_use}_k{n_clusters}.json")
            import json
            with open(metrics_file, 'w') as f:
                # Convert numpy values to Python types
                metrics_dict = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                                for k, v in metrics.items()}
                json.dump(metrics_dict, f, indent=2)
            print(f"Clustering metrics saved to: {metrics_file}")

    # Return results
    results = {
        'cluster_labels': cluster_labels,
        'clustering_model': clustering_model,
        'excitation_used': excitation_to_use,
        'encoded_features': encoded_features,
        'spatial_shapes': spatial_shapes,
        'n_clusters': n_clusters
    }

    if metrics is not None:
        results['metrics'] = metrics

    return results


def visualize_cluster_profiles(
        cluster_results,
        dataset,
        original_data=None,
        output_dir=None
):
    """
    Visualize the spectral profiles of each cluster.

    Args:
        cluster_results: Results from run_pixel_wise_clustering
        dataset: MaskedHyperspectralDataset instance
        original_data: Optional dictionary with original data
        output_dir: Directory to save visualizations

    Returns:
        Dictionary with cluster profile statistics
    """
    # Get cluster labels and excitation
    cluster_labels = cluster_results['cluster_labels']
    excitation_used = cluster_results['excitation_used']

    # Get original data if not provided
    if original_data is None:
        original_data = dataset.get_all_data()

    # Get emission wavelengths
    emission_wavelengths = {}
    if hasattr(dataset, 'emission_wavelengths'):
        emission_wavelengths = dataset.emission_wavelengths

    # Get unique cluster IDs (excluding -1 which is used for masked areas)
    unique_clusters = sorted([c for c in np.unique(cluster_labels) if c >= 0])
    print(f"Analyzing profiles for {len(unique_clusters)} clusters")

    # Create a figure for spectral profiles
    plt.figure(figsize=(12, 8))

    # Store cluster statistics
    cluster_stats = {}

    # Get first excitation to determine marker styles
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    # Process each excitation
    for i, ex in enumerate(original_data.keys()):
        # Get data for this excitation
        data = original_data[ex].cpu().numpy()

        # Get emission wavelengths for x-axis
        if ex in emission_wavelengths:
            x_values = emission_wavelengths[ex]
        else:
            x_values = np.arange(data.shape[2])

        # Use a different marker for each excitation
        marker = markers[i % len(markers)]

        # Calculate mean spectrum for each cluster
        for cluster_id in unique_clusters:
            # Create mask for this cluster
            cluster_mask = cluster_labels == cluster_id

            # Skip if no pixels in this cluster
            if not np.any(cluster_mask):
                continue

            # Get data for this cluster
            cluster_data = data[cluster_mask]

            # Calculate mean spectrum (ignoring NaNs)
            mean_spectrum = np.nanmean(cluster_data, axis=0)

            # Calculate standard deviation (ignoring NaNs)
            std_spectrum = np.nanstd(cluster_data, axis=0)

            # Store statistics
            if cluster_id not in cluster_stats:
                cluster_stats[cluster_id] = {}

            cluster_stats[cluster_id][ex] = {
                'mean': mean_spectrum,
                'std': std_spectrum,
                'count': np.sum(cluster_mask)
            }

            # Plot mean spectrum
            label = f"Cluster {cluster_id}, Ex={ex}nm" if i == 0 else f"Ex={ex}nm"
            plt.plot(x_values, mean_spectrum, marker=marker, label=label if i == 0 else None,
                     color=plt.cm.tab10(cluster_id % 10))

            # Plot error bars for first few excitations to avoid clutter
            if i < 3:  # Limit to first 3 excitations
                plt.fill_between(
                    x_values,
                    mean_spectrum - std_spectrum,
                    mean_spectrum + std_spectrum,
                    alpha=0.2,
                    color=plt.cm.tab10(cluster_id % 10)
                )

    # Add legend for clusters
    plt.legend(loc='best')
    plt.xlabel('Emission Wavelength (nm)' if len(emission_wavelengths) > 0 else 'Emission Band Index')
    plt.ylabel('Intensity')
    plt.title(f'Spectral Profiles by Cluster (Excitation {excitation_used}nm)')
    plt.grid(True, alpha=0.3)

    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, f"cluster_profiles_ex{excitation_used}.png"),
                    dpi=300, bbox_inches='tight')

    plt.close()

    # Create a bar chart of cluster sizes
    plt.figure(figsize=(10, 6))
    cluster_sizes = [np.sum(cluster_labels == c) for c in unique_clusters]

    plt.bar(
        [f"Cluster {c}" for c in unique_clusters],
        cluster_sizes,
        color=[plt.cm.tab10(c % 10) for c in unique_clusters]
    )

    plt.xlabel('Cluster')
    plt.ylabel('Number of Pixels')
    plt.title(f'Cluster Sizes (Excitation {excitation_used}nm)')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', alpha=0.3)

    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, f"cluster_sizes_ex{excitation_used}.png"),
                    dpi=300, bbox_inches='tight')

    plt.close()

    return cluster_stats


def evaluate_clustering_quality(
        features,
        cluster_labels,
        original_data=None,
        mask=None,
        distance_metric='euclidean',
        output_dir=None
):
    """
    Calculate multiple metrics to evaluate clustering quality.

    Args:
        features: Feature array used for clustering (from encoder)
        cluster_labels: Generated cluster labels
        original_data: Optional original hyperspectral data
        mask: Optional binary mask (1=valid, 0=masked)
        distance_metric: Distance metric for silhouette calculation
        output_dir: Directory to save visualizations

    Returns:
        Dictionary of evaluation metrics
    """
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import ndimage

    print("Calculating clustering quality metrics...")

    # Reshape features if needed
    if len(features.shape) == 3:  # [n_features, height, width]
        n_features, height, width = features.shape
        features_reshaped = features.reshape(n_features, -1).T
    else:
        features_reshaped = features

    if len(cluster_labels.shape) == 2:  # [height, width]
        labels_flat = cluster_labels.flatten()
    else:
        labels_flat = cluster_labels

    # Apply mask if provided
    if mask is not None:
        if isinstance(mask, np.ndarray):
            mask_flat = mask.flatten() if len(mask.shape) == 2 else mask
            valid_indices = np.where(mask_flat > 0)[0]
            if features_reshaped.shape[0] == len(mask_flat):
                features_reshaped = features_reshaped[valid_indices]
            labels_flat = labels_flat[valid_indices]

    # Valid indices (exclude -1 which is used for masked areas)
    valid = labels_flat >= 0
    features_valid = features_reshaped[valid]
    labels_valid = labels_flat[valid]

    # Check if we have enough clusters for evaluation
    unique_clusters = np.unique(labels_valid)
    n_clusters = len(unique_clusters)

    # Initialize results
    metrics = {
        'n_clusters': n_clusters,
        'n_samples': len(labels_valid)
    }

    # Skip metric calculation if only one cluster
    if n_clusters <= 1:
        print("Warning: Only one cluster found. Cannot calculate clustering metrics.")
        return metrics

    try:
        # Calculate internal metrics
        metrics['silhouette_score'] = silhouette_score(features_valid, labels_valid, metric=distance_metric)
        metrics['davies_bouldin_score'] = davies_bouldin_score(features_valid, labels_valid)
        metrics['calinski_harabasz_score'] = calinski_harabasz_score(features_valid, labels_valid)

        print(f"Silhouette Score: {metrics['silhouette_score']:.4f} (higher is better)")
        print(f"Davies-Bouldin Index: {metrics['davies_bouldin_score']:.4f} (lower is better)")
        print(f"Calinski-Harabasz Index: {metrics['calinski_harabasz_score']:.4f} (higher is better)")
    except Exception as e:
        print(f"Error calculating internal metrics: {str(e)}")

    # Calculate spatial coherence if we have 2D labels
    if len(cluster_labels.shape) == 2:
        try:
            # Create a neighbor count matrix (how many neighbors have same label)
            spatial_coherence = np.zeros_like(cluster_labels, dtype=float)

            for cluster_id in unique_clusters:
                # Create binary mask for this cluster
                cluster_mask = (cluster_labels == cluster_id).astype(np.int32)

                # Count neighbors with same label (3x3 kernel minus center)
                kernel = np.ones((3, 3), dtype=np.int32)
                kernel[1, 1] = 0  # Remove center
                neighbor_count = ndimage.convolve(cluster_mask, kernel, mode='constant', cval=0)

                # Max possible neighbors is 8 (3x3 kernel minus center)
                neighbor_ratio = neighbor_count / 8.0

                # Set spatial coherence for this cluster
                spatial_coherence[cluster_labels == cluster_id] = neighbor_ratio[cluster_labels == cluster_id]

            # Apply mask if provided
            if mask is not None:
                spatial_coherence = spatial_coherence * mask

            # Calculate average spatial coherence
            metrics['spatial_coherence'] = np.mean(spatial_coherence[cluster_labels >= 0])

            print(f"Spatial Coherence: {metrics['spatial_coherence']:.4f} (higher is better)")

            # Create visualization of spatial coherence
            if output_dir is not None:
                plt.figure(figsize=(10, 8))
                plt.imshow(spatial_coherence, cmap='viridis', vmin=0, vmax=1)
                plt.colorbar(label='Spatial Coherence')
                plt.title('Spatial Coherence Map')
                plt.axis('off')
                plt.savefig(os.path.join(output_dir, "spatial_coherence_map.png"), dpi=300, bbox_inches='tight')
                plt.close()
        except Exception as e:
            print(f"Error calculating spatial coherence: {str(e)}")

    # Calculate spectral metrics if original data provided
    if original_data is not None:
        try:
            # Get data from first excitation wavelength
            first_ex = list(original_data.keys())[0]
            if isinstance(original_data[first_ex], torch.Tensor):
                spectral_data = original_data[first_ex].cpu().numpy()
            else:
                spectral_data = original_data[first_ex]

            # Calculate average spectral angle between clusters
            if len(spectral_data.shape) == 3:  # [height, width, bands]
                h, w, bands = spectral_data.shape
                spectral_data_flat = spectral_data.reshape(-1, bands)

                # Apply mask if needed
                if mask is not None:
                    mask_flat = mask.flatten()
                    spectral_data_flat = spectral_data_flat[mask_flat > 0]

                # Calculate cluster centroids in spectral space
                centroids = []
                for cluster_id in unique_clusters:
                    centroid = np.mean(spectral_data_flat[labels_valid == cluster_id], axis=0)
                    centroids.append(centroid)

                # Calculate minimum spectral angle between any two centroids
                min_angle = float('inf')
                for i in range(len(centroids)):
                    for j in range(i + 1, len(centroids)):
                        # Spectral angle between centroids
                        dot_product = np.dot(centroids[i], centroids[j])
                        norm_i = np.linalg.norm(centroids[i])
                        norm_j = np.linalg.norm(centroids[j])

                        # Avoid division by zero
                        if norm_i > 0 and norm_j > 0:
                            # Ensure dot product / (norm_i * norm_j) is in [-1, 1]
                            cos_angle = min(1.0, max(-1.0, dot_product / (norm_i * norm_j)))
                            angle = np.arccos(cos_angle)
                            min_angle = min(min_angle, angle)

                # Convert to degrees and store
                min_angle_degrees = np.degrees(min_angle)
                metrics['min_spectral_angle'] = min_angle_degrees

                print(f"Minimum Spectral Angle: {min_angle_degrees:.2f} degrees (higher is better)")

        except Exception as e:
            print(f"Error calculating spectral metrics: {str(e)}")

    return metrics
# =============================================================================
# 5. VISUALIZATION UTILITIES
# =============================================================================

def create_rgb_visualization(
        data_dict,
        emission_wavelengths,
        r_band=650,
        g_band=550,
        b_band=450,
        excitation=None,
        mask=None,
        normalization='global',
        output_dir=None
):
    """
    Create RGB false color visualizations from hyperspectral data.

    Args:
        data_dict: Dictionary mapping excitation wavelengths to data tensors
        emission_wavelengths: Dictionary mapping excitations to emission wavelengths
        r_band, g_band, b_band: Target wavelengths for R, G, B channels
        excitation: Specific excitation to visualize (if None, creates for all)
        mask: Optional binary mask to apply (1=valid, 0=masked)
        normalization: Normalization method ('global', 'local', 'percentile')
        output_dir: Directory to save visualizations

    Returns:
        Dictionary mapping excitations to RGB images
    """
    # Create output directory if needed
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    # Determine which excitations to process
    if excitation is not None:
        excitations = [excitation] if excitation in data_dict else []
    else:
        excitations = list(data_dict.keys())

    # Check if we have any valid excitations
    if not excitations:
        print("No valid excitations found for RGB visualization")
        return {}

    # Create a figure for comparison
    num_ex = len(excitations)
    fig, axes = plt.subplots(1, num_ex, figsize=(num_ex * 6, 5))
    if num_ex == 1:
        axes = [axes]

    # Store RGB images
    rgb_dict = {}

    # Track global min and max if using global normalization
    global_min = float('inf')
    global_max = float('-inf')

    # First pass to determine global range if needed
    if normalization == 'global':
        for ex in excitations:
            # Get data and convert to numpy if needed
            if isinstance(data_dict[ex], torch.Tensor):
                data = data_dict[ex].cpu().numpy()
            else:
                data = data_dict[ex]

            # Get band indices
            if ex in emission_wavelengths:
                wavelengths = emission_wavelengths[ex]
                r_idx = np.argmin(np.abs(np.array(wavelengths) - r_band))
                g_idx = np.argmin(np.abs(np.array(wavelengths) - g_band))
                b_idx = np.argmin(np.abs(np.array(wavelengths) - b_band))
            else:
                # Use indices directly if wavelengths not available
                num_bands = data.shape[2]
                r_idx = int(num_bands * 0.8)  # Red ~ 80% through the bands
                g_idx = int(num_bands * 0.5)  # Green ~ middle
                b_idx = int(num_bands * 0.2)  # Blue ~ 20% through the bands

            # Get RGB values
            r_values = data[:, :, r_idx].flatten()
            g_values = data[:, :, g_idx].flatten()
            b_values = data[:, :, b_idx].flatten()

            # Apply mask if provided
            if mask is not None:
                mask_flat = mask.flatten()
                r_values = r_values[mask_flat > 0]
                g_values = g_values[mask_flat > 0]
                b_values = b_values[mask_flat > 0]

            # Remove NaNs
            r_values = r_values[~np.isnan(r_values)]
            g_values = g_values[~np.isnan(g_values)]
            b_values = b_values[~np.isnan(b_values)]

            # Update global min and max
            local_min = min(np.min(r_values), np.min(g_values), np.min(b_values))
            local_max = max(np.max(r_values), np.max(g_values), np.max(b_values))

            global_min = min(global_min, local_min)
            global_max = max(global_max, local_max)

    # Second pass to create RGB images
    for i, ex in enumerate(excitations):
        # Get data and convert to numpy if needed
        if isinstance(data_dict[ex], torch.Tensor):
            data = data_dict[ex].cpu().numpy()
        else:
            data = data_dict[ex]

        # Get band indices
        if ex in emission_wavelengths:
            wavelengths = emission_wavelengths[ex]
            r_idx = np.argmin(np.abs(np.array(wavelengths) - r_band))
            g_idx = np.argmin(np.abs(np.array(wavelengths) - g_band))
            b_idx = np.argmin(np.abs(np.array(wavelengths) - b_band))

            # Print the actual wavelengths used
            r_wl = wavelengths[r_idx]
            g_wl = wavelengths[g_idx]
            b_wl = wavelengths[b_idx]
            print(f"Ex={ex}nm: Using R={r_wl}nm, G={g_wl}nm, B={b_wl}nm")
        else:
            # Use indices directly if wavelengths not available
            num_bands = data.shape[2]
            r_idx = int(num_bands * 0.8)
            g_idx = int(num_bands * 0.5)
            b_idx = int(num_bands * 0.2)
            print(f"Ex={ex}nm: Using band indices R={r_idx}, G={g_idx}, B={b_idx}")

        # Create RGB image
        rgb = np.stack([
            data[:, :, r_idx],  # R channel
            data[:, :, g_idx],  # G channel
            data[:, :, b_idx]  # B channel
        ], axis=2)

        # Apply mask if provided
        if mask is not None:
            # Create a mask with 3 channels
            mask_rgb = np.stack([mask, mask, mask], axis=2)

            # Set masked areas to black
            rgb = rgb * mask_rgb

        # Apply normalization
        if normalization == 'global':
            # Use global min and max
            rgb_normalized = (rgb - global_min) / (global_max - global_min + 1e-8)
        elif normalization == 'local':
            # Normalize each image separately
            local_min = np.nanmin(rgb)
            local_max = np.nanmax(rgb)
            rgb_normalized = (rgb - local_min) / (local_max - local_min + 1e-8)
        elif normalization == 'percentile':
            # Use percentile-based normalization to handle outliers
            p1, p99 = np.nanpercentile(rgb, [1, 99])
            rgb_normalized = (rgb - p1) / (p99 - p1 + 1e-8)
        else:
            rgb_normalized = rgb

        # Clip to [0, 1] range
        rgb_normalized = np.clip(rgb_normalized, 0, 1)

        # Replace NaNs with zeros
        rgb_normalized = np.nan_to_num(rgb_normalized, nan=0.0)

        # Store RGB image
        rgb_dict[ex] = rgb_normalized

        # Plot
        axes[i].imshow(rgb_normalized)
        axes[i].set_title(f'Excitation {ex}nm')
        axes[i].axis('off')

    # Adjust layout
    plt.tight_layout()

    # Save figure if output directory provided
    if output_dir is not None:
        filename = f"rgb_visualization_{'_'.join([str(ex) for ex in excitations])}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')

        # Save individual RGB images
        for ex, rgb in rgb_dict.items():
            plt.figure(figsize=(8, 8))
            plt.imshow(rgb)
            plt.title(f'Excitation {ex}nm')
            plt.axis('off')
            plt.savefig(os.path.join(output_dir, f"rgb_ex{ex}.png"), dpi=300, bbox_inches='tight')
            plt.close()

    plt.close(fig)

    return rgb_dict


def visualize_reconstruction_comparison(
        original_data,
        reconstructed_data,
        excitation,
        emission_wavelengths=None,
        mask=None,
        r_band=650,
        g_band=550,
        b_band=450,
        output_dir=None
):
    """
    Create a side-by-side comparison of original and reconstructed data.

    Args:
        original_data: Original hyperspectral data tensor
        reconstructed_data: Reconstructed hyperspectral data tensor
        excitation: Excitation wavelength
        emission_wavelengths: List of emission wavelengths
        mask: Optional binary mask (1=valid, 0=masked)
        r_band, g_band, b_band: Target wavelengths for RGB visualization
        output_dir: Directory to save visualizations

    Returns:
        Dictionary with comparison metrics
    """
    # Convert tensors to numpy if needed
    if isinstance(original_data, torch.Tensor):
        original_data = original_data.cpu().numpy()

    if isinstance(reconstructed_data, torch.Tensor):
        reconstructed_data = reconstructed_data.cpu().numpy()

    # Create output directory if needed
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    # Get the number of bands
    num_bands = original_data.shape[2]

    # Create a figure for comparison
    plt.figure(figsize=(18, 10))

    # Create grid for subplots
    grid = plt.GridSpec(2, 3, hspace=0.3, wspace=0.3)

    # 1. RGB comparison
    if emission_wavelengths is not None:
        # Find closest bands for RGB
        r_idx = np.argmin(np.abs(np.array(emission_wavelengths) - r_band))
        g_idx = np.argmin(np.abs(np.array(emission_wavelengths) - g_band))
        b_idx = np.argmin(np.abs(np.array(emission_wavelengths) - b_band))

        # Print the actual wavelengths used
        r_wl = emission_wavelengths[r_idx]
        g_wl = emission_wavelengths[g_idx]
        b_wl = emission_wavelengths[b_idx]
        print(f"Using R={r_wl}nm, G={g_wl}nm, B={b_wl}nm")
    else:
        # Use evenly spaced bands if wavelengths not provided
        r_idx = int(num_bands * 0.8)  # Red ~ 80% through the bands
        g_idx = int(num_bands * 0.5)  # Green ~ middle
        b_idx = int(num_bands * 0.2)  # Blue ~ 20% through the bands

    # Create RGB images
    rgb_original = np.stack([
        original_data[:, :, r_idx],
        original_data[:, :, g_idx],
        original_data[:, :, b_idx]
    ], axis=2)

    rgb_recon = np.stack([
        reconstructed_data[:, :, r_idx],
        reconstructed_data[:, :, g_idx],
        reconstructed_data[:, :, b_idx]
    ], axis=2)

    # Apply mask if provided
    if mask is not None:
        mask_rgb = np.stack([mask, mask, mask], axis=2)
        rgb_original = rgb_original * mask_rgb
        rgb_recon = rgb_recon * mask_rgb

    # Normalize for visualization (use same scale for both)
    valid_min = np.nanmin([np.nanmin(rgb_original), np.nanmin(rgb_recon)])
    valid_max = np.nanmax([np.nanmax(rgb_original), np.nanmax(rgb_recon)])

    rgb_original_norm = np.clip((rgb_original - valid_min) / (valid_max - valid_min + 1e-8), 0, 1)
    rgb_recon_norm = np.clip((rgb_recon - valid_min) / (valid_max - valid_min + 1e-8), 0, 1)

    # Replace NaNs with zeros
    rgb_original_norm = np.nan_to_num(rgb_original_norm, nan=0.0)
    rgb_recon_norm = np.nan_to_num(rgb_recon_norm, nan=0.0)

    # Calculate RGB difference
    rgb_diff = np.abs(rgb_original_norm - rgb_recon_norm)
    rgb_diff_enhanced = np.clip(rgb_diff * 5, 0, 1)  # Enhance for visibility

    # Plot RGB images
    ax1 = plt.subplot(grid[0, 0])
    ax1.imshow(rgb_original_norm)
    ax1.set_title('Original')
    ax1.axis('off')

    ax2 = plt.subplot(grid[0, 1])
    ax2.imshow(rgb_recon_norm)
    ax2.set_title('Reconstructed')
    ax2.axis('off')

    ax3 = plt.subplot(grid[0, 2])
    im3 = ax3.imshow(rgb_diff_enhanced)
    ax3.set_title('Difference (enhanced 5x)')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    # 2. Spectral profile comparison
    # Choose center point for spectrum visualization
    center_y, center_x = original_data.shape[0] // 2, original_data.shape[1] // 2

    # Extract spectra
    original_spectrum = original_data[center_y, center_x, :]
    recon_spectrum = reconstructed_data[center_y, center_x, :]

    # x-axis values
    x_values = emission_wavelengths if emission_wavelengths is not None else np.arange(num_bands)

    # Calculate error metrics
    rmse = np.sqrt(np.nanmean((original_spectrum - recon_spectrum) ** 2))
    mae = np.nanmean(np.abs(original_spectrum - recon_spectrum))

    # Plot spectrum
    ax4 = plt.subplot(grid[1, :2])
    ax4.plot(x_values, original_spectrum, 'b-', label='Original', linewidth=2)
    ax4.plot(x_values, recon_spectrum, 'r--', label='Reconstructed', linewidth=2)
    ax4.set_title(f'Spectral Profile at Center Point ({center_y}, {center_x}), RMSE={rmse:.4f}')
    ax4.set_xlabel('Emission Wavelength (nm)' if emission_wavelengths is not None else 'Emission Band Index')
    ax4.set_ylabel('Intensity')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # 3. Error distribution
    # Calculate error metrics for each pixel (ignoring NaNs)
    error_map = np.sqrt(np.nanmean((original_data - reconstructed_data) ** 2, axis=2))

    # Apply mask if provided
    if mask is not None:
        error_map = error_map * mask

    # Plot error map
    ax5 = plt.subplot(grid[1, 2])
    im5 = ax5.imshow(error_map, cmap='hot')
    ax5.set_title(f'RMSE Map (Mean: {np.nanmean(error_map):.4f})')
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

    # Set overall title
    plt.suptitle(f'Reconstruction Comparison (Excitation {excitation}nm)', fontsize=16)

    # Save figure if output directory provided
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, f"reconstruction_comparison_ex{excitation}.png"),
                    dpi=300, bbox_inches='tight')

    plt.close()

    # Calculate global metrics
    if mask is not None:
        # Expand mask to match dimensions
        mask_expanded = np.repeat(mask[:, :, np.newaxis], num_bands, axis=2)

        # Calculate metrics only on valid pixels
        mse = np.nansum(((original_data - reconstructed_data) ** 2) * mask_expanded) / np.sum(mask_expanded)
        mae = np.nansum(np.abs(original_data - reconstructed_data) * mask_expanded) / np.sum(mask_expanded)
    else:
        # Calculate metrics on all pixels
        mse = np.nanmean((original_data - reconstructed_data) ** 2)
        mae = np.nanmean(np.abs(original_data - reconstructed_data))

    rmse = np.sqrt(mse)
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')

    # Return metrics
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'psnr': psnr,
        'rgb_metrics': {
            'rmse_r': np.sqrt(np.nanmean((original_data[:, :, r_idx] - reconstructed_data[:, :, r_idx]) ** 2)),
            'rmse_g': np.sqrt(np.nanmean((original_data[:, :, g_idx] - reconstructed_data[:, :, g_idx]) ** 2)),
            'rmse_b': np.sqrt(np.nanmean((original_data[:, :, b_idx] - reconstructed_data[:, :, b_idx]) ** 2))
        }
    }

    # Print metrics
    print(f"Reconstruction Metrics for Excitation {excitation}nm:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  RGB Channel RMSE - R: {metrics['rgb_metrics']['rmse_r']:.4f}, " +
          f"G: {metrics['rgb_metrics']['rmse_g']:.4f}, B: {metrics['rgb_metrics']['rmse_b']:.4f}")

    return metrics


def overlay_clusters_on_rgb(
        cluster_labels,
        rgb_image,
        alpha=0.5,
        mask=None,
        output_path=None
):
    """
    Create a visualization with cluster labels overlaid on an RGB image.

    Args:
        cluster_labels: Array with cluster labels (2D)
        rgb_image: RGB image as numpy array (HxWx3)
        alpha: Transparency of the overlay
        mask: Optional binary mask (1=valid, 0=masked)
        output_path: Path to save the visualization

    Returns:
        Overlay image as numpy array
    """
    # Convert cluster labels to a colormap
    unique_clusters = np.unique(cluster_labels)

    # Skip -1 which is typically used for masked areas
    unique_clusters = [c for c in unique_clusters if c >= 0]
    n_clusters = len(unique_clusters)

    # Create a colormap for clusters
    cluster_cmap = plt.cm.get_cmap('tab10', max(10, n_clusters))

    # Create an empty overlay
    overlay = np.zeros((*cluster_labels.shape, 4))  # RGBA

    # Fill with cluster colors
    for i, cluster_id in enumerate(unique_clusters):
        # Find pixels belonging to this cluster
        mask_cluster = cluster_labels == cluster_id
        if not np.any(mask_cluster):
            continue

        # Get color for this cluster
        color = cluster_cmap(i % 10)

        # Set color and alpha for this cluster
        overlay[mask_cluster] = (*color[:3], alpha)

    # Set transparent for masked areas or invalid clusters
    invalid_mask = np.logical_or(cluster_labels < 0, np.isnan(cluster_labels))
    overlay[invalid_mask] = (0, 0, 0, 0)

    # Apply additional mask if provided
    if mask is not None:
        # Ensure mask has same shape as cluster labels
        if mask.shape != cluster_labels.shape:
            # Resize mask if needed
            from scipy.ndimage import zoom
            mask_resized = zoom(mask,
                                (cluster_labels.shape[0] / mask.shape[0],
                                 cluster_labels.shape[1] / mask.shape[1]),
                                order=0)
            # Ensure binary mask
            mask_resized = (mask_resized > 0.5).astype(np.uint8)
        else:
            mask_resized = mask

        # Set transparent for masked areas
        overlay[mask_resized == 0] = (0, 0, 0, 0)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Show RGB image
    ax.imshow(rgb_image)

    # Add overlay
    ax.imshow(overlay, alpha=overlay[..., 3])

    # Add colorbar with axes reference
    sm = plt.cm.ScalarMappable(cmap=cluster_cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, ticks=np.arange(n_clusters))  # Added ax parameter
    cbar.set_ticklabels([f'Cluster {c}' for c in unique_clusters])

    ax.set_title('Cluster Overlay on RGB Image')
    ax.axis('off')

    # Save if path provided
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Overlay image saved to {output_path}")

    plt.close()

    # Create a composite for return
    composite = np.copy(rgb_image)

    # Apply overlay
    for i in range(overlay.shape[0]):
        for j in range(overlay.shape[1]):
            a = overlay[i, j, 3]
            if a > 0:
                composite[i, j] = (1 - a) * composite[i, j] + a * overlay[i, j, :3]

    return composite


# =============================================================================
# 6. COMPLETE WORKFLOW FUNCTIONS
# =============================================================================

def complete_hyperspectral_workflow(
        data_path,
        mask_path=None,
        output_dir="hyperspectral_results",
        n_clusters=5,
        excitation_to_use=None,
        normalize=True,
        downscale_factor=1,
        num_epochs=50,
        learning_rate=0.001,
        chunk_size=64,
        chunk_overlap=8,
        early_stopping_patience=5,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        calculate_metrics=True
):
    """
    Run the complete hyperspectral data analysis workflow from loading to clustering.

    Args:
        data_path: Path to the hyperspectral data pickle file
        mask_path: Optional path to a binary mask file
        output_dir: Directory to save all outputs
        n_clusters: Number of clusters for the clustering step
        excitation_to_use: Specific excitation to use for clustering
        normalize: Whether to normalize the data
        downscale_factor: Factor to downscale the spatial dimensions
        num_epochs: Number of epochs for training
        learning_rate: Learning rate for training
        chunk_size: Size of spatial chunks for processing
        chunk_overlap: Overlap between adjacent chunks
        early_stopping_patience: Patience for early stopping
        device: Device to use for computation

    Returns:
        Dictionary with results from each step
    """
    print(f"Starting complete hyperspectral analysis workflow...")
    print(f"Using device: {device}")

    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load data and mask
    print("\n--- Step 1: Loading Data ---")
    data_dict = load_hyperspectral_data(data_path)

    mask = None
    if mask_path is not None:
        mask = load_mask(mask_path)

    # Step 2: Create dataset
    print("\n--- Step 2: Creating Dataset ---")
    dataset = MaskedHyperspectralDataset(
        data_dict=data_dict,
        mask=mask,
        normalize=normalize,
        downscale_factor=downscale_factor
    )

    # Get spatial dimensions
    height, width = dataset.get_spatial_dimensions()
    print(f"Data dimensions after processing: {height}x{width}")

    # Step 3: Create model
    print("\n--- Step 3: Creating Model ---")
    all_data = dataset.get_all_data()

    model = HyperspectralCAEWithMasking(
        excitations_data={ex: data.numpy() for ex, data in all_data.items()},
        k1=20,
        k3=20,
        filter_size=5,
        sparsity_target=0.1,
        sparsity_weight=1.0,
        dropout_rate=0.5
    )

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Step 4: Train the model
    print("\n--- Step 4: Training Model ---")
    model_dir = os.path.join(output_dir, "model")

    model, losses = train_with_masking(
        model=model,
        dataset=dataset,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        batch_size=1,
        device=device,
        early_stopping_patience=early_stopping_patience,
        mask=dataset.processed_mask,
        output_dir=model_dir
    )

    # Step 5: Evaluate the model
    print("\n--- Step 5: Evaluating Model ---")
    eval_dir = os.path.join(output_dir, "evaluation")

    evaluation_results = evaluate_model_with_masking(
        model=model,
        dataset=dataset,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        device=device,
        output_dir=eval_dir
    )

    # Step 6: Generate RGB visualizations
    print("\n--- Step 6: Creating RGB Visualizations ---")
    vis_dir = os.path.join(output_dir, "visualizations")

    # Get reconstructed data
    reconstructions = evaluation_results['reconstructions']

    # Create RGB visualizations for original data
    original_rgb = create_rgb_visualization(
        data_dict=all_data,
        emission_wavelengths=dataset.emission_wavelengths,
        mask=dataset.processed_mask,
        output_dir=vis_dir
    )

    # Create RGB visualizations for reconstructed data
    recon_rgb = create_rgb_visualization(
        data_dict=reconstructions,
        emission_wavelengths=dataset.emission_wavelengths,
        mask=dataset.processed_mask,
        output_dir=vis_dir
    )

    # For each excitation, create side-by-side comparison
    for ex in all_data:
        if ex in reconstructions:
            visualize_reconstruction_comparison(
                original_data=all_data[ex],
                reconstructed_data=reconstructions[ex],
                excitation=ex,
                emission_wavelengths=dataset.emission_wavelengths.get(ex, None),
                mask=dataset.processed_mask,
                output_dir=vis_dir
            )

    # Step 7: Run clustering
    print("\n--- Step 7: Running Pixel-wise Clustering ---")
    cluster_dir = os.path.join(output_dir, "clustering")

    cluster_results = run_pixel_wise_clustering(
        model=model,
        dataset=dataset,
        n_clusters=n_clusters,
        excitation_to_use=excitation_to_use,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        device=device,
        output_dir=cluster_dir,
        calculate_metrics=calculate_metrics  # Pass parameter
    )

    # Step 8: Analyze cluster profiles
    print("\n--- Step 8: Analyzing Cluster Profiles ---")

    cluster_stats = visualize_cluster_profiles(
        cluster_results=cluster_results,
        dataset=dataset,
        original_data=all_data,
        output_dir=cluster_dir
    )

    # Step 9: Create cluster overlay on RGB
    print("\n--- Step 9: Creating Cluster Overlay ---")

    # Determine which excitation to use for RGB background
    ex_for_rgb = excitation_to_use if excitation_to_use in original_rgb else next(iter(original_rgb.keys()))

    overlay = overlay_clusters_on_rgb(
        cluster_labels=cluster_results['cluster_labels'],
        rgb_image=original_rgb[ex_for_rgb],
        mask=dataset.processed_mask,
        output_path=os.path.join(cluster_dir, "cluster_overlay.png")
    )

    # Save results
    print("\n--- Workflow Complete! ---")
    print(f"All results saved to {output_dir}")

    # Return all results
    workflow_results = {
        'dataset': dataset,
        'model': model,
        'training_losses': losses,
        'evaluation': evaluation_results,
        'original_rgb': original_rgb,
        'reconstructed_rgb': recon_rgb,
        'clustering': cluster_results,
        'cluster_stats': cluster_stats,
        'overlay': overlay
    }

    return workflow_results


def compare_preprocessing_methods(
        input_data_paths,
        preprocessing_configs,
        output_dir,
        n_clusters=5,
        chunk_size=64,
        device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Compare different preprocessing methods based on clustering quality.

    Args:
        input_data_paths: Dictionary mapping config names to preprocessed data paths
        preprocessing_configs: List of preprocessing configuration dictionaries
        output_dir: Directory to save comparison results
        n_clusters: Number of clusters for K-means
        chunk_size: Size of spatial chunks for processing
        device: Device to use for computation

    Returns:
        DataFrame with comparison results
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    import json
    from pathlib import Path

    print(f"Comparing {len(preprocessing_configs)} preprocessing methods...")

    os.makedirs(output_dir, exist_ok=True)

    # Store results for each method
    results = []

    # Process each preprocessing method
    for config in preprocessing_configs:
        config_name = config['name']
        config_path = input_data_paths[config_name]

        print(f"\n=== Processing {config_name} ===")
        print(f"Loading data from: {config_path}")

        # Create subdirectory for this config
        config_dir = os.path.join(output_dir, config_name)
        os.makedirs(config_dir, exist_ok=True)

        try:
            # Step 1: Load data
            with open(config_path, 'rb') as f:
                data_dict = pickle.load(f)

            # Step 2: Create dataset
            print("Creating dataset...")
            mask = config.get('mask_path', None)
            if mask is not None:
                mask = np.load(mask)

            dataset = MaskedHyperspectralDataset(
                data_dict=data_dict,
                mask=mask,
                normalize=config.get('normalize', True),
                downscale_factor=config.get('downscale_factor', 1)
            )

            # Step 3: Create model
            print("Creating model...")
            all_data = dataset.get_all_data()

            model = HyperspectralCAEWithMasking(
                excitations_data={ex: data.numpy() for ex, data in all_data.items()},
                k1=20,
                k3=20,
                filter_size=5,
                sparsity_target=0.1,
                sparsity_weight=1.0,
                dropout_rate=0.5
            )

            # Step 4: Train model
            model_dir = os.path.join(config_dir, "model")

            model, losses = train_with_masking(
                model=model,
                dataset=dataset,
                num_epochs=config.get('num_epochs', 50),
                learning_rate=config.get('learning_rate', 0.001),
                chunk_size=chunk_size,
                chunk_overlap=chunk_size // 8,
                batch_size=1,
                device=device,
                early_stopping_patience=config.get('early_stopping_patience', 5),
                mask=dataset.processed_mask,
                output_dir=model_dir
            )

            # Step 5: Run clustering
            cluster_dir = os.path.join(config_dir, "clustering")

            clustering_results = run_pixel_wise_clustering(
                model=model,
                dataset=dataset,
                n_clusters=n_clusters,
                excitation_to_use=config.get('excitation_to_use', None),
                chunk_size=chunk_size,
                chunk_overlap=chunk_size // 8,
                device=device,
                output_dir=cluster_dir,
                calculate_metrics=True
            )

            # Extract metrics
            metrics = clustering_results.get('metrics', {})

            # Save metrics to file
            metrics_file = os.path.join(config_dir, "clustering_metrics.json")
            with open(metrics_file, 'w') as f:
                # Convert numpy values to Python types
                metrics_dict = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                                for k, v in metrics.items()}
                json.dump(metrics_dict, f, indent=2)

            # Add to results with config info
            result_entry = {
                'config_name': config_name,
                'config': config,
                'metrics': metrics,
                'best_loss': min(losses) if losses else float('inf')
            }

            results.append(result_entry)

            print(f"Completed processing for {config_name}")
            print(f"Metrics: {metrics}")

        except Exception as e:
            print(f"Error processing {config_name}: {str(e)}")
            import traceback
            traceback.print_exc()

    # Create comparison dataframe
    comparison_data = []

    for result in results:
        row = {
            'Preprocessing': result['config_name'],
            'Autoencoder Loss': result['best_loss']
        }

        if 'metrics' in result and result['metrics']:
            for metric, value in result['metrics'].items():
                if isinstance(value, (int, float, np.number)):
                    row[metric] = value

        comparison_data.append(row)

    # Create DataFrame
    df = pd.DataFrame(comparison_data)

    # Save to CSV
    csv_path = os.path.join(output_dir, "preprocessing_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"Comparison results saved to {csv_path}")

    # Create visualizations
    for metric in df.columns:
        if metric not in ['Preprocessing']:
            try:
                plt.figure(figsize=(12, 6))
                plt.bar(df['Preprocessing'], df[metric])
                plt.title(f'Comparison of {metric} across preprocessing methods')
                plt.xlabel('Preprocessing Method')
                plt.ylabel(metric)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"comparison_{metric.replace(' ', '_')}.png"))
                plt.close()
            except Exception as e:
                print(f"Error creating chart for {metric}: {str(e)}")

    try:
        metrics = [col for col in df.columns if col != 'Preprocessing']

        # Create radar chart
        from matplotlib.path import Path as MplPath
        from matplotlib.spines import Spine
        from matplotlib.projections.polar import PolarAxes
        from matplotlib.projections import register_projection

        def radar_factory(num_vars, frame='circle'):
            """Create a radar chart with `num_vars` axes."""
            # Calculate evenly-spaced axis angles
            theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

            class RadarAxes(PolarAxes):
                name = 'radar'

                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.set_theta_zero_location('N')

                def fill(self, *args, **kwargs):
                    """Override fill so that line is closed by default"""
                    closed = kwargs.pop('closed', True)
                    return super().fill(closed=closed, *args, **kwargs)

                def plot(self, *args, **kwargs):
                    """Override plot so that line is closed by default"""
                    lines = super().plot(*args, **kwargs)
                    for line in lines:
                        self._close_line(line)
                    return lines

                def _close_line(self, line):
                    x, y = line.get_data()
                    # FIXME: markers at x[0], y[0] get doubled-up
                    if x[0] != x[-1]:
                        x = np.concatenate((x, [x[0]]))
                        y = np.concatenate((y, [y[0]]))
                        line.set_data(x, y)

                def set_varlabels(self, labels):
                    self.set_thetagrids(np.degrees(theta), labels)

                def _gen_axes_patch(self):
                    # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
                    # in axes coordinates.
                    if frame == 'circle':
                        return Circle((0.5, 0.5), 0.5)
                    elif frame == 'polygon':
                        return RegularPolygon((0.5, 0.5), num_vars,
                                              radius=.5, edgecolor="k")
                    else:
                        raise ValueError("unknown value for 'frame': %s" % frame)

                def _gen_axes_spines(self):
                    if frame == 'circle':
                        return super()._gen_axes_spines()
                    elif frame == 'polygon':
                        # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                        spine = Spine(axes=self,
                                      spine_type='circle',
                                      path=MplPath.unit_regular_polygon(num_vars))
                        # unit_regular_polygon returns a polygon of radius 1 centered at
                        # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                        # 0.5) in axes coordinates.
                        spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                            + self.transAxes)
                        return {'polar': spine}
                    else:
                        raise ValueError("unknown value for 'frame': %s" % frame)

            register_projection(RadarAxes)
            return theta

        # Normalize metrics to [0, 1] range for radar chart
        df_radar = df.copy()

        for metric in metrics:
            if df_radar[metric].min() == df_radar[metric].max():
                # Skip metrics with no variation
                continue

            # Determine if higher or lower is better
            higher_better = metric in ['silhouette_score', 'calinski_harabasz_score',
                                       'spatial_coherence', 'min_spectral_angle']

            if higher_better:
                # Normalize such that higher values are better
                df_radar[metric] = (df_radar[metric] - df_radar[metric].min()) / \
                                   (df_radar[metric].max() - df_radar[metric].min())
            else:
                # For metrics where lower is better (like Davies-Bouldin or Loss)
                # Invert so that lower values map to higher normalized values
                df_radar[metric] = 1 - (df_radar[metric] - df_radar[metric].min()) / \
                                   (df_radar[metric].max() - df_radar[metric].min())

        # Keep only metrics with variation
        varied_metrics = [m for m in metrics if df_radar[m].min() != df_radar[m].max()]

        if varied_metrics:
            # Create radar chart
            theta = radar_factory(len(varied_metrics), frame='polygon')

            fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection='radar'))
            fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

            colors = plt.cm.viridis(np.linspace(0, 1, len(df_radar)))

            for i, (idx, row) in enumerate(df_radar.iterrows()):
                values = [row[m] for m in varied_metrics]
                ax.plot(theta, values, color=colors[i])
                ax.fill(theta, values, facecolor=colors[i], alpha=0.25)

            ax.set_varlabels(varied_metrics)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1])

            # Add legend
            legend = plt.legend(df_radar['Preprocessing'], loc=(0.9, 0.9),
                                labelspacing=0.1, fontsize='small')

            plt.title('Comparison of Preprocessing Methods (Normalized Metrics)')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "radar_comparison.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating radar chart: {str(e)}")
        import traceback
        traceback.print_exc()

    return df


def create_feature_selected_datasets(
        input_pkl_path,
        output_dir,
        n_features=30,
        methods=None,
        create_visualizations=True
):
    """
    Apply multiple feature selection methods to hyperspectral data and save as separate PKL files.

    Args:
        input_pkl_path: Path to preprocessed PKL file
        output_dir: Directory to save results
        n_features: Number of features/wavelengths to select
        methods: List of methods to apply (default: all)
        create_visualizations: Whether to create visualizations

    Returns:
        Dictionary mapping method names to output PKL paths
    """
    import os
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.feature_selection import mutual_info_regression, f_classif
    import copy

    # Define all available methods
    all_methods = [
        'pca', 'spectral_variability', 'spectral_gradient',
        'spatial_variability', 'mutual_information', 'band_ratio',
        'spectral_angle', 'correlation_minimization', 'spectral_contrast',
        'all_wavelengths'  # Baseline using all wavelengths
    ]

    # Use specified methods or all methods
    methods = methods or all_methods

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    # Load input data
    print(f"Loading data from {input_pkl_path}...")
    with open(input_pkl_path, 'rb') as f:
        data_dict = pickle.load(f)

    # Track output PKL paths
    output_paths = {}

    # Get excitation wavelengths
    if 'excitation_wavelengths' in data_dict:
        excitation_wavelengths = data_dict['excitation_wavelengths']
    else:
        excitation_wavelengths = list(data_dict['data'].keys())

    # Get data mapping - handle different data structures
    if 'data' in data_dict and isinstance(data_dict['data'], dict):
        data_mapping = data_dict['data']
    else:
        data_mapping = data_dict

    # Process each method
    for method in methods:
        print(f"\n=== Processing method: {method} ===")

        # Create a deep copy of the data
        selected_data = copy.deepcopy(data_dict)
        if 'data' in selected_data and isinstance(selected_data['data'], dict):
            selected_data_mapping = selected_data['data']
        else:
            selected_data_mapping = selected_data

        # Initialize wavelength importance tracking
        wavelength_importance = {}
        selected_wavelength_indices = {}

        # Skip feature selection for baseline method
        if method == 'all_wavelengths':
            # No feature selection - use all wavelengths
            output_path = os.path.join(output_dir, f"data_all_wavelengths.pkl")
            with open(output_path, 'wb') as f:
                pickle.dump(selected_data, f)
            output_paths[method] = output_path
            continue

        # Apply feature selection for each excitation wavelength
        for ex_str in excitation_wavelengths:
            if isinstance(ex_str, (int, float)):
                ex_str = str(ex_str)

            if ex_str not in data_mapping:
                print(f"  Skipping excitation {ex_str} - not found in data")
                continue

            # Get data cube and wavelengths
            cube = data_mapping[ex_str]['cube']

            if 'wavelengths' in data_mapping[ex_str]:
                wavelengths = data_mapping[ex_str]['wavelengths']
            else:
                wavelengths = list(range(cube.shape[2]))

            # Apply feature selection based on method
            if method == 'pca':
                # Reshape to [pixels, wavelengths]
                h, w, bands = cube.shape
                reshaped = cube.reshape(-1, bands)

                # Remove NaN values
                valid_mask = ~np.isnan(reshaped).any(axis=1)
                valid_data = reshaped[valid_mask]

                if len(valid_data) > 0:
                    # Apply PCA
                    pca = PCA(n_components=min(n_features, bands, len(valid_data)))
                    pca.fit(valid_data)

                    # Get feature importance from component loadings
                    loadings = np.abs(pca.components_)
                    importance = loadings.sum(axis=0)

                    # Select top wavelengths
                    top_indices = np.argsort(-importance)[:n_features]

                    # Store importance
                    wavelength_importance[ex_str] = {
                        'importance': importance,
                        'explained_variance': pca.explained_variance_ratio_
                    }
                else:
                    # Fallback if all data is NaN
                    top_indices = np.arange(min(n_features, bands))

            elif method == 'spectral_variability':
                # Calculate variance of each wavelength across all pixels
                h, w, bands = cube.shape
                reshaped = cube.reshape(-1, bands)

                # Calculate variance, ignoring NaNs
                variance = np.nanvar(reshaped, axis=0)

                # Select wavelengths with highest variance
                top_indices = np.argsort(-variance)[:n_features]

                # Store importance
                wavelength_importance[ex_str] = {'importance': variance}

            elif method == 'spectral_gradient':
                # Calculate gradient between adjacent wavelengths
                h, w, bands = cube.shape
                reshaped = cube.reshape(-1, bands)

                # Calculate average gradient magnitude
                gradients = np.zeros(bands)
                for i in range(1, bands):
                    # Difference between adjacent bands
                    diff = np.abs(reshaped[:, i] - reshaped[:, i - 1])
                    gradients[i] = np.nanmean(diff)

                # First band gets same as second
                gradients[0] = gradients[1]

                # Select wavelengths with highest gradients
                top_indices = np.argsort(-gradients)[:n_features]

                # Store importance
                wavelength_importance[ex_str] = {'importance': gradients}

            elif method == 'spatial_variability':
                # Calculate spatial variance within each wavelength
                h, w, bands = cube.shape
                spatial_variance = np.zeros(bands)

                for b in range(bands):
                    # Get this band
                    band = cube[:, :, b]

                    # Calculate spatial variance (local vs global mean)
                    # Using a 3x3 window
                    from scipy.ndimage import uniform_filter
                    local_mean = uniform_filter(band, size=3, mode='reflect')
                    global_mean = np.nanmean(band)

                    # Variance between local and global mean
                    variance = np.nanmean((local_mean - global_mean) ** 2)
                    spatial_variance[b] = variance

                # Select wavelengths with highest spatial variance
                top_indices = np.argsort(-spatial_variance)[:n_features]

                # Store importance
                wavelength_importance[ex_str] = {'importance': spatial_variance}

            elif method == 'mutual_information':
                # Calculate mutual information between each wavelength and all others
                h, w, bands = cube.shape
                reshaped = cube.reshape(-1, bands)

                # Remove NaN values
                valid_mask = ~np.isnan(reshaped).any(axis=1)
                valid_data = reshaped[valid_mask]

                if len(valid_data) > 0:
                    # Initialize mutual information scores
                    mi_scores = np.zeros(bands)

                    # Calculate MI of each band with all others
                    for i in range(bands):
                        # Use this band as target
                        target = valid_data[:, i]

                        # Calculate MI with all other bands
                        other_bands = np.delete(valid_data, i, axis=1)
                        mi = mutual_info_regression(other_bands, target)

                        # Store average MI
                        mi_scores[i] = np.mean(mi)

                    # Select bands with highest mutual information
                    top_indices = np.argsort(-mi_scores)[:n_features]

                    # Store importance
                    wavelength_importance[ex_str] = {'importance': mi_scores}
                else:
                    # Fallback if all data is NaN
                    top_indices = np.arange(min(n_features, bands))

            elif method == 'band_ratio':
                # Calculate ratios between bands to find most informative combinations
                h, w, bands = cube.shape
                reshaped = cube.reshape(-1, bands)

                # Initialize scores
                ratio_scores = np.zeros(bands)

                # For each band, calculate average ratio variance with other bands
                for i in range(bands):
                    ratio_vars = []
                    for j in range(bands):
                        if i != j:
                            # Calculate ratio, avoiding division by zero
                            ratio = np.divide(
                                reshaped[:, i],
                                reshaped[:, j],
                                out=np.zeros_like(reshaped[:, i]),
                                where=reshaped[:, j] != 0
                            )
                            # Calculate variance of ratio
                            ratio_vars.append(np.nanvar(ratio))

                    # Store average ratio variance
                    if ratio_vars:
                        ratio_scores[i] = np.mean(ratio_vars)

                # Select bands with highest ratio variance
                top_indices = np.argsort(-ratio_scores)[:n_features]

                # Store importance
                wavelength_importance[ex_str] = {'importance': ratio_scores}

            elif method == 'spectral_angle':
                # Calculate average spectral angle between pixels
                h, w, bands = cube.shape
                reshaped = cube.reshape(-1, bands)

                # Remove NaN values
                valid_mask = ~np.isnan(reshaped).any(axis=1)
                valid_data = reshaped[valid_mask]

                if len(valid_data) > 0:
                    # Calculate mean spectrum
                    mean_spectrum = np.mean(valid_data, axis=0)

                    # Calculate spectral angle for each band
                    angle_importance = np.zeros(bands)

                    for i in range(bands):
                        # Set this band to zero in a copy of the mean spectrum
                        mod_spectrum = mean_spectrum.copy()
                        mod_spectrum[i] = 0

                        # Calculate dot product (normalized)
                        norm_mean = np.linalg.norm(mean_spectrum)
                        norm_mod = np.linalg.norm(mod_spectrum)

                        if norm_mean > 0 and norm_mod > 0:
                            dot_product = np.dot(mean_spectrum, mod_spectrum)
                            cos_angle = dot_product / (norm_mean * norm_mod)
                            # Ensure it's in valid range due to numerical issues
                            cos_angle = np.clip(cos_angle, -1.0, 1.0)
                            angle = np.arccos(cos_angle)
                            angle_importance[i] = angle

                    # Select bands that cause largest spectral angle change
                    top_indices = np.argsort(-angle_importance)[:n_features]

                    # Store importance
                    wavelength_importance[ex_str] = {'importance': angle_importance}
                else:
                    # Fallback if all data is NaN
                    top_indices = np.arange(min(n_features, bands))

            elif method == 'correlation_minimization':
                # Select wavelengths with minimal correlation
                h, w, bands = cube.shape
                reshaped = cube.reshape(-1, bands)

                # Remove NaN values
                valid_mask = ~np.isnan(reshaped).any(axis=1)
                valid_data = reshaped[valid_mask]

                if len(valid_data) > 0:
                    # Calculate correlation matrix
                    corr_matrix = np.corrcoef(valid_data.T)

                    # Replace NaNs with 1 (perfect correlation)
                    corr_matrix = np.nan_to_num(corr_matrix, nan=1.0)

                    # Greedy approach: select bands with minimal correlation to already selected
                    selected = []
                    # Start with band with highest variance
                    variance = np.nanvar(valid_data, axis=0)
                    first_idx = np.argmax(variance)
                    selected.append(first_idx)

                    # Add bands one by one
                    while len(selected) < min(n_features, bands):
                        # Calculate average absolute correlation with selected bands
                        avg_corr = np.zeros(bands)
                        for i in range(bands):
                            if i not in selected:
                                corrs = [np.abs(corr_matrix[i, j]) for j in selected]
                                avg_corr[i] = np.mean(corrs)
                            else:
                                avg_corr[i] = 1.0  # Max correlation for already selected

                        # Select band with minimum correlation to already selected
                        next_idx = np.argmin(avg_corr)
                        selected.append(next_idx)

                    top_indices = np.array(selected)

                    # Create importance scores (inverse of average correlation)
                    importance = np.zeros(bands)
                    for i in range(bands):
                        corrs = [np.abs(corr_matrix[i, j]) for j in range(bands) if i != j]
                        avg_corr = np.mean(corrs)
                        importance[i] = 1.0 - avg_corr  # Higher value = less correlation

                    # Store importance
                    wavelength_importance[ex_str] = {'importance': importance}
                else:
                    # Fallback if all data is NaN
                    top_indices = np.arange(min(n_features, bands))

            elif method == 'spectral_contrast':
                # Select wavelengths with highest contrast between regions
                h, w, bands = cube.shape

                # Calculate contrast score for each wavelength
                contrast_scores = np.zeros(bands)

                for b in range(bands):
                    # Get this band
                    band = cube[:, :, b]

                    # Calculate local contrast
                    from scipy.ndimage import gaussian_filter
                    blurred = gaussian_filter(band, sigma=2)
                    contrast = np.nanmean(np.abs(band - blurred))
                    contrast_scores[b] = contrast

                # Select wavelengths with highest contrast
                top_indices = np.argsort(-contrast_scores)[:n_features]

                # Store importance
                wavelength_importance[ex_str] = {'importance': contrast_scores}

            # Store selected indices
            selected_wavelength_indices[ex_str] = top_indices

            # Keep only selected wavelengths in output data
            selected_data_mapping[ex_str]['cube'] = cube[:, :, top_indices]

            if 'wavelengths' in selected_data_mapping[ex_str]:
                selected_data_mapping[ex_str]['wavelengths'] = [wavelengths[i] for i in top_indices]

            print(f"  Excitation {ex_str}: Selected {len(top_indices)} wavelengths")

        # Store wavelength selection information in metadata
        if 'metadata' not in selected_data:
            selected_data['metadata'] = {}

        selected_data['metadata']['wavelength_selection'] = {
            'method': method,
            'n_features': n_features,
            'selected_indices': selected_wavelength_indices,
            'importance': wavelength_importance
        }

        # Save selected data
        output_path = os.path.join(output_dir, f"data_{method}_{n_features}.pkl")
        with open(output_path, 'wb') as f:
            pickle.dump(selected_data, f)

        output_paths[method] = output_path
        print(f"  Selected data saved to {output_path}")

        # Create visualizations
        if create_visualizations:
            # Create directory for this method
            method_vis_dir = os.path.join(vis_dir, method)
            os.makedirs(method_vis_dir, exist_ok=True)

            # Plot wavelength importance for each excitation
            for ex_str, importance_dict in wavelength_importance.items():
                if 'importance' in importance_dict:
                    importance = importance_dict['importance']

                    plt.figure(figsize=(12, 6))
                    if 'wavelengths' in data_mapping[ex_str]:
                        wavelengths = data_mapping[ex_str]['wavelengths']
                        plt.bar(wavelengths, importance)
                        plt.xlabel('Wavelength (nm)')
                    else:
                        plt.bar(range(len(importance)), importance)
                        plt.xlabel('Band Index')

                    plt.ylabel('Importance Score')
                    plt.title(f'Wavelength Importance - {method.replace("_", " ").title()} - Ex {ex_str}nm')
                    plt.grid(alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(method_vis_dir, f"importance_ex{ex_str}.png"))
                    plt.close()

                    # Plot selected wavelengths
                    top_indices = selected_wavelength_indices[ex_str]
                    plt.figure(figsize=(12, 6))

                    if 'wavelengths' in data_mapping[ex_str]:
                        wavelengths = data_mapping[ex_str]['wavelengths']
                        all_x = wavelengths
                        selected_x = [wavelengths[i] for i in top_indices]
                    else:
                        all_x = range(len(importance))
                        selected_x = top_indices

                    # Plot all wavelengths
                    plt.plot(all_x, importance, 'b-', alpha=0.5, label='All Wavelengths')

                    # Highlight selected wavelengths
                    selected_y = [importance[i] for i in top_indices]
                    plt.scatter(selected_x, selected_y, color='red', s=50, label=f'Selected ({len(top_indices)})')

                    plt.xlabel('Wavelength (nm)' if 'wavelengths' in data_mapping[ex_str] else 'Band Index')
                    plt.ylabel('Importance Score')
                    plt.title(f'Selected Wavelengths - {method.replace("_", " ").title()} - Ex {ex_str}nm')
                    plt.legend()
                    plt.grid(alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(method_vis_dir, f"selected_ex{ex_str}.png"))
                    plt.close()

    # Create summary visualization
    if create_visualizations:
        # Compare number of selected features across methods
        methods_to_compare = [m for m in methods if m != 'all_wavelengths']
        if methods_to_compare:
            try:
                # Get a common excitation for comparison
                common_ex = None
                for ex in excitation_wavelengths:
                    if str(ex) in data_mapping:
                        common_ex = str(ex)
                        break

                if common_ex:
                    # Create comparison plot
                    plt.figure(figsize=(14, 8))

                    # Get wavelengths for this excitation
                    if 'wavelengths' in data_mapping[common_ex]:
                        wavelengths = data_mapping[common_ex]['wavelengths']
                        x_values = wavelengths
                        x_label = 'Wavelength (nm)'
                    else:
                        band_count = data_mapping[common_ex]['cube'].shape[2]
                        x_values = range(band_count)
                        x_label = 'Band Index'

                    # Plot histogram of selected wavelengths
                    selection_counts = np.zeros(len(x_values))

                    for method in methods_to_compare:
                        if method in output_paths:
                            # Load data to get selection info
                            with open(output_paths[method], 'rb') as f:
                                method_data = pickle.load(f)

                            # Get selected indices
                            if 'metadata' in method_data and 'wavelength_selection' in method_data['metadata']:
                                selection_info = method_data['metadata']['wavelength_selection']
                                if 'selected_indices' in selection_info and common_ex in selection_info[
                                    'selected_indices']:
                                    indices = selection_info['selected_indices'][common_ex]
                                    # Increment count for each selected wavelength
                                    for idx in indices:
                                        if idx < len(selection_counts):
                                            selection_counts[idx] += 1

                    # Plot histogram
                    plt.bar(x_values, selection_counts, alpha=0.7)
                    plt.xlabel(x_label)
                    plt.ylabel('Selection Count')
                    plt.title(f'Wavelength Selection Frequency Across Methods (Ex {common_ex}nm)')
                    plt.grid(alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(vis_dir, "wavelength_selection_frequency.png"))
                    plt.close()
            except Exception as e:
                print(f"Error creating summary visualization: {str(e)}")

    return output_paths


def compare_feature_selection_methods(
        input_dir,
        output_dir,
        n_clusters=5,
        methods=None,
        chunk_size=64,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        num_epochs = 50
):
    """
    Compare different feature selection methods using clustering.

    Args:
        input_dir: Directory containing feature-selected PKL files
        output_dir: Directory to save comparison results
        n_clusters: Number of clusters for K-means
        methods: List of methods to compare (default: all PKL files in input_dir)
        chunk_size: Size of spatial chunks for processing
        device: Computing device

    Returns:
        DataFrame with comparison results
    """
    import os
    import glob
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    import pickle
    import json

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get input files if methods not specified
    if methods is None:
        # Get all PKL files in input directory
        pkl_files = glob.glob(os.path.join(input_dir, "data_*.pkl"))
        methods = [Path(f).stem.split('_', 1)[1] for f in pkl_files]

    # Map methods to input files
    input_files = {}
    for method in methods:
        method_file = os.path.join(input_dir, f"data_{method}.pkl")
        if os.path.exists(method_file):
            input_files[method] = method_file
        else:
            print(f"Warning: File not found for method {method}")

    print(f"Comparing {len(input_files)} feature selection methods...")

    # Initialize results
    results = []

    # Process each method
    for method, input_file in input_files.items():
        print(f"\n=== Processing method: {method} ===")

        # Create method output directory
        method_dir = os.path.join(output_dir, method)
        os.makedirs(method_dir, exist_ok=True)

        try:
            # Process this feature selection method using the autoencoder pipeline
            workflow_results = complete_hyperspectral_workflow(
                data_path=input_file,
                output_dir=method_dir,
                n_clusters=n_clusters,
                normalize=True,
                chunk_size=chunk_size,
                device=device,
                calculate_metrics=True,
                num_epochs=num_epochs
            )

            # Extract clustering metrics
            if 'clustering' in workflow_results and 'metrics' in workflow_results['clustering']:
                metrics = workflow_results['clustering']['metrics']
            else:
                metrics = {}

            # Extract best loss
            if 'training_losses' in workflow_results:
                losses = workflow_results['training_losses']
                best_loss = min(losses) if losses else float('inf')
            else:
                best_loss = float('inf')

            # Add feature selection info
            with open(input_file, 'rb') as f:
                data = pickle.load(f)

            feature_info = {}
            if 'metadata' in data and 'wavelength_selection' in data['metadata']:
                feature_info = data['metadata']['wavelength_selection']

            # Add to results
            result_entry = {
                'method': method,
                'input_file': input_file,
                'output_dir': method_dir,
                'metrics': metrics,
                'best_loss': best_loss,
                'feature_info': feature_info
            }

            results.append(result_entry)

            print(f"Completed processing for method: {method}")
            if metrics:
                print(f"Metrics: {metrics}")

        except Exception as e:
            print(f"Error processing method {method}: {str(e)}")
            import traceback
            traceback.print_exc()

    # Create comparison dataframe
    comparison_data = []

    for result in results:
        row = {
            'Method': result['method'],
            'Autoencoder Loss': result['best_loss']
        }

        # Add metrics to row
        if 'metrics' in result and result['metrics']:
            for metric, value in result['metrics'].items():
                if isinstance(value, (int, float, np.number)):
                    row[metric] = value

        comparison_data.append(row)

    # Create DataFrame
    df = pd.DataFrame(comparison_data)

    # Save to CSV
    csv_path = os.path.join(output_dir, "feature_selection_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"Comparison results saved to {csv_path}")

    # Create visualizations
    for metric in df.columns:
        if metric not in ['Method']:
            try:
                plt.figure(figsize=(12, 6))
                plt.bar(df['Method'], df[metric])
                plt.title(f'Comparison of {metric} across Feature Selection Methods')
                plt.xlabel('Feature Selection Method')
                plt.ylabel(metric)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"comparison_{metric.replace(' ', '_')}.png"))
                plt.close()
            except Exception as e:
                print(f"Error creating chart for {metric}: {str(e)}")

    # Create radar chart for overall comparison
    try:
        # Select metrics for radar chart
        metrics_to_plot = [col for col in df.columns if col not in ['Method']]

        if len(metrics_to_plot) >= 3:  # Need at least 3 metrics for radar chart
            from matplotlib.path import Path as MplPath
            from matplotlib.spines import Spine
            from matplotlib.transforms import Affine2D
            from matplotlib.projections.polar import PolarAxes
            from matplotlib.projections import register_projection
            from matplotlib.patches import Circle, RegularPolygon

            # Normalize metrics for radar chart
            df_radar = df.copy()

            for metric in metrics_to_plot:
                # Skip metrics with no variation
                if df_radar[metric].min() == df_radar[metric].max():
                    continue

                # Determine if higher is better
                higher_better = metric in ['silhouette_score', 'calinski_harabasz_score', 'spatial_coherence']

                if higher_better:
                    # Normalize such that higher values are better (0-1 scale)
                    df_radar[metric] = (df_radar[metric] - df_radar[metric].min()) / \
                                       (df_radar[metric].max() - df_radar[metric].min())
                else:
                    # For metrics where lower is better (like DB index)
                    # Invert so 1 is best, 0 is worst
                    df_radar[metric] = 1 - (df_radar[metric] - df_radar[metric].min()) / \
                                       (df_radar[metric].max() - df_radar[metric].min())

            # Filter to metrics with variation
            valid_metrics = [m for m in metrics_to_plot if df_radar[m].min() != df_radar[m].max()]

            if valid_metrics:
                # Create radar chart factory
                def radar_factory(num_vars, frame='circle'):
                    """Create a radar chart with `num_vars` axes."""
                    # Calculate evenly-spaced axis angles
                    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

                    class RadarAxes(PolarAxes):
                        name = 'radar'

                        def __init__(self, *args, **kwargs):
                            super().__init__(*args, **kwargs)
                            self.set_theta_zero_location('N')

                        def fill(self, *args, **kwargs):
                            """Override fill so that line is closed by default"""
                            closed = kwargs.pop('closed', True)
                            return super().fill(closed=closed, *args, **kwargs)

                        def plot(self, *args, **kwargs):
                            """Override plot so that line is closed by default"""
                            lines = super().plot(*args, **kwargs)
                            for line in lines:
                                self._close_line(line)
                            return lines

                        def _close_line(self, line):
                            x, y = line.get_data()
                            # FIXME: markers at x[0], y[0] get doubled-up
                            if x[0] != x[-1]:
                                x = np.concatenate((x, [x[0]]))
                                y = np.concatenate((y, [y[0]]))
                                line.set_data(x, y)

                        def set_varlabels(self, labels):
                            self.set_thetagrids(np.degrees(theta), labels)

                        def _gen_axes_patch(self):
                            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
                            # in axes coordinates.
                            if frame == 'circle':
                                return Circle((0.5, 0.5), 0.5)
                            elif frame == 'polygon':
                                return RegularPolygon((0.5, 0.5), num_vars,
                                                      radius=.5, edgecolor="k")
                            else:
                                raise ValueError("unknown value for 'frame': %s" % frame)

                        def _gen_axes_spines(self):
                            if frame == 'circle':
                                return super()._gen_axes_spines()
                            elif frame == 'polygon':
                                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                                spine = Spine(axes=self,
                                              spine_type='circle',
                                              path=MplPath.unit_regular_polygon(num_vars))
                                # Unit regular polygon returns a polygon of radius 1 centered at
                                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                                # 0.5) in axes coordinates.
                                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                                    + self.transAxes)
                                return {'polar': spine}
                            else:
                                raise ValueError("unknown value for 'frame': %s" % frame)

                    register_projection(RadarAxes)
                    return theta

                # Create radar chart
                theta = radar_factory(len(valid_metrics), frame='polygon')

                fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection='radar'))
                fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

                colors = plt.cm.tab10(np.linspace(0, 1, len(df_radar)))

                for i, (_, row) in enumerate(df_radar.iterrows()):
                    values = [row[m] for m in valid_metrics]
                    ax.plot(theta, values, color=colors[i % 10])
                    ax.fill(theta, values, facecolor=colors[i % 10], alpha=0.25)

                ax.set_varlabels(valid_metrics)
                ax.set_yticks([0, 0.25, 0.5, 0.75, 1])

                # Add legend
                legend = plt.legend(df_radar['Method'], loc=(0.9, 0.9),
                                    labelspacing=0.1, fontsize='small')

                plt.title('Comparison of Feature Selection Methods (Normalized Metrics)')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "radar_comparison.png"))
                plt.close()
    except Exception as e:
        print(f"Error creating radar chart: {str(e)}")
        import traceback
        traceback.print_exc()

    return df