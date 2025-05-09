"""
Autoencoder module for hyperspectral data processing.

This module provides a convolutional autoencoder architecture
optimized for hyperspectral data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


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
