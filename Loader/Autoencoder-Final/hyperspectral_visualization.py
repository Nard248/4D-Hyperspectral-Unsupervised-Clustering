"""
Hyperspectral Data Visualization

This module provides functions for visualizing hyperspectral data and model results,
including spectral profiles, spatial slices, and false color visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Tuple, Optional, Union


def visualize_training_loss(losses):
    """
    Visualize the training loss curve.

    Args:
        losses: List of training losses
    """
    plt.figure(figsize=(10, 5))
    plt.plot(losses, marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Use log scale to better visualize loss decrease
    plt.show()


def visualize_emission_spectrum(
        original_data,
        reconstructed_data,
        excitation_wavelength,
        y=None,
        x=None,
        wavelengths=None
):
    """
    Visualize the original and reconstructed emission spectrum at a specific spatial location.

    Args:
        original_data: Original data tensor [height, width, emission_bands]
        reconstructed_data: Reconstructed data tensor [height, width, emission_bands]
        excitation_wavelength: Excitation wavelength value
        y, x: Spatial coordinates (if None, uses the center of the image)
        wavelengths: List of emission wavelengths (if available)

    Returns:
        RMSE value for the spectrum
    """
    # Get dimensions
    height, width, num_bands = original_data.shape

    # Use center coordinates if not specified
    if y is None:
        y = height // 2
    if x is None:
        x = width // 2

    # Extract spectra
    if isinstance(original_data, torch.Tensor):
        original_spectrum = original_data[y, x, :].cpu().numpy()
    else:
        original_spectrum = original_data[y, x, :]

    if isinstance(reconstructed_data, torch.Tensor):
        reconstructed_spectrum = reconstructed_data[y, x, :].cpu().numpy()
    else:
        reconstructed_spectrum = reconstructed_data[y, x, :]

    # Set up x-axis values
    x_values = wavelengths if wavelengths is not None else np.arange(num_bands)
    x_label = 'Emission Wavelength (nm)' if wavelengths is not None else 'Emission Band Index'

    # Calculate RMSE
    rmse = np.sqrt(np.mean((original_spectrum - reconstructed_spectrum) ** 2))

    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(x_values, original_spectrum, 'b-', label='Original', linewidth=2)
    plt.plot(x_values, reconstructed_spectrum, 'r--', label='Reconstructed', linewidth=2)

    # Add RMSE text to plot
    plt.text(0.05, 0.95, f'RMSE: {rmse:.4f}', transform=plt.gca().transAxes,
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    plt.title(f'Emission Spectrum at Position ({y}, {x}), Excitation {excitation_wavelength}nm')
    plt.xlabel(x_label)
    plt.ylabel('Intensity (Normalized [0-1])')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    return rmse


def visualize_multiple_spectra(
        original_data,
        reconstructed_data,
        excitation_wavelength,
        positions=None,
        wavelengths=None
):
    """
    Visualize emission spectra at multiple spatial locations.

    Args:
        original_data: Original data tensor [height, width, emission_bands]
        reconstructed_data: Reconstructed data tensor [height, width, emission_bands]
        excitation_wavelength: Excitation wavelength value
        positions: List of (y, x) coordinates to visualize (if None, uses 4 positions)
        wavelengths: List of emission wavelengths (if available)

    Returns:
        List of RMSE values for each position
    """
    # Get dimensions
    height, width, num_bands = original_data.shape

    # Use default positions if not specified
    if positions is None:
        h_quarter, w_quarter = height // 4, width // 4
        positions = [
            (h_quarter, w_quarter),
            (h_quarter, width - w_quarter),
            (height - h_quarter, w_quarter),
            (height - h_quarter, width - w_quarter)
        ]

    # Set up x-axis values
    x_values = wavelengths if wavelengths is not None else np.arange(num_bands)
    x_label = 'Emission Wavelength (nm)' if wavelengths is not None else 'Emission Band Index'

    # Convert to numpy if tensors
    if isinstance(original_data, torch.Tensor):
        original_data = original_data.cpu().numpy()
    if isinstance(reconstructed_data, torch.Tensor):
        reconstructed_data = reconstructed_data.cpu().numpy()

    # Create figure
    fig, axes = plt.subplots(len(positions), 1, figsize=(12, 4 * len(positions)))
    if len(positions) == 1:
        axes = [axes]

    # Plot each position
    rmse_values = []
    for i, (y, x) in enumerate(positions):
        # Extract spectra
        original_spectrum = original_data[y, x, :]
        reconstructed_spectrum = reconstructed_data[y, x, :]

        # Calculate RMSE
        rmse = np.sqrt(np.mean((original_spectrum - reconstructed_spectrum) ** 2))
        rmse_values.append(rmse)

        # Plot
        axes[i].plot(x_values, original_spectrum, 'b-', label='Original', linewidth=2)
        axes[i].plot(x_values, reconstructed_spectrum, 'r--', label='Reconstructed', linewidth=2)
        axes[i].set_title(f'Position ({y}, {x}), RMSE: {rmse:.4f}')
        axes[i].set_xlabel(x_label)
        axes[i].set_ylabel('Intensity [0-1]')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()

    plt.tight_layout()
    plt.show()

    return rmse_values


def visualize_spatial_slice(
        original_data,
        reconstructed_data,
        excitation_wavelength,
        emission_idx=None,
        cmap='viridis'
):
    """
    Visualize a spatial slice of the original and reconstructed data.

    Args:
        original_data: Original data tensor [height, width, emission_bands]
        reconstructed_data: Reconstructed data tensor [height, width, emission_bands]
        excitation_wavelength: Excitation wavelength value
        emission_idx: Index of emission band to visualize (if None, uses the middle band)
        cmap: Colormap for visualization

    Returns:
        Dictionary with evaluation metrics
    """
    # Get dimensions
    height, width, num_bands = original_data.shape

    # Use middle emission band if not specified
    if emission_idx is None:
        emission_idx = num_bands // 2

    # Convert to numpy if tensors
    if isinstance(original_data, torch.Tensor):
        original_data = original_data.cpu().numpy()
    if isinstance(reconstructed_data, torch.Tensor):
        reconstructed_data = reconstructed_data.cpu().numpy()

    # Extract spatial slices
    original_slice = original_data[:, :, emission_idx]
    reconstructed_slice = reconstructed_data[:, :, emission_idx]

    # Calculate absolute difference
    diff = np.abs(original_slice - reconstructed_slice)

    # Calculate evaluation metrics
    rmse = np.sqrt(np.mean((original_slice - reconstructed_slice) ** 2))
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot original
    im0 = axes[0].imshow(original_slice, cmap=cmap, vmin=0, vmax=1)
    axes[0].set_title(f'Original (Ex={excitation_wavelength}nm, Em Band={emission_idx})')
    plt.colorbar(im0, ax=axes[0])

    # Plot reconstruction
    im1 = axes[1].imshow(reconstructed_slice, cmap=cmap, vmin=0, vmax=1)
    axes[1].set_title('Reconstructed')
    plt.colorbar(im1, ax=axes[1])

    # Plot difference
    im2 = axes[2].imshow(diff, cmap='hot')
    axes[2].set_title(f'Absolute Difference (RMSE: {rmse:.4f}, Max: {max_diff:.4f})')
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.show()

    return {
        'rmse': rmse,
        'max_diff': max_diff,
        'mean_diff': mean_diff
    }


def visualize_feature_maps(
        model,
        data_dict,
        num_maps=16,
        cmap='viridis',
        device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Visualize the feature maps from the encoder.

    Args:
        model: Trained HyperspectralCAEVariable model
        data_dict: Dictionary mapping excitation wavelengths to data tensors
        num_maps: Number of feature maps to visualize
        cmap: Colormap for visualization
        device: Device to use for computation

    Returns:
        Dictionary with feature map statistics
    """
    # Ensure model is in eval mode
    model.eval()

    # Add batch dimension to data
    batch_data = {ex: data.unsqueeze(0).to(device) for ex, data in data_dict.items()}

    # Get encoded representation
    with torch.no_grad():
        encoded = model.encode(batch_data)

    # Convert to numpy for plotting
    encoded_np = encoded.cpu().numpy()[0]  # Remove batch dimension

    # Determine grid size
    grid_size = int(np.ceil(np.sqrt(num_maps)))

    # Create figure
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))

    # Flatten axes for easy indexing
    axes = axes.flatten()

    # Plot feature maps
    for i in range(min(num_maps, encoded_np.shape[0])):
        # Take a slice from the middle of the emission dimension
        middle_slice = encoded_np[i, 0]  # Now emission dimension is 1

        # Calculate statistics for this feature map
        mean_val = np.mean(middle_slice)
        std_val = np.std(middle_slice)
        max_val = np.max(middle_slice)
        min_val = np.min(middle_slice)

        # Plot
        im = axes[i].imshow(middle_slice, cmap=cmap)
        axes[i].set_title(f'Feature {i + 1}\nμ={mean_val:.2f}, σ={std_val:.2f}')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    # Hide any unused subplots
    for i in range(num_maps, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

    # Return statistics about feature maps
    stats = {
        'means': np.mean(encoded_np, axis=(1, 2, 3)),
        'stds': np.std(encoded_np, axis=(1, 2, 3)),
        'maxs': np.max(encoded_np, axis=(1, 2, 3)),
        'mins': np.min(encoded_np, axis=(1, 2, 3))
    }

    return stats


def create_rgb_from_hyperspectral(data, emission_wavelengths, r_band=650, g_band=550, b_band=450,
                                  normalization_range=None):
    """
    Create an RGB visualization from hyperspectral data.

    Args:
        data: Hyperspectral data with shape [height, width, emission_bands]
        emission_wavelengths: List of emission wavelengths
        r_band, g_band, b_band: Target wavelengths for R, G, B channels (in nm)
        normalization_range: Optional tuple (min, max) for consistent normalization

    Returns:
        RGB image as numpy array
    """
    # Convert to numpy if tensor
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    # Get closest wavelength indices for R, G, B
    r_idx = np.argmin(np.abs(np.array(emission_wavelengths) - r_band))
    g_idx = np.argmin(np.abs(np.array(emission_wavelengths) - g_band))
    b_idx = np.argmin(np.abs(np.array(emission_wavelengths) - b_band))

    # Create RGB image
    rgb = np.stack([
        data[:, :, r_idx],  # R channel
        data[:, :, g_idx],  # G channel
        data[:, :, b_idx]  # B channel
    ], axis=2)

    # Scale to [0, 1] for visualization
    if normalization_range is None:
        # Normalize based on this data only
        rgb_min = np.min(rgb)
        rgb_max = np.max(rgb)
    else:
        # Use provided normalization range
        rgb_min, rgb_max = normalization_range

    rgb_normalized = (rgb - rgb_min) / (rgb_max - rgb_min + 1e-8)
    # Clip to ensure values are in [0, 1]
    rgb_normalized = np.clip(rgb_normalized, 0, 1)

    print(f"Using emission bands: R={emission_wavelengths[r_idx]}nm, "
          f"G={emission_wavelengths[g_idx]}nm, B={emission_wavelengths[b_idx]}nm")

    return rgb_normalized


def visualize_reconstruction_comparison(original_data, reconstructed_data,
                                        emission_wavelengths, excitation_wavelength,
                                        r_band=650, g_band=550, b_band=450,
                                        use_consistent_normalization=True):
    """
    Create a side-by-side comparison of original and reconstructed data.

    Args:
        original_data: Original hyperspectral data [height, width, emission_bands]
        reconstructed_data: Reconstructed hyperspectral data [height, width, emission_bands]
        emission_wavelengths: List of emission wavelengths
        excitation_wavelength: Excitation wavelength (for title)
        r_band, g_band, b_band: Target wavelengths for R, G, B channels
        use_consistent_normalization: Whether to use the same normalization for both images

    Returns:
        Dictionary with RGB visualization data and RMSE values
    """
    # Convert to numpy if tensors
    if isinstance(original_data, torch.Tensor):
        original_data_np = original_data.cpu().numpy()
    else:
        original_data_np = original_data

    if isinstance(reconstructed_data, torch.Tensor):
        reconstructed_data_np = reconstructed_data.cpu().numpy()
    else:
        reconstructed_data_np = reconstructed_data

    # Calculate normalization range if using consistent normalization
    if use_consistent_normalization:
        # Get R, G, B indices
        r_idx = np.argmin(np.abs(np.array(emission_wavelengths) - r_band))
        g_idx = np.argmin(np.abs(np.array(emission_wavelengths) - g_band))
        b_idx = np.argmin(np.abs(np.array(emission_wavelengths) - b_band))

        # Get min and max across both original and reconstructed for these bands
        combined_min = min(
            np.min(original_data_np[:, :, r_idx]),
            np.min(original_data_np[:, :, g_idx]),
            np.min(original_data_np[:, :, b_idx]),
            np.min(reconstructed_data_np[:, :, r_idx]),
            np.min(reconstructed_data_np[:, :, g_idx]),
            np.min(reconstructed_data_np[:, :, b_idx])
        )

        combined_max = max(
            np.max(original_data_np[:, :, r_idx]),
            np.max(original_data_np[:, :, g_idx]),
            np.max(original_data_np[:, :, b_idx]),
            np.max(reconstructed_data_np[:, :, r_idx]),
            np.max(reconstructed_data_np[:, :, g_idx]),
            np.max(reconstructed_data_np[:, :, b_idx])
        )

        normalization_range = (combined_min, combined_max)

        print(f"Using consistent normalization range: [{combined_min:.4f}, {combined_max:.4f}]")
    else:
        normalization_range = None
        print("Using separate normalization for original and reconstructed images")

    # Create RGB visualizations
    rgb_original = create_rgb_from_hyperspectral(
        original_data_np, emission_wavelengths, r_band, g_band, b_band, normalization_range)

    rgb_reconstructed = create_rgb_from_hyperspectral(
        reconstructed_data_np, emission_wavelengths, r_band, g_band, b_band, normalization_range)

    # Calculate difference
    diff = np.abs(rgb_original - rgb_reconstructed)
    # Enhance the difference for better visibility (optional)
    enhanced_diff = 5 * diff  # Multiply by 5 to make differences more visible
    enhanced_diff = np.clip(enhanced_diff, 0, 1)  # Clip to [0, 1]

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot original
    axes[0].imshow(rgb_original)
    axes[0].set_title(f'Original (Ex={excitation_wavelength}nm)')
    axes[0].axis('off')

    # Plot reconstruction
    axes[1].imshow(rgb_reconstructed)
    axes[1].set_title('Reconstructed')
    axes[1].axis('off')

    # Plot difference
    im = axes[2].imshow(enhanced_diff)
    axes[2].set_title('Difference (enhanced 5x)')
    axes[2].axis('off')
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

    # Calculate RMSE per channel
    rmse_r = np.sqrt(np.mean((rgb_original[:, :, 0] - rgb_reconstructed[:, :, 0]) ** 2))
    rmse_g = np.sqrt(np.mean((rgb_original[:, :, 1] - rgb_reconstructed[:, :, 1]) ** 2))
    rmse_b = np.sqrt(np.mean((rgb_original[:, :, 2] - rgb_reconstructed[:, :, 2]) ** 2))
    rmse_overall = np.sqrt(np.mean((rgb_original - rgb_reconstructed) ** 2))

    print(f"RMSE per channel: R={rmse_r:.4f}, G={rmse_g:.4f}, B={rmse_b:.4f}, Overall={rmse_overall:.4f}")

    return {
        'rgb_original': rgb_original,
        'rgb_reconstructed': rgb_reconstructed,
        'enhanced_diff': enhanced_diff,
        'rmse': {
            'r': rmse_r,
            'g': rmse_g,
            'b': rmse_b,
            'overall': rmse_overall
        }
    }


def visualize_multiple_excitations(model, all_data, emission_wavelengths,
                                   r_band=650, g_band=550, b_band=450,
                                   use_consistent_normalization=True,
                                   device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Create RGB visualizations for multiple excitation wavelengths.

    Args:
        model: Trained HyperspectralCAEVariable model
        all_data: Dictionary mapping excitation wavelengths to data tensors
        emission_wavelengths: Dictionary mapping excitation wavelengths to emission wavelengths
        r_band, g_band, b_band: Target wavelengths for R, G, B channels
        use_consistent_normalization: Whether to use the same normalization for all images
        device: Device to use for computation

    Returns:
        Dictionary of RMSE values for each excitation
    """
    # Ensure model is in eval mode
    model.eval()

    # Generate reconstructions for all excitations
    reconstructions = {}
    for ex, data in all_data.items():
        # Add batch dimension
        data_batch = {ex: data.unsqueeze(0).to(device)}

        # Generate reconstruction
        with torch.no_grad():
            output = model(data_batch)

        # Store reconstruction if available
        if ex in output:
            reconstructions[ex] = output[ex][0].cpu()  # Remove batch dimension

    # Count excitations and determine grid layout
    excitations_to_show = sorted([ex for ex in all_data.keys() if ex in reconstructions])
    num_to_show = min(9, len(excitations_to_show))  # Show maximum 9 excitations

    # Select excitations to visualize (choose evenly spaced ones)
    if len(excitations_to_show) <= num_to_show:
        selected_excitations = excitations_to_show
    else:
        step = len(excitations_to_show) // num_to_show
        indices = list(range(0, len(excitations_to_show), step))[:num_to_show]
        selected_excitations = [excitations_to_show[i] for i in indices]

    # Determine grid layout
    grid_size = int(np.ceil(np.sqrt(num_to_show)))

    # For consistent normalization across all excitations
    if use_consistent_normalization:
        # Find global min and max for RGB bands across all excitations
        global_min = float('inf')
        global_max = float('-inf')

        for ex in selected_excitations:
            # Get original and reconstructed data
            original_np = all_data[ex].numpy()
            reconstructed_np = reconstructions[ex].numpy()

            # Get R, G, B indices for this excitation
            r_idx = np.argmin(np.abs(np.array(emission_wavelengths[ex]) - r_band))
            g_idx = np.argmin(np.abs(np.array(emission_wavelengths[ex]) - g_band))
            b_idx = np.argmin(np.abs(np.array(emission_wavelengths[ex]) - b_band))

            # Update global min/max
            for idx in [r_idx, g_idx, b_idx]:
                global_min = min(global_min, np.min(original_np[:, :, idx]), np.min(reconstructed_np[:, :, idx]))
                global_max = max(global_max, np.max(original_np[:, :, idx]), np.max(reconstructed_np[:, :, idx]))

        print(f"Using global normalization range: [{global_min:.4f}, {global_max:.4f}]")

    # Create a figure for RGB visualizations
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 6, grid_size * 4))
    if grid_size == 1:
        axes = np.array([[axes]])
    axes = axes.flatten()

    # Store RMSE values
    rmse_values = {}

    for i, ex in enumerate(selected_excitations):
        if i >= len(axes):
            break

        # Get original and reconstructed data
        original = all_data[ex].numpy()
        reconstructed = reconstructions[ex].numpy()

        # Get normalization range for this excitation
        if use_consistent_normalization:
            normalization_range = (global_min, global_max)
        else:
            # Use separate normalization for each excitation
            r_idx = np.argmin(np.abs(np.array(emission_wavelengths[ex]) - r_band))
            g_idx = np.argmin(np.abs(np.array(emission_wavelengths[ex]) - g_band))
            b_idx = np.argmin(np.abs(np.array(emission_wavelengths[ex]) - b_band))

            min_val = min(
                np.min(original[:, :, r_idx]), np.min(original[:, :, g_idx]), np.min(original[:, :, b_idx]),
                np.min(reconstructed[:, :, r_idx]), np.min(reconstructed[:, :, g_idx]),
                np.min(reconstructed[:, :, b_idx])
            )
            max_val = max(
                np.max(original[:, :, r_idx]), np.max(original[:, :, g_idx]), np.max(original[:, :, b_idx]),
                np.max(reconstructed[:, :, r_idx]), np.max(reconstructed[:, :, g_idx]),
                np.max(reconstructed[:, :, b_idx])
            )
            normalization_range = (min_val, max_val)

        # Create RGB visualization
        rgb_original = create_rgb_from_hyperspectral(
            original, emission_wavelengths[ex], r_band, g_band, b_band, normalization_range)

        rgb_reconstructed = create_rgb_from_hyperspectral(
            reconstructed, emission_wavelengths[ex], r_band, g_band, b_band, normalization_range)

        # Calculate RMSE
        rmse = np.sqrt(np.mean((rgb_original - rgb_reconstructed) ** 2))
        rmse_values[ex] = rmse

        # Create a side-by-side comparison
        comparison = np.hstack([rgb_original, rgb_reconstructed])

        # Plot
        axes[i].imshow(comparison)
        axes[i].set_title(f'Ex={ex}nm (RMSE: {rmse:.4f})')
        axes[i].set_xlabel('Original | Reconstructed')
        axes[i].axis('off')

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.suptitle('RGB Visualizations of Original vs Reconstructed Data', fontsize=16)
    plt.subplots_adjust(top=0.92)
    plt.show()

    return rmse_values


def visualize_all_spectral_bands(original_data, reconstructed_data,
                                 excitation_wavelength, emission_wavelengths=None,
                                 grid_size=5, cmap='viridis'):
    """
    Visualize all or a subset of spectral bands as spatial maps.

    Args:
        original_data: Original hyperspectral data [height, width, emission_bands]
        reconstructed_data: Reconstructed hyperspectral data [height, width, emission_bands]
        excitation_wavelength: Excitation wavelength value
        emission_wavelengths: List of emission wavelengths (if available)
        grid_size: Number of bands to show on each side of the grid
        cmap: Colormap for visualization

    Returns:
        Dictionary with RMSE values for each band
    """
    # Convert to numpy if tensors
    if isinstance(original_data, torch.Tensor):
        original_data = original_data.cpu().numpy()
    if isinstance(reconstructed_data, torch.Tensor):
        reconstructed_data = reconstructed_data.cpu().numpy()

    # Get dimensions
    height, width, num_bands = original_data.shape

    # Determine bands to visualize
    if num_bands <= grid_size * grid_size:
        # Show all bands if they fit in the grid
        band_indices = list(range(num_bands))
    else:
        # Sample bands evenly across the range
        step = num_bands // (grid_size * grid_size)
        band_indices = list(range(0, num_bands, step))[:grid_size * grid_size]

    # Create figure
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
    axes = axes.flatten()

    # Store RMSE values
    rmse_values = {}

    # Plot each band
    for i, band_idx in enumerate(band_indices):
        if i >= len(axes):
            break

        # Get wavelength label if available
        if emission_wavelengths is not None:
            wavelength = emission_wavelengths[band_idx]
            band_label = f"{wavelength}nm"
        else:
            band_label = f"Band {band_idx}"

        # Extract band images
        orig_band = original_data[:, :, band_idx]
        recon_band = reconstructed_data[:, :, band_idx]

        # Calculate RMSE
        rmse = np.sqrt(np.mean((orig_band - recon_band) ** 2))
        rmse_values[band_idx] = rmse

        # Calculate absolute difference
        diff = np.abs(orig_band - recon_band)

        # Create composite image (orig | recon | diff)
        # Normalize each component for better visualization
        vmin, vmax = 0, 1  # Assuming [0,1] normalized data

        # Create RGB-like image with three channels
        composite = np.zeros((height, width * 3, 3))

        # Original band in first column (gray)
        gray_orig = np.stack([orig_band, orig_band, orig_band], axis=2)
        composite[:, :width, :] = gray_orig

        # Reconstructed band in second column (gray)
        gray_recon = np.stack([recon_band, recon_band, recon_band], axis=2)
        composite[:, width:2 * width, :] = gray_recon

        # Difference in third column (heatmap)
        # Normalize difference to [0,1] for better visualization
        norm_diff = diff / max(np.max(diff), 0.001)
        # Create a heatmap (red channel)
        heat_diff = np.zeros((height, width, 3))
        heat_diff[:, :, 0] = norm_diff  # Red channel for error
        composite[:, 2 * width:, :] = heat_diff

        # Plot composite
        axes[i].imshow(composite)
        axes[i].set_title(f'{band_label} (RMSE: {rmse:.4f})')
        axes[i].axis('off')

        # Add column labels on the first row
        if i < grid_size:
            axes[i].text(width // 2, -10, 'Original', ha='center', va='top')
            axes[i].text(width + width // 2, -10, 'Reconstructed', ha='center', va='top')
            axes[i].text(2 * width + width // 2, -10, 'Difference', ha='center', va='top')

    # Hide any unused subplots
    for j in range(len(band_indices), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.suptitle(f'Spectral Bands Comparison (Ex={excitation_wavelength}nm)', fontsize=16)
    plt.subplots_adjust(top=0.95)
    plt.show()

    return rmse_values