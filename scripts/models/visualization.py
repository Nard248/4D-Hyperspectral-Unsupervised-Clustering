"""
Visualization utilities for hyperspectral data.

This module provides functions for visualizing hyperspectral data,
reconstruction results, and clustering outputs.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch

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


def overlay_clusters_with_consistent_colors(
        cluster_labels,
        rgb_image,
        alpha=0.5,
        mask=None,
        output_path=None,
        color_mapping=None
):
    """
    Create a visualization with cluster labels overlaid on an RGB image using consistent colors.

    Args:
        cluster_labels: Array with cluster labels (2D)
        rgb_image: RGB image as numpy array (HxWx3)
        alpha: Transparency of the overlay
        mask: Optional binary mask (1=valid, 0=masked)
        output_path: Path to save the visualization
        color_mapping: Dictionary mapping cluster IDs to colors for consistency

    Returns:
        Overlay image as numpy array
    """
    # Get unique cluster IDs
    unique_clusters = np.unique(cluster_labels)

    # If no color mapping provided, create a default mapping
    if color_mapping is None:
        # Create default colors for clusters
        color_mapping = {}
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        for i, cluster_id in enumerate([c for c in unique_clusters if c >= 0]):
            color_mapping[cluster_id] = colors[i % len(colors)]

    # Create an empty overlay
    overlay = np.zeros((*cluster_labels.shape, 4))  # RGBA

    # Fill with cluster colors using the mapping
    for cluster_id in unique_clusters:
        if cluster_id >= 0:  # Skip -1 (masked areas) for the overlay
            # Find pixels belonging to this cluster
            mask_cluster = cluster_labels == cluster_id
            if not np.any(mask_cluster):
                continue

            # Get color for this cluster
            color = color_mapping[cluster_id]

            # Set color and alpha for this cluster
            overlay[mask_cluster] = (*color[:3], alpha)

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
    fig, ax = plt.subplots(figsize=(16, 14))

    # Show RGB image
    ax.imshow(rgb_image)

    # Add overlay
    ax.imshow(overlay, alpha=overlay[..., 3])

    # No colorbar - as requested
    ax.set_title('Cluster Overlay on RGB Image', fontsize=24)
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

def visualize_4d_cluster_profiles_consistent(
        cluster_results,
        dataset,
        original_data=None,
        output_dir=None,
        color_mapping=None
):
    """
    Visualize the spectral profiles of each cluster for 4D data with consistent colors.

    Args:
        cluster_results: Results from run_4d_pixel_wise_clustering
        dataset: MaskedHyperspectralDataset instance
        original_data: Optional dictionary with original data
        output_dir: Directory to save visualizations
        color_mapping: Dictionary mapping cluster IDs to colors for consistency

    Returns:
        Dictionary with cluster profile statistics
    """
    # Get cluster labels and excitations
    cluster_labels = cluster_results['cluster_labels']
    excitations_used = cluster_results['excitations_used']
    n_clusters = cluster_results['n_clusters']  # Use this to ensure we show ALL clusters

    # Get unique cluster IDs (excluding -1 which is used for masked areas)
    unique_clusters = sorted([c for c in np.unique(cluster_labels) if c >= 0])
    print(f"Analyzing profiles for {len(unique_clusters)} clusters using {len(excitations_used)} excitations")

    # If no color mapping provided, create a default mapping
    if color_mapping is None:
        # Create default colors for ALL possible clusters (0 to n_clusters-1)
        color_mapping = {}
        colors = plt.cm.tab10(np.linspace(0, 1, max(10, n_clusters)))
        for cluster_id in range(n_clusters):
            color_mapping[cluster_id] = colors[cluster_id % len(colors)]

    # Get original data if not provided
    if original_data is None:
        original_data = dataset.get_all_data()

    # Get emission wavelengths
    emission_wavelengths = {}
    if hasattr(dataset, 'emission_wavelengths'):
        emission_wavelengths = dataset.emission_wavelengths

    # Store cluster statistics
    cluster_stats = {}

    # Process each excitation and create individual plots
    for ex in excitations_used:
        # Create a figure for spectral profiles for this excitation
        plt.figure(figsize=(16, 12))

        # Get data for this excitation
        data = original_data[ex].cpu().numpy()

        # Get emission wavelengths for x-axis
        if ex in emission_wavelengths:
            x_values = emission_wavelengths[ex]
        else:
            x_values = np.arange(data.shape[2])

        # Calculate mean spectrum for each cluster using exact same colors
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

            # Plot mean spectrum with consistent color from mapping
            plt.plot(x_values, mean_spectrum, marker='o',
                     label=f"Cluster {cluster_id}",
                     color=color_mapping[cluster_id][:3],  # Use same color as cluster map
                     linewidth=3)

            # Plot error bands
            plt.fill_between(
                x_values,
                mean_spectrum - std_spectrum,
                mean_spectrum + std_spectrum,
                alpha=0.2,
                color=color_mapping[cluster_id][:3]
            )

        # Add legend, labels, etc.
        plt.legend(loc='best', fontsize=16)
        plt.xlabel('Emission Wavelength (nm)' if len(emission_wavelengths) > 0 else 'Emission Band Index', fontsize=18)
        plt.ylabel('Intensity', fontsize=18)
        plt.title(f'Spectral Profiles by Cluster (Excitation {ex}nm)', fontsize=24)
        plt.grid(True, alpha=0.3)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        if output_dir is not None:
            plt.savefig(os.path.join(output_dir, f"cluster_profiles_ex{ex}.png"),
                        dpi=300, bbox_inches='tight')

        plt.close()

    # Create a combined visualization showing cluster profiles across excitations
    plt.figure(figsize=(20, 16))

    # For each cluster, create a subplot showing mean intensity across excitations
    # Use n_clusters to ensure we display ALL possible clusters
    rows = int(np.ceil(n_clusters / 2))

    for cluster_id in range(n_clusters):  # Loop through ALL possible cluster IDs
        plt.subplot(rows, 2, cluster_id + 1)

        # Check if this cluster has any data
        if cluster_id in cluster_stats:
            # Collect mean intensities across all excitations
            ex_values = []
            mean_intensities = []
            std_intensities = []

            for ex in excitations_used:
                if ex in cluster_stats[cluster_id]:
                    ex_values.append(ex)
                    mean_intensities.append(np.mean(cluster_stats[cluster_id][ex]['mean']))
                    std_intensities.append(np.mean(cluster_stats[cluster_id][ex]['std']))

            # Plot mean intensity vs excitation wavelength with consistent color
            plt.errorbar(ex_values, mean_intensities, yerr=std_intensities,
                         marker='o', linestyle='-', capsize=3,
                         color=color_mapping[cluster_id][:3],
                         linewidth=3,
                         markersize=8)

            # Add count information
            if ex_values:  # Only if we have data
                first_ex = ex_values[0]
                count = cluster_stats[cluster_id][first_ex]['count']
                plt.text(0.02, 0.95, f"Pixels: {count}",
                         transform=plt.gca().transAxes, fontsize=12,
                         va='top', ha='left', bbox=dict(facecolor='white', alpha=0.7))
        else:
            # This cluster has no pixels - show a message
            plt.text(0.5, 0.5, f"No pixels assigned",
                     ha='center', va='center', transform=plt.gca().transAxes,
                     fontsize=16, color='gray')

        plt.title(f'Cluster {cluster_id}', fontsize=20)
        plt.xlabel('Excitation Wavelength (nm)', fontsize=16)
        plt.ylabel('Mean Intensity', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

    plt.tight_layout()
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, "4d_cluster_excitation_profiles.png"),
                    dpi=300, bbox_inches='tight')

    plt.close()

    # Create a bar chart of cluster sizes with consistent colors
    plt.figure(figsize=(16, 10))

    # Create a list of all possible cluster IDs (0 to n_clusters-1)
    all_cluster_ids = list(range(n_clusters))

    # Get sizes for all clusters (including ones with no pixels)
    sizes = []
    colors = []
    x_labels = []

    for cluster_id in all_cluster_ids:
        # Get count (0 if this cluster doesn't exist in the results)
        size = np.sum(cluster_labels == cluster_id)
        sizes.append(size)

        # Get color from color mapping
        color = color_mapping[cluster_id][:3]
        colors.append(color)

        # Add label
        x_labels.append(f'Cluster {cluster_id}')

    # Create the bars with exactly the same colors as used in cluster maps
    x_pos = np.arange(len(all_cluster_ids))
    plt.bar(x_pos, sizes, color=colors)

    plt.xlabel('Cluster', fontsize=18)
    plt.ylabel('Number of Pixels', fontsize=18)
    plt.title(f'Cluster Sizes (4D Clustering, K={n_clusters})', fontsize=24)
    plt.xticks(x_pos, x_labels, rotation=45, fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, axis='y', alpha=0.3)

    # Add value labels on top of each bar
    for i, v in enumerate(sizes):
        if v > 0:  # Only add text for non-zero values
            plt.text(i, v + max(sizes)*0.01, str(v),
                    color='black', fontweight='bold', ha='center', fontsize=12)

    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, f"4d_cluster_sizes.png"),
                    dpi=300, bbox_inches='tight')

    plt.close()

    return cluster_stats
