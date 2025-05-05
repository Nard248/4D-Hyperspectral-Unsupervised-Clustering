import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
from sklearn.cluster import MiniBatchKMeans
from pathlib import Path
import time
from typing import Dict, List, Tuple, Optional, Union, Any

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

torch.manual_seed(42)
np.random.seed(42)


class Hyperspectral4DClusteringProcessor:
    """
    Efficient processor for hyperspectral data that preserves 4D information
    by using feature averaging instead of dimension reduction.
    """

    def __init__(
            self,
            data_path: str,
            mask_path: Optional[str] = None,
            output_dir: str = "results",
            downscale_factor: int = 1,
            normalize: bool = True,
            n_clusters: int = 5,
            max_samples: int = 100000  # Max samples to use for clustering
    ):
        """
        Initialize the processor.

        Args:
            data_path: Path to the pickle file containing hyperspectral data
            mask_path: Optional path to a numpy file containing a binary mask
            output_dir: Directory to save results
            downscale_factor: Factor to downscale the spatial dimensions
            normalize: Whether to normalize the data
            n_clusters: Number of clusters for K-means
            max_samples: Maximum number of samples to use for clustering (for memory efficiency)
        """
        self.data_path = data_path
        self.mask_path = mask_path
        self.output_dir = output_dir
        self.downscale_factor = downscale_factor
        self.normalize = normalize
        self.n_clusters = n_clusters
        self.max_samples = max_samples

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize attributes
        self.data_dict = None
        self.mask = None
        self.excitation_wavelengths = []
        self.processed_data = {}
        self.emission_wavelengths = {}
        self.height = None
        self.width = None
        self.cluster_results = {}
        self.combined_cluster_labels = None

    def load_data(self):
        """Load hyperspectral data and mask."""
        print(f"Loading data from {self.data_path}...")
        start_time = time.time()

        with open(self.data_path, 'rb') as f:
            self.data_dict = pickle.load(f)

        # Extract excitation wavelengths
        if 'excitation_wavelengths' in self.data_dict:
            self.excitation_wavelengths = self.data_dict['excitation_wavelengths']
        else:
            # Try to extract from data keys
            if 'data' in self.data_dict and isinstance(self.data_dict['data'], dict):
                self.excitation_wavelengths = sorted([float(k) for k in self.data_dict['data'].keys()])
            else:
                self.excitation_wavelengths = sorted(
                    [float(k) for k in self.data_dict.keys() if k not in ['metadata', 'excitation_wavelengths']])

        print(f"Found {len(self.excitation_wavelengths)} excitation wavelengths")

        # Load mask if provided
        if self.mask_path is not None:
            print(f"Loading mask from {self.mask_path}...")
            self.mask = np.load(self.mask_path)

            # Ensure mask is binary
            if not np.all(np.isin(self.mask, [0, 1])):
                print("Warning: Mask is not binary. Converting to binary (0 and 1).")
                self.mask = (self.mask != 0).astype(np.uint8)

            valid_pixels = np.sum(self.mask)
            total_pixels = self.mask.size
            print(f"Mask loaded: {valid_pixels}/{total_pixels} valid pixels ({valid_pixels / total_pixels * 100:.2f}%)")

        print(f"Data loading completed in {time.time() - start_time:.2f} seconds")

    def process_data(self, excitation_subset: Optional[List[float]] = None):
        """
        Process and normalize the hyperspectral data.

        Args:
            excitation_subset: Optional list of excitation wavelengths to process
                              (if None, process all)
        """
        print("Processing hyperspectral data...")
        start_time = time.time()

        # Determine data source (handle different pickle structures)
        data_source = self.data_dict
        if 'data' in self.data_dict and isinstance(self.data_dict['data'], dict):
            data_source = self.data_dict['data']

        # Select excitations to process
        if excitation_subset is None:
            # Process all excitations
            excitations_to_process = self.excitation_wavelengths
        else:
            # Process only the specified subset
            excitations_to_process = [
                ex for ex in self.excitation_wavelengths
                if ex in excitation_subset
            ]

            if not excitations_to_process:
                print("Warning: None of the specified excitations were found in the data.")
                # Fall back to a default subset
                excitations_to_process = self.excitation_wavelengths[:min(4, len(self.excitation_wavelengths))]

        print(f"Processing {len(excitations_to_process)} excitation wavelengths")

        # Get the first excitation to determine dimensions
        first_ex = str(excitations_to_process[0])
        if first_ex in data_source:
            first_data = data_source[first_ex]['cube']
            self.height, self.width = first_data.shape[0], first_data.shape[1]

            # Store emission wavelengths if available
            if 'wavelengths' in data_source[first_ex]:
                self.emission_wavelengths[first_ex] = data_source[first_ex]['wavelengths']
        else:
            print(f"Warning: Excitation {first_ex} not found in data source")
            return

        print(f"Original data dimensions: {self.height}x{self.width}")

        # Apply downscaling
        if self.downscale_factor > 1:
            self.height = self.height // self.downscale_factor
            self.width = self.width // self.downscale_factor
            print(f"Downscaled dimensions: {self.height}x{self.width}")

        # Process mask
        if self.mask is not None:
            # Downscale mask if needed
            if self.downscale_factor > 1:
                from scipy.ndimage import zoom
                self.processed_mask = zoom(self.mask, 1 / self.downscale_factor, order=0)
                # Ensure binary mask
                self.processed_mask = (self.processed_mask > 0.5).astype(np.float32)
            else:
                self.processed_mask = self.mask.astype(np.float32)
        else:
            # Create default mask where all pixels are valid
            self.processed_mask = np.ones((self.height, self.width), dtype=np.float32)

        # Collect all data for normalization
        all_valid_values = []

        for ex in excitations_to_process:
            ex_str = str(ex)
            if ex_str not in data_source:
                print(f"Warning: Excitation {ex} not found in data source")
                continue

            # Get cube data
            cube = data_source[ex_str]['cube']

            # Apply downscaling if needed
            if self.downscale_factor > 1:
                num_bands = cube.shape[2]
                downscaled_data = np.zeros((self.height, self.width, num_bands))

                for h in range(self.height):
                    for w in range(self.width):
                        h_start = h * self.downscale_factor
                        h_end = min((h + 1) * self.downscale_factor, cube.shape[0])
                        w_start = w * self.downscale_factor
                        w_end = min((w + 1) * self.downscale_factor, cube.shape[1])

                        block = cube[h_start:h_end, w_start:w_end, :]
                        downscaled_data[h, w, :] = np.mean(block, axis=(0, 1))

                processed = downscaled_data
            else:
                processed = cube

            # Store emission wavelengths
            if 'wavelengths' in data_source[ex_str]:
                self.emission_wavelengths[ex_str] = data_source[ex_str]['wavelengths']

            # Create a mask showing which values are valid (not NaN)
            valid_mask = ~np.isnan(processed)

            # Store the data and valid mask
            self.processed_data[ex_str] = {
                'data': processed,
                'valid_mask': valid_mask
            }

            # Collect valid values for normalization
            if self.normalize:
                all_valid_values.extend(processed[valid_mask].flatten())

        # Normalize if requested
        if self.normalize and all_valid_values:
            # Calculate global min/max
            global_min = np.min(all_valid_values)
            global_max = np.max(all_valid_values)

            print(f"Global data range: [{global_min:.4f}, {global_max:.4f}]")

            # Normalize each excitation
            for ex_str in self.processed_data:
                data = self.processed_data[ex_str]['data']
                valid_mask = self.processed_data[ex_str]['valid_mask']

                # Normalize only valid values
                normalized_data = np.copy(data)
                normalized_data[valid_mask] = (data[valid_mask] - global_min) / (global_max - global_min)

                self.processed_data[ex_str]['data'] = normalized_data

            print("Data normalized to range [0, 1]")

        print(f"Data processing completed in {time.time() - start_time:.2f} seconds")

    def create_rgb_visualization(self, r_band=650, g_band=550, b_band=450):
        """
        Create RGB false color visualizations from hyperspectral data.

        Args:
            r_band, g_band, b_band: Target wavelengths for R, G, B channels

        Returns:
            Dictionary mapping excitation wavelengths to RGB images
        """
        print("Creating RGB visualizations...")
        start_time = time.time()

        # Create directory for visualizations
        vis_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        # Track global min and max for consistent normalization
        global_min = float('inf')
        global_max = float('-inf')

        # First pass to determine global range
        for ex_str in self.processed_data:
            data = self.processed_data[ex_str]['data']

            # Get band indices
            if ex_str in self.emission_wavelengths:
                wavelengths = self.emission_wavelengths[ex_str]
                r_idx = np.argmin(np.abs(np.array(wavelengths) - r_band))
                g_idx = np.argmin(np.abs(np.array(wavelengths) - g_band))
                b_idx = np.argmin(np.abs(np.array(wavelengths) - b_band))
            else:
                # Use indices proportionally if wavelengths not available
                bands = data.shape[2]
                r_idx = int(bands * 0.8)  # Red ~ 80% through bands
                g_idx = int(bands * 0.5)  # Green ~ middle
                b_idx = int(bands * 0.2)  # Blue ~ 20% through bands

            # Get channel data
            r_channel = data[:, :, r_idx]
            g_channel = data[:, :, g_idx]
            b_channel = data[:, :, b_idx]

            # Update global min/max
            local_min = min(np.nanmin(r_channel), np.nanmin(g_channel), np.nanmin(b_channel))
            local_max = max(np.nanmax(r_channel), np.nanmax(g_channel), np.nanmax(b_channel))

            global_min = min(global_min, local_min)
            global_max = max(global_max, local_max)

        # Second pass to create RGB images
        rgb_images = {}

        for ex_str in self.processed_data:
            data = self.processed_data[ex_str]['data']

            # Get band indices
            if ex_str in self.emission_wavelengths:
                wavelengths = self.emission_wavelengths[ex_str]
                r_idx = np.argmin(np.abs(np.array(wavelengths) - r_band))
                g_idx = np.argmin(np.abs(np.array(wavelengths) - g_band))
                b_idx = np.argmin(np.abs(np.array(wavelengths) - b_band))
            else:
                bands = data.shape[2]
                r_idx = int(bands * 0.8)
                g_idx = int(bands * 0.5)
                b_idx = int(bands * 0.2)

            # Create RGB image
            rgb = np.stack([
                data[:, :, r_idx],  # R channel
                data[:, :, g_idx],  # G channel
                data[:, :, b_idx]  # B channel
            ], axis=2)

            # Apply mask
            rgb = rgb * self.processed_mask[:, :, np.newaxis]

            # Normalize
            rgb_normalized = np.clip((rgb - global_min) / (global_max - global_min + 1e-8), 0, 1)

            # Replace NaNs with zeros
            rgb_normalized = np.nan_to_num(rgb_normalized, nan=0.0)

            # Store RGB image
            rgb_images[ex_str] = rgb_normalized

            # Save individual RGB image
            plt.figure(figsize=(8, 8))
            plt.imshow(rgb_normalized)
            plt.title(f'RGB Composite (Ex {ex_str}nm)')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f"rgb_ex{ex_str}.png"), dpi=300)
            plt.close()

        print(f"RGB visualization completed in {time.time() - start_time:.2f} seconds")
        return rgb_images

    def run_clustering(self, excitation_subset: Optional[List[str]] = None):
        """
        Run clustering on the hyperspectral data, preserving 4D information.
        Now with proper NaN handling.

        Args:
            excitation_subset: Optional list of excitation wavelengths to use for clustering
                              (if None, use all processed excitations)
        """
        print("Running clustering...")
        start_time = time.time()

        # Create directory for clustering results
        cluster_dir = os.path.join(self.output_dir, "clustering")
        os.makedirs(cluster_dir, exist_ok=True)

        # Select excitations to process
        excitations_to_process = []

        if excitation_subset is not None:
            # Try to match the specified excitations
            for ex in excitation_subset:
                # Try different formats
                if str(ex) in self.processed_data:
                    excitations_to_process.append(str(ex))
                elif ex in self.processed_data:
                    excitations_to_process.append(ex)
                # Try rounding to one decimal
                elif str(round(ex, 1)) in self.processed_data:
                    excitations_to_process.append(str(round(ex, 1)))

        # If no matches or no subset specified, use all available
        if not excitations_to_process:
            excitations_to_process = list(self.processed_data.keys())
            print(f"Using all available excitations: {excitations_to_process}")
        else:
            print(f"Using specified excitations: {excitations_to_process}")

        print(f"Running clustering on {len(excitations_to_process)} excitation wavelengths")

        # 1. Run individual clustering for each excitation
        for ex_str in excitations_to_process:
            print(f"  Clustering excitation {ex_str}...")
            data = self.processed_data[ex_str]['data']

            # Reshape to [pixels, bands]
            height, width, bands = data.shape
            reshaped = data.reshape(height * width, bands)

            # Apply mask and handle NaN values
            if self.processed_mask is not None:
                mask_flat = self.processed_mask.flatten()
                valid_indices = np.where(mask_flat > 0)[0]
                features = reshaped[valid_indices]
            else:
                valid_indices = np.arange(height * width)
                features = reshaped

            # Handle NaN values - CRITICAL FIX
            # First check if there are any NaNs
            if np.any(np.isnan(features)):
                print(f"    Found NaN values in data. Handling NaNs...")

                # Option 1: Remove samples with any NaN values
                nan_rows = np.any(np.isnan(features), axis=1)
                clean_features = features[~nan_rows]
                clean_indices = valid_indices[~nan_rows]

                print(f"    Removed {np.sum(nan_rows)} samples with NaNs. Remaining: {len(clean_features)}")

                # If we removed too many, fallback to imputation
                if len(clean_features) < 1000:  # Arbitrary threshold
                    print("    Too many NaNs, using imputation instead...")
                    # Replace NaNs with zeros
                    features = np.nan_to_num(features, nan=0.0)
                    clean_features = features
                    clean_indices = valid_indices
            else:
                clean_features = features
                clean_indices = valid_indices

            # Check if we need to subsample for memory efficiency
            if len(clean_features) > self.max_samples:
                print(f"    Subsampling from {len(clean_features)} to {self.max_samples} pixels for efficiency")
                sample_indices = np.random.choice(len(clean_features), self.max_samples, replace=False)
                features_subset = clean_features[sample_indices]

                # Run clustering on the subset
                kmeans = MiniBatchKMeans(
                    n_clusters=self.n_clusters,
                    batch_size=1000,
                    max_iter=100,
                    random_state=42
                )
                kmeans.fit(features_subset)

                # Predict on all clean data
                labels = kmeans.predict(clean_features)
            else:
                # Run clustering on all clean data
                kmeans = MiniBatchKMeans(
                    n_clusters=self.n_clusters,
                    batch_size=1000,
                    max_iter=100,
                    random_state=42
                )
                labels = kmeans.fit_predict(clean_features)

            # Create a cluster map
            cluster_map = np.ones((height, width)) * -1  # Default to -1 (masked)
            for i, idx in enumerate(clean_indices):
                y, x = idx // width, idx % width
                cluster_map[y, x] = labels[i]

            # Store the result
            self.cluster_results[ex_str] = cluster_map

            # Save the cluster map
            plt.figure(figsize=(8, 8))
            plt.imshow(cluster_map, cmap='tab10', interpolation='nearest')
            plt.colorbar(label='Cluster')
            plt.title(f'Clustering (Ex {ex_str}nm)')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(cluster_dir, f"cluster_map_ex{ex_str}.png"), dpi=300)
            plt.close()

        # 2. Run combined clustering using feature averaging across excitations
        print("  Running combined 4D clustering using feature averaging...")

        # Get data dimensions from the first excitation
        first_ex = excitations_to_process[0]
        height, width, _ = self.processed_data[first_ex]['data'].shape

        # Step 1: Identify the excitation with the most representative spectral resolution
        band_counts = {ex: self.processed_data[ex]['data'].shape[2] for ex in excitations_to_process}
        representative_ex = max(band_counts, key=band_counts.get)
        rep_bands = band_counts[representative_ex]

        print(f"  Using excitation {representative_ex} with {rep_bands} bands as representative")

        # Step 2: Create a 4D feature tensor with standardized dimensions
        # Determine the number of valid pixels
        if self.processed_mask is not None:
            n_valid_pixels = np.sum(self.processed_mask)
        else:
            n_valid_pixels = height * width

        print(f"  Creating 4D feature tensor for {n_valid_pixels} valid pixels")

        # Create arrays to store indices and features
        valid_indices = []

        # Get the valid pixel indices
        if self.processed_mask is not None:
            mask_flat = self.processed_mask.flatten()
            for i in range(len(mask_flat)):
                if mask_flat[i] > 0:
                    valid_indices.append(i)
        else:
            valid_indices = list(range(height * width))

        # Average across excitations to create a single emission spectrum
        avg_features = np.zeros((len(valid_indices), rep_bands))
        valid_pixel_mask = np.ones(len(valid_indices), dtype=bool)  # Track valid pixels

        print("  Extracting features from all excitations for each pixel...")

        for i, pixel_idx in enumerate(valid_indices):
            # Convert flat index to 2D coordinates
            y, x = pixel_idx // width, pixel_idx % width

            # Collect spectra from all excitations
            pixel_spectra = []
            pixel_has_nans = False

            for ex in excitations_to_process:
                data = self.processed_data[ex]['data']
                spectrum = data[y, x, :]

                # Check for NaNs in this spectrum
                if np.any(np.isnan(spectrum)):
                    pixel_has_nans = True
                    continue  # Skip this excitation for this pixel

                # Resample to match the representative excitation if needed
                if spectrum.shape[0] != rep_bands:
                    # Simple linear interpolation for resampling
                    old_indices = np.linspace(0, 1, spectrum.shape[0])
                    new_indices = np.linspace(0, 1, rep_bands)
                    try:
                        spectrum = np.interp(new_indices, old_indices, spectrum)
                    except:
                        # Skip if interpolation fails
                        continue

                pixel_spectra.append(spectrum)

            # If we have no valid spectra for this pixel, mark it as invalid
            if not pixel_spectra:
                valid_pixel_mask[i] = False
                continue

            # Average the spectra
            avg_spectrum = np.nanmean(pixel_spectra, axis=0)

            # Final NaN check - if still NaNs after averaging, mark as invalid
            if np.any(np.isnan(avg_spectrum)):
                valid_pixel_mask[i] = False
                continue

            avg_features[i] = avg_spectrum

            # Progress update
            if i % 10000 == 0:
                print(f"    Processed {i}/{len(valid_indices)} pixels")

        # Filter out invalid pixels
        final_features = avg_features[valid_pixel_mask]
        final_indices = np.array(valid_indices)[valid_pixel_mask]

        print(f"  Final valid pixels for clustering: {len(final_features)}/{len(valid_indices)}")

        # Check if we need to subsample for memory efficiency
        if len(final_features) > self.max_samples:
            print(f"  Subsampling from {len(final_features)} to {self.max_samples} pixels for efficiency")
            sample_indices = np.random.choice(len(final_features), self.max_samples, replace=False)
            features_subset = final_features[sample_indices]

            # Run clustering on the subset
            kmeans = MiniBatchKMeans(
                n_clusters=self.n_clusters,
                batch_size=min(1000, self.max_samples),
                max_iter=100,
                random_state=42
            )
            kmeans.fit(features_subset)

            # Predict on all data
            labels = kmeans.predict(final_features)
        else:
            # Run clustering on all data
            kmeans = MiniBatchKMeans(
                n_clusters=self.n_clusters,
                batch_size=1000,
                max_iter=100,
                random_state=42
            )
            labels = kmeans.fit_predict(final_features)

        # Create a cluster map
        combined_cluster_map = np.ones((height, width)) * -1  # Default to -1 (masked)
        for i, pixel_idx in enumerate(final_indices):
            y, x = pixel_idx // width, pixel_idx % width
            combined_cluster_map[y, x] = labels[i]

        # Store the combined result
        self.combined_cluster_labels = combined_cluster_map

        # Save the combined cluster map
        plt.figure(figsize=(8, 8))
        plt.imshow(combined_cluster_map, cmap='tab10', interpolation='nearest')
        plt.colorbar(label='Cluster')
        plt.title('Combined 4D Clustering')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(cluster_dir, "combined_cluster_map.png"), dpi=300)
        plt.close()

        print(f"Clustering completed in {time.time() - start_time:.2f} seconds")
    def create_overlays(self, rgb_images):
        """
        Create cluster overlays on RGB images.

        Args:
            rgb_images: Dictionary mapping excitation wavelengths to RGB images
        """
        print("Creating cluster overlays...")
        start_time = time.time()

        # Create directory for overlays
        overlay_dir = os.path.join(self.output_dir, "overlays")
        os.makedirs(overlay_dir, exist_ok=True)

        # Create function for overlay creation
        def create_overlay(rgb_image, cluster_labels, alpha=0.5):
            # Get unique clusters (exclude -1 for masked areas)
            unique_clusters = sorted([c for c in np.unique(cluster_labels) if c >= 0])
            n_clusters = len(unique_clusters)

            # Create a colormap for clusters
            cluster_cmap = plt.cm.get_cmap('tab10', max(10, n_clusters))

            # Create empty overlay (RGBA)
            overlay = np.zeros((*cluster_labels.shape, 4))

            # Fill with cluster colors
            for i, cluster_id in enumerate(unique_clusters):
                mask_cluster = cluster_labels == cluster_id
                color = cluster_cmap(i % 10)
                overlay[mask_cluster] = (*color[:3], alpha)

            # Set transparent for masked areas
            mask = cluster_labels < 0
            overlay[mask] = (0, 0, 0, 0)

            return overlay

        # Choose a reference excitation for overlays
        reference_ex = list(rgb_images.keys())[0]
        reference_rgb = rgb_images[reference_ex]

        # 1. Create individual cluster overlays
        for ex_str, cluster_map in self.cluster_results.items():
            if ex_str in rgb_images:
                rgb_image = rgb_images[ex_str]
            else:
                rgb_image = reference_rgb

            # Create overlay
            overlay = create_overlay(rgb_image, cluster_map)

            # Create figure
            plt.figure(figsize=(12, 10))
            plt.imshow(rgb_image)
            plt.imshow(overlay, alpha=overlay[..., 3])
            plt.title(f'Cluster Overlay (Ex {ex_str}nm)')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(overlay_dir, f"overlay_ex{ex_str}.png"), dpi=300)
            plt.close()

        # 2. Create combined cluster overlay
        if self.combined_cluster_labels is not None:
            # Create overlay
            overlay = create_overlay(reference_rgb, self.combined_cluster_labels)

            # Create figure
            plt.figure(figsize=(12, 10))
            plt.imshow(reference_rgb)
            plt.imshow(overlay, alpha=overlay[..., 3])
            plt.title('Combined 4D Cluster Overlay')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(overlay_dir, "overlay_combined.png"), dpi=300)
            plt.close()

        print(f"Overlays created in {time.time() - start_time:.2f} seconds")

    def create_side_by_side_comparison(self, rgb_images):
        """
        Create a side-by-side comparison of original RGB and clustering results.

        Args:
            rgb_images: Dictionary mapping excitation wavelengths to RGB images
        """
        print("Creating side-by-side comparison...")
        start_time = time.time()

        # Choose excitations to display
        if len(self.cluster_results) > 4:
            # Display first 4 excitations for clarity
            excitations_to_display = sorted(list(self.cluster_results.keys()))[:4]
        else:
            excitations_to_display = sorted(list(self.cluster_results.keys()))

        # Choose reference RGB image
        reference_ex = list(rgb_images.keys())[0]
        reference_rgb = rgb_images[reference_ex]

        # Create figure with original RGB and all clustering results
        n_plots = len(excitations_to_display) + 2  # +2 for RGB and combined
        fig, axes = plt.subplots(1, n_plots, figsize=(n_plots * 4, 5))

        # Plot original RGB
        axes[0].imshow(reference_rgb)
        axes[0].set_title(f'RGB Composite\n(Ex {reference_ex}nm)')
        axes[0].axis('off')

        # Plot individual clusterings
        for i, ex_str in enumerate(excitations_to_display):
            axes[i + 1].imshow(self.cluster_results[ex_str], cmap='tab10')
            axes[i + 1].set_title(f'Clustering\n(Ex {ex_str}nm)')
            axes[i + 1].axis('off')

        # Plot combined clustering
        if self.combined_cluster_labels is not None:
            axes[-1].imshow(self.combined_cluster_labels, cmap='tab10')
            axes[-1].set_title('Combined 4D\nClustering')
            axes[-1].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "full_comparison.png"), dpi=300)
        plt.close()

        print(f"Side-by-side comparison created in {time.time() - start_time:.2f} seconds")

    def run_pipeline(self, excitation_subset=None):
        """
        Run the complete pipeline.

        Args:
            excitation_subset: Optional list of excitation wavelengths to process
                              (if None, use all excitations)
        """
        print("\n=== Starting hyperspectral processing pipeline ===")
        total_start_time = time.time()

        # 1. Load data
        self.load_data()

        # 2. Process data
        self.process_data(excitation_subset)

        # 3. Create RGB visualizations
        rgb_images = self.create_rgb_visualization()

        # 4. Run clustering
        self.run_clustering(excitation_subset)

        # 5. Create overlays
        self.create_overlays(rgb_images)

        # 6. Create side-by-side comparison
        self.create_side_by_side_comparison(rgb_images)

        total_time = time.time() - total_start_time
        print(f"\n=== Pipeline completed in {total_time:.2f} seconds ===")