import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
from scipy import stats
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time
import datetime
import warnings


class HyperspectralAnalyzerWithMask:
    """Class for analyzing hyperspectral data stored in HDF5 format with mask support"""

    def __init__(self, h5_file_path):
        """
        Initialize the analyzer with the path to the HDF5 file

        Args:
            h5_file_path: Path to the HDF5 file containing hyperspectral data
        """
        self.h5_file_path = h5_file_path
        self.data_dict = {}
        self.stats = {}
        self.emission_wavelengths = {}
        self.excitation_wavelengths = []
        self.metadata = {}
        self.mask = None

        # Load the data and mask
        self._load_data()

    def _load_data(self):
        """Load data and mask from the HDF5 file"""
        with h5py.File(self.h5_file_path, 'r') as f:
            # Extract metadata
            if 'metadata' in f:
                self.metadata = {key: f['metadata'].attrs[key] for key in f['metadata'].attrs}

            # Load mask if present
            if 'mask' in f:
                self.mask = f['mask']['data'][:]
                print(f"Loaded mask from HDF5 file. Mask has {np.sum(self.mask)} masked pixels "
                      f"({(np.sum(self.mask) / self.mask.size) * 100:.2f}% of total)")

            # Read each excitation group
            for group_name in f.keys():
                if group_name.startswith('excitation_'):
                    # Extract excitation wavelength from group name
                    excitation_wavelength = int(group_name.split('_')[1])
                    self.excitation_wavelengths.append(excitation_wavelength)

                    group = f[group_name]
                    excitation_data = {
                        'wavelengths': group['wavelengths'][:],
                        'cubes': {}
                    }

                    # Store emission wavelengths for this excitation
                    self.emission_wavelengths[excitation_wavelength] = group['wavelengths'][:]

                    # Read individual cubes
                    for dataset_name in group.keys():
                        if dataset_name.startswith('cube_'):
                            index = int(dataset_name.split('_')[1])
                            cube_group = group[dataset_name]
                            cube_data = cube_group['data'][:]

                            # Extract header information
                            header = {key: cube_group.attrs[key] for key in cube_group.attrs}

                            excitation_data['cubes'][index] = {
                                'data': cube_data,
                                'header': header
                            }
                        elif dataset_name == 'average_cube':
                            excitation_data['average_cube'] = group[dataset_name][:]

                    self.data_dict[excitation_wavelength] = excitation_data

            # Sort excitation wavelengths
            self.excitation_wavelengths.sort()

        # If no mask was found, print a warning
        if self.mask is None:
            warnings.warn("No mask found in the HDF5 file. Analysis will be performed on the entire image.",
                          UserWarning)

    def analyze_dataset(self, output_dir='analysis_results_masked',
                        filename_prefix='masked_analysis_'):
        """
        Perform comprehensive analysis of the dataset using the mask

        Args:
            output_dir: Directory to save analysis results
            filename_prefix: Prefix for output filenames
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Time the analysis
        start_time = time.time()

        # Open a file to write results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"{filename_prefix}hyperspectral_analysis_{timestamp}.txt")

        with open(output_file, 'w') as f:
            # Write header
            f.write("=" * 80 + "\n")
            f.write("HYPERSPECTRAL DATA ANALYSIS REPORT (MASKED REGION ONLY)\n")
            f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"HDF5 File: {os.path.basename(self.h5_file_path)}\n")
            f.write("=" * 80 + "\n\n")

            # Mask information
            f.write("1. MASK INFORMATION\n")
            f.write("-" * 40 + "\n")

            if self.mask is not None:
                masked_pixels = np.sum(self.mask)
                total_pixels = self.mask.size
                percent_masked = (masked_pixels / total_pixels) * 100
                f.write(f"Mask present: Yes\n")
                f.write(f"Total pixels in image: {total_pixels}\n")
                f.write(f"Masked pixels (region of interest): {masked_pixels}\n")
                f.write(f"Percentage of image masked: {percent_masked:.2f}%\n")

                # Save a visualization of the mask
                plt.figure(figsize=(8, 6))
                plt.imshow(self.mask, cmap='gray')
                plt.title('Analysis Mask (White = Region of Interest)')
                plt.colorbar(label='Mask Value')
                mask_file = os.path.join(output_dir, f"{filename_prefix}mask.png")
                plt.savefig(mask_file, dpi=150)
                plt.close()
                f.write(f"Mask visualization saved to: {os.path.basename(mask_file)}\n")
            else:
                f.write(f"Mask present: No\n")
                f.write(f"Analysis performed on entire image\n")

            # General dataset information
            f.write("\n2. GENERAL DATASET INFORMATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Number of excitation wavelengths: {len(self.excitation_wavelengths)}\n")
            f.write(f"Excitation wavelengths: {self.excitation_wavelengths} nm\n")

            # Calculate the mean number of emission wavelengths
            num_emission_wavelengths = [len(self.emission_wavelengths[ex]) for ex in self.excitation_wavelengths]
            f.write(f"Mean number of emission wavelengths per excitation: {np.mean(num_emission_wavelengths):.1f}\n")
            f.write(
                f"Range of emission wavelengths: {np.min([wave.min() for wave in self.emission_wavelengths.values()]):.1f} - {np.max([wave.max() for wave in self.emission_wavelengths.values()]):.1f} nm\n")

            # Calculate dataset size
            total_size_bytes = 0
            num_cubes = 0
            for ex, data in self.data_dict.items():
                for idx, cube in data['cubes'].items():
                    total_size_bytes += cube['data'].nbytes
                    num_cubes += 1

            f.write(f"Number of individual data cubes: {num_cubes}\n")
            f.write(f"Total data size: {total_size_bytes / (1024 ** 3):.2f} GB\n")

            if self.metadata:
                f.write("\nMetadata:\n")
                for key, value in self.metadata.items():
                    f.write(f"  {key}: {value}\n")

            # Detailed analysis per excitation wavelength
            f.write("\n\n3. ANALYSIS BY EXCITATION WAVELENGTH (MASKED REGION ONLY)\n")
            f.write("-" * 40 + "\n")

            # Create dataframes to store summary statistics
            excitation_summary = []

            for excitation in self.excitation_wavelengths:
                f.write(f"\n3.{self.excitation_wavelengths.index(excitation) + 1}. Excitation: {excitation} nm\n")
                data = self.data_dict[excitation]

                # Emission wavelengths for this excitation
                emission_waves = data['wavelengths']
                f.write(f"  Number of emission wavelengths: {len(emission_waves)}\n")
                f.write(f"  Emission range: {emission_waves.min():.1f} - {emission_waves.max():.1f} nm\n")

                # Cube information
                num_cubes = len(data['cubes'])
                f.write(f"  Number of sample cubes: {num_cubes}\n")

                # Get first cube to extract shape information
                first_cube_key = list(data['cubes'].keys())[0]
                first_cube = data['cubes'][first_cube_key]['data']
                cube_shape = first_cube.shape
                f.write(f"  Cube dimensions (bands, height, width): {cube_shape}\n")

                # Use the average cube for statistics if available
                if 'average_cube' in data:
                    avg_cube = data['average_cube']

                    # Apply mask if available
                    if self.mask is not None:
                        masked_avg_cube = avg_cube.copy()

                        # For each band, extract only the masked pixels
                        # We'll use the mask to index directly
                        masked_data = [masked_avg_cube[b, self.mask == 1] for b in range(masked_avg_cube.shape[0])]

                        # Compute statistics on masked regions only
                        all_masked_values = np.concatenate(masked_data)

                        stats_dict = {
                            'min': np.min(all_masked_values),
                            'max': np.max(all_masked_values),
                            'mean': np.mean(all_masked_values),
                            'median': np.median(all_masked_values),
                            'std': np.std(all_masked_values),
                            'var': np.var(all_masked_values),
                            'skewness': stats.skew(all_masked_values),
                            'kurtosis': stats.kurtosis(all_masked_values),
                            'signal_range': np.max(all_masked_values) - np.min(all_masked_values),
                            'signal_to_noise': np.mean(all_masked_values) / np.std(all_masked_values) if np.std(
                                all_masked_values) > 0 else float('inf')
                        }
                    else:
                        # If no mask, use the entire cube
                        stats_dict = {
                            'min': np.min(avg_cube),
                            'max': np.max(avg_cube),
                            'mean': np.mean(avg_cube),
                            'median': np.median(avg_cube),
                            'std': np.std(avg_cube),
                            'var': np.var(avg_cube),
                            'skewness': stats.skew(avg_cube.flatten()),
                            'kurtosis': stats.kurtosis(avg_cube.flatten()),
                            'signal_range': np.max(avg_cube) - np.min(avg_cube),
                            'signal_to_noise': np.mean(avg_cube) / np.std(avg_cube) if np.std(avg_cube) > 0 else float(
                                'inf')
                        }

                    f.write("\n  Statistics for masked region:\n")
                    f.write(f"    Minimum value: {stats_dict['min']:.4f}\n")
                    f.write(f"    Maximum value: {stats_dict['max']:.4f}\n")
                    f.write(f"    Mean value: {stats_dict['mean']:.4f}\n")
                    f.write(f"    Median value: {stats_dict['median']:.4f}\n")
                    f.write(f"    Standard deviation: {stats_dict['std']:.4f}\n")
                    f.write(f"    Variance: {stats_dict['var']:.4f}\n")
                    f.write(f"    Skewness: {stats_dict['skewness']:.4f}\n")
                    f.write(f"    Kurtosis: {stats_dict['kurtosis']:.4f}\n")
                    f.write(f"    Signal range: {stats_dict['signal_range']:.4f}\n")
                    f.write(f"    Signal-to-noise ratio: {stats_dict['signal_to_noise']:.4f}\n")

                    # Store the statistics for summary
                    excitation_summary.append({
                        'Excitation': excitation,
                        'Min': stats_dict['min'],
                        'Max': stats_dict['max'],
                        'Mean': stats_dict['mean'],
                        'Median': stats_dict['median'],
                        'StdDev': stats_dict['std'],
                        'SNR': stats_dict['signal_to_noise'],
                        'NumEmissionBands': len(emission_waves)
                    })

                    # Analyze the spectral dimension for masked region
                    f.write("\n  Spectral analysis (emission bands, masked region only):\n")

                    # Get band-wise statistics for masked region
                    if self.mask is not None:
                        # Calculate statistics for each band over just the masked pixels
                        band_means = np.array(
                            [np.mean(masked_avg_cube[b, self.mask == 1]) for b in range(masked_avg_cube.shape[0])])
                        band_stds = np.array(
                            [np.std(masked_avg_cube[b, self.mask == 1]) for b in range(masked_avg_cube.shape[0])])
                        band_maxes = np.array(
                            [np.max(masked_avg_cube[b, self.mask == 1]) for b in range(masked_avg_cube.shape[0])])
                        band_mins = np.array(
                            [np.min(masked_avg_cube[b, self.mask == 1]) for b in range(masked_avg_cube.shape[0])])
                    else:
                        # If no mask, calculate over entire image
                        band_means = np.mean(avg_cube, axis=(1, 2))
                        band_stds = np.std(avg_cube, axis=(1, 2))
                        band_maxes = np.max(avg_cube, axis=(1, 2))
                        band_mins = np.min(avg_cube, axis=(1, 2))

                    # Calculate SNR for each band
                    band_snrs = np.zeros_like(band_means)
                    for i in range(len(band_means)):
                        band_snrs[i] = band_means[i] / band_stds[i] if band_stds[i] > 0 else float('inf')

                    # Find emission wavelengths with highest mean signal
                    top_bands_by_mean = np.argsort(band_means)[-5:][::-1]
                    f.write("    Top 5 emission bands by mean signal:\n")
                    for rank, band_idx in enumerate(top_bands_by_mean):
                        f.write(
                            f"      #{rank + 1}: {emission_waves[band_idx]:.1f} nm, Mean: {band_means[band_idx]:.4f}, SNR: {band_snrs[band_idx]:.4f}\n")

                    # Find emission wavelengths with highest SNR
                    top_bands_by_snr = np.argsort(band_snrs)[-5:][::-1]
                    f.write("    Top 5 emission bands by signal-to-noise ratio:\n")
                    for rank, band_idx in enumerate(top_bands_by_snr):
                        f.write(
                            f"      #{rank + 1}: {emission_waves[band_idx]:.1f} nm, SNR: {band_snrs[band_idx]:.4f}, Mean: {band_means[band_idx]:.4f}\n")

                    # Save spectral profile plot
                    plt.figure(figsize=(10, 6))
                    plt.plot(emission_waves, band_means)
                    plt.fill_between(emission_waves, band_means - band_stds, band_means + band_stds, alpha=0.3)
                    plt.xlabel('Emission Wavelength (nm)')
                    plt.ylabel('Mean Signal (Masked Region Only)')
                    plt.title(f'Spectral Profile for Excitation {excitation} nm (Masked Region)')
                    plt.grid(True, alpha=0.3)
                    spectral_plot_file = os.path.join(output_dir,
                                                      f"{filename_prefix}spectral_profile_ex{excitation}nm.png")
                    plt.savefig(spectral_plot_file, dpi=150)
                    plt.close()
                    f.write(f"\n    Spectral profile saved to: {os.path.basename(spectral_plot_file)}\n")

                    # Calculate and visualize 2D distribution of signal intensity for top emission band
                    top_band_idx = top_bands_by_mean[0]
                    top_band_wavelength = emission_waves[top_band_idx]

                    # Create a masked version of the image for visualization
                    if self.mask is not None:
                        # Get the image for the top emission band
                        band_image = avg_cube[top_band_idx]

                        # Create a version where non-ROI areas are set to NaN for better visualization
                        masked_band_image = np.copy(band_image)
                        masked_band_image[self.mask != 1] = np.nan

                        # Plot both the original and masked images
                        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                        # Original image with ROI outline
                        im0 = axes[0].imshow(band_image, cmap='viridis')
                        # Add mask boundary
                        from skimage import measure
                        contours = measure.find_contours(self.mask, 0.5)
                        for contour in contours:
                            axes[0].plot(contour[:, 1], contour[:, 0], 'r-', linewidth=2)
                        axes[0].set_title(f'Full Image with ROI Outline\nEmission {top_band_wavelength:.1f} nm')
                        fig.colorbar(im0, ax=axes[0], label='Signal Intensity')

                        # Masked image showing only ROI
                        im1 = axes[1].imshow(masked_band_image, cmap='viridis')
                        axes[1].set_title(f'Masked Region Only\nEmission {top_band_wavelength:.1f} nm')
                        fig.colorbar(im1, ax=axes[1], label='Signal Intensity')

                        plt.tight_layout()
                        spatial_plot_file = os.path.join(output_dir,
                                                         f"{filename_prefix}spatial_distribution_ex{excitation}nm_em{top_band_wavelength:.1f}nm.png")
                        plt.savefig(spatial_plot_file, dpi=150)
                        plt.close()
                        f.write(
                            f"\n    Spatial distribution visualization saved to: {os.path.basename(spatial_plot_file)}\n")

                    # Calculate band-to-band correlations for most interesting bands
                    interesting_bands = np.unique(np.concatenate([top_bands_by_mean, top_bands_by_snr]))
                    if len(interesting_bands) > 1:
                        f.write("\n    Correlations between top emission bands (masked region only):\n")
                        correlation_matrix = np.zeros((len(interesting_bands), len(interesting_bands)))

                        # Pre-compute all masked band data for efficiency
                        if self.mask is not None:
                            masked_band_data = [masked_avg_cube[band_idx, self.mask == 1] for band_idx in
                                                interesting_bands]
                        else:
                            masked_band_data = [avg_cube[band_idx].flatten() for band_idx in interesting_bands]

                        for i, idx1 in enumerate(range(len(interesting_bands))):
                            band1 = interesting_bands[idx1]
                            for j, idx2 in enumerate(range(i + 1, len(interesting_bands))):
                                band2 = interesting_bands[idx2]

                                # Get data for the two bands
                                band1_data = masked_band_data[idx1]
                                band2_data = masked_band_data[idx2]

                                # Calculate correlation
                                correlation = np.corrcoef(band1_data, band2_data)[0, 1]
                                f.write(
                                    f"      {emission_waves[band1]:.1f} nm <-> {emission_waves[band2]:.1f} nm: r = {correlation:.4f}\n")

                                # Store in matrix for visualization
                                correlation_matrix[idx1, idx2] = correlation
                                correlation_matrix[idx2, idx1] = correlation  # Mirror

                        # Set diagonal to 1.0 (self-correlation)
                        np.fill_diagonal(correlation_matrix, 1.0)

                        # Visualize the correlation matrix
                        plt.figure(figsize=(10, 8))
                        band_labels = [f"{emission_waves[band]:.1f}" for band in interesting_bands]
                        sns.heatmap(correlation_matrix, annot=True, fmt=".2f",
                                    xticklabels=band_labels, yticklabels=band_labels,
                                    cmap="coolwarm", vmin=-1, vmax=1)
                        plt.title(f'Band-to-Band Correlation Matrix (Excitation {excitation} nm, Masked Region)')
                        plt.xlabel('Emission Wavelength (nm)')
                        plt.ylabel('Emission Wavelength (nm)')
                        correlation_plot_file = os.path.join(output_dir,
                                                             f"{filename_prefix}correlation_matrix_ex{excitation}nm.png")
                        plt.savefig(correlation_plot_file, dpi=150)
                        plt.close()
                        f.write(f"\n    Correlation matrix saved to: {os.path.basename(correlation_plot_file)}\n")

            # Create a summary dataframe for excitation wavelengths
            excitation_df = pd.DataFrame(excitation_summary)

            # Save the summary dataframe to a CSV file
            excitation_csv = os.path.join(output_dir, f"{filename_prefix}excitation_summary.csv")
            excitation_df.to_csv(excitation_csv, index=False)
            f.write(f"\nExcitation summary statistics saved to: {os.path.basename(excitation_csv)}\n")

            # Comparative analysis across excitation wavelengths
            f.write("\n\n4. COMPARATIVE ANALYSIS ACROSS EXCITATION WAVELENGTHS (MASKED REGION ONLY)\n")
            f.write("-" * 40 + "\n")

            # Compare mean signal across excitation wavelengths
            f.write("\n4.1. Mean Signal Comparison:\n")
            max_mean_ex = excitation_df.loc[excitation_df['Mean'].idxmax()]
            min_mean_ex = excitation_df.loc[excitation_df['Mean'].idxmin()]
            f.write(
                f"  Highest mean signal: Excitation {max_mean_ex['Excitation']} nm (Mean: {max_mean_ex['Mean']:.4f})\n")
            f.write(
                f"  Lowest mean signal: Excitation {min_mean_ex['Excitation']} nm (Mean: {min_mean_ex['Mean']:.4f})\n")
            f.write(f"  Ratio of highest to lowest mean: {max_mean_ex['Mean'] / min_mean_ex['Mean']:.2f}\n")

            # Compare SNR across excitation wavelengths
            f.write("\n4.2. Signal-to-Noise Ratio Comparison:\n")
            max_snr_ex = excitation_df.loc[excitation_df['SNR'].idxmax()]
            min_snr_ex = excitation_df.loc[excitation_df['SNR'].idxmin()]
            f.write(f"  Highest SNR: Excitation {max_snr_ex['Excitation']} nm (SNR: {max_snr_ex['SNR']:.4f})\n")
            f.write(f"  Lowest SNR: Excitation {min_snr_ex['Excitation']} nm (SNR: {min_snr_ex['SNR']:.4f})\n")
            f.write(f"  Ratio of highest to lowest SNR: {max_snr_ex['SNR'] / min_snr_ex['SNR']:.2f}\n")

            # Visualize excitation comparison
            plt.figure(figsize=(12, 6))
            sns.barplot(x='Excitation', y='Mean', data=excitation_df)
            plt.title('Mean Signal by Excitation Wavelength (Masked Region Only)')
            plt.xlabel('Excitation Wavelength (nm)')
            plt.ylabel('Mean Signal')
            plt.grid(True, alpha=0.3)
            excitation_plot_file = os.path.join(output_dir, f"{filename_prefix}excitation_comparison_mean.png")
            plt.savefig(excitation_plot_file, dpi=150)
            plt.close()

            plt.figure(figsize=(12, 6))
            sns.barplot(x='Excitation', y='SNR', data=excitation_df)
            plt.title('Signal-to-Noise Ratio by Excitation Wavelength (Masked Region Only)')
            plt.xlabel('Excitation Wavelength (nm)')
            plt.ylabel('Signal-to-Noise Ratio')
            plt.grid(True, alpha=0.3)
            snr_plot_file = os.path.join(output_dir, f"{filename_prefix}excitation_comparison_snr.png")
            plt.savefig(snr_plot_file, dpi=150)
            plt.close()

            f.write(
                f"\n  Excitation comparison plots saved to: {os.path.basename(excitation_plot_file)} and {os.path.basename(snr_plot_file)}\n")

            # Find potentially most informative excitation-emission combinations
            f.write("\n\n5. POTENTIAL HIGH-INFORMATION EXCITATION-EMISSION COMBINATIONS (MASKED REGION ONLY)\n")
            f.write("-" * 40 + "\n")

            # Create a dataframe to store best emission bands for each excitation
            best_combinations = []

            for excitation in self.excitation_wavelengths:
                data = self.data_dict[excitation]
                emission_waves = data['wavelengths']

                if 'average_cube' in data:
                    avg_cube = data['average_cube']

                    # Calculate band-wise statistics for masked region
                    if self.mask is not None:
                        # Pre-compute masked data for each band
                        masked_avg_cube = avg_cube.copy()
                        band_means = np.array([np.mean(masked_avg_cube[b, self.mask == 1])
                                               for b in range(masked_avg_cube.shape[0])])
                        band_stds = np.array([np.std(masked_avg_cube[b, self.mask == 1])
                                              for b in range(masked_avg_cube.shape[0])])
                    else:
                        # If no mask, use the entire image
                        band_means = np.mean(avg_cube, axis=(1, 2))
                        band_stds = np.std(avg_cube, axis=(1, 2))

                    # Calculate SNR for each band
                    band_snrs = np.zeros_like(band_means)
                    for i in range(len(band_means)):
                        band_snrs[i] = band_means[i] / band_stds[i] if band_stds[i] > 0 else float('inf')

                    # Calculate a composite score for each band (combining mean and SNR)
                    # Normalize the metrics first
                    norm_means = (band_means - np.min(band_means)) / (
                                np.max(band_means) - np.min(band_means)) if np.max(band_means) > np.min(
                        band_means) else np.zeros_like(band_means)
                    norm_snrs = (band_snrs - np.min(band_snrs)) / (np.max(band_snrs) - np.min(band_snrs)) if np.max(
                        band_snrs) > np.min(band_snrs) else np.zeros_like(band_snrs)

                    # Composite score (equal weight to mean and SNR)
                    composite_scores = 0.5 * norm_means + 0.5 * norm_snrs

                    # Find top 3 bands by composite score
                    top_bands = np.argsort(composite_scores)[-3:][::-1]

                    for rank, band_idx in enumerate(top_bands):
                        best_combinations.append({
                            'Excitation': excitation,
                            'Emission': emission_waves[band_idx],
                            'Mean': band_means[band_idx],
                            'SNR': band_snrs[band_idx],
                            'CompositeScore': composite_scores[band_idx],
                            'Rank': rank + 1
                        })

            # Create dataframe of best combinations
            best_df = pd.DataFrame(best_combinations)

            # Sort by composite score
            best_df_sorted = best_df.sort_values('CompositeScore', ascending=False)

            # Display top 10 combinations overall
            f.write("\n5.1. Top 10 excitation-emission combinations by composite score (masked region only):\n")
            for i, row in best_df_sorted.head(10).iterrows():
                f.write(f"  #{i + 1}: Excitation {row['Excitation']} nm, Emission {row['Emission']:.1f} nm")
                f.write(f" (Score: {row['CompositeScore']:.4f}, Mean: {row['Mean']:.4f}, SNR: {row['SNR']:.4f})\n")

            # Save the best combinations to CSV
            best_csv = os.path.join(output_dir, f"{filename_prefix}best_excitation_emission_combinations.csv")
            best_df.to_csv(best_csv, index=False)
            f.write(f"\nBest excitation-emission combinations saved to: {os.path.basename(best_csv)}\n")

            # Visualize top combinations
            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(best_df_sorted['Mean'], best_df_sorted['SNR'],
                                  c=best_df_sorted['CompositeScore'], cmap='viridis',
                                  s=100, alpha=0.7)

            # Add labels for top 5 points
            for i, row in best_df_sorted.head(5).iterrows():
                plt.annotate(f"Ex:{row['Excitation']}, Em:{row['Emission']:.1f}",
                             (row['Mean'], row['SNR']),
                             xytext=(10, 5), textcoords='offset points',
                             fontsize=9, fontweight='bold')

            plt.colorbar(scatter, label='Composite Score')
            plt.xlabel('Mean Signal')
            plt.ylabel('Signal-to-Noise Ratio (SNR)')
            plt.title('Excitation-Emission Combinations by Information Content (Masked Region Only)')
            plt.grid(True, alpha=0.3)
            combinations_plot_file = os.path.join(output_dir, f"{filename_prefix}best_combinations.png")
            plt.savefig(combinations_plot_file, dpi=150)
            plt.close()
            f.write(f"\nBest combinations visualization saved to: {os.path.basename(combinations_plot_file)}\n")

            # Time the analysis
            end_time = time.time()
            analysis_time = end_time - start_time
            f.write(f"\nAnalysis completed in {analysis_time:.2f} seconds.\n")

        print(f"Analysis complete. Results saved to {output_file}")
        return output_file

    def analyze_excitation_emission_matrix(self, output_dir='analysis_results_masked',
                                           filename_prefix='masked_'):
        """
        Create and analyze an excitation-emission matrix (EEM) for the masked region

        Args:
            output_dir: Directory to save analysis results
            filename_prefix: Prefix for output filenames
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Create a list to store the EEM data
        eem_data = []

        # Process each excitation wavelength
        for excitation in self.excitation_wavelengths:
            data = self.data_dict[excitation]
            emission_waves = data['wavelengths']

            if 'average_cube' in data:
                avg_cube = data['average_cube']

                # Calculate band-wise statistics for masked region
                if self.mask is not None:
                    # Pre-compute masked data for each band
                    masked_avg_cube = avg_cube.copy()
                    band_means = np.array([np.mean(masked_avg_cube[b, self.mask == 1])
                                           for b in range(masked_avg_cube.shape[0])])
                else:
                    # If no mask, use the entire image
                    band_means = np.mean(avg_cube, axis=(1, 2))

                # Add each excitation-emission combination
                for i, emission in enumerate(emission_waves):
                    eem_data.append({
                        'Excitation': excitation,
                        'Emission': emission,
                        'Intensity': band_means[i]
                    })

        # Create a dataframe from the EEM data
        eem_df = pd.DataFrame(eem_data)

        # Save the EEM data
        eem_csv = os.path.join(output_dir, f"{filename_prefix}excitation_emission_matrix.csv")
        eem_df.to_csv(eem_csv, index=False)

        # Create a pivot table for visualization
        eem_pivot = eem_df.pivot_table(values='Intensity', index='Emission', columns='Excitation')

        # Plot the EEM
        plt.figure(figsize=(12, 10))

        # Use log scale for better visualization
        sns.heatmap(eem_pivot, cmap='viridis', norm=LogNorm())
        plt.title('Excitation-Emission Matrix (EEM) - Masked Region Only')
        plt.xlabel('Excitation Wavelength (nm)')
        plt.ylabel('Emission Wavelength (nm)')

        # Save the plot
        eem_plot = os.path.join(output_dir, f"{filename_prefix}excitation_emission_matrix.png")
        plt.savefig(eem_plot, dpi=150)
        plt.close()

        # Create a non-log version for comparison
        plt.figure(figsize=(12, 10))
        sns.heatmap(eem_pivot, cmap='viridis')
        plt.title('Excitation-Emission Matrix (EEM) - Masked Region Only (Linear Scale)')
        plt.xlabel('Excitation Wavelength (nm)')
        plt.ylabel('Emission Wavelength (nm)')

        # Save the plot
        eem_plot_linear = os.path.join(output_dir, f"{filename_prefix}excitation_emission_matrix_linear.png")
        plt.savefig(eem_plot_linear, dpi=150)
        plt.close()

        print(f"EEM analysis complete. Results saved to {eem_csv}, {eem_plot}, and {eem_plot_linear}")
        return eem_csv, eem_plot

    def analyze_pca(self, output_dir='analysis_results_masked',
                    filename_prefix='masked_'):
        """
        Perform Principal Component Analysis (PCA) on the spectral dimension for masked region

        Args:
            output_dir: Directory to save analysis results
            filename_prefix: Prefix for output filenames
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Open a file to write results
        pca_output_file = os.path.join(output_dir, f"{filename_prefix}pca_analysis.txt")

        with open(pca_output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("PRINCIPAL COMPONENT ANALYSIS (PCA) REPORT - MASKED REGION ONLY\n")
            f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            # Perform PCA for each excitation wavelength
            for excitation in self.excitation_wavelengths:
                f.write(f"PCA Analysis for Excitation {excitation} nm (Masked Region Only)\n")
                f.write("-" * 40 + "\n")

                data = self.data_dict[excitation]
                emission_waves = data['wavelengths']

                if 'average_cube' in data:
                    avg_cube = data['average_cube']

                    if self.mask is not None:
                        # For masked PCA, we need to reshape the data differently
                        # We'll extract only the pixels within the mask

                        # Get mask indices
                        mask_indices = np.where(self.mask == 1)
                        num_masked_pixels = len(mask_indices[0])

                        # Create a matrix where each row is a masked pixel and each column is a band
                        masked_pixels = np.zeros((num_masked_pixels, avg_cube.shape[0]))

                        for i in range(avg_cube.shape[0]):  # For each band
                            band_data = avg_cube[i]
                            masked_pixels[:, i] = band_data[mask_indices]

                        # Now perform PCA on the masked pixels
                        # Standardize the data
                        scaler = StandardScaler()
                        pixels_scaled = scaler.fit_transform(masked_pixels)

                        # Apply PCA
                        pca = PCA()
                        pca.fit(pixels_scaled)

                        num_bands = avg_cube.shape[0]
                    else:
                        # If no mask, use the original approach
                        num_bands = avg_cube.shape[0]
                        pixels = avg_cube.reshape(num_bands, -1).T  # Each row is a pixel, each column is a band

                        # Standardize the data
                        scaler = StandardScaler()
                        pixels_scaled = scaler.fit_transform(pixels)

                        # Apply PCA
                        pca = PCA()
                        pca.fit(pixels_scaled)

                    # Number of components needed to explain 95% variance
                    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
                    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1

                    f.write(f"Number of emission bands: {num_bands}\n")
                    f.write(f"Number of components needed for 95% variance: {n_components_95}\n")
                    f.write(
                        f"Percentage of variance explained by first component: {pca.explained_variance_ratio_[0] * 100:.2f}%\n")

                    # Top 5 components
                    f.write("\nTop 5 Principal Components:\n")
                    for i in range(min(5, len(pca.explained_variance_ratio_))):
                        f.write(f"PC{i + 1}: {pca.explained_variance_ratio_[i] * 100:.2f}% of variance\n")

                    # Plot the cumulative explained variance
                    plt.figure(figsize=(10, 6))
                    plt.plot(np.arange(1, len(cumulative_variance) + 1), cumulative_variance * 100)
                    plt.xlabel('Number of Components')
                    plt.ylabel('Cumulative Explained Variance (%)')
                    plt.title(f'PCA Cumulative Explained Variance (Excitation {excitation} nm, Masked Region)')
                    plt.axhline(y=95, color='r', linestyle='--')
                    plt.grid(True, alpha=0.3)
                    plt.xlim(0, min(20, len(cumulative_variance)))  # Show only first 20 components

                    # Save the plot
                    pca_plot = os.path.join(output_dir, f"{filename_prefix}pca_variance_ex{excitation}nm.png")
                    plt.savefig(pca_plot, dpi=150)
                    plt.close()

                    f.write(f"\nPCA variance plot saved to: {os.path.basename(pca_plot)}\n")

                    # Analyze the loadings of the first PC to identify most important bands
                    pc1_loadings = pca.components_[0]
                    pc1_loadings_abs = np.abs(pc1_loadings)
                    top_band_indices = np.argsort(pc1_loadings_abs)[-5:][::-1]

                    f.write("\nMost important emission bands according to PC1 (Masked Region):\n")
                    for rank, band_idx in enumerate(top_band_indices):
                        f.write(
                            f"#{rank + 1}: {emission_waves[band_idx]:.1f} nm (Loading: {pc1_loadings[band_idx]:.4f})\n")

                    # Plot the loadings of the first 3 PCs
                    plt.figure(figsize=(12, 6))
                    for i in range(min(3, len(pca.components_))):
                        plt.plot(emission_waves, pca.components_[i],
                                 label=f'PC{i + 1} ({pca.explained_variance_ratio_[i] * 100:.1f}%)')

                    plt.xlabel('Emission Wavelength (nm)')
                    plt.ylabel('PCA Loading')
                    plt.title(f'PCA Loadings by Emission Wavelength (Excitation {excitation} nm, Masked Region)')
                    plt.legend()
                    plt.grid(True, alpha=0.3)

                    # Save the plot
                    pca_loadings_plot = os.path.join(output_dir, f"{filename_prefix}pca_loadings_ex{excitation}nm.png")
                    plt.savefig(pca_loadings_plot, dpi=150)
                    plt.close()

                    f.write(f"\nPCA loadings plot saved to: {os.path.basename(pca_loadings_plot)}\n\n")

        print(f"PCA analysis complete. Results saved to {pca_output_file}")
        return pca_output_file


def main():
    """Main function to run the hyperspectral analysis with mask"""
    # Set input and output paths
    h5_file_path = "kiwi_hyperspectral_4d_data_with_mask.h5"
    output_dir = "analysis_results_masked"

    # Check if the H5 file exists
    if not os.path.exists(h5_file_path):
        print(f"Error: HDF5 file not found at {h5_file_path}")
        return

    # Create analyzer and run analyses
    analyzer = HyperspectralAnalyzerWithMask(h5_file_path)

    print("Starting comprehensive dataset analysis for masked region...")
    analysis_report = analyzer.analyze_dataset(output_dir)

    print("Generating excitation-emission matrix (EEM) for masked region...")
    eem_csv, eem_plot = analyzer.analyze_excitation_emission_matrix(output_dir)

    print("Performing principal component analysis (PCA) for masked region...")
    pca_report = analyzer.analyze_pca(output_dir)

    print("\nAll analyses complete!")
    print(f"Main analysis report: {analysis_report}")
    print(f"EEM data: {eem_csv}")
    print(f"EEM plot: {eem_plot}")
    print(f"PCA analysis report: {pca_report}")


if __name__ == "__main__":
    main()