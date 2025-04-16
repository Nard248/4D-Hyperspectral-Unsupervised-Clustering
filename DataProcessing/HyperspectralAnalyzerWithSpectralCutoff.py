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


class HyperspectralAnalyzerWithSpectralCutoff:
    """Class for analyzing hyperspectral data with spectral cutoff to remove excitation scattering artifacts"""

    def __init__(self, h5_file_path, cutoff_offset=20, use_mask=True):
        """
        Initialize the analyzer with spectral cutoff functionality

        Args:
            h5_file_path: Path to the HDF5 file containing hyperspectral data
            cutoff_offset: Offset in nm to add to excitation wavelength for cutoff (default: 20nm)
            use_mask: Whether to use mask if available in the HDF5 file (default: True)
        """
        self.h5_file_path = h5_file_path
        self.cutoff_offset = cutoff_offset
        self.use_mask = use_mask
        self.data_dict = {}
        self.filtered_data_dict = {}  # Will hold spectral data after cutoff
        self.cutoff_indices = {}  # Will store cutoff indices for each excitation
        self.stats = {}
        self.emission_wavelengths = {}
        self.filtered_emission_wavelengths = {}  # Will hold filtered wavelengths
        self.excitation_wavelengths = []
        self.metadata = {}
        self.mask = None

        # Load the data
        self._load_data()

        # Apply spectral cutoff filtering
        self._apply_spectral_cutoff()

    def _load_data(self):
        """Load data and mask (if available) from the HDF5 file"""
        with h5py.File(self.h5_file_path, 'r') as f:
            # Extract metadata
            if 'metadata' in f:
                self.metadata = {key: f['metadata'].attrs[key] for key in f['metadata'].attrs}

            # Load mask if present and use_mask is True
            if self.use_mask and 'mask' in f:
                self.mask = f['mask']['data'][:]
                print(f"Loaded mask from HDF5 file. Mask has {np.sum(self.mask)} masked pixels "
                      f"({(np.sum(self.mask) / self.mask.size) * 100:.2f}% of total)")
            else:
                if self.use_mask:
                    warnings.warn("No mask found in the HDF5 file. Analysis will be performed on the entire image.",
                                  UserWarning)

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

    def _apply_spectral_cutoff(self):
        """
        Apply spectral cutoff to remove excitation artifacts.
        Filters emission spectra to only include wavelengths >= (excitation + cutoff_offset)
        """
        for excitation in self.excitation_wavelengths:
            data = self.data_dict[excitation]
            emission_waves = data['wavelengths']

            # Calculate cutoff wavelength
            cutoff_wavelength = excitation + self.cutoff_offset

            # Find cutoff index (first wavelength >= cutoff_wavelength)
            cutoff_idx = np.argmax(emission_waves >= cutoff_wavelength)

            # Store cutoff index for this excitation
            self.cutoff_indices[excitation] = cutoff_idx

            # Filter emission wavelengths
            filtered_waves = emission_waves[cutoff_idx:]
            self.filtered_emission_wavelengths[excitation] = filtered_waves

            # Create filtered data structure
            filtered_data = {
                'wavelengths': filtered_waves,
                'cubes': {}
            }

            # Filter cube data
            for cube_idx, cube in data['cubes'].items():
                filtered_cube_data = cube['data'][cutoff_idx:, :, :]
                filtered_data['cubes'][cube_idx] = {
                    'data': filtered_cube_data,
                    'header': cube['header']
                }

            # Filter average cube if available
            if 'average_cube' in data:
                filtered_data['average_cube'] = data['average_cube'][cutoff_idx:, :, :]

            # Store filtered data
            self.filtered_data_dict[excitation] = filtered_data

            print(f"Excitation {excitation}nm: Cutoff at {cutoff_wavelength}nm (removed {cutoff_idx} bands)")

    def visualize_spectral_cutoff(self, output_dir='analysis_results_cutoff'):
        """
        Create visualizations showing the effect of spectral cutoff

        Args:
            output_dir: Directory to save analysis results
        """
        os.makedirs(output_dir, exist_ok=True)

        for excitation in self.excitation_wavelengths:
            data = self.data_dict[excitation]
            filtered_data = self.filtered_data_dict[excitation]

            # Get wavelengths
            full_waves = data['wavelengths']
            filtered_waves = filtered_data['wavelengths']

            # Get average cube data if available
            if 'average_cube' in data:
                avg_cube = data['average_cube']
                avg_filtered_cube = filtered_data['average_cube']

                # Calculate band means
                if self.mask is not None and np.any(self.mask):
                    # Calculate with mask
                    full_band_means = np.array([np.mean(avg_cube[b, self.mask == 1])
                                                for b in range(avg_cube.shape[0])])
                    full_band_stds = np.array([np.std(avg_cube[b, self.mask == 1])
                                               for b in range(avg_cube.shape[0])])

                    filtered_band_means = np.array([np.mean(avg_filtered_cube[b, self.mask == 1])
                                                    for b in range(avg_filtered_cube.shape[0])])
                    filtered_band_stds = np.array([np.std(avg_filtered_cube[b, self.mask == 1])
                                                   for b in range(avg_filtered_cube.shape[0])])
                else:
                    # Calculate without mask
                    full_band_means = np.mean(avg_cube, axis=(1, 2))
                    full_band_stds = np.std(avg_cube, axis=(1, 2))

                    filtered_band_means = np.mean(avg_filtered_cube, axis=(1, 2))
                    filtered_band_stds = np.std(avg_filtered_cube, axis=(1, 2))

                # Create figure showing cutoff
                plt.figure(figsize=(12, 8))

                # Plot full spectrum with cutoff region highlighted
                plt.plot(full_waves, full_band_means, 'b-', label='Full Spectrum')
                plt.fill_between(full_waves, full_band_means - full_band_stds,
                                 full_band_means + full_band_stds, color='b', alpha=0.2)

                # Highlight cutoff region
                cutoff_wavelength = excitation + self.cutoff_offset
                plt.axvspan(excitation - 5, cutoff_wavelength, color='r', alpha=0.2,
                            label=f'Cutoff Region (Ex: {excitation}nm + {self.cutoff_offset}nm)')
                plt.axvline(x=cutoff_wavelength, color='r', linestyle='--')

                # Plot filtered spectrum
                plt.plot(filtered_waves, filtered_band_means, 'g-', linewidth=2, label='Filtered Spectrum')
                plt.fill_between(filtered_waves, filtered_band_means - filtered_band_stds,
                                 filtered_band_means + filtered_band_stds, color='g', alpha=0.2)

                plt.xlabel('Emission Wavelength (nm)')
                plt.ylabel('Mean Signal')
                title = f'Spectral Cutoff Effect - Excitation {excitation}nm'
                if self.mask is not None and np.any(self.mask):
                    title += ' (Masked Region Only)'
                plt.title(title)
                plt.legend()
                plt.grid(True, alpha=0.3)

                # Highlight excitation wavelength
                plt.axvline(x=excitation, color='orange', linestyle='-', label='Excitation Wavelength')

                # Save plot
                plt.savefig(os.path.join(output_dir, f'spectral_cutoff_ex{excitation}nm.png'), dpi=150)
                plt.close()

                print(f"Generated spectral cutoff visualization for excitation {excitation}nm")

    def analyze_dataset(self, output_dir='analysis_results_cutoff'):
        """
        Perform comprehensive analysis of the dataset using spectrally filtered data

        Args:
            output_dir: Directory to save analysis results
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Time the analysis
        start_time = time.time()

        # Open a file to write results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"cutoff_analysis_{timestamp}.txt")

        with open(output_file, 'w') as f:
            # Write header
            f.write("=" * 80 + "\n")
            f.write("HYPERSPECTRAL DATA ANALYSIS REPORT WITH SPECTRAL CUTOFF\n")
            f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"HDF5 File: {os.path.basename(self.h5_file_path)}\n")
            f.write(f"Cutoff Offset: {self.cutoff_offset}nm\n")
            f.write("=" * 80 + "\n\n")

            # Mask information
            if self.mask is not None and np.any(self.mask):
                f.write("1. MASK INFORMATION\n")
                f.write("-" * 40 + "\n")

                masked_pixels = np.sum(self.mask)
                total_pixels = self.mask.size
                percent_masked = (masked_pixels / total_pixels) * 100
                f.write(f"Mask present: Yes\n")
                f.write(f"Total pixels in image: {total_pixels}\n")
                f.write(f"Masked pixels (region of interest): {masked_pixels}\n")
                f.write(f"Percentage of image masked: {percent_masked:.2f}%\n\n")

            # Spectral cutoff information
            f.write("2. SPECTRAL CUTOFF INFORMATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Cutoff offset from excitation: {self.cutoff_offset}nm\n\n")

            f.write("Excitation-specific cutoffs:\n")
            for excitation in self.excitation_wavelengths:
                cutoff_wavelength = excitation + self.cutoff_offset
                cutoff_idx = self.cutoff_indices[excitation]
                full_bands = len(self.emission_wavelengths[excitation])
                remaining_bands = len(self.filtered_emission_wavelengths[excitation])
                percent_remaining = (remaining_bands / full_bands) * 100

                f.write(f"  Excitation {excitation}nm: Cutoff at {cutoff_wavelength}nm\n")
                f.write(f"    Removed {cutoff_idx} of {full_bands} bands ({100 - percent_remaining:.1f}% removed)\n")
                f.write(f"    Remaining emission range: {self.filtered_emission_wavelengths[excitation][0]:.1f} - ")
                f.write(f"{self.filtered_emission_wavelengths[excitation][-1]:.1f}nm\n")

            # General dataset information
            f.write("\n3. GENERAL DATASET INFORMATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Number of excitation wavelengths: {len(self.excitation_wavelengths)}\n")
            f.write(f"Excitation wavelengths: {self.excitation_wavelengths} nm\n")

            # Calculate dataset size after cutoff
            total_size_bytes = 0
            num_cubes = 0
            for ex, data in self.filtered_data_dict.items():
                for idx, cube in data['cubes'].items():
                    total_size_bytes += cube['data'].nbytes
                    num_cubes += 1

            f.write(f"Number of individual data cubes: {num_cubes}\n")
            f.write(f"Total filtered data size: {total_size_bytes / (1024 ** 3):.2f} GB\n")

            if self.metadata:
                f.write("\nMetadata:\n")
                for key, value in self.metadata.items():
                    f.write(f"  {key}: {value}\n")

            # Detailed analysis per excitation wavelength
            f.write("\n\n4. ANALYSIS BY EXCITATION WAVELENGTH (WITH SPECTRAL CUTOFF)\n")
            f.write("-" * 40 + "\n")

            # Create dataframes to store summary statistics
            excitation_summary = []

            for excitation in self.excitation_wavelengths:
                f.write(f"\n4.{self.excitation_wavelengths.index(excitation) + 1}. Excitation: {excitation} nm\n")
                data = self.filtered_data_dict[excitation]

                # Emission wavelengths for this excitation
                emission_waves = data['wavelengths']
                original_waves = self.emission_wavelengths[excitation]

                f.write(f"  Number of emission wavelengths (after cutoff): {len(emission_waves)}\n")
                f.write(
                    f"  Emission range (after cutoff): {emission_waves.min():.1f} - {emission_waves.max():.1f} nm\n")
                f.write(f"  Original emission range: {original_waves.min():.1f} - {original_waves.max():.1f} nm\n")

                # Cube information
                num_cubes = len(data['cubes'])
                f.write(f"  Number of sample cubes: {num_cubes}\n")

                # Get first cube to extract shape information
                first_cube_key = list(data['cubes'].keys())[0]
                first_cube = data['cubes'][first_cube_key]['data']
                cube_shape = first_cube.shape
                f.write(f"  Filtered cube dimensions (bands, height, width): {cube_shape}\n")

                # Use the average cube for statistics if available
                if 'average_cube' in data:
                    avg_cube = data['average_cube']

                    # Apply mask if available
                    if self.mask is not None and np.any(self.mask):
                        # For each band, extract only the masked pixels
                        masked_data = [avg_cube[b, self.mask == 1] for b in range(avg_cube.shape[0])]
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

                    region_str = "masked region" if self.mask is not None and np.any(self.mask) else "entire image"
                    f.write(f"\n  Statistics for {region_str} (after spectral cutoff):\n")
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
                        'NumEmissionBands': len(emission_waves),
                        'CutoffWavelength': excitation + self.cutoff_offset
                    })

                    # Analyze the spectral dimension for filtered data
                    f.write("\n  Spectral analysis (emission bands, after cutoff):\n")

                    # Get band-wise statistics for filtered data
                    if self.mask is not None and np.any(self.mask):
                        # Calculate statistics for each band over just the masked pixels
                        band_means = np.array(
                            [np.mean(avg_cube[b, self.mask == 1]) for b in range(avg_cube.shape[0])])
                        band_stds = np.array(
                            [np.std(avg_cube[b, self.mask == 1]) for b in range(avg_cube.shape[0])])
                        band_maxes = np.array(
                            [np.max(avg_cube[b, self.mask == 1]) for b in range(avg_cube.shape[0])])
                        band_mins = np.array(
                            [np.min(avg_cube[b, self.mask == 1]) for b in range(avg_cube.shape[0])])
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
                    f.write("    Top 5 emission bands by mean signal (after cutoff):\n")
                    for rank, band_idx in enumerate(top_bands_by_mean):
                        f.write(
                            f"      #{rank + 1}: {emission_waves[band_idx]:.1f} nm, Mean: {band_means[band_idx]:.4f}, SNR: {band_snrs[band_idx]:.4f}\n")

                    # Find emission wavelengths with highest SNR
                    top_bands_by_snr = np.argsort(band_snrs)[-5:][::-1]
                    f.write("    Top 5 emission bands by signal-to-noise ratio (after cutoff):\n")
                    for rank, band_idx in enumerate(top_bands_by_snr):
                        f.write(
                            f"      #{rank + 1}: {emission_waves[band_idx]:.1f} nm, SNR: {band_snrs[band_idx]:.4f}, Mean: {band_means[band_idx]:.4f}\n")

                    # Save spectral profile plot
                    plt.figure(figsize=(10, 6))
                    plt.plot(emission_waves, band_means)
                    plt.fill_between(emission_waves, band_means - band_stds, band_means + band_stds, alpha=0.3)
                    plt.xlabel('Emission Wavelength (nm)')
                    y_label = 'Mean Signal'
                    if self.mask is not None and np.any(self.mask):
                        y_label += ' (Masked Region Only)'
                    plt.ylabel(y_label)
                    title = f'Spectral Profile for Excitation {excitation} nm (After Cutoff at {excitation + self.cutoff_offset} nm)'
                    plt.title(title)
                    plt.grid(True, alpha=0.3)

                    # Add dotted vertical line at cutoff
                    plt.axvline(x=excitation + self.cutoff_offset, color='r', linestyle='--',
                                label=f'Cutoff ({excitation + self.cutoff_offset} nm)')
                    plt.legend()

                    spectral_plot_file = os.path.join(output_dir,
                                                      f"cutoff_spectral_profile_ex{excitation}nm.png")
                    plt.savefig(spectral_plot_file, dpi=150)
                    plt.close()
                    f.write(f"\n    Spectral profile saved to: {os.path.basename(spectral_plot_file)}\n")

            # Create a summary dataframe for excitation wavelengths
            excitation_df = pd.DataFrame(excitation_summary)

            # Save the summary dataframe to a CSV file
            excitation_csv = os.path.join(output_dir, f"cutoff_excitation_summary.csv")
            excitation_df.to_csv(excitation_csv, index=False)
            f.write(f"\nExcitation summary statistics saved to: {os.path.basename(excitation_csv)}\n")

            # Comparative analysis across excitation wavelengths with spectral cutoff
            f.write("\n\n5. COMPARATIVE ANALYSIS ACROSS EXCITATION WAVELENGTHS (WITH SPECTRAL CUTOFF)\n")
            f.write("-" * 40 + "\n")

            # Compare mean signal across excitation wavelengths
            f.write("\n5.1. Mean Signal Comparison (after cutoff):\n")
            max_mean_ex = excitation_df.loc[excitation_df['Mean'].idxmax()]
            min_mean_ex = excitation_df.loc[excitation_df['Mean'].idxmin()]
            f.write(
                f"  Highest mean signal: Excitation {max_mean_ex['Excitation']} nm (Mean: {max_mean_ex['Mean']:.4f})\n")
            f.write(
                f"  Lowest mean signal: Excitation {min_mean_ex['Excitation']} nm (Mean: {min_mean_ex['Mean']:.4f})\n")
            f.write(f"  Ratio of highest to lowest mean: {max_mean_ex['Mean'] / min_mean_ex['Mean']:.2f}\n")

            # Compare SNR across excitation wavelengths
            f.write("\n5.2. Signal-to-Noise Ratio Comparison (after cutoff):\n")
            max_snr_ex = excitation_df.loc[excitation_df['SNR'].idxmax()]
            min_snr_ex = excitation_df.loc[excitation_df['SNR'].idxmin()]
            f.write(f"  Highest SNR: Excitation {max_snr_ex['Excitation']} nm (SNR: {max_snr_ex['SNR']:.4f})\n")
            f.write(f"  Lowest SNR: Excitation {min_snr_ex['Excitation']} nm (SNR: {min_snr_ex['SNR']:.4f})\n")
            f.write(f"  Ratio of highest to lowest SNR: {max_snr_ex['SNR'] / min_snr_ex['SNR']:.2f}\n")

            # Visualize excitation comparison
            plt.figure(figsize=(12, 6))
            sns.barplot(x='Excitation', y='Mean', data=excitation_df)
            plt.title('Mean Signal by Excitation Wavelength (After Spectral Cutoff)')
            plt.xlabel('Excitation Wavelength (nm)')
            plt.ylabel('Mean Signal')
            plt.grid(True, alpha=0.3)
            excitation_plot_file = os.path.join(output_dir, f"cutoff_excitation_comparison_mean.png")
            plt.savefig(excitation_plot_file, dpi=150)
            plt.close()

            plt.figure(figsize=(12, 6))
            sns.barplot(x='Excitation', y='SNR', data=excitation_df)
            plt.title('Signal-to-Noise Ratio by Excitation Wavelength (After Spectral Cutoff)')
            plt.xlabel('Excitation Wavelength (nm)')
            plt.ylabel('Signal-to-Noise Ratio')
            plt.grid(True, alpha=0.3)
            snr_plot_file = os.path.join(output_dir, f"cutoff_excitation_comparison_snr.png")
            plt.savefig(snr_plot_file, dpi=150)
            plt.close()

            f.write(
                f"\n  Excitation comparison plots saved to: {os.path.basename(excitation_plot_file)} and {os.path.basename(snr_plot_file)}\n")

            # Time the analysis
            end_time = time.time()
            analysis_time = end_time - start_time
            f.write(f"\nAnalysis completed in {analysis_time:.2f} seconds.\n")

        print(f"Analysis with spectral cutoff complete. Results saved to {output_file}")
        return output_file

    def analyze_excitation_emission_matrix(self, output_dir='analysis_results_cutoff'):
        """
        Create and analyze an excitation-emission matrix (EEM) with spectral cutoff applied

        Args:
            output_dir: Directory to save analysis results
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Create a list to store the EEM data
        eem_data = []

        # Process each excitation wavelength using filtered data
        for excitation in self.excitation_wavelengths:
            data = self.filtered_data_dict[excitation]
            emission_waves = data['wavelengths']

            if 'average_cube' in data:
                avg_cube = data['average_cube']

                # Calculate band-wise statistics for the appropriate region
                if self.mask is not None and np.any(self.mask):
                    # Calculate with mask
                    band_means = np.array([np.mean(avg_cube[b, self.mask == 1])
                                           for b in range(avg_cube.shape[0])])
                else:
                    # If no mask, use the entire image
                    band_means = np.mean(avg_cube, axis=(1, 2))

                # Add each excitation-emission combination after cutoff
                for i, emission in enumerate(emission_waves):
                    eem_data.append({
                        'Excitation': excitation,
                        'Emission': emission,
                        'Intensity': band_means[i],
                        'Cutoff': emission >= (excitation + self.cutoff_offset)
                    })

        # Create a dataframe from the EEM data
        eem_df = pd.DataFrame(eem_data)

        # Save the EEM data
        eem_csv = os.path.join(output_dir, "cutoff_excitation_emission_matrix.csv")
        eem_df.to_csv(eem_csv, index=False)

        # Create a pivot table for visualization
        eem_pivot = eem_df.pivot_table(values='Intensity', index='Emission', columns='Excitation')

        # Plot the EEM
        plt.figure(figsize=(12, 10))

        # Use log scale for better visualization
        sns.heatmap(eem_pivot, cmap='viridis', norm=LogNorm())
        plt.title('Excitation-Emission Matrix (EEM) - After Spectral Cutoff')
        plt.xlabel('Excitation Wavelength (nm)')
        plt.ylabel('Emission Wavelength (nm)')

        # Add diagonal line showing the cutoff boundary
        cutoff_x = []
        cutoff_y = []
        for ex in self.excitation_wavelengths:
            cutoff_x.append(ex)
            cutoff_y.append(ex + self.cutoff_offset)

        # Convert to axis coordinates (this is approximate, may need adjustment)
        plt.plot(cutoff_x, cutoff_y, 'r--', linewidth=2,
                 label=f'Spectral Cutoff (Ex + {self.cutoff_offset}nm)')
        plt.legend(loc='upper left')

        # Save the plot
        eem_plot = os.path.join(output_dir, "cutoff_excitation_emission_matrix.png")
        plt.savefig(eem_plot, dpi=150)
        plt.close()

        # Create a non-log version for comparison
        plt.figure(figsize=(12, 10))
        sns.heatmap(eem_pivot, cmap='viridis')
        plt.title('Excitation-Emission Matrix (EEM) - After Spectral Cutoff (Linear Scale)')
        plt.xlabel('Excitation Wavelength (nm)')
        plt.ylabel('Emission Wavelength (nm)')

        # Add diagonal line showing the cutoff boundary
        plt.plot(cutoff_x, cutoff_y, 'r--', linewidth=2,
                 label=f'Spectral Cutoff (Ex + {self.cutoff_offset}nm)')
        plt.legend(loc='upper left')

        # Save the plot
        eem_plot_linear = os.path.join(output_dir, "cutoff_excitation_emission_matrix_linear.png")
        plt.savefig(eem_plot_linear, dpi=150)
        plt.close()

        print(f"EEM analysis complete. Results saved to {eem_csv}, {eem_plot}, and {eem_plot_linear}")
        return eem_csv, eem_plot

    def analyze_pca(self, output_dir='analysis_results_cutoff'):
        """
        Perform Principal Component Analysis (PCA) on the spectral dimension after cutoff

        Args:
            output_dir: Directory to save analysis results
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Open a file to write results
        pca_output_file = os.path.join(output_dir, "cutoff_pca_analysis.txt")

        with open(pca_output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("PRINCIPAL COMPONENT ANALYSIS (PCA) REPORT - WITH SPECTRAL CUTOFF\n")
            f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Cutoff Offset: {self.cutoff_offset}nm\n")
            f.write("=" * 80 + "\n\n")

            # Perform PCA for each excitation wavelength
            for excitation in self.excitation_wavelengths:
                f.write(
                    f"PCA Analysis for Excitation {excitation} nm (After Cutoff at {excitation + self.cutoff_offset} nm)\n")
                f.write("-" * 40 + "\n")

                data = self.filtered_data_dict[excitation]
                emission_waves = data['wavelengths']

                if 'average_cube' in data:
                    avg_cube = data['average_cube']

                    if self.mask is not None and np.any(self.mask):
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
                        region_str = "masked region"
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
                        region_str = "entire image"

                    # Number of components needed to explain 95% variance
                    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
                    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1

                    f.write(f"Number of emission bands (after cutoff): {num_bands}\n")
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
                    plt.title(
                        f'PCA Cumulative Explained Variance (Excitation {excitation} nm, After Cutoff, {region_str})')
                    plt.axhline(y=95, color='r', linestyle='--')
                    plt.grid(True, alpha=0.3)
                    plt.xlim(0, min(20, len(cumulative_variance)))  # Show only first 20 components

                    # Save the plot
                    pca_plot = os.path.join(output_dir, f"cutoff_pca_variance_ex{excitation}nm.png")
                    plt.savefig(pca_plot, dpi=150)
                    plt.close()

                    f.write(f"\nPCA variance plot saved to: {os.path.basename(pca_plot)}\n")

                    # Analyze the loadings of the first PC to identify most important bands
                    pc1_loadings = pca.components_[0]
                    pc1_loadings_abs = np.abs(pc1_loadings)
                    top_band_indices = np.argsort(pc1_loadings_abs)[-5:][::-1]

                    f.write("\nMost important emission bands according to PC1 (After Cutoff):\n")
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
                    plt.title(
                        f'PCA Loadings by Emission Wavelength (Excitation {excitation} nm, After Cutoff, {region_str})')
                    plt.legend()
                    plt.grid(True, alpha=0.3)

                    # Save the plot
                    pca_loadings_plot = os.path.join(output_dir, f"cutoff_pca_loadings_ex{excitation}nm.png")
                    plt.savefig(pca_loadings_plot, dpi=150)
                    plt.close()

                    f.write(f"\nPCA loadings plot saved to: {os.path.basename(pca_loadings_plot)}\n\n")

        print(f"PCA analysis with spectral cutoff complete. Results saved to {pca_output_file}")
        return pca_output_file


def main():
    """Main function to run the hyperspectral analysis with spectral cutoff"""
    # Set input and output paths
    h5_file_path = "kiwi_hyperspectral_4d_data_with_mask.h5"
    output_dir = "analysis_results_cutoff"

    # Default cutoff offset is 20nm, but can be changed
    cutoff_offset = 20

    # Check if the H5 file exists
    if not os.path.exists(h5_file_path):
        print(f"Error: HDF5 file not found at {h5_file_path}")
        return

    # Create analyzer with spectral cutoff
    analyzer = HyperspectralAnalyzerWithSpectralCutoff(h5_file_path, cutoff_offset=cutoff_offset, use_mask=True)

    # First visualize the effect of cutoff for better understanding
    print("Generating spectral cutoff visualizations...")
    analyzer.visualize_spectral_cutoff(output_dir)

    print("Starting comprehensive dataset analysis with spectral cutoff...")
    analysis_report = analyzer.analyze_dataset(output_dir)

    print("Generating excitation-emission matrix (EEM) with spectral cutoff...")
    eem_csv, eem_plot = analyzer.analyze_excitation_emission_matrix(output_dir)

    print("Performing principal component analysis (PCA) with spectral cutoff...")
    pca_report = analyzer.analyze_pca(output_dir)

    print("\nAll analyses complete!")
    print(f"Main analysis report: {analysis_report}")
    print(f"EEM data: {eem_csv}")
    print(f"EEM plot: {eem_plot}")
    print(f"PCA analysis report: {pca_report}")


if __name__ == "__main__":
    main()