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


class HyperspectralAnalyzer:
    """Class for analyzing hyperspectral data stored in HDF5 format"""

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

        # Load the data
        self._load_data()

    def _load_data(self):
        """Load data from the HDF5 file"""
        with h5py.File(self.h5_file_path, 'r') as f:
            # Extract metadata
            if 'metadata' in f:
                self.metadata = {key: f['metadata'].attrs[key] for key in f['metadata'].attrs}

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

    def analyze_dataset(self, output_dir='analysis_results'):
        """
        Perform comprehensive analysis of the dataset

        Args:
            output_dir: Directory to save analysis results
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Time the analysis
        start_time = time.time()

        # Open a file to write results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"hyperspectral_analysis_{timestamp}.txt")

        with open(output_file, 'w') as f:
            # Write header
            f.write("=" * 80 + "\n")
            f.write("HYPERSPECTRAL DATA ANALYSIS REPORT\n")
            f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"HDF5 File: {os.path.basename(self.h5_file_path)}\n")
            f.write("=" * 80 + "\n\n")

            # General dataset information
            f.write("1. GENERAL DATASET INFORMATION\n")
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
            f.write("\n\n2. ANALYSIS BY EXCITATION WAVELENGTH\n")
            f.write("-" * 40 + "\n")

            # Create dataframes to store summary statistics
            excitation_summary = []

            for excitation in self.excitation_wavelengths:
                f.write(f"\n2.{self.excitation_wavelengths.index(excitation) + 1}. Excitation: {excitation} nm\n")
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

                    # Compute overall statistics for this excitation
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

                    f.write("\n  Statistics for average cube:\n")
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

                    # Analyze the spectral dimension
                    f.write("\n  Spectral analysis (emission bands):\n")

                    # Get band-wise statistics
                    band_means = np.mean(avg_cube, axis=(1, 2))
                    band_stds = np.std(avg_cube, axis=(1, 2))
                    band_maxes = np.max(avg_cube, axis=(1, 2))
                    band_mins = np.min(avg_cube, axis=(1, 2))
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
                    plt.ylabel('Mean Signal')
                    plt.title(f'Spectral Profile for Excitation {excitation} nm')
                    plt.grid(True, alpha=0.3)
                    spectral_plot_file = os.path.join(output_dir, f'spectral_profile_ex{excitation}nm.png')
                    plt.savefig(spectral_plot_file, dpi=150)
                    plt.close()
                    f.write(f"\n    Spectral profile saved to: {os.path.basename(spectral_plot_file)}\n")

                    # Calculate band-to-band correlations for most interesting bands
                    interesting_bands = np.unique(np.concatenate([top_bands_by_mean, top_bands_by_snr]))
                    if len(interesting_bands) > 1:
                        f.write("\n    Correlations between top emission bands:\n")
                        for i, band1 in enumerate(interesting_bands):
                            for band2 in interesting_bands[i + 1:]:
                                band1_data = avg_cube[band1].flatten()
                                band2_data = avg_cube[band2].flatten()
                                correlation = np.corrcoef(band1_data, band2_data)[0, 1]
                                f.write(
                                    f"      {emission_waves[band1]:.1f} nm <-> {emission_waves[band2]:.1f} nm: r = {correlation:.4f}\n")

            # Create a summary dataframe for excitation wavelengths
            excitation_df = pd.DataFrame(excitation_summary)

            # Save the summary dataframe to a CSV file
            excitation_csv = os.path.join(output_dir, "excitation_summary.csv")
            excitation_df.to_csv(excitation_csv, index=False)
            f.write(f"\nExcitation summary statistics saved to: {os.path.basename(excitation_csv)}\n")

            # Comparative analysis across excitation wavelengths
            f.write("\n\n3. COMPARATIVE ANALYSIS ACROSS EXCITATION WAVELENGTHS\n")
            f.write("-" * 40 + "\n")

            # Compare mean signal across excitation wavelengths
            f.write("\n3.1. Mean Signal Comparison:\n")
            max_mean_ex = excitation_df.loc[excitation_df['Mean'].idxmax()]
            min_mean_ex = excitation_df.loc[excitation_df['Mean'].idxmin()]
            f.write(
                f"  Highest mean signal: Excitation {max_mean_ex['Excitation']} nm (Mean: {max_mean_ex['Mean']:.4f})\n")
            f.write(
                f"  Lowest mean signal: Excitation {min_mean_ex['Excitation']} nm (Mean: {min_mean_ex['Mean']:.4f})\n")
            f.write(f"  Ratio of highest to lowest mean: {max_mean_ex['Mean'] / min_mean_ex['Mean']:.2f}\n")

            # Compare SNR across excitation wavelengths
            f.write("\n3.2. Signal-to-Noise Ratio Comparison:\n")
            max_snr_ex = excitation_df.loc[excitation_df['SNR'].idxmax()]
            min_snr_ex = excitation_df.loc[excitation_df['SNR'].idxmin()]
            f.write(f"  Highest SNR: Excitation {max_snr_ex['Excitation']} nm (SNR: {max_snr_ex['SNR']:.4f})\n")
            f.write(f"  Lowest SNR: Excitation {min_snr_ex['Excitation']} nm (SNR: {min_snr_ex['SNR']:.4f})\n")
            f.write(f"  Ratio of highest to lowest SNR: {max_snr_ex['SNR'] / min_snr_ex['SNR']:.2f}\n")

            # Find potentially most informative excitation-emission combinations
            f.write("\n\n4. POTENTIAL HIGH-INFORMATION EXCITATION-EMISSION COMBINATIONS\n")
            f.write("-" * 40 + "\n")

            # Create a dataframe to store best emission bands for each excitation
            best_combinations = []

            for excitation in self.excitation_wavelengths:
                data = self.data_dict[excitation]
                emission_waves = data['wavelengths']

                if 'average_cube' in data:
                    avg_cube = data['average_cube']

                    # Calculate band-wise statistics
                    band_means = np.mean(avg_cube, axis=(1, 2))
                    band_stds = np.std(avg_cube, axis=(1, 2))
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
            f.write("\n4.1. Top 10 excitation-emission combinations by composite score:\n")
            for i, row in best_df_sorted.head(10).iterrows():
                f.write(f"  #{i + 1}: Excitation {row['Excitation']} nm, Emission {row['Emission']:.1f} nm")
                f.write(f" (Score: {row['CompositeScore']:.4f}, Mean: {row['Mean']:.4f}, SNR: {row['SNR']:.4f})\n")

            # Save the best combinations to CSV
            best_csv = os.path.join(output_dir, "best_excitation_emission_combinations.csv")
            best_df.to_csv(best_csv, index=False)
            f.write(f"\nBest excitation-emission combinations saved to: {os.path.basename(best_csv)}\n")

            # Time the analysis
            end_time = time.time()
            analysis_time = end_time - start_time
            f.write(f"\nAnalysis completed in {analysis_time:.2f} seconds.\n")

        print(f"Analysis complete. Results saved to {output_file}")
        return output_file

    def analyze_excitation_emission_matrix(self, output_dir='analysis_results'):
        """
        Create and analyze an excitation-emission matrix (EEM)

        Args:
            output_dir: Directory to save analysis results
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

                # Calculate band-wise statistics
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
        eem_csv = os.path.join(output_dir, "excitation_emission_matrix.csv")
        eem_df.to_csv(eem_csv, index=False)

        # Create a pivot table for visualization
        eem_pivot = eem_df.pivot_table(values='Intensity', index='Emission', columns='Excitation')

        # Plot the EEM
        plt.figure(figsize=(12, 10))

        # Use log scale for better visualization
        sns.heatmap(eem_pivot, cmap='viridis', norm=LogNorm())
        plt.title('Excitation-Emission Matrix (EEM)')
        plt.xlabel('Excitation Wavelength (nm)')
        plt.ylabel('Emission Wavelength (nm)')

        # Save the plot
        eem_plot = os.path.join(output_dir, "excitation_emission_matrix.png")
        plt.savefig(eem_plot, dpi=150)
        plt.close()

        print(f"EEM analysis complete. Results saved to {eem_csv} and {eem_plot}")
        return eem_csv, eem_plot

    def analyze_pca(self, output_dir='analysis_results'):
        """
        Perform Principal Component Analysis (PCA) on the spectral dimension

        Args:
            output_dir: Directory to save analysis results
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Open a file to write results
        pca_output_file = os.path.join(output_dir, "pca_analysis.txt")

        with open(pca_output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("PRINCIPAL COMPONENT ANALYSIS (PCA) REPORT\n")
            f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            # Perform PCA for each excitation wavelength
            for excitation in self.excitation_wavelengths:
                f.write(f"PCA Analysis for Excitation {excitation} nm\n")
                f.write("-" * 40 + "\n")

                data = self.data_dict[excitation]
                emission_waves = data['wavelengths']

                if 'average_cube' in data:
                    avg_cube = data['average_cube']

                    # Reshape the cube for PCA
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
                    plt.title(f'PCA Cumulative Explained Variance (Excitation {excitation} nm)')
                    plt.axhline(y=95, color='r', linestyle='--')
                    plt.grid(True, alpha=0.3)
                    plt.xlim(0, min(20, len(cumulative_variance)))  # Show only first 20 components

                    # Save the plot
                    pca_plot = os.path.join(output_dir, f"pca_variance_ex{excitation}nm.png")
                    plt.savefig(pca_plot, dpi=150)
                    plt.close()

                    f.write(f"\nPCA variance plot saved to: {os.path.basename(pca_plot)}\n")

                    # Analyze the loadings of the first PC to identify most important bands
                    pc1_loadings = pca.components_[0]
                    pc1_loadings_abs = np.abs(pc1_loadings)
                    top_band_indices = np.argsort(pc1_loadings_abs)[-5:][::-1]

                    f.write("\nMost important emission bands according to PC1:\n")
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
                    plt.title(f'PCA Loadings by Emission Wavelength (Excitation {excitation} nm)')
                    plt.legend()
                    plt.grid(True, alpha=0.3)

                    # Save the plot
                    pca_loadings_plot = os.path.join(output_dir, f"pca_loadings_ex{excitation}nm.png")
                    plt.savefig(pca_loadings_plot, dpi=150)
                    plt.close()

                    f.write(f"\nPCA loadings plot saved to: {os.path.basename(pca_loadings_plot)}\n\n")

        print(f"PCA analysis complete. Results saved to {pca_output_file}")
        return pca_output_file

    def analyze_sample_variability(self, output_dir='analysis_results'):
        """
        Analyze variability between sample cubes for the same excitation wavelength

        Args:
            output_dir: Directory to save analysis results
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Open a file to write results
        variability_output_file = os.path.join(output_dir, "sample_variability_analysis.txt")

        with open(variability_output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("SAMPLE VARIABILITY ANALYSIS REPORT\n")
            f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            # Analyze variability for each excitation wavelength
            for excitation in self.excitation_wavelengths:
                f.write(f"Sample Variability for Excitation {excitation} nm\n")
                f.write("-" * 40 + "\n")

                data = self.data_dict[excitation]
                emission_waves = data['wavelengths']
                cubes = data['cubes']

                if len(cubes) > 1:
                    # Calculate mean and standard deviation across samples
                    sample_values = np.array([cube['data'] for cube in cubes.values()])
                    sample_mean = np.mean(sample_values, axis=0)
                    sample_std = np.std(sample_values, axis=0)

                    # Calculate coefficient of variation (CV) as a measure of variability
                    # CV = std / mean (expressed as percentage)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        cv = np.where(sample_mean != 0, (sample_std / sample_mean) * 100, 0)

                    # Calculate overall statistics
                    mean_cv = np.nanmean(cv)
                    median_cv = np.nanmedian(cv)
                    max_cv = np.nanmax(cv)

                    f.write(f"Number of samples: {len(cubes)}\n")
                    f.write(f"Mean coefficient of variation: {mean_cv:.2f}%\n")
                    f.write(f"Median coefficient of variation: {median_cv:.2f}%\n")
                    f.write(f"Maximum coefficient of variation: {max_cv:.2f}%\n")

                    # Calculate band-wise variability
                    band_mean_cv = np.nanmean(cv, axis=(1, 2))

                    # Identify bands with highest and lowest variability
                    most_stable_bands = np.argsort(band_mean_cv)[:5]
                    most_variable_bands = np.argsort(band_mean_cv)[-5:][::-1]

                    f.write("\nMost stable emission bands (lowest CV):\n")
                    for rank, band_idx in enumerate(most_stable_bands):
                        f.write(f"#{rank + 1}: {emission_waves[band_idx]:.1f} nm (CV: {band_mean_cv[band_idx]:.2f}%)\n")

                    f.write("\nMost variable emission bands (highest CV):\n")
                    for rank, band_idx in enumerate(most_variable_bands):
                        f.write(f"#{rank + 1}: {emission_waves[band_idx]:.1f} nm (CV: {band_mean_cv[band_idx]:.2f}%)\n")

                    # Plot band-wise variability
                    plt.figure(figsize=(10, 6))
                    plt.plot(emission_waves, band_mean_cv)
                    plt.xlabel('Emission Wavelength (nm)')
                    plt.ylabel('Mean Coefficient of Variation (%)')
                    plt.title(f'Band-wise Variability across Samples (Excitation {excitation} nm)')
                    plt.grid(True, alpha=0.3)

                    # Save the plot
                    variability_plot = os.path.join(output_dir, f"sample_variability_ex{excitation}nm.png")
                    plt.savefig(variability_plot, dpi=150)
                    plt.close()

                    f.write(f"\nVariability plot saved to: {os.path.basename(variability_plot)}\n\n")
                else:
                    f.write(f"Insufficient samples for variability analysis\n\n")

        print(f"Sample variability analysis complete. Results saved to {variability_output_file}")
        return variability_output_file


def main():
    """Main function to run the hyperspectral analysis"""
    # Set input and output paths
    h5_file_path = "kiwi_hyperspectral_4d_data.h5"
    output_dir = "analysis_results"

    # Check if the H5 file exists
    if not os.path.exists(h5_file_path):
        print(f"Error: HDF5 file not found at {h5_file_path}")
        return

    # Create analyzer and run analyses
    analyzer = HyperspectralAnalyzer(h5_file_path)

    print("Starting comprehensive dataset analysis...")
    analysis_report = analyzer.analyze_dataset(output_dir)

    print("Generating excitation-emission matrix (EEM)...")
    eem_csv, eem_plot = analyzer.analyze_excitation_emission_matrix(output_dir)

    print("Performing principal component analysis (PCA)...")
    pca_report = analyzer.analyze_pca(output_dir)

    print("Analyzing sample variability...")
    variability_report = analyzer.analyze_sample_variability(output_dir)

    print("\nAll analyses complete!")
    print(f"Main analysis report: {analysis_report}")
    print(f"EEM data: {eem_csv}")
    print(f"EEM plot: {eem_plot}")
    print(f"PCA analysis report: {pca_report}")
    print(f"Sample variability report: {variability_report}")


if __name__ == "__main__":
    main()