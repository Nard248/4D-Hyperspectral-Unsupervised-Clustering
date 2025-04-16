import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics.cluster import mutual_info_score
import pandas as pd
from itertools import product
import os


class InfoTheoryAnalyzer:
    """Class for analyzing hyperspectral data using information theory metrics"""

    def __init__(self, h5_file_path):
        """
        Initialize with the path to the HDF5 file

        Args:
            h5_file_path: Path to the HDF5 file containing hyperspectral data
        """
        self.h5_file_path = h5_file_path
        self.data_dict = {}
        self.excitation_wavelengths = []
        self.emission_wavelengths = {}
        self.entropy_scores = {}
        self.mi_matrix = None
        self.best_combinations = None

        # Load the data
        self.load_data()

    def load_data(self):
        """Load the hyperspectral data from the HDF5 file"""
        with h5py.File(self.h5_file_path, 'r') as f:
            # Read each excitation group
            for group_name in f.keys():
                if group_name.startswith('excitation_'):
                    # Extract excitation wavelength from group name
                    excitation_wavelength = int(group_name.split('_')[1])
                    self.excitation_wavelengths.append(excitation_wavelength)

                    group = f[group_name]
                    excitation_data = {
                        'wavelengths': group['wavelengths'][:],
                        'average_cube': None
                    }

                    # Store emission wavelengths for this excitation
                    self.emission_wavelengths[excitation_wavelength] = group['wavelengths'][:]

                    # Load the average cube if available
                    if 'average_cube' in group:
                        excitation_data['average_cube'] = group['average_cube'][:]

                    self.data_dict[excitation_wavelength] = excitation_data

            # Sort excitation wavelengths
            self.excitation_wavelengths.sort()

            # Check if mask is present
            self.mask = None
            if 'mask' in f:
                self.mask = f['mask']['data'][:]
                print(f"Mask found with {np.sum(self.mask)} masked pixels")

    def calculate_entropy(self):
        """
        Calculate Shannon entropy for each excitation-emission combination

        This quantifies how much information (variability) exists in each slice
        Higher entropy = more complex intensity distribution = potentially more information
        """
        results = []

        for ex_wave in self.excitation_wavelengths:
            if 'average_cube' not in self.data_dict[ex_wave] or self.data_dict[ex_wave]['average_cube'] is None:
                print(f"No average cube found for excitation {ex_wave} nm")
                continue

            cube = self.data_dict[ex_wave]['average_cube']
            em_waves = self.emission_wavelengths[ex_wave]

            for band_idx, em_wave in enumerate(em_waves):
                # Get the 2D image for this excitation-emission combination
                img = cube[band_idx]

                # Apply mask if available
                if self.mask is not None:
                    img = img[self.mask == 1]

                # Calculate entropy - first bin the data to get a histogram
                # Normalize and flatten the image
                img_flat = img.flatten()

                # Remove NaN or infinite values
                img_flat = img_flat[np.isfinite(img_flat)]

                if len(img_flat) == 0:
                    # Skip if no valid data
                    continue

                # Calculate entropy using scipy.stats
                entropy = stats.entropy(np.histogram(img_flat, bins=256)[0])

                # Calculate additional statistics
                mean_val = np.mean(img_flat)
                std_val = np.std(img_flat)
                min_val = np.min(img_flat)
                max_val = np.max(img_flat)
                dynamic_range = max_val - min_val
                snr = mean_val / std_val if std_val > 0 else float('inf')

                # Store results
                results.append({
                    'Excitation': ex_wave,
                    'Emission': em_wave,
                    'Entropy': entropy,
                    'Mean': mean_val,
                    'StdDev': std_val,
                    'DynamicRange': dynamic_range,
                    'SNR': snr
                })

        # Convert to DataFrame for easier analysis
        self.entropy_scores = pd.DataFrame(results)
        return self.entropy_scores

    def calculate_mutual_information(self, n_top=20):
        """
        Calculate mutual information between the top n_top excitation-emission combinations

        This helps identify redundancy between different wavelength combinations
        Low MI = less redundancy = more complementary information

        Args:
            n_top: Number of top entropy bands to consider for MI calculation

        Returns:
            Mutual information matrix for the top bands
        """
        if self.entropy_scores is None:
            self.calculate_entropy()

        # Get the top n bands by entropy
        top_bands = self.entropy_scores.sort_values('Entropy', ascending=False).head(n_top)

        # Initialize mutual information matrix
        n = len(top_bands)
        mi_matrix = np.zeros((n, n))

        # Create a list to store the actual images for faster access
        images = []
        band_info = []

        for i, row in top_bands.iterrows():
            ex_wave = row['Excitation']
            em_wave = row['Emission']

            # Find the band index for this emission wavelength
            em_waves = self.emission_wavelengths[ex_wave]
            band_idx = np.argmin(np.abs(em_waves - em_wave))

            # Get the image
            img = self.data_dict[ex_wave]['average_cube'][band_idx]

            # Apply mask if available
            if self.mask is not None:
                img_masked = img[self.mask == 1]
            else:
                img_masked = img.flatten()

            # Remove NaN or infinite values and bin the data
            img_masked = img_masked[np.isfinite(img_masked)]
            if len(img_masked) > 0:
                # Bin the data into discrete bins for mutual information calculation
                hist, bin_edges = np.histogram(img_masked, bins=256)
                digitized = np.digitize(img_masked, bin_edges[:-1])
                images.append(digitized)
                band_info.append((ex_wave, em_wave))

        # Calculate mutual information between each pair of bands
        for i in range(len(images)):
            for j in range(len(images)):
                # Mutual information is symmetric, only calculate upper triangle
                if i <= j:
                    try:
                        mi = mutual_info_score(images[i], images[j])
                        mi_matrix[i, j] = mi
                        mi_matrix[j, i] = mi  # Mirror the value (symmetric)
                    except ValueError as e:
                        print(f"Error calculating MI between bands {i} and {j}: {e}")
                        mi_matrix[i, j] = 0
                        mi_matrix[j, i] = 0

        self.mi_matrix = mi_matrix
        self.mi_band_info = band_info
        return mi_matrix, band_info

    def find_best_combinations(self, n_bands=5, method='entropy'):
        """
        Find the best excitation-emission combinations using different criteria

        Args:
            n_bands: Number of bands to select
            method: Method to use ('entropy', 'entropy_mi', 'snr')

        Returns:
            List of selected bands
        """
        if self.entropy_scores is None:
            self.calculate_entropy()

        if method == 'entropy':
            # Simply select the top n_bands by entropy
            best_bands = self.entropy_scores.sort_values('Entropy', ascending=False).head(n_bands)

        elif method == 'snr':
            # Select the top n_bands by SNR
            best_bands = self.entropy_scores.sort_values('SNR', ascending=False).head(n_bands)

        elif method == 'entropy_mi':
            # Use a greedy approach that maximizes entropy while minimizing mutual information
            # First calculate MI if not already done
            if self.mi_matrix is None:
                self.calculate_mutual_information(n_top=min(30, len(self.entropy_scores)))

            # Start with the highest entropy band
            selected_indices = [0]  # Start with the highest entropy band
            available_indices = list(range(1, len(self.mi_band_info)))

            # Greedily select bands with high entropy and low mutual information with already selected bands
            while len(selected_indices) < n_bands and available_indices:
                best_score = -float('inf')
                best_idx = -1

                for idx in available_indices:
                    # Calculate average mutual information with already selected bands
                    avg_mi = np.mean([self.mi_matrix[idx, j] for j in selected_indices])

                    # Score = entropy - redundancy penalty
                    # We want high entropy and low mutual information
                    score = self.entropy_scores.iloc[idx]['Entropy'] - avg_mi

                    if score > best_score:
                        best_score = score
                        best_idx = idx

                if best_idx >= 0:
                    selected_indices.append(best_idx)
                    available_indices.remove(best_idx)
                else:
                    break

            # Get the corresponding bands
            best_bands = pd.DataFrame([
                {
                    'Excitation': self.mi_band_info[i][0],
                    'Emission': self.mi_band_info[i][1],
                    'Entropy': self.entropy_scores[
                        (self.entropy_scores['Excitation'] == self.mi_band_info[i][0]) &
                        (self.entropy_scores['Emission'] == self.mi_band_info[i][1])
                        ]['Entropy'].values[0]
                }
                for i in selected_indices
            ])

        else:
            raise ValueError(f"Unknown method: {method}")

        self.best_combinations = best_bands
        return best_bands

    def plot_entropy_heatmap(self, output_dir='analysis_results'):
        """
        Create a heatmap visualization of entropy for all excitation-emission combinations

        Args:
            output_dir: Directory to save the plot
        """
        if self.entropy_scores is None:
            self.calculate_entropy()

        # Create pivot table for heatmap
        pivot = self.entropy_scores.pivot_table(values='Entropy', index='Emission', columns='Excitation')

        # Create the plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(pivot, cmap='viridis',
                    xticklabels=sorted(self.entropy_scores['Excitation'].unique()),
                    yticklabels=sorted(self.entropy_scores['Emission'].unique())[
                                ::10])  # Show every 10th emission wavelength for readability

        plt.title('Shannon Entropy for Each Excitation-Emission Combination')
        plt.xlabel('Excitation Wavelength (nm)')
        plt.ylabel('Emission Wavelength (nm)')

        # Save the plot
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, 'entropy_heatmap.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()

        print(f"Entropy heatmap saved to {plot_path}")
        return plot_path

    def plot_mutual_information_heatmap(self, output_dir='analysis_results'):
        """
        Create a heatmap visualization of mutual information between top bands

        Args:
            output_dir: Directory to save the plot
        """
        if self.mi_matrix is None:
            self.calculate_mutual_information()

        # Create the plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.mi_matrix, cmap='coolwarm', annot=True, fmt=".2f",
                    xticklabels=[f"Ex:{ex} Em:{em}" for ex, em in self.mi_band_info],
                    yticklabels=[f"Ex:{ex} Em:{em}" for ex, em in self.mi_band_info])

        # Rotate x-axis labels for readability
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)

        plt.title('Mutual Information Between Top Excitation-Emission Combinations')

        # Save the plot
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, 'mutual_information_heatmap.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Mutual information heatmap saved to {plot_path}")
        return plot_path

    def plot_best_combinations(self, output_dir='analysis_results'):
        """
        Plot the best excitation-emission combinations on the EEM matrix

        Args:
            output_dir: Directory to save the plot
        """
        if self.best_combinations is None:
            self.find_best_combinations()

        if self.entropy_scores is None:
            self.calculate_entropy()

        # Create pivot table for background heatmap
        pivot = self.entropy_scores.pivot_table(values='Entropy', index='Emission', columns='Excitation')

        # Create the plot
        plt.figure(figsize=(12, 10))

        # Plot the background heatmap of entropy
        sns.heatmap(pivot, cmap='viridis', alpha=0.7,
                    xticklabels=sorted(self.entropy_scores['Excitation'].unique()),
                    yticklabels=sorted(self.entropy_scores['Emission'].unique())[::10])

        # Overlay the best combinations as scatter points
        for i, row in self.best_combinations.iterrows():
            # Convert wavelengths to plot coordinates (reverse lookup in the pivot index/columns)
            x_idx = list(pivot.columns).index(row['Excitation'])
            y_idx = list(pivot.index).index(row['Emission'])

            # Add marker with label
            plt.scatter(x_idx + 0.5, y_idx + 0.5, color='red', s=100, marker='*', edgecolor='white')
            plt.text(x_idx + 0.7, y_idx + 0.5, f"{i + 1}", color='white', fontweight='bold')

        plt.title('Best Excitation-Emission Combinations')
        plt.xlabel('Excitation Wavelength (nm)')
        plt.ylabel('Emission Wavelength (nm)')

        # Add a legend explaining the ranking
        top_ex_em = []
        for i, row in self.best_combinations.iterrows():
            top_ex_em.append(f"{i + 1}: Ex:{row['Excitation']}nm, Em:{row['Emission']:.1f}nm")

        plt.figtext(0.5, 0.01, '\n'.join(top_ex_em), ha='center', bbox=dict(facecolor='white', alpha=0.8))

        # Save the plot
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, 'best_combinations.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Best combinations plot saved to {plot_path}")
        return plot_path

    def visualize_best_emission_spectra(self, output_dir='analysis_results'):
        """
        Visualize emission spectra for the best excitation wavelengths

        Args:
            output_dir: Directory to save the plot
        """
        if self.best_combinations is None:
            self.find_best_combinations()

        # Find unique excitation wavelengths in the best combinations
        best_excitations = self.best_combinations['Excitation'].unique()

        # Create a plot for each best excitation wavelength
        plt.figure(figsize=(15, 10))

        for ex_wave in best_excitations:
            # Get emission wavelengths for this excitation
            em_waves = self.emission_wavelengths[ex_wave]

            # Get the data cube for this excitation
            cube = self.data_dict[ex_wave]['average_cube']

            # Calculate mean intensity for each emission wavelength
            if self.mask is not None:
                mean_intensities = [np.mean(cube[i][self.mask == 1]) for i in range(len(em_waves))]
            else:
                mean_intensities = [np.mean(cube[i]) for i in range(len(em_waves))]

            # Plot the emission spectrum
            plt.plot(em_waves, mean_intensities, label=f'Ex: {ex_wave} nm')

            # Mark the best emission wavelengths for this excitation
            best_emissions = self.best_combinations[self.best_combinations['Excitation'] == ex_wave]['Emission'].values
            for em_wave in best_emissions:
                # Find the closest emission wavelength index
                em_idx = np.argmin(np.abs(em_waves - em_wave))
                plt.scatter(em_waves[em_idx], mean_intensities[em_idx],
                            color='red', marker='*', s=100, edgecolor='black')

        plt.xlabel('Emission Wavelength (nm)')
        plt.ylabel('Mean Intensity')
        plt.title('Emission Spectra for Best Excitation Wavelengths')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save the plot
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, 'best_emission_spectra.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()

        print(f"Emission spectra plot saved to {plot_path}")
        return plot_path

    def visualize_band_images(self, n_best=5, output_dir='analysis_results'):
        """
        Visualize the actual images for the best excitation-emission combinations

        Args:
            n_best: Number of best combinations to show
            output_dir: Directory to save the plot
        """
        if self.best_combinations is None:
            self.find_best_combinations()

        # Limit to the top n_best combinations
        best_n = self.best_combinations.head(n_best)

        # Create a figure with n_best subplots
        fig, axes = plt.subplots(1, n_best, figsize=(4 * n_best, 4))
        if n_best == 1:
            axes = [axes]  # Make sure axes is iterable

        for i, (_, row) in enumerate(best_n.iterrows()):
            ex_wave = row['Excitation']
            em_wave = row['Emission']

            # Find the band index for this emission wavelength
            em_waves = self.emission_wavelengths[ex_wave]
            band_idx = np.argmin(np.abs(em_waves - em_wave))

            # Get the image
            img = self.data_dict[ex_wave]['average_cube'][band_idx]

            # Apply mask if available
            if self.mask is not None:
                # Create a masked version for display (replace non-ROI with NaN)
                display_img = img.copy()
                display_img[self.mask != 1] = np.nan
            else:
                display_img = img

            # Plot the image
            im = axes[i].imshow(display_img, cmap='viridis')
            axes[i].set_title(f"Ex: {ex_wave}nm\nEm: {em_wave:.1f}nm")
            axes[i].axis('off')

            # Add colorbar
            plt.colorbar(im, ax=axes[i], label='Intensity')

        plt.suptitle('Top Informative Excitation-Emission Combinations', fontsize=16)
        plt.tight_layout()

        # Save the plot
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, 'best_combination_images.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()

        print(f"Band images saved to {plot_path}")
        return plot_path


def run_information_theory_analysis(h5_file_path, output_dir='analysis_results', n_bands=5):
    """
    Run the complete information theory-based analysis workflow

    Args:
        h5_file_path: Path to the HDF5 file containing hyperspectral data
        output_dir: Directory to save results
        n_bands: Number of top bands to identify

    Returns:
        Dictionary with results including best combinations and file paths
    """
    # Create the analyzer
    analyzer = InfoTheoryAnalyzer(h5_file_path)

    # Calculate entropy scores
    entropy_scores = analyzer.calculate_entropy()
    print(f"Calculated entropy for {len(entropy_scores)} excitation-emission combinations")

    # Calculate mutual information for top bands
    mi_matrix, band_info = analyzer.calculate_mutual_information(n_top=min(30, len(entropy_scores)))
    print(f"Calculated mutual information for top bands")

    # Find best combinations using entropy and MI
    best_combinations_entropy = analyzer.find_best_combinations(n_bands=n_bands, method='entropy')
    print(f"\nTop {n_bands} combinations by entropy:")
    print(best_combinations_entropy)

    # Reset and find best combinations using entropy-mi method
    best_combinations_entropy_mi = analyzer.find_best_combinations(n_bands=n_bands, method='entropy_mi')
    print(f"\nTop {n_bands} combinations by entropy with minimum redundancy:")
    print(best_combinations_entropy_mi)

    # Reset and find best combinations using SNR
    best_combinations_snr = analyzer.find_best_combinations(n_bands=n_bands, method='snr')
    print(f"\nTop {n_bands} combinations by SNR:")
    print(best_combinations_snr)

    # Create visualization plots
    entropy_heatmap_path = analyzer.plot_entropy_heatmap(output_dir)
    mi_heatmap_path = analyzer.plot_mutual_information_heatmap(output_dir)
    best_combinations_path = analyzer.plot_best_combinations(output_dir)
    emission_spectra_path = analyzer.visualize_best_emission_spectra(output_dir)
    band_images_path = analyzer.visualize_band_images(n_best=n_bands, output_dir=output_dir)

    # Return results
    results = {
        'best_combinations_entropy': best_combinations_entropy,
        'best_combinations_entropy_mi': best_combinations_entropy_mi,
        'best_combinations_snr': best_combinations_snr,
        'entropy_heatmap_path': entropy_heatmap_path,
        'mi_heatmap_path': mi_heatmap_path,
        'best_combinations_path': best_combinations_path,
        'emission_spectra_path': emission_spectra_path,
        'band_images_path': band_images_path
    }

    return results


if __name__ == "__main__":
    # Example usage:
    h5_file_path = "../kiwi_hyperspectral_4d_data_with_mask.h5"
    output_dir = "information_theory_results"
    results = run_information_theory_analysis(h5_file_path, output_dir, n_bands=10)