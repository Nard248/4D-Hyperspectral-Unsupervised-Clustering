import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import os
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE
import umap
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap


class ClusteringFeatureSelector:
    """Class for clustering and feature selection on hyperspectral data"""

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
        self.unfolded_data = None
        self.bands_data = None
        self.band_labels = None
        self.cluster_labels = None
        self.selected_bands = None

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

    def prepare_bands_data(self):
        """
        Prepare data for band clustering by creating a matrix where each band is a row
        and columns represent spatial pixels

        Returns:
            bands_data: Matrix of bands x pixels
            band_labels: List of (excitation, emission) tuples for each band
        """
        # First, check if we have all the necessary data
        if not self.data_dict:
            raise ValueError("No data loaded. Please call load_data() first.")

        # Count total number of emission wavelengths across all excitations
        total_bands = sum(len(self.emission_wavelengths[ex]) for ex in self.excitation_wavelengths)

        # Get spatial dimensions from the first data cube
        first_ex = self.excitation_wavelengths[0]
        if 'average_cube' not in self.data_dict[first_ex] or self.data_dict[first_ex]['average_cube'] is None:
            raise ValueError(f"No average cube found for excitation {first_ex} nm")

        height, width = self.data_dict[first_ex]['average_cube'].shape[1:]
        total_pixels = height * width

        # If mask is present, count only masked pixels
        if self.mask is not None:
            total_pixels = np.sum(self.mask)

        # Initialize the bands data matrix
        # Rows = bands (excitation-emission combinations), Columns = pixels
        bands_data = np.zeros((total_bands, total_pixels))

        # Keep track of band labels
        band_labels = []

        # Current row index
        row_idx = 0

        # For each excitation wavelength
        for ex_wave in self.excitation_wavelengths:
            # Get emission wavelengths for this excitation
            em_waves = self.emission_wavelengths[ex_wave]

            # Get the data cube for this excitation
            cube = self.data_dict[ex_wave]['average_cube']

            # For each emission wavelength
            for band_idx, em_wave in enumerate(em_waves):
                # Get the 2D image for this excitation-emission combination
                img = cube[band_idx]

                # If mask is present, extract only masked pixels
                if self.mask is not None:
                    pixel_values = img[self.mask == 1]
                else:
                    pixel_values = img.flatten()

                # Store in the bands data matrix
                bands_data[row_idx, :] = pixel_values

                # Store the label for this band
                band_labels.append((ex_wave, em_wave))

                # Increment row index
                row_idx += 1

        # Store the bands data
        self.bands_data = bands_data
        self.band_labels = band_labels

        print(f"Bands data shape: {bands_data.shape}")
        return bands_data, band_labels

    def compute_band_similarity_matrix(self, metric='correlation'):
        """
        Compute a similarity matrix between all bands

        Args:
            metric: Distance metric to use ('correlation', 'euclidean', etc.)

        Returns:
            Similarity matrix
        """
        if self.bands_data is None:
            self.prepare_bands_data()

        # Compute pairwise distances
        distances = pdist(self.bands_data, metric=metric)

        # Convert to a square matrix
        dist_matrix = squareform(distances)

        # If using correlation, convert to similarity (1 - correlation distance)
        if metric == 'correlation':
            sim_matrix = 1 - dist_matrix
        else:
            # For other metrics, normalize to [0, 1] and invert
            sim_matrix = 1 - (dist_matrix / np.max(dist_matrix))

        self.similarity_matrix = sim_matrix
        return sim_matrix

    def cluster_bands(self, n_clusters=5, method='kmeans'):
        """
        Cluster the bands based on their similarity

        Args:
            n_clusters: Number of clusters to form
            method: Clustering method ('kmeans' or 'spectral')

        Returns:
            Cluster labels for each band
        """
        if self.bands_data is None:
            self.prepare_bands_data()

        # Standardize the bands data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.bands_data)

        # Apply clustering
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=0)
            cluster_labels = clusterer.fit_predict(scaled_data)
        elif method == 'spectral':
            # Compute similarity matrix if not already done
            if not hasattr(self, 'similarity_matrix'):
                self.compute_band_similarity_matrix(metric='correlation')

            # Convert similarity to affinity (ensure non-negative)
            affinity = np.maximum(self.similarity_matrix, 0)

            clusterer = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=0)
            cluster_labels = clusterer.fit_predict(affinity)
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        # Store the cluster labels
        self.cluster_labels = cluster_labels
        self.n_clusters = n_clusters

        # Evaluate clustering quality with silhouette score
        try:
            silhouette_avg = silhouette_score(scaled_data, cluster_labels)
            print(f"Silhouette score: {silhouette_avg:.3f}")
        except:
            print("Silhouette score calculation failed")

        return cluster_labels

    def visualize_clusters_umap(self, output_dir='analysis_results'):
        """
        Visualize the band clusters using UMAP

        Args:
            output_dir: Directory to save the plot

        Returns:
            Path to the saved plot
        """
        if self.bands_data is None or self.cluster_labels is None:
            raise ValueError("No clustering results available. Please call cluster_bands() first.")

        # Apply UMAP for visualization
        reducer = umap.UMAP(random_state=0)
        embedding = reducer.fit_transform(StandardScaler().fit_transform(self.bands_data))

        # Create scatter plot colored by cluster
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=self.cluster_labels, cmap='tab10', s=100, alpha=0.8)

        # Add labels for selected points
        for cluster_id in range(self.n_clusters):
            # Find the center of each cluster in the embedding
            cluster_points = embedding[self.cluster_labels == cluster_id]
            if len(cluster_points) > 0:
                center = np.mean(cluster_points, axis=0)
                plt.text(center[0], center[1], f'Cluster {cluster_id + 1}', fontsize=12, fontweight='bold',
                         ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))

        plt.title(f'UMAP Visualization of {self.n_clusters} Band Clusters')
        plt.colorbar(scatter, label='Cluster')

        # Save the plot
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, 'band_clusters_umap.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()

        return plot_path

    def visualize_cluster_similarity_matrix(self, output_dir='analysis_results'):
        """
        Visualize the similarity matrix with cluster annotations

        Args:
            output_dir: Directory to save the plot

        Returns:
            Path to the saved plot
        """
        if not hasattr(self, 'similarity_matrix') or self.cluster_labels is None:
            # Compute similarity matrix if not done yet
            self.compute_band_similarity_matrix()

        # Create the plot
        plt.figure(figsize=(14, 12))

        # Sort by cluster for better visualization
        sort_idx = np.argsort(self.cluster_labels)
        sorted_matrix = self.similarity_matrix[sort_idx, :][:, sort_idx]
        sorted_labels = self.cluster_labels[sort_idx]

        # Get boundaries of clusters
        boundaries = []
        for i in range(self.n_clusters):
            if i < self.n_clusters - 1:
                # Find the index where the next cluster starts
                boundaries.append(np.where(sorted_labels == i)[0][-1] + 0.5)

        # Plot the heatmap
        sns.heatmap(sorted_matrix, cmap='coolwarm', center=0)

        # Add lines to mark cluster boundaries
        for boundary in boundaries:
            plt.axhline(y=boundary, color='black', linestyle='-')
            plt.axvline(x=boundary, color='black', linestyle='-')

        plt.title('Band Similarity Matrix (Sorted by Cluster)')

        # Save the plot
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, 'cluster_similarity_matrix.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()

        return plot_path

    def select_representative_bands(self, method='variance'):
        """
        Select representative bands from each cluster

        Args:
            method: Method to select representative bands ('variance', 'centroid', 'max_similarity')

        Returns:
            Selected bands as a list of (excitation, emission) tuples
        """
        if self.bands_data is None or self.cluster_labels is None:
            raise ValueError("No clustering results available. Please call cluster_bands() first.")

        selected_bands = []

        # For each cluster
        for cluster_id in range(self.n_clusters):
            # Get indices of bands in this cluster
            cluster_indices = np.where(self.cluster_labels == cluster_id)[0]

            # Get the data for these bands
            cluster_data = self.bands_data[cluster_indices, :]

            # Select a representative band based on the chosen method
            if method == 'variance':
                # Select the band with highest variance
                band_variances = np.var(cluster_data, axis=1)
                representative_idx = cluster_indices[np.argmax(band_variances)]

            elif method == 'centroid':
                # Select the band closest to the cluster centroid
                centroid = np.mean(cluster_data, axis=0)
                distances = np.sqrt(np.sum((cluster_data - centroid) ** 2, axis=1))
                representative_idx = cluster_indices[np.argmin(distances)]

            elif method == 'max_similarity':
                # Select the band with maximum total similarity to other bands in the cluster
                if not hasattr(self, 'similarity_matrix'):
                    self.compute_band_similarity_matrix()

                # Extract the submatrix for this cluster
                submatrix = self.similarity_matrix[cluster_indices][:, cluster_indices]

                # Sum similarities for each band (excluding self-similarity)
                np.fill_diagonal(submatrix, 0)  # Zero out self-similarity
                total_similarities = np.sum(submatrix, axis=1)

                representative_idx = cluster_indices[np.argmax(total_similarities)]

            else:
                raise ValueError(f"Unknown selection method: {method}")

            # Get the excitation-emission values for this band
            selected_bands.append(self.band_labels[representative_idx])

        # Store the selected bands
        self.selected_bands = selected_bands

        # Create a DataFrame for easier reading
        selected_df = pd.DataFrame([
            {'Cluster': i + 1, 'Excitation': ex, 'Emission': em}
            for i, (ex, em) in enumerate(selected_bands)
        ])

        print("Selected bands:")
        print(selected_df)

        return selected_bands, selected_df

    def visualize_cluster_spectra(self, output_dir='analysis_results'):
        """
        Visualize the spectra of bands in each cluster

        Args:
            output_dir: Directory to save the plot

        Returns:
            List of paths to the saved plots
        """
        if self.bands_data is None or self.cluster_labels is None:
            raise ValueError("No clustering results available. Please call cluster_bands() first.")

        plot_paths = []

        # Prepare data to plot emission spectra for each excitation
        ex_clusters = {}

        # For each cluster
        for cluster_id in range(self.n_clusters):
            # Get indices of bands in this cluster
            cluster_indices = np.where(self.cluster_labels == cluster_id)[0]

            # Group by excitation wavelength
            for idx in cluster_indices:
                ex_wave, em_wave = self.band_labels[idx]

                if ex_wave not in ex_clusters:
                    ex_clusters[ex_wave] = {}

                if cluster_id not in ex_clusters[ex_wave]:
                    ex_clusters[ex_wave][cluster_id] = []

                ex_clusters[ex_wave][cluster_id].append((em_wave, idx))

        # For each excitation wavelength
        for ex_wave in sorted(ex_clusters.keys()):
            plt.figure(figsize=(12, 8))

            # For each cluster that has bands with this excitation
            for cluster_id, points in ex_clusters[ex_wave].items():
                # Sort points by emission wavelength
                points.sort(key=lambda x: x[0])

                # Extract emission wavelengths and corresponding indices
                em_waves = [p[0] for p in points]
                indices = [p[1] for p in points]

                # Get the data for these bands
                spectra = self.bands_data[indices, :]

                # Calculate mean and std for each spectrum
                mean_spectra = np.mean(spectra, axis=1)

                # Plot the spectra
                plt.plot(em_waves, mean_spectra, 'o-', label=f'Cluster {cluster_id + 1}')

            # Mark the selected band for this excitation if available
            if hasattr(self, 'selected_bands'):
                for i, (ex, em) in enumerate(self.selected_bands):
                    if ex == ex_wave:
                        # Find the index of this band
                        band_idx = self.band_labels.index((ex, em))

                        # Get the mean value for this band
                        mean_val = np.mean(self.bands_data[band_idx, :])

                        # Mark it on the plot
                        plt.scatter([em], [mean_val], color='red', s=100, marker='*', edgecolor='black')
                        plt.text(em, mean_val, f'Selected', fontsize=10, ha='right', va='bottom')

            plt.xlabel('Emission Wavelength (nm)')
            plt.ylabel('Mean Signal')
            plt.title(f'Cluster Spectra for Excitation {ex_wave} nm')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Save the plot
            os.makedirs(output_dir, exist_ok=True)
            plot_path = os.path.join(output_dir, f'cluster_spectra_ex{ex_wave}.png')
            plt.savefig(plot_path, dpi=150)
            plt.close()

            plot_paths.append(plot_path)

        return plot_paths

    def visualize_eem_clusters(self, output_dir='analysis_results'):
        """
        Visualize the clusters on an excitation-emission matrix

        Args:
            output_dir: Directory to save the plot

        Returns:
            Path to the saved plot
        """
        if self.bands_data is None or self.cluster_labels is None:
            raise ValueError("No clustering results available. Please call cluster_bands() first.")

        # Create a DataFrame with band info
        df = pd.DataFrame([
            {'Excitation': ex, 'Emission': em, 'Cluster': cluster + 1}
            for (ex, em), cluster in zip(self.band_labels, self.cluster_labels)
        ])

        # Create the plot
        plt.figure(figsize=(12, 10))

        # Plot each cluster with a different color
        for cluster_id in range(self.n_clusters):
            cluster_df = df[df['Cluster'] == cluster_id + 1]
            plt.scatter(cluster_df['Emission'], cluster_df['Excitation'],
                        s=50, label=f'Cluster {cluster_id + 1}')

        # Mark selected bands if available
        if hasattr(self, 'selected_bands'):
            for i, (ex, em) in enumerate(self.selected_bands):
                plt.scatter([em], [ex], color='red', s=100, marker='*', edgecolor='black')
                plt.text(em, ex, f'C{i + 1}', fontsize=12, fontweight='bold', ha='right', va='bottom')

        plt.xlabel('Emission Wavelength (nm)')
        plt.ylabel('Excitation Wavelength (nm)')
        plt.title('Excitation-Emission Clusters')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save the plot
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, 'eem_clusters.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()

        return plot_path

    def visualize_selected_bands(self, n_best=5, output_dir='analysis_results'):
        """
        Visualize the actual images for the selected bands

        Args:
            n_best: Number of selected bands to show (limit if there are many clusters)
            output_dir: Directory to save the plot

        Returns:
            Path to the saved plot
        """
        if not hasattr(self, 'selected_bands'):
            raise ValueError("No selected bands available. Please call select_representative_bands() first.")

        # Limit to n_best if needed
        bands_to_show = self.selected_bands[:n_best]
        n_bands = len(bands_to_show)

        # Create a figure with n_bands subplots
        fig, axes = plt.subplots(1, n_bands, figsize=(4 * n_bands, 4))
        if n_bands == 1:
            axes = [axes]  # Make sure axes is iterable

        for i, (ex_wave, em_wave) in enumerate(bands_to_show):
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
            axes[i].set_title(f"Cluster {i + 1}\nEx: {ex_wave}nm\nEm: {em_wave:.1f}nm")
            axes[i].axis('off')

            # Add colorbar
            plt.colorbar(im, ax=axes[i], label='Intensity')

        plt.suptitle('Selected Representative Bands', fontsize=16)
        plt.tight_layout()

        # Save the plot
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, 'selected_band_images.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()

        return plot_path

    def rank_bands_by_variance(self):
        """
        Rank all bands by their variance

        Returns:
            DataFrame with bands ranked by variance
        """
        if self.bands_data is None:
            self.prepare_bands_data()

        # Calculate variance for each band
        band_variances = np.var(self.bands_data, axis=1)

        # Create a DataFrame with band info and variance
        variance_df = pd.DataFrame([
            {'Excitation': ex, 'Emission': em, 'Variance': var}
            for (ex, em), var in zip(self.band_labels, band_variances)
        ])

        # Sort by variance in descending order
        variance_df = variance_df.sort_values('Variance', ascending=False)

        return variance_df

    def run_feature_selection(self, n_features=10, method='variance'):
        """
        Select the most important bands using a specific feature selection method

        Args:
            n_features: Number of features to select
            method: Feature selection method ('variance' or 'clustering')

        Returns:
            DataFrame with selected bands
        """
        if method == 'variance':
            # Simply select the top n_features bands by variance
            variance_df = self.rank_bands_by_variance()
            selected_df = variance_df.head(n_features)

            # Store the selected bands
            self.variance_selected_bands = [(row['Excitation'], row['Emission'])
                                            for _, row in selected_df.iterrows()]

        elif method == 'clustering':
            # If clustering hasn't been done yet, do it with n_features clusters
            if self.cluster_labels is None or self.n_clusters != n_features:
                self.cluster_bands(n_clusters=n_features)

            # Select representative bands
            selected_bands, selected_df = self.select_representative_bands()

        else:
            raise ValueError(f"Unknown feature selection method: {method}")

        print(f"Selected {n_features} bands using {method} method:")
        print(selected_df)

        return selected_df

    def visualize_feature_selection_comparison(self, n_features=5, output_dir='analysis_results'):
        """
        Compare different feature selection methods

        Args:
            n_features: Number of features to select
            output_dir: Directory to save the plot

        Returns:
            Path to the saved plot
        """
        # Get band features using different methods
        variance_df = self.run_feature_selection(n_features, method='variance')
        self.cluster_bands(n_clusters=n_features)
        cluster_df = self.run_feature_selection(n_features, method='clustering')

        # Create a summary DataFrame
        summary_data = []

        # Add variance-based selections
        for i, (_, row) in enumerate(variance_df.iterrows()):
            summary_data.append({
                'Method': 'Variance',
                'Rank': i + 1,
                'Excitation': row['Excitation'],
                'Emission': row['Emission'],
                'Value': row['Variance']
            })

        # Add cluster-based selections
        for i, (_, row) in enumerate(cluster_df.iterrows()):
            summary_data.append({
                'Method': 'Clustering',
                'Rank': i + 1,
                'Excitation': row['Excitation'],
                'Emission': row['Emission'],
                'Value': row['Cluster']  # Just use cluster number as value
            })

        summary_df = pd.DataFrame(summary_data)

        # Create the plot to compare selections
        plt.figure(figsize=(12, 10))

        # Plot variance-based selections
        variance_data = summary_df[summary_df['Method'] == 'Variance']
        plt.scatter(variance_data['Emission'], variance_data['Excitation'],
                    color='blue', s=100, marker='o', label='Variance-based')

        # Plot cluster-based selections
        cluster_data = summary_df[summary_df['Method'] == 'Clustering']
        plt.scatter(cluster_data['Emission'], cluster_data['Excitation'],
                    color='red', s=100, marker='s', label='Cluster-based')

        # Add labels for each point
        for i, row in variance_data.iterrows():
            plt.text(row['Emission'], row['Excitation'], f"V{row['Rank']}",
                     color='blue', fontsize=10, ha='right', va='bottom')

        for i, row in cluster_data.iterrows():
            plt.text(row['Emission'], row['Excitation'], f"C{row['Rank']}",
                     color='red', fontsize=10, ha='left', va='top')

        plt.xlabel('Emission Wavelength (nm)')
        plt.ylabel('Excitation Wavelength (nm)')
        plt.title('Comparison of Feature Selection Methods')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save the plot
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, 'feature_selection_comparison.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()

        return plot_path, summary_df


def run_clustering_feature_selection(h5_file_path, output_dir='analysis_results', n_clusters=5):
    """
    Run the complete clustering and feature selection workflow

    Args:
        h5_file_path: Path to the HDF5 file containing hyperspectral data
        output_dir: Directory to save results
        n_clusters: Number of clusters/features to find

    Returns:
        Dictionary with results
    """
    # Create the analyzer
    analyzer = ClusteringFeatureSelector(h5_file_path)

    # Prepare data for analysis
    analyzer.prepare_bands_data()

    # Compute band similarity
    analyzer.compute_band_similarity_matrix()

    # Run clustering
    analyzer.cluster_bands(n_clusters=n_clusters)

    # Select representative bands
    selected_bands, selected_df = analyzer.select_representative_bands(method='centroid')

    # Rank bands by variance
    variance_df = analyzer.rank_bands_by_variance()
    print("\nTop bands by variance:")
    print(variance_df.head(n_clusters))

    # Visualize results
    umap_plot = analyzer.visualize_clusters_umap(output_dir)
    similarity_plot = analyzer.visualize_cluster_similarity_matrix(output_dir)
    spectra_plots = analyzer.visualize_cluster_spectra(output_dir)
    eem_plot = analyzer.visualize_eem_clusters(output_dir)
    band_images_plot = analyzer.visualize_selected_bands(output_dir=output_dir)

    # Compare feature selection methods
    comparison_plot, summary_df = analyzer.visualize_feature_selection_comparison(n_features=n_clusters,
                                                                                  output_dir=output_dir)

    # Return results
    results = {
        'selected_bands': selected_bands,
        'selected_df': selected_df,
        'variance_df': variance_df,
        'umap_plot': umap_plot,
        'similarity_plot': similarity_plot,
        'spectra_plots': spectra_plots,
        'eem_plot': eem_plot,
        'band_images_plot': band_images_plot,
        'comparison_plot': comparison_plot,
        'summary_df': summary_df
    }

    return results


if __name__ == "__main__":
    # Example usage:
    h5_file_path = "../kiwi_hyperspectral_4d_data_with_mask.h5"
    output_dir = "clustering_feature_selection_results"
    results = run_clustering_feature_selection(h5_file_path, output_dir, n_clusters=5)