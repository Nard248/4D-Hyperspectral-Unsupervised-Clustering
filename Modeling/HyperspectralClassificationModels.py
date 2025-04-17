import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, NMF
import umap
from sklearn.metrics import silhouette_score


class HyperspectralClustering:
    """
    A class for applying different clustering algorithms to 4D hyperspectral data
    """

    def __init__(self):
        self.model = None
        self.labels = None
        self.data = None
        self.scaler = StandardScaler()

    def preprocess_data(self, hypercube, format_type='flatten_pixels'):
        """
        Preprocess 4D hyperspectral data into a format suitable for clustering

        Parameters:
        -----------
        hypercube : numpy.ndarray
            4D hyperspectral data with shape (excitations, emissions, height, width)
        format_type : str
            Type of formatting to apply:
            - 'flatten_pixels': Flatten to (height*width, excitations*emissions)
            - 'pixel_spectra': Extract excitation-emission matrix for each pixel (height*width, excitations, emissions)
            - 'tensor': Keep the original tensor format

        Returns:
        --------
        formatted_data : numpy.ndarray
            The formatted data ready for clustering
        """
        self.original_shape = hypercube.shape
        excitations, emissions, height, width = hypercube.shape

        if format_type == 'flatten_pixels':
            # Reshape to (height*width, excitations*emissions)
            formatted_data = hypercube.reshape(excitations * emissions, height * width).T
            # Scale the data
            formatted_data = self.scaler.fit_transform(formatted_data)

        elif format_type == 'pixel_spectra':
            # For each pixel, extract full excitation-emission matrix
            # Results in: (height*width, excitations, emissions)
            formatted_data = hypercube.transpose(2, 3, 0, 1).reshape(height * width, excitations, emissions)
            # Reshape to (height*width, excitations*emissions) for standard clustering
            formatted_data = formatted_data.reshape(height * width, excitations * emissions)
            # Scale the data
            formatted_data = self.scaler.fit_transform(formatted_data)

        elif format_type == 'tensor':
            # Keep original structure
            formatted_data = hypercube

        self.data = formatted_data
        return formatted_data

    def apply_kmeans(self, n_clusters=5, random_state=42):
        """
        Apply K-means clustering to the preprocessed data

        Parameters:
        -----------
        n_clusters : int
            Number of clusters
        random_state : int
            Random seed for reproducibility

        Returns:
        --------
        labels : numpy.ndarray
            Cluster labels for each sample
        """
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.labels = self.model.fit_predict(self.data)
        return self.labels

    def apply_dbscan(self, eps=0.5, min_samples=5):
        """
        Apply DBSCAN clustering to the preprocessed data

        Parameters:
        -----------
        eps : float
            The maximum distance between two samples for them to be considered as in the same neighborhood
        min_samples : int
            The number of samples in a neighborhood for a point to be considered as a core point

        Returns:
        --------
        labels : numpy.ndarray
            Cluster labels for each sample
        """
        self.model = DBSCAN(eps=eps, min_samples=min_samples)
        self.labels = self.model.fit_predict(self.data)
        return self.labels

    def apply_hierarchical(self, n_clusters=5, linkage='ward'):
        """
        Apply hierarchical clustering to the preprocessed data

        Parameters:
        -----------
        n_clusters : int
            Number of clusters
        linkage : str
            Linkage criterion to use

        Returns:
        --------
        labels : numpy.ndarray
            Cluster labels for each sample
        """
        self.model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        self.labels = self.model.fit_predict(self.data)
        return self.labels

    def apply_gmm(self, n_components=5, random_state=42):
        """
        Apply Gaussian Mixture Model clustering to the preprocessed data

        Parameters:
        -----------
        n_components : int
            Number of mixture components
        random_state : int
            Random seed for reproducibility

        Returns:
        --------
        labels : numpy.ndarray
            Cluster labels for each sample
        """
        self.model = GaussianMixture(n_components=n_components, random_state=random_state)
        self.labels = self.model.fit_predict(self.data)
        return self.labels

    def visualize_clusters(self, reduction_method='pca', show_plot=True):
        """
        Visualize the clustering results using dimensionality reduction

        Parameters:
        -----------
        reduction_method : str
            Method to use for dimensionality reduction ('pca' or 'umap')
        show_plot : bool
            Whether to display the plot

        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object
        """
        if self.labels is None:
            raise ValueError("No clustering has been performed yet")

        # Apply dimensionality reduction
        if reduction_method == 'pca':
            reducer = PCA(n_components=2)
            reduced_data = reducer.fit_transform(self.data)
        elif reduction_method == 'umap':
            reducer = umap.UMAP(n_components=2, random_state=42)
            reduced_data = reducer.fit_transform(self.data)
        else:
            raise ValueError("Unknown reduction method. Use 'pca' or 'umap'.")

        # Plot results
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=self.labels, cmap='viridis', alpha=0.7)

        # Add color bar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Cluster')

        # Set labels and title
        ax.set_xlabel(f'{reduction_method.upper()} Component 1')
        ax.set_ylabel(f'{reduction_method.upper()} Component 2')
        ax.set_title(f'Clustering Results Visualized with {reduction_method.upper()}')

        if show_plot:
            plt.show()

        return fig

    def visualize_spatial_clusters(self, show_plot=True):
        """
        Visualize the clustering results spatially

        Parameters:
        -----------
        show_plot : bool
            Whether to display the plot

        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object
        """
        if self.labels is None:
            raise ValueError("No clustering has been performed yet")

        # Reshape cluster labels to original spatial dimensions
        _, _, height, width = self.original_shape
        spatial_labels = self.labels.reshape(height, width)

        # Plot results
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(spatial_labels, cmap='viridis')

        # Add color bar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Cluster')

        # Set title
        ax.set_title('Spatial Distribution of Clusters')

        if show_plot:
            plt.show()

        return fig

    def evaluate_clustering(self, n_clusters_range=range(2, 11)):
        """
        Evaluate clustering performance using silhouette score for different numbers of clusters

        Parameters:
        -----------
        n_clusters_range : range or list
            Range of number of clusters to evaluate

        Returns:
        --------
        silhouette_scores : list
            Silhouette scores for each number of clusters
        """
        silhouette_scores = []

        for n_clusters in n_clusters_range:
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(self.data)

            # Calculate silhouette score
            score = silhouette_score(self.data, labels)
            silhouette_scores.append(score)
            print(f'Number of clusters: {n_clusters}, Silhouette score: {score:.4f}')

        # Plot silhouette scores
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(n_clusters_range, silhouette_scores, 'o-')
        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel('Silhouette Score')
        ax.set_title('Silhouette Score vs. Number of Clusters')
        ax.grid(True)
        plt.show()

        return silhouette_scores