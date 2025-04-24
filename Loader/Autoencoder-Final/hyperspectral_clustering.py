"""
Hyperspectral Data Clustering and Classification

This module provides functions for clustering and classifying hyperspectral data
using the encoded features from the convolutional autoencoder.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.colors as mcolors
from typing import Dict, List, Tuple, Optional, Union


def extract_encoded_features(
        model,
        data_dict,
        device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Extract encoded features from the convolutional autoencoder for all excitation wavelengths.

    Args:
        model: Trained HyperspectralCAEVariable model
        data_dict: Dictionary mapping excitation wavelengths to data tensors
        device: Device to use for computation

    Returns:
        Dictionary mapping excitation wavelengths to encoded features
    """
    # Ensure model is in eval mode
    model.eval()

    # Move model to the specified device
    model = model.to(device)  # Add this line to fix the error

    # Store encoded features for each excitation
    encoded_features = {}
    spatial_shapes = {}

    with torch.no_grad():
        for ex, data in data_dict.items():
            # Add batch dimension
            data_batch = {ex: data.unsqueeze(0).to(device)}

            # Extract encoded representation
            encoded = model.encode(data_batch)

            # Get the encoded features
            features = encoded.cpu().numpy()[0]  # Remove batch dimension

            # Store spatial dimensions to use for reconstructing cluster maps
            spatial_shapes[ex] = (features.shape[1], features.shape[2])

            # Reshape to (channels, height*width)
            num_channels = features.shape[0]
            features_reshaped = features.reshape(num_channels, -1)

            # Store features
            encoded_features[ex] = features_reshaped

    return encoded_features, spatial_shapes


def prepare_features_for_clustering(
        encoded_features,
        combine_excitations=True,
        reduction_method='none',
        n_components=2
):
    """
    Prepare encoded features for clustering by optionally combining excitation wavelengths
    and reducing dimensionality.

    Args:
        encoded_features: Dictionary mapping excitations to encoded features
        combine_excitations: Whether to combine all excitation wavelengths
        reduction_method: Dimensionality reduction method ('none', 'pca', 'tsne')
        n_components: Number of components for dimensionality reduction

    Returns:
        Prepared features and excitation info
    """
    if combine_excitations:
        # Combine features from all excitations
        all_features = []
        excitation_indices = []
        excitation_list = list(encoded_features.keys())

        for i, ex in enumerate(excitation_list):
            features = encoded_features[ex]
            all_features.append(features.T)  # Transpose to (samples, features)
            excitation_indices.extend([i] * features.shape[1])

        # Stack all features
        combined_features = np.vstack(all_features)
        excitation_indices = np.array(excitation_indices)

        # Apply dimensionality reduction if requested
        if reduction_method == 'pca' and combined_features.shape[1] > n_components:
            print(f"Applying PCA to reduce from {combined_features.shape[1]} to {n_components} dimensions")
            pca = PCA(n_components=n_components)
            reduced_features = pca.fit_transform(combined_features)
            explained_var = pca.explained_variance_ratio_
            print(f"Explained variance: {sum(explained_var):.2%}")

            return reduced_features, excitation_indices, excitation_list

        elif reduction_method == 'tsne':
            print(f"Applying t-SNE to reduce to {n_components} dimensions")
            tsne = TSNE(n_components=n_components, random_state=42)
            reduced_features = tsne.fit_transform(combined_features)

            return reduced_features, excitation_indices, excitation_list

        else:
            return combined_features, excitation_indices, excitation_list

    else:
        # Process each excitation separately
        reduced_features = {}

        for ex, features in encoded_features.items():
            features_transposed = features.T  # (samples, features)

            # Apply dimensionality reduction if requested
            if reduction_method == 'pca' and features_transposed.shape[1] > n_components:
                pca = PCA(n_components=n_components)
                reduced = pca.fit_transform(features_transposed)
                reduced_features[ex] = reduced

            elif reduction_method == 'tsne':
                tsne = TSNE(n_components=n_components, random_state=42)
                reduced = tsne.fit_transform(features_transposed)
                reduced_features[ex] = reduced

            else:
                reduced_features[ex] = features_transposed

        return reduced_features, None, list(encoded_features.keys())


def cluster_features(
        features,
        method='kmeans',
        n_clusters=5,
        eps=0.5,  # For DBSCAN
        min_samples=5,  # For DBSCAN
        random_state=42
):
    """
    Apply clustering to the prepared features.

    Args:
        features: Prepared features for clustering
        method: Clustering method ('kmeans', 'dbscan', 'agglomerative', 'gmm')
        n_clusters: Number of clusters (for KMeans, Agglomerative, GMM)
        eps: Epsilon parameter for DBSCAN
        min_samples: Minimum samples parameter for DBSCAN
        random_state: Random state for reproducibility

    Returns:
        Cluster labels and clustering model
    """
    # Initialize clustering algorithm
    if method == 'kmeans':
        print(f"Applying K-means clustering with {n_clusters} clusters")
        clustering = KMeans(n_clusters=n_clusters, random_state=random_state)

    elif method == 'dbscan':
        print(f"Applying DBSCAN clustering with eps={eps}, min_samples={min_samples}")
        clustering = DBSCAN(eps=eps, min_samples=min_samples)

    elif method == 'agglomerative':
        print(f"Applying Agglomerative clustering with {n_clusters} clusters")
        clustering = AgglomerativeClustering(n_clusters=n_clusters)

    elif method == 'gmm':
        print(f"Applying Gaussian Mixture Model with {n_clusters} components")
        clustering = GaussianMixture(n_components=n_clusters, random_state=random_state)

    else:
        raise ValueError(f"Unknown clustering method: {method}")

    # Fit clustering model
    if isinstance(features, dict):
        # If features is a dictionary (separate excitations)
        cluster_labels = {}
        for ex, ex_features in features.items():
            cluster_labels[ex] = clustering.fit_predict(ex_features)
    else:
        # If features is a combined array (all excitations)
        cluster_labels = clustering.fit_predict(features)

    return cluster_labels, clustering


def evaluate_clustering(features, labels):
    """
    Evaluate clustering quality using various metrics.

    Args:
        features: Features used for clustering
        labels: Cluster labels

    Returns:
        Dictionary of clustering quality metrics
    """
    # Check if there are at least 2 clusters
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return {
            'silhouette': np.nan,
            'davies_bouldin': np.nan,
            'calinski_harabasz': np.nan
        }

    # Check if all data points have been assigned to clusters
    if -1 in unique_labels:
        # Remove noise points (label -1) for metric calculation
        mask = labels != -1
        features_filtered = features[mask]
        labels_filtered = labels[mask]

        # Check if we still have at least 2 clusters after filtering
        if len(np.unique(labels_filtered)) < 2:
            return {
                'silhouette': np.nan,
                'davies_bouldin': np.nan,
                'calinski_harabasz': np.nan
            }
    else:
        features_filtered = features
        labels_filtered = labels

    try:
        # Calculate clustering quality metrics
        silhouette = silhouette_score(features_filtered, labels_filtered)
        davies_bouldin = davies_bouldin_score(features_filtered, labels_filtered)
        calinski_harabasz = calinski_harabasz_score(features_filtered, labels_filtered)

        return {
            'silhouette': silhouette,  # Higher is better, range [-1, 1]
            'davies_bouldin': davies_bouldin,  # Lower is better
            'calinski_harabasz': calinski_harabasz  # Higher is better
        }
    except Exception as e:
        print(f"Error calculating clustering metrics: {str(e)}")
        return {
            'silhouette': np.nan,
            'davies_bouldin': np.nan,
            'calinski_harabasz': np.nan
        }


def visualize_clusters_2d(
        features,
        labels,
        excitation_indices=None,
        excitation_wavelengths=None,
        title="Cluster Visualization",
        marker_size=10,
        show_legend=True
):
    """
    Visualize clustering results in 2D.

    Args:
        features: 2D array of features for visualization
        labels: Cluster labels
        excitation_indices: Optional array mapping points to excitation wavelengths
        excitation_wavelengths: Optional list of excitation wavelength values
        title: Plot title
        marker_size: Size of markers in the plot
        show_legend: Whether to show the legend

    Returns:
        Matplotlib figure
    """
    # Make sure features has at least 2 dimensions
    if features.shape[1] < 2:
        print("Error: Features must have at least 2 dimensions for 2D visualization")
        return None

    # Create figure
    plt.figure(figsize=(12, 10))

    # Get unique cluster labels
    unique_labels = np.unique(labels)

    # Create colormap for clusters
    n_clusters = len(unique_labels)
    if -1 in unique_labels:  # If there are noise points (DBSCAN)
        colors = ['gray'] + list(plt.cm.tab10(np.linspace(0, 1, n_clusters - 1)))
    else:
        colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

    # Add a different plotting mode if excitation indices are provided
    if excitation_indices is not None and excitation_wavelengths is not None:
        # Plot points colored by cluster but with different markers by excitation
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x']

        for i, label in enumerate(unique_labels):
            # Get points in this cluster
            mask = labels == label
            cluster_points = features[mask]

            # For each excitation wavelength
            for j, ex_wavelength in enumerate(excitation_wavelengths):
                # Get points in this cluster and this excitation
                ex_mask = excitation_indices[mask] == j
                if np.any(ex_mask):
                    ex_points = cluster_points[ex_mask]
                    plt.scatter(
                        ex_points[:, 0],
                        ex_points[:, 1],
                        s=marker_size,
                        c=[colors[i]],
                        marker=markers[j % len(markers)],
                        label=f"Cluster {label}, Ex={ex_wavelength}nm"
                    )
    else:
        # Simple plot with just cluster colors
        for i, label in enumerate(unique_labels):
            mask = labels == label
            cluster_points = features[mask]

            if label == -1:
                # Noise points (DBSCAN)
                plt.scatter(
                    cluster_points[:, 0],
                    cluster_points[:, 1],
                    s=marker_size // 2,  # Smaller markers for noise
                    c='gray',
                    marker='.',
                    label="Noise"
                )
            else:
                plt.scatter(
                    cluster_points[:, 0],
                    cluster_points[:, 1],
                    s=marker_size,
                    c=[colors[i]],
                    label=f"Cluster {label}"
                )

    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    if show_legend:
        # Create a more compact legend if there are many clusters or excitations
        if excitation_indices is not None and len(unique_labels) * len(excitation_wavelengths) > 10:
            # First, legend for clusters
            cluster_handles = []
            for i, label in enumerate(unique_labels):
                cluster_handles.append(plt.Line2D([0], [0], marker='o', color=colors[i],
                                                  label=f"Cluster {label}", linestyle=''))

            # Then, legend for excitations
            excitation_handles = []
            for j, ex_wavelength in enumerate(excitation_wavelengths):
                excitation_handles.append(plt.Line2D([0], [0], marker=markers[j % len(markers)],
                                                     color='black', label=f"Ex={ex_wavelength}nm",
                                                     linestyle=''))

            # Create two legends
            first_legend = plt.legend(handles=cluster_handles, loc='upper right',
                                      title="Clusters", bbox_to_anchor=(1.15, 1))
            plt.gca().add_artist(first_legend)
            plt.legend(handles=excitation_handles, loc='upper right',
                       title="Excitations", bbox_to_anchor=(1.15, 0.7))
        else:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.grid(True, alpha=0.3)

    return plt.gcf()


def map_clusters_to_image(
        labels,
        spatial_shapes,
        excitation_wavelengths,
        combined=True
):
    """
    Map cluster labels back to spatial positions to create cluster maps.

    Args:
        labels: Cluster labels
        spatial_shapes: Dictionary mapping excitations to spatial dimensions
        excitation_wavelengths: List of excitation wavelengths
        combined: Whether the labels are from combined excitations

    Returns:
        Dictionary mapping excitations to cluster maps
    """
    cluster_maps = {}

    if combined:
        # Labels are for combined excitations
        excitation_indices = np.arange(len(excitation_wavelengths))

        start_idx = 0
        for i, ex in enumerate(excitation_wavelengths):
            # Get spatial shape for this excitation
            height, width = spatial_shapes[ex]
            num_pixels = height * width

            # Get labels for this excitation
            ex_labels = labels[start_idx:start_idx + num_pixels]

            # Reshape to spatial dimensions
            cluster_map = ex_labels.reshape(height, width)
            cluster_maps[ex] = cluster_map

            start_idx += num_pixels
    else:
        # Labels are separate for each excitation
        for ex in excitation_wavelengths:
            if ex in labels:
                # Get spatial shape for this excitation
                height, width = spatial_shapes[ex]

                # Reshape to spatial dimensions
                cluster_map = labels[ex].reshape(height, width)
                cluster_maps[ex] = cluster_map

    return cluster_maps


def visualize_cluster_maps(
        cluster_maps,
        excitation_wavelengths,
        cols=3,
        cmap='tab10'
):
    """
    Visualize cluster maps for multiple excitation wavelengths.

    Args:
        cluster_maps: Dictionary mapping excitations to cluster maps
        excitation_wavelengths: List of excitation wavelengths
        cols: Number of columns in the grid
        cmap: Colormap for visualization

    Returns:
        Matplotlib figure
    """
    # Determine number of excitations to plot
    n_excitations = len([ex for ex in excitation_wavelengths if ex in cluster_maps])

    if n_excitations == 0:
        print("No cluster maps available for visualization")
        return None

    # Calculate grid dimensions
    rows = (n_excitations + cols - 1) // cols

    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))

    # Handle single row or column case
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)

    # Flatten axes for easy indexing
    axes_flat = axes.flatten()

    # Plot each cluster map
    idx = 0
    for i, ex in enumerate(excitation_wavelengths):
        if ex in cluster_maps:
            if idx >= len(axes_flat):
                break

            # Get cluster map
            cluster_map = cluster_maps[ex]

            # Plot cluster map
            im = axes_flat[idx].imshow(cluster_map, cmap=cmap, interpolation='nearest')
            axes_flat[idx].set_title(f"Excitation {ex}nm Clusters")

            # Add color bar
            cbar = plt.colorbar(im, ax=axes_flat[idx])
            cbar.set_label("Cluster ID")

            # Increment index
            idx += 1

    # Hide any unused subplots
    for j in range(idx, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.tight_layout()

    return fig


def compare_cluster_maps(
        cluster_maps,
        excitation_wavelengths,
        original_data_dict=None,
        emission_idx=None,
        spectral_profiles=True
):
    """
    Compare cluster maps across excitation wavelengths with original data and spectral profiles.

    Args:
        cluster_maps: Dictionary mapping excitations to cluster maps
        excitation_wavelengths: List of excitation wavelengths
        original_data_dict: Optional dictionary of original hyperspectral data
        emission_idx: Optional emission band index for original data visualization
        spectral_profiles: Whether to show spectral profiles for each cluster

    Returns:
        Matplotlib figure
    """
    # Determine number of excitations to compare
    excitations_to_plot = [ex for ex in excitation_wavelengths if ex in cluster_maps]
    n_excitations = len(excitations_to_plot)

    if n_excitations == 0:
        print("No cluster maps available for comparison")
        return None

    # Determine number of rows and columns
    if original_data_dict is not None:
        cols = 2  # One for cluster map, one for original data
    else:
        cols = 1  # Just cluster maps

    # Add an extra row for spectral profiles if requested
    if spectral_profiles and original_data_dict is not None:
        extra_rows = 1
    else:
        extra_rows = 0

    # Create figure
    fig = plt.figure(figsize=(cols * 6, n_excitations * 4 + extra_rows * 4))

    # Define colormap for clusters
    cluster_cmap = plt.get_cmap('tab10')

    # Plot each excitation
    for i, ex in enumerate(excitations_to_plot):
        # Get cluster map
        cluster_map = cluster_maps[ex]

        # Get unique clusters
        unique_clusters = np.unique(cluster_map)
        n_clusters = len(unique_clusters)

        # Plot cluster map
        ax1 = plt.subplot(n_excitations + extra_rows, cols, i * cols + 1)
        im1 = ax1.imshow(cluster_map, cmap='tab10', interpolation='nearest')
        ax1.set_title(f"Excitation {ex}nm: {n_clusters} Clusters")

        # Add color bar
        cbar = plt.colorbar(im1, ax=ax1)
        cbar.set_label("Cluster ID")

        # Plot original data if available
        if original_data_dict is not None and ex in original_data_dict:
            original_data = original_data_dict[ex]

            # Determine which emission band to show
            if emission_idx is None:
                # Use middle band
                emission_idx = original_data.shape[2] // 2

            # Get the spectral slice
            if isinstance(original_data, torch.Tensor):
                original_slice = original_data[:, :, emission_idx].cpu().numpy()
            else:
                original_slice = original_data[:, :, emission_idx]

            # Plot original data
            ax2 = plt.subplot(n_excitations + extra_rows, cols, i * cols + 2)
            im2 = ax2.imshow(original_slice, cmap='viridis')
            ax2.set_title(f"Original Data (Ex={ex}nm, Em Band={emission_idx})")

            # Add color bar
            cbar2 = plt.colorbar(im2, ax=ax2)
            cbar2.set_label("Intensity")

    # Add spectral profiles for each cluster if requested
    if spectral_profiles and original_data_dict is not None:
        # Create a subplot for spectral profiles
        ax_spectral = plt.subplot(n_excitations + extra_rows, 1, n_excitations + 1)

        # Plot spectral profiles for each cluster in each excitation
        for i, ex in enumerate(excitations_to_plot):
            if ex not in original_data_dict:
                continue

            # Get cluster map and original data
            cluster_map = cluster_maps[ex]
            original_data = original_data_dict[ex]

            if isinstance(original_data, torch.Tensor):
                original_data = original_data.cpu().numpy()

            # Get unique clusters
            unique_clusters = np.unique(cluster_map)

            # Get spectral profiles for each cluster
            for j, cluster_id in enumerate(unique_clusters):
                if cluster_id < 0:  # Skip noise points
                    continue

                # Create mask for this cluster
                mask = cluster_map == cluster_id

                # Calculate mean spectrum for this cluster
                cluster_spectrum = np.mean(original_data[mask], axis=0)

                # Plot spectrum
                if len(excitations_to_plot) > 1:
                    # Use different line styles for different excitations
                    linestyle = ['-', '--', ':', '-.'][i % 4]
                    ax_spectral.plot(
                        range(len(cluster_spectrum)),
                        cluster_spectrum,
                        color=cluster_cmap(j % 10),
                        linestyle=linestyle,
                        label=f"Ex={ex}nm, Cluster {cluster_id}"
                    )
                else:
                    # Just use different colors for clusters
                    ax_spectral.plot(
                        range(len(cluster_spectrum)),
                        cluster_spectrum,
                        color=cluster_cmap(j % 10),
                        label=f"Cluster {cluster_id}"
                    )

        ax_spectral.set_xlabel("Emission Band Index")
        ax_spectral.set_ylabel("Mean Intensity")
        ax_spectral.set_title("Spectral Profiles by Cluster")
        ax_spectral.grid(True, alpha=0.3)
        ax_spectral.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    return fig


def create_cluster_distribution_chart(
        labels,
        cluster_maps,
        excitation_wavelengths,
        combined=True
):
    """
    Create a chart showing the distribution of points across clusters.

    Args:
        labels: Cluster labels
        cluster_maps: Dictionary mapping excitations to cluster maps
        excitation_wavelengths: List of excitation wavelengths
        combined: Whether the labels are from combined excitations

    Returns:
        Matplotlib figure
    """
    # Get unique clusters
    if combined:
        unique_clusters = np.unique(labels)
    else:
        # Combine all cluster labels
        all_labels = []
        for ex in excitation_wavelengths:
            if ex in labels:
                all_labels.extend(labels[ex])
        unique_clusters = np.unique(all_labels)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Prepare data for bar chart
    if combined:
        # Count points per cluster across all excitations
        cluster_counts = {cluster: np.sum(labels == cluster) for cluster in unique_clusters}

        # If we have excitation indices, break down by excitation
        excitation_breakdown = {}

        # For each excitation, count points in each cluster
        for ex in excitation_wavelengths:
            if ex in cluster_maps:
                ex_map = cluster_maps[ex]
                ex_breakdown = {cluster: np.sum(ex_map == cluster) for cluster in unique_clusters}
                excitation_breakdown[ex] = ex_breakdown
    else:
        # Count points per cluster for each excitation
        excitation_breakdown = {}
        for ex in excitation_wavelengths:
            if ex in labels:
                ex_labels = labels[ex]
                ex_breakdown = {cluster: np.sum(ex_labels == cluster) for cluster in unique_clusters}
                excitation_breakdown[ex] = ex_breakdown

        # Calculate overall counts
        cluster_counts = {cluster: 0 for cluster in unique_clusters}
        for ex_breakdown in excitation_breakdown.values():
            for cluster, count in ex_breakdown.items():
                cluster_counts[cluster] += count

    # Sort clusters by ID
    sorted_clusters = sorted(unique_clusters)

    # Create bar positions
    bar_width = 0.8 / len(excitation_wavelengths)
    positions = np.arange(len(sorted_clusters))

    # Plot bars for each excitation
    for i, ex in enumerate(excitation_wavelengths):
        if ex in excitation_breakdown:
            # Get counts for this excitation
            ex_counts = [excitation_breakdown[ex].get(cluster, 0) for cluster in sorted_clusters]

            # Calculate offset for this excitation's bars
            offset = (i - len(excitation_wavelengths) / 2 + 0.5) * bar_width

            # Plot bars
            ax.bar(
                positions + offset,
                ex_counts,
                bar_width,
                label=f"Ex={ex}nm",
                alpha=0.7
            )

    # Set x-axis labels and ticks
    ax.set_xticks(positions)
    ax.set_xticklabels([f"Cluster {cluster}" for cluster in sorted_clusters])

    # Set title and labels
    ax.set_title("Distribution of Points Across Clusters")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Points")

    # Add legend
    ax.legend()

    # Add grid
    ax.grid(True, axis='y', alpha=0.3)

    return fig


def run_hyperspectral_clustering(
        model,
        dataset,
        clustering_config=None,
        device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Run the complete hyperspectral clustering pipeline.

    Args:
        model: Trained HyperspectralCAEVariable model
        dataset: HyperspectralDataset instance
        clustering_config: Dictionary with clustering parameters
        device: Device to use for computation

    Returns:
        Dictionary with clustering results
    """
    # Default clustering configuration
    default_config = {
        'method': 'kmeans',
        'n_clusters': 5,
        'eps': 0.5,
        'min_samples': 5,
        'combine_excitations': True,
        'reduction_method': 'pca',
        'n_components': 8
    }

    model = model.to(device)

    # Use default config if none provided
    config = default_config.copy()
    if clustering_config is not None:
        config.update(clustering_config)

    # Get data
    all_data = dataset.get_all_data()

    # Extract encoded features
    print("Extracting encoded features...")
    encoded_features, spatial_shapes = extract_encoded_features(model, all_data, device)

    # Prepare features for clustering
    print("Preparing features for clustering...")
    prepared_features, excitation_indices, excitation_wavelengths = prepare_features_for_clustering(
        encoded_features,
        combine_excitations=config['combine_excitations'],
        reduction_method=config['reduction_method'],
        n_components=config['n_components']
    )

    # Apply clustering
    print(f"Applying {config['method']} clustering...")
    cluster_labels, clustering_model = cluster_features(
        prepared_features,
        method=config['method'],
        n_clusters=config['n_clusters'],
        eps=config['eps'],
        min_samples=config['min_samples']
    )

    # Evaluate clustering quality
    if isinstance(prepared_features, dict):
        # Evaluate each excitation separately
        quality_metrics = {}
        for ex in prepared_features:
            if ex in cluster_labels:
                metrics = evaluate_clustering(prepared_features[ex], cluster_labels[ex])
                quality_metrics[ex] = metrics
    else:
        # Evaluate combined clustering
        quality_metrics = evaluate_clustering(prepared_features, cluster_labels)

    # Map clusters back to images
    print("Mapping clusters to spatial positions...")
    cluster_maps = map_clusters_to_image(
        cluster_labels,
        spatial_shapes,
        excitation_wavelengths,
        combined=config['combine_excitations']
    )

    # Return results
    return {
        'encoded_features': encoded_features,
        'prepared_features': prepared_features,
        'excitation_indices': excitation_indices,
        'excitation_wavelengths': excitation_wavelengths,
        'cluster_labels': cluster_labels,
        'clustering_model': clustering_model,
        'quality_metrics': quality_metrics,
        'cluster_maps': cluster_maps,
        'spatial_shapes': spatial_shapes,
        'config': config
    }


def visualize_clustering_results(
        clustering_results,
        original_data_dict=None,
        emission_idx=None
):
    """
    Create visualizations for clustering results.

    Args:
        clustering_results: Results from run_hyperspectral_clustering
        original_data_dict: Optional dictionary of original hyperspectral data
        emission_idx: Optional emission band index for original data visualization

    Returns:
        Dictionary of visualization figures
    """
    # Extract necessary components from results
    prepared_features = clustering_results['prepared_features']
    cluster_labels = clustering_results['cluster_labels']
    excitation_indices = clustering_results['excitation_indices']
    excitation_wavelengths = clustering_results['excitation_wavelengths']
    cluster_maps = clustering_results['cluster_maps']
    quality_metrics = clustering_results['quality_metrics']
    config = clustering_results['config']

    # Create visualizations dictionary
    visualizations = {}

    # 1. Cluster visualization in feature space (if dimensionality is appropriate)
    if config['reduction_method'] != 'none' and config['n_components'] >= 2:
        if config['combine_excitations']:
            # Only visualize if features were reduced to 2D or 3D
            if isinstance(prepared_features, np.ndarray) and prepared_features.shape[1] >= 2:
                # Use only first 2 dimensions for visualization
                viz_features = prepared_features[:, :2]

                # Create cluster visualization
                print("Creating cluster visualization in feature space...")
                cluster_viz = visualize_clusters_2d(
                    viz_features,
                    cluster_labels,
                    excitation_indices,
                    excitation_wavelengths,
                    title=f"{config['method'].upper()} Clustering Results"
                )

                visualizations['cluster_2d'] = cluster_viz
        else:
            # Create visualizations for each excitation
            for ex in excitation_wavelengths:
                if ex in prepared_features and ex in cluster_labels:
                    # Use only first 2 dimensions
                    viz_features = prepared_features[ex][:, :2]

                    # Create cluster visualization
                    ex_cluster_viz = visualize_clusters_2d(
                        viz_features,
                        cluster_labels[ex],
                        title=f"{config['method'].upper()} Clustering for Ex={ex}nm"
                    )

                    visualizations[f'cluster_2d_{ex}'] = ex_cluster_viz

    # 2. Cluster maps visualization
    print("Creating cluster maps visualization...")
    cluster_maps_viz = visualize_cluster_maps(
        cluster_maps,
        excitation_wavelengths
    )

    visualizations['cluster_maps'] = cluster_maps_viz

    # 3. Cluster comparison with original data
    if original_data_dict is not None:
        print("Creating cluster comparison visualization...")
        comparison_viz = compare_cluster_maps(
            cluster_maps,
            excitation_wavelengths,
            original_data_dict,
            emission_idx,
            spectral_profiles=True
        )

        visualizations['comparison'] = comparison_viz

    # 4. Cluster distribution chart
    print("Creating cluster distribution chart...")
    distribution_viz = create_cluster_distribution_chart(
        cluster_labels,
        cluster_maps,
        excitation_wavelengths,
        combined=config['combine_excitations']
    )

    visualizations['distribution'] = distribution_viz

    # 5. Clustering quality metrics visualization
    if isinstance(quality_metrics, dict) and len(quality_metrics) > 1:
        # Multiple excitations with separate metrics
        print("Creating clustering quality metrics visualization...")

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Prepare data
        ex_list = list(quality_metrics.keys())
        silhouette_scores = [quality_metrics[ex]['silhouette'] for ex in ex_list]
        davies_bouldin_scores = [quality_metrics[ex]['davies_bouldin'] for ex in ex_list]
        calinski_harabasz_scores = [quality_metrics[ex]['calinski_harabasz'] for ex in ex_list]

        # Plot metrics
        axes[0].bar(ex_list, silhouette_scores)
        axes[0].set_title("Silhouette Score (higher is better)")
        axes[0].set_xlabel("Excitation Wavelength (nm)")
        axes[0].set_ylabel("Score")
        axes[0].grid(True, alpha=0.3)

        axes[1].bar(ex_list, davies_bouldin_scores)
        axes[1].set_title("Davies-Bouldin Score (lower is better)")
        axes[1].set_xlabel("Excitation Wavelength (nm)")
        axes[1].set_ylabel("Score")
        axes[1].grid(True, alpha=0.3)

        axes[2].bar(ex_list, calinski_harabasz_scores)
        axes[2].set_title("Calinski-Harabasz Score (higher is better)")
        axes[2].set_xlabel("Excitation Wavelength (nm)")
        axes[2].set_ylabel("Score")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        visualizations['quality_metrics'] = fig

    return visualizations