import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
from pathlib import Path
import pickle


def visualize_reconstructions(original, reconstructed, n_samples=5, excitation_idx=None, emission_idx=None,
                              save_path=None, figsize=(15, 6)):
    """
    Visualize original images and their reconstructions.

    Args:
        original: Original data [batch, excitations, emissions, height, width]
        reconstructed: Reconstructed data [batch, excitations, emissions, height, width]
        n_samples: Number of random samples to visualize
        excitation_idx: Specific excitation index (if None, uses first)
        emission_idx: Specific emission index (if None, uses mean over emissions)
        save_path: Path to save visualization
        figsize: Figure size

    Returns:
        Figure object
    """
    n_samples = min(n_samples, original.shape[0])

    # Default to first excitation if not specified
    if excitation_idx is None:
        excitation_idx = 0

    # Extract data for the specified excitation
    orig_ex = original[:, excitation_idx]
    recon_ex = reconstructed[:, excitation_idx]

    fig, axes = plt.subplots(2, n_samples, figsize=figsize)

    for i in range(n_samples):
        # Get original and reconstructed data
        orig_sample = orig_ex[i]
        recon_sample = recon_ex[i]

        # If emission index is specified, use that specific band
        if emission_idx is not None:
            orig_img = orig_sample[emission_idx]
            recon_img = recon_sample[emission_idx]
            title_suffix = f"(Ex:{excitation_idx}, Em:{emission_idx})"
        else:
            # Otherwise, use mean over emission dimension
            orig_img = np.mean(orig_sample, axis=0)
            recon_img = np.mean(recon_sample, axis=0)
            title_suffix = f"(Ex:{excitation_idx}, Mean Em)"

        # Display original
        axes[0, i].imshow(orig_img, cmap='viridis')
        axes[0, i].set_title(f"Original {title_suffix}" if i == 0 else "Original")
        axes[0, i].axis('off')

        # Display reconstruction
        axes[1, i].imshow(recon_img, cmap='viridis')
        axes[1, i].set_title(f"Reconstructed {title_suffix}" if i == 0 else "Reconstructed")
        axes[1, i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Reconstruction visualization saved to {save_path}")

    return fig


def visualize_cluster_map(cluster_labels, height, width, n_clusters=10, save_path=None, figsize=(10, 8)):
    """
    Visualize clustering results as a spatial map.

    Args:
        cluster_labels: Cluster assignments
        height: Image height
        width: Image width
        n_clusters: Number of clusters
        save_path: Path to save visualization
        figsize: Figure size

    Returns:
        Figure object
    """
    # Reshape labels to match image dimensions
    if len(cluster_labels.shape) == 1:
        cluster_image = cluster_labels.reshape(height, width)
    else:
        cluster_image = cluster_labels

    # Create colormap
    cmap = plt.cm.get_cmap('tab10', n_clusters)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot cluster map
    im = ax.imshow(cluster_image, cmap=cmap, vmin=0, vmax=n_clusters - 1)

    # Add colorbar with discrete labels
    cbar = plt.colorbar(im, ax=ax, ticks=np.arange(n_clusters))
    cbar.set_label('Cluster')

    ax.set_title(f'Hyperspectral Clustering Result ({n_clusters} clusters)')
    ax.set_xlabel('Pixel X')
    ax.set_ylabel('Pixel Y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Cluster map saved to {save_path}")

    return fig


def visualize_individual_clusters(cluster_labels, height, width, n_clusters=10, save_path=None, figsize=(15, 10)):
    """
    Visualize each cluster separately as binary masks.

    Args:
        cluster_labels: Cluster assignments
        height: Image height
        width: Image width
        n_clusters: Number of clusters
        save_path: Path to save visualization
        figsize: Figure size

    Returns:
        Figure object
    """
    # Reshape labels to match image dimensions
    if len(cluster_labels.shape) == 1:
        cluster_image = cluster_labels.reshape(height, width)
    else:
        cluster_image = cluster_labels

    # Determine grid dimensions
    n_cols = min(4, n_clusters)
    n_rows = (n_clusters + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Flatten axes for easy indexing if only one row
    if n_rows == 1:
        axes = [axes]

    # Create a mask for each cluster
    for i in range(n_clusters):
        row = i // n_cols
        col = i % n_cols

        # Create mask for this cluster
        mask = (cluster_image == i).astype(float)

        # Plot mask
        ax = axes[row][col] if n_rows > 1 else axes[col]
        ax.imshow(mask, cmap='viridis')
        ax.set_title(f'Cluster {i}')
        ax.axis('off')

        # Add text with percentage of pixels in this cluster
        percentage = np.sum(mask) / (height * width) * 100
        ax.text(0.05, 0.95, f"{percentage:.1f}% of pixels",
                transform=ax.transAxes, color='white', fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round',
                                                   facecolor='black', alpha=0.5))

    # Hide unused subplots
    for i in range(n_clusters, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row][col] if n_rows > 1 else axes[col]
        ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Individual clusters visualization saved to {save_path}")

    return fig


def visualize_feature_space(features, cluster_labels, n_clusters=10, method='pca', save_path=None, figsize=(12, 10)):
    """
    Visualize feature space using dimensionality reduction.

    Args:
        features: Latent features
        cluster_labels: Cluster assignments
        n_clusters: Number of clusters
        method: Dimensionality reduction method ('pca' or 'tsne')
        save_path: Path to save visualization
        figsize: Figure size

    Returns:
        Figure object
    """
    # Apply dimensionality reduction
    if method.lower() == 'pca':
        reducer = PCA(n_components=2)
        title_prefix = 'PCA'
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
        title_prefix = 't-SNE'
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pca' or 'tsne'.")

    # Reduce dimensionality
    reduced_features = reducer.fit_transform(features)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create colormap
    cmap = plt.cm.get_cmap('tab10', n_clusters)

    # Create scatter plot
    scatter = ax.scatter(
        reduced_features[:, 0], reduced_features[:, 1],
        c=cluster_labels, cmap=cmap, alpha=0.7, s=5
    )

    # Add legend
    legend_elements = [
        mpatches.Patch(color=cmap(i), label=f'Cluster {i}')
        for i in range(n_clusters)
    ]
    ax.legend(handles=legend_elements, loc='best')

    ax.set_title(f'{title_prefix} Visualization of Feature Space')
    ax.set_xlabel(f'{title_prefix} Dimension 1')
    ax.set_ylabel(f'{title_prefix} Dimension 2')
    ax.grid(True, alpha=0.3)

    # Add cluster centroids if PCA (since centroids match the feature space)
    if method.lower() == 'pca' and hasattr(reducer, 'transform'):
        try:
            # If kmeans model is available in globals (a bit hacky)
            if 'kmeans' in globals() and hasattr(globals()['kmeans'], 'cluster_centers_'):
                centroids = globals()['kmeans'].cluster_centers_
                centroid_reduced = reducer.transform(centroids)

                ax.scatter(
                    centroid_reduced[:, 0], centroid_reduced[:, 1],
                    marker='X', s=100, c=range(n_clusters), cmap=cmap,
                    edgecolors='black', linewidth=2
                )
        except Exception as e:
            print(f"Error plotting centroids: {e}")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"{title_prefix} visualization saved to {save_path}")

    return fig


def visualize_spectral_signatures(original_data, cluster_labels, data_dict, height, width,
                                  n_clusters=10, max_excitations=5, save_path=None, figsize=(15, 10)):
    """
    Visualize average spectral signatures for each cluster.

    Args:
        original_data: Original data or path to pickle file
        cluster_labels: Cluster assignments
        data_dict: Data dictionary from HyperspectralDataset
        height: Image height
        width: Image width
        n_clusters: Number of clusters
        max_excitations: Maximum number of excitations to plot
        save_path: Path to save visualization
        figsize: Figure size

    Returns:
        Figure object
    """
    # Reshape labels to match image dimensions
    if len(cluster_labels.shape) == 1:
        cluster_image = cluster_labels.reshape(height, width)
    else:
        cluster_image = cluster_labels

    # Load data_dict if string is provided
    if isinstance(original_data, str):
        with open(original_data, 'rb') as f:
            data_dict = pickle.load(f)
    else:
        data_dict = data_dict

    # Get excitation wavelengths
    excitation_wavelengths = sorted([float(ex) for ex in data_dict['data'].keys()])

    # Limit number of excitations to plot
    if max_excitations is not None and max_excitations < len(excitation_wavelengths):
        # Choose evenly spaced excitations
        excitation_indices = np.linspace(0, len(excitation_wavelengths) - 1, max_excitations, dtype=int)
        excitation_wavelengths = [excitation_wavelengths[i] for i in excitation_indices]

    # Set up subplots for each excitation
    n_ex_cols = min(3, len(excitation_wavelengths))
    n_ex_rows = (len(excitation_wavelengths) + n_ex_cols - 1) // n_ex_cols

    fig, axes = plt.subplots(n_ex_rows, n_ex_cols, figsize=figsize)

    # Flatten axes for easy indexing if only one row
    if n_ex_rows == 1:
        axes = [axes]

    # Colors for clusters
    cluster_colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

    # For each excitation, plot average spectrum for each cluster
    for ex_idx, ex in enumerate(excitation_wavelengths):
        row = ex_idx // n_ex_cols
        col = ex_idx % n_ex_cols

        # Get the current axis
        ax = axes[row][col] if n_ex_rows > 1 else axes[col]

        ex_str = str(ex)
        cube = data_dict['data'][ex_str]['cube']
        wavelengths = data_dict['data'][ex_str]['wavelengths']

        # Plot spectrum for each cluster
        for cluster_idx in range(n_clusters):
            # Get mask for this cluster
            mask = cluster_image == cluster_idx

            # Skip if cluster is empty
            if not np.any(mask):
                continue

            # Extract data for this cluster
            cluster_data = cube[mask]

            # Calculate mean spectrum
            mean_spectrum = np.mean(cluster_data, axis=0)

            # Plot with cluster-specific color
            ax.plot(wavelengths, mean_spectrum, '-',
                    color=cluster_colors[cluster_idx], linewidth=2,
                    label=f'Cluster {cluster_idx}')

        ax.set_xlabel('Emission Wavelength (nm)')
        ax.set_ylabel('Mean Intensity')
        ax.set_title(f'Excitation: {ex} nm')
        ax.grid(True, alpha=0.3)

        # Add legend to first plot only
        if ex_idx == 0:
            ax.legend(loc='best')

    # Hide unused subplots
    for i in range(len(excitation_wavelengths), n_ex_rows * n_ex_cols):
        row = i // n_ex_cols
        col = i % n_ex_cols
        ax = axes[row][col] if n_ex_rows > 1 else axes[col]
        ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Spectral signatures visualization saved to {save_path}")

    return fig


def create_summary_visualization(result, original_data, reconstructed_data, data_dict, height, width,
                                 n_clusters=10, save_dir='results', prefix='cluster_'):
    """
    Create a comprehensive summary of clustering results.

    Args:
        result: Dictionary with clustering results
        original_data: Original data
        reconstructed_data: Reconstructed data
        data_dict: Data dictionary from HyperspectralDataset
        height: Image height
        width: Image width
        n_clusters: Number of clusters
        save_dir: Directory to save visualizations
        prefix: Prefix for saved files

    Returns:
        Dictionary with paths to saved visualizations
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Create visualizations
    paths = {}

    # 1. Reconstruction quality
    recon_path = os.path.join(save_dir, f"{prefix}reconstructions.png")
    visualize_reconstructions(
        original_data, reconstructed_data, n_samples=5,
        save_path=recon_path
    )
    paths['reconstructions'] = recon_path

    # 2. Cluster map
    cluster_map_path = os.path.join(save_dir, f"{prefix}map.png")
    visualize_cluster_map(
        result['cluster_labels'], height, width, n_clusters,
        save_path=cluster_map_path
    )
    paths['cluster_map'] = cluster_map_path

    # 3. Individual clusters
    individual_clusters_path = os.path.join(save_dir, f"{prefix}individual.png")
    visualize_individual_clusters(
        result['cluster_labels'], height, width, n_clusters,
        save_path=individual_clusters_path
    )
    paths['individual_clusters'] = individual_clusters_path

    # 4. Feature space visualization - PCA
    pca_path = os.path.join(save_dir, f"{prefix}pca.png")
    visualize_feature_space(
        result['features'], result['cluster_labels'], n_clusters, method='pca',
        save_path=pca_path
    )
    paths['pca'] = pca_path

    # 5. Feature space visualization - t-SNE (if not too many samples)
    if len(result['features']) <= 10000:
        tsne_path = os.path.join(save_dir, f"{prefix}tsne.png")
        try:
            visualize_feature_space(
                result['features'], result['cluster_labels'], n_clusters, method='tsne',
                save_path=tsne_path
            )
            paths['tsne'] = tsne_path
        except Exception as e:
            print(f"Error creating t-SNE visualization: {e}")

    # 6. Spectral signatures
    spectra_path = os.path.join(save_dir, f"{prefix}spectra.png")
    visualize_spectral_signatures(
        data_dict, result['cluster_labels'], data_dict, height, width, n_clusters,
        save_path=spectra_path
    )
    paths['spectra'] = spectra_path

    # 7. Save metrics as text file
    metrics_path = os.path.join(save_dir, f"{prefix}metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write(f"Clustering Metrics (n_clusters={n_clusters}):\n")
        f.write("-" * 40 + "\n")
        for metric_name, value in result['metrics'].items():
            f.write(f"{metric_name}: {value:.4f}\n")
    paths['metrics'] = metrics_path

    # 8. Save results as pickle for further analysis
    results_path = os.path.join(save_dir, f"{prefix}results.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(result, f)
    paths['results'] = results_path

    print(f"All visualizations saved to {save_dir}")

    return paths