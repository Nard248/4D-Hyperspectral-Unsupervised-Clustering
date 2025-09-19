"""
Concatenation-based clustering for hyperspectral data analysis.
This module processes masked hyperspectral data by concatenating spectral intensities
at valid pixel locations and applying clustering algorithms.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Union
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import seaborn as sns
from tqdm import tqdm
try:
    from .ground_truth_validation import (
        extract_ground_truth_from_png,
        calculate_clustering_accuracy,
        visualize_clustering_vs_ground_truth,
        compare_multiple_clusterings_to_ground_truth
    )
except ImportError:
    from ground_truth_validation import (
        extract_ground_truth_from_png,
        calculate_clustering_accuracy,
        visualize_clustering_vs_ground_truth,
        compare_multiple_clusterings_to_ground_truth
    )


def load_masked_data(file_path: Union[str, Path]) -> Dict:
    """
    Load masked hyperspectral data from a pickle file.
    
    Args:
        file_path: Path to the pickle file containing masked data
        
    Returns:
        Dictionary containing the loaded data
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def concatenate_hyperspectral_data_improved(
    data_dict: Dict,
    global_normalize: bool = True,
    normalization_method: str = 'global_percentile'
) -> Tuple[pd.DataFrame, np.ndarray, Dict]:
    """
    Improved concatenation of hyperspectral data with proper coordinate handling.
    Only processes valid pixels and applies global normalization to spectral features.

    Args:
        data_dict: Dictionary containing hyperspectral data
        global_normalize: Whether to apply global normalization to spectral features
        normalization_method: 'global_percentile', 'global_minmax', 'global_standard', or 'none'

    Returns:
        Tuple of (concatenated_df, valid_mask, metadata)
    """
    print("Starting improved data concatenation...")

    # Extract data and metadata
    if 'data' in data_dict:
        excitation_data = data_dict['data']
    else:
        excitation_data = data_dict

    # Get excitation wavelengths
    if 'excitation_wavelengths' in data_dict:
        excitation_wavelengths = data_dict['excitation_wavelengths']
    else:
        excitation_wavelengths = sorted([float(k) for k in excitation_data.keys() if k != 'metadata'])

    print(f"Found {len(excitation_wavelengths)} excitation wavelengths: {excitation_wavelengths}")

    # Get spatial dimensions from first excitation
    first_ex = str(excitation_wavelengths[0])
    first_cube = excitation_data[first_ex]['cube']
    height, width = first_cube.shape[:2]

    print(f"Spatial dimensions: {height} x {width}")

    # Find valid (non-masked) pixels across all excitations
    print("Finding valid pixels...")
    valid_mask_flat = np.ones(height * width, dtype=bool)

    for ex in excitation_wavelengths:
        ex_str = str(ex)
        cube = excitation_data[ex_str]['cube']

        # Check for NaN or masked values
        cube_flat = cube.reshape(-1, cube.shape[-1])

        # A pixel is invalid if all emissions are NaN for this excitation
        invalid_pixels = np.all(np.isnan(cube_flat), axis=1)
        valid_mask_flat &= ~invalid_pixels

    n_valid = np.sum(valid_mask_flat)
    print(f"Found {n_valid} valid pixels out of {height * width} total ({100 * n_valid / (height * width):.1f}%)")

    # Get coordinates for valid pixels only (keep as integers)
    y_coords, x_coords = np.where(valid_mask_flat.reshape(height, width))

    print(f"Valid pixel coordinates range: x[{x_coords.min()}-{x_coords.max()}], y[{y_coords.min()}-{y_coords.max()}]")

    # Build spectral data for valid pixels only
    print("Extracting spectral data for valid pixels...")
    spectral_data_list = []
    column_names = []

    for ex in excitation_wavelengths:
        ex_str = str(ex)
        cube = excitation_data[ex_str]['cube']
        wavelengths = excitation_data[ex_str]['wavelengths']

        # Extract data only for valid pixels
        for i, em in enumerate(wavelengths):
            column_name = f"ex_{ex:.0f}_em_{em:.0f}"
            column_names.append(column_name)

            # Extract emission data for valid pixels
            emission_data = cube[y_coords, x_coords, i]
            # Replace any remaining NaNs with 0
            emission_data = np.nan_to_num(emission_data, 0)
            spectral_data_list.append(emission_data)

    # Stack all spectral data
    spectral_data = np.column_stack(spectral_data_list)
    print(f"Extracted spectral data shape: {spectral_data.shape}")

    # Apply global normalization to ALL spectral values if requested
    if global_normalize:
        print(f"Applying global normalization using method: {normalization_method}")

        if normalization_method == 'global_percentile':
            # Use percentiles to handle outliers
            p5 = np.percentile(spectral_data[spectral_data > 0], 5) if np.any(spectral_data > 0) else 0
            p95 = np.percentile(spectral_data[spectral_data > 0], 95) if np.any(spectral_data > 0) else 1
            spectral_data = np.clip((spectral_data - p5) / (p95 - p5 + 1e-10), 0, 1)
            print(f"  Used percentiles: P5={p5:.2f}, P95={p95:.2f}")

        elif normalization_method == 'global_minmax':
            # Global min-max scaling
            data_min = np.min(spectral_data)
            data_max = np.max(spectral_data)
            spectral_data = (spectral_data - data_min) / (data_max - data_min + 1e-10)
            print(f"  Global range: [{data_min:.2f}, {data_max:.2f}]")

        elif normalization_method == 'global_standard':
            # Global standardization
            data_mean = np.mean(spectral_data)
            data_std = np.std(spectral_data)
            spectral_data = (spectral_data - data_mean) / (data_std + 1e-10)
            print(f"  Global statistics: mean={data_mean:.2f}, std={data_std:.2f}")

        else:  # 'none'
            print("  No normalization applied")

    # Create DataFrame with integer coordinates and normalized spectral data
    df_data = {
        'x': x_coords.astype(int),  # Keep as integers
        'y': y_coords.astype(int),  # Keep as integers
    }

    # Add spectral columns
    for i, col_name in enumerate(column_names):
        df_data[col_name] = spectral_data[:, i]

    df = pd.DataFrame(df_data)

    print(f"Created DataFrame with shape: {df.shape}")
    print(f"  - {n_valid} valid pixels (rows)")
    print(f"  - {len(df.columns)} features (columns): 2 integer coordinates + {len(column_names)} spectral features")
    print(f"  - Coordinate types: x={df['x'].dtype}, y={df['y'].dtype}")
    print(f"  - Spectral data range: [{spectral_data.min():.4f}, {spectral_data.max():.4f}]")

    # Create metadata
    metadata = {
        'height': height,
        'width': width,
        'excitation_wavelengths': excitation_wavelengths,
        'n_valid_pixels': n_valid,
        'n_total_pixels': height * width,
        'global_normalized': global_normalize,
        'normalization_method': normalization_method if global_normalize else 'none'
    }

    # Store emission wavelengths per excitation
    metadata['emission_wavelengths'] = {}
    for ex in excitation_wavelengths:
        ex_str = str(ex)
        metadata['emission_wavelengths'][ex] = excitation_data[ex_str]['wavelengths']

    return df, valid_mask_flat.reshape(height, width), metadata


# Keep the old function for backward compatibility
def concatenate_hyperspectral_data(
    data_dict: Dict,
    normalize: bool = True,
    scale: bool = True
) -> Tuple[pd.DataFrame, np.ndarray, Dict]:
    """
    DEPRECATED: Use concatenate_hyperspectral_data_improved() instead.
    This function has issues with coordinate normalization and inefficient processing.
    """
    print("WARNING: Using deprecated function. Consider switching to concatenate_hyperspectral_data_improved()")
    return concatenate_hyperspectral_data_improved(
        data_dict,
        global_normalize=normalize,
        normalization_method='global_percentile'
    )


def perform_clustering(
    df: pd.DataFrame,
    n_clusters: int = 5,
    method: str = 'kmeans',
    use_pca: bool = False,
    n_components: int = 10,
    random_state: int = 42
) -> Tuple[np.ndarray, Dict]:
    """
    Perform clustering on concatenated hyperspectral data.
    
    Args:
        df: DataFrame with concatenated spectral features
        n_clusters: Number of clusters
        method: Clustering method ('kmeans' supported for now)
        use_pca: Whether to apply PCA before clustering
        n_components: Number of PCA components if use_pca is True
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (cluster_labels, metrics_dict)
    """
    print(f"\nPerforming {method} clustering with {n_clusters} clusters...")
    
    # Extract spectral features (exclude x, y coordinates)
    spectral_columns = [col for col in df.columns if col not in ['x', 'y']]
    X = df[spectral_columns].values
    
    print(f"Clustering on {X.shape[0]} samples with {X.shape[1]} features")
    
    # Apply PCA if requested
    if use_pca:
        print(f"Applying PCA with {n_components} components...")
        pca = PCA(n_components=n_components, random_state=random_state)
        X_transformed = pca.fit_transform(X)
        
        explained_var = np.sum(pca.explained_variance_ratio_)
        print(f"PCA explained variance: {explained_var:.2%}")
        
        X = X_transformed
    
    # Perform clustering
    if method.lower() == 'kmeans':
        print("Running KMeans...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        
        # Calculate inertia
        inertia = kmeans.inertia_
        print(f"KMeans inertia: {inertia:.2f}")
    else:
        raise ValueError(f"Unsupported clustering method: {method}")
    
    # Calculate clustering metrics
    print("Calculating clustering metrics...")
    metrics = {}
    
    # Silhouette score
    if len(np.unique(labels)) > 1:
        metrics['silhouette_score'] = silhouette_score(X, labels, sample_size=min(10000, len(labels)))
        print(f"Silhouette Score: {metrics['silhouette_score']:.4f}")
    
    # Davies-Bouldin score (lower is better)
    metrics['davies_bouldin_score'] = davies_bouldin_score(X, labels)
    print(f"Davies-Bouldin Score: {metrics['davies_bouldin_score']:.4f}")
    
    # Calinski-Harabasz score (higher is better)
    metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, labels)
    print(f"Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.2f}")
    
    if method.lower() == 'kmeans':
        metrics['inertia'] = inertia
        metrics['cluster_centers'] = kmeans.cluster_centers_
    
    # Count samples per cluster
    unique_labels, counts = np.unique(labels, return_counts=True)
    metrics['cluster_counts'] = dict(zip(unique_labels, counts))
    
    print("\nCluster distribution:")
    for label, count in metrics['cluster_counts'].items():
        percentage = 100 * count / len(labels)
        print(f"  Cluster {label}: {count} pixels ({percentage:.1f}%)")
    
    return labels, metrics


def reconstruct_cluster_map(
    cluster_labels: np.ndarray,
    df: pd.DataFrame,
    valid_mask: np.ndarray,
    metadata: Dict
) -> np.ndarray:
    """
    Reconstruct the 2D cluster map from cluster labels.
    
    Args:
        cluster_labels: 1D array of cluster labels for valid pixels
        df: DataFrame with pixel coordinates
        valid_mask: 2D boolean mask of valid pixels
        metadata: Metadata dictionary with spatial dimensions
        
    Returns:
        2D array with cluster labels (-1 for masked pixels)
    """
    height = metadata['height']
    width = metadata['width']
    
    # Initialize cluster map with -1 (masked pixels)
    cluster_map = np.full((height, width), -1, dtype=int)
    
    # Fill in cluster labels for valid pixels
    x_coords = df['x'].values.astype(int)
    y_coords = df['y'].values.astype(int)
    
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        cluster_map[y, x] = cluster_labels[i]
    
    return cluster_map


def visualize_clustering_results(
    cluster_map: np.ndarray,
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    metadata: Dict,
    metrics: Dict,
    save_path: Optional[Path] = None
):
    """
    Create comprehensive visualization of clustering results.
    
    Args:
        cluster_map: 2D array with cluster labels
        df: DataFrame with concatenated data
        cluster_labels: 1D array of cluster labels
        metadata: Metadata dictionary
        metrics: Clustering metrics dictionary
        save_path: Optional path to save the figure
    """
    n_clusters = len(np.unique(cluster_labels[cluster_labels >= 0]))
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Cluster map
    ax1 = plt.subplot(2, 3, 1)
    im = ax1.imshow(cluster_map, cmap='tab10', interpolation='nearest')
    ax1.set_title('Cluster Map', fontsize=14, fontweight='bold')
    ax1.axis('off')
    plt.colorbar(im, ax=ax1, label='Cluster ID')
    
    # 2. Cluster size distribution
    ax2 = plt.subplot(2, 3, 2)
    cluster_counts = metrics['cluster_counts']
    bars = ax2.bar(range(len(cluster_counts)), list(cluster_counts.values()))
    ax2.set_xlabel('Cluster ID')
    ax2.set_ylabel('Number of Pixels')
    ax2.set_title('Cluster Size Distribution', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(cluster_counts)))
    ax2.set_xticklabels([f'C{i}' for i in range(len(cluster_counts))])
    
    # Add percentage labels on bars
    total_pixels = sum(cluster_counts.values())
    for i, (bar, count) in enumerate(zip(bars, cluster_counts.values())):
        percentage = 100 * count / total_pixels
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * max(cluster_counts.values()),
                f'{percentage:.1f}%', ha='center', va='bottom')
    
    # 3. Mean spectral profiles per cluster
    ax3 = plt.subplot(2, 3, 3)
    spectral_columns = [col for col in df.columns if col not in ['x', 'y']]
    
    # Sample spectral columns for visualization (every 5th)
    sample_indices = np.arange(0, len(spectral_columns), 5)
    sampled_columns = [spectral_columns[i] for i in sample_indices]
    
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        mean_spectrum = df.loc[mask, sampled_columns].mean().values
        ax3.plot(mean_spectrum, label=f'Cluster {cluster_id}', linewidth=2)
    
    ax3.set_xlabel('Spectral Feature Index')
    ax3.set_ylabel('Mean Intensity')
    ax3.set_title('Mean Spectral Profiles (Sampled)', fontsize=14, fontweight='bold')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # 4. Clustering metrics
    ax4 = plt.subplot(2, 3, 4)
    ax4.axis('off')
    metrics_text = f"Clustering Metrics (K={n_clusters}):\n\n"
    if 'silhouette_score' in metrics:
        metrics_text += f"Silhouette Score: {metrics['silhouette_score']:.4f}\n"
    metrics_text += f"Davies-Bouldin Score: {metrics['davies_bouldin_score']:.4f}\n"
    metrics_text += f"Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.1f}\n"
    if 'inertia' in metrics:
        metrics_text += f"KMeans Inertia: {metrics['inertia']:.2f}\n"
    
    ax4.text(0.1, 0.5, metrics_text, fontsize=12, transform=ax4.transAxes,
            verticalalignment='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 5. Spatial coherence visualization
    ax5 = plt.subplot(2, 3, 5)
    
    # Calculate spatial coherence (neighboring pixels in same cluster)
    coherence_map = np.zeros_like(cluster_map, dtype=float)
    for i in range(1, cluster_map.shape[0] - 1):
        for j in range(1, cluster_map.shape[1] - 1):
            if cluster_map[i, j] >= 0:
                center_cluster = cluster_map[i, j]
                neighbors = [
                    cluster_map[i-1, j], cluster_map[i+1, j],
                    cluster_map[i, j-1], cluster_map[i, j+1]
                ]
                same_cluster_count = sum(1 for n in neighbors if n == center_cluster)
                coherence_map[i, j] = same_cluster_count / 4
    
    im2 = ax5.imshow(coherence_map, cmap='RdYlGn', vmin=0, vmax=1)
    ax5.set_title('Spatial Coherence Map', fontsize=14, fontweight='bold')
    ax5.axis('off')
    plt.colorbar(im2, ax=ax5, label='Coherence')
    
    # 6. Excitation-Emission heatmap for selected cluster
    ax6 = plt.subplot(2, 3, 6)
    
    # Select the largest cluster for detailed view
    largest_cluster = max(cluster_counts.keys(), key=lambda k: cluster_counts[k])
    mask = cluster_labels == largest_cluster
    
    # Create a matrix of mean intensities per excitation-emission pair
    excitation_wavelengths = metadata['excitation_wavelengths']
    n_ex = len(excitation_wavelengths)
    
    # Get max number of emissions
    max_em = max(len(metadata['emission_wavelengths'][ex]) for ex in excitation_wavelengths)
    
    heatmap_data = np.full((n_ex, max_em), np.nan)
    
    for i, ex in enumerate(excitation_wavelengths):
        emissions = metadata['emission_wavelengths'][ex]
        for j, em in enumerate(emissions):
            col_name = f"ex_{ex:.0f}_em_{em:.0f}"
            if col_name in df.columns:
                heatmap_data[i, j] = df.loc[mask, col_name].mean()
    
    im3 = ax6.imshow(heatmap_data, aspect='auto', cmap='viridis')
    ax6.set_xlabel('Emission Index')
    ax6.set_ylabel('Excitation Wavelength (nm)')
    ax6.set_title(f'Mean Intensity Matrix - Cluster {largest_cluster}', fontsize=14, fontweight='bold')
    ax6.set_yticks(range(n_ex))
    ax6.set_yticklabels([f'{ex:.0f}' for ex in excitation_wavelengths])
    plt.colorbar(im3, ax=ax6, label='Mean Intensity')
    
    plt.suptitle('Concatenation-Based Clustering Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()


def compare_clustering_k_values(
    df: pd.DataFrame,
    valid_mask: np.ndarray,
    metadata: Dict,
    k_values: List[int] = [3, 4, 5, 6, 7, 8],
    ground_truth: Optional[np.ndarray] = None,
    save_dir: Optional[Path] = None
) -> Dict:
    """
    Compare clustering results for different k values.
    
    Args:
        df: DataFrame with concatenated data
        valid_mask: Valid pixel mask
        metadata: Metadata dictionary
        k_values: List of k values to test
        ground_truth: Optional ground truth labels for validation
        save_dir: Directory to save results
        
    Returns:
        Dictionary with results for each k value
    """
    results = {}
    
    # Create figure for comparison
    fig, axes = plt.subplots(2, len(k_values), figsize=(4*len(k_values), 8))
    
    # Store metrics for comparison plot
    silhouette_scores = []
    davies_bouldin_scores = []
    calinski_harabasz_scores = []
    ground_truth_scores = [] if ground_truth is not None else None
    
    for idx, k in enumerate(k_values):
        print(f"\n{'='*50}")
        print(f"Testing k = {k}")
        print('='*50)
        
        # Perform clustering
        labels, metrics = perform_clustering(df, n_clusters=k)
        
        # Reconstruct cluster map
        cluster_map = reconstruct_cluster_map(labels, df, valid_mask, metadata)
        
        # Store results
        results[k] = {
            'labels': labels,
            'metrics': metrics,
            'cluster_map': cluster_map
        }
        
        # Calculate ground truth accuracy if available
        if ground_truth is not None:
            gt_metrics = calculate_clustering_accuracy(cluster_map, ground_truth, valid_mask)
            results[k]['ground_truth_metrics'] = gt_metrics
            ground_truth_scores.append(gt_metrics.get('purity', 0))
            print(f"Ground Truth Purity: {gt_metrics.get('purity', 0):.4f}")
            print(f"Ground Truth ARI: {gt_metrics.get('adjusted_rand_score', 0):.4f}")
        
        # Store metrics for comparison
        if 'silhouette_score' in metrics:
            silhouette_scores.append(metrics['silhouette_score'])
        else:
            silhouette_scores.append(0)
        davies_bouldin_scores.append(metrics['davies_bouldin_score'])
        calinski_harabasz_scores.append(metrics['calinski_harabasz_score'])
        
        # Plot cluster map
        ax1 = axes[0, idx]
        im = ax1.imshow(cluster_map, cmap='tab10')
        ax1.set_title(f'k = {k}')
        ax1.axis('off')
        
        # Plot cluster sizes
        ax2 = axes[1, idx]
        cluster_counts = list(metrics['cluster_counts'].values())
        ax2.bar(range(len(cluster_counts)), cluster_counts)
        ax2.set_xlabel('Cluster')
        ax2.set_ylabel('Count')
        ax2.set_title(f'Distribution (k={k})')
    
    plt.suptitle('Clustering Comparison - Different K Values', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_dir:
        save_path = save_dir / 'k_comparison_maps.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison maps to {save_path}")
    
    plt.show()
    
    # Create metrics comparison plot
    n_plots = 4 if ground_truth is not None else 3
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
    
    # Silhouette score
    axes[0].plot(k_values, silhouette_scores, 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of Clusters (k)')
    axes[0].set_ylabel('Silhouette Score')
    axes[0].set_title('Silhouette Score vs K (Higher is Better)')
    axes[0].grid(True, alpha=0.3)
    
    # Davies-Bouldin score
    axes[1].plot(k_values, davies_bouldin_scores, 'o-', linewidth=2, markersize=8, color='orange')
    axes[1].set_xlabel('Number of Clusters (k)')
    axes[1].set_ylabel('Davies-Bouldin Score')
    axes[1].set_title('Davies-Bouldin Score vs K (Lower is Better)')
    axes[1].grid(True, alpha=0.3)
    
    # Calinski-Harabasz score
    axes[2].plot(k_values, calinski_harabasz_scores, 'o-', linewidth=2, markersize=8, color='green')
    axes[2].set_xlabel('Number of Clusters (k)')
    axes[2].set_ylabel('Calinski-Harabasz Score')
    axes[2].set_title('Calinski-Harabasz Score vs K (Higher is Better)')
    axes[2].grid(True, alpha=0.3)
    
    # Ground truth purity if available
    if ground_truth is not None and ground_truth_scores:
        axes[3].plot(k_values, ground_truth_scores, 'o-', linewidth=2, markersize=8, color='purple')
        axes[3].set_xlabel('Number of Clusters (k)')
        axes[3].set_ylabel('Purity Score')
        axes[3].set_title('Ground Truth Purity vs K (Higher is Better)')
        axes[3].grid(True, alpha=0.3)
        
        # Mark the best k based on purity
        best_k_idx = np.argmax(ground_truth_scores)
        best_k = k_values[best_k_idx]
        best_purity = ground_truth_scores[best_k_idx]
        axes[3].plot(best_k, best_purity, 'r*', markersize=15, label=f'Best k={best_k}')
        axes[3].legend()
    
    plt.suptitle('Clustering Metrics Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_dir:
        save_path = save_dir / 'k_comparison_metrics.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved metrics comparison to {save_path}")
    
    plt.show()
    
    # Find optimal k based on silhouette score
    if silhouette_scores:
        optimal_k = k_values[np.argmax(silhouette_scores)]
        print(f"\nOptimal k based on Silhouette Score: {optimal_k}")
    
    return results


def main():
    """
    Main function to run the concatenation-based clustering pipeline.
    """
    # Set up paths
    base_dir = Path(r"C:\Users\meloy\PycharmProjects\Capstone")
    data_path = base_dir / "data" / "processed" / "Lichens" / "lichens_data_masked.pkl"
    output_dir = base_dir / "wavelength_analysis" / "concatenation_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Concatenation-Based Clustering Pipeline")
    print("="*60)
    
    # Load data
    print("\n1. Loading masked data...")
    data_dict = load_masked_data(data_path)
    
    # Concatenate data
    print("\n2. Concatenating hyperspectral data...")
    df, valid_mask, metadata = concatenate_hyperspectral_data(
        data_dict,
        normalize=True,
        scale=True
    )
    
    # Save concatenated data
    concat_path = output_dir / "concatenated_data.csv"
    print(f"\n3. Saving concatenated data to {concat_path}...")
    df.to_csv(concat_path, index=False)
    
    # Perform clustering with default k
    default_k = 5
    print(f"\n4. Performing clustering with k={default_k}...")
    labels, metrics = perform_clustering(
        df,
        n_clusters=default_k,
        method='kmeans',
        use_pca=False
    )
    
    # Reconstruct cluster map
    print("\n5. Reconstructing cluster map...")
    cluster_map = reconstruct_cluster_map(labels, df, valid_mask, metadata)
    
    # Visualize results
    print("\n6. Creating visualizations...")
    vis_path = output_dir / f"clustering_results_k{default_k}.png"
    visualize_clustering_results(
        cluster_map,
        df,
        labels,
        metadata,
        metrics,
        save_path=vis_path
    )
    
    # Compare different k values
    print("\n7. Comparing different k values...")
    k_values = [3, 4, 5, 6, 7, 8, 9, 10]
    comparison_results = compare_clustering_k_values(
        df,
        valid_mask,
        metadata,
        k_values=k_values,
        save_dir=output_dir
    )
    
    # Save results
    print("\n8. Saving results...")
    results_dict = {
        'default_k': default_k,
        'default_labels': labels,
        'default_metrics': metrics,
        'default_cluster_map': cluster_map,
        'comparison_results': {k: {
            'metrics': v['metrics'],
            'cluster_map': v['cluster_map'].tolist()
        } for k, v in comparison_results.items()},
        'metadata': metadata
    }
    
    results_path = output_dir / "clustering_results.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(results_dict, f)
    
    print(f"\nResults saved to {results_path}")
    
    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    print("="*60)
    
    return df, cluster_map, metrics


if __name__ == "__main__":
    df, cluster_map, metrics = main()