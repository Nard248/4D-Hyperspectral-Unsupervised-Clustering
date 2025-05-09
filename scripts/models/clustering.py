"""
Clustering module for hyperspectral data.

This module provides functions for extracting features and
performing optimized clustering on hyperspectral data.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

from .training import create_spatial_chunks

def extract_encoded_features(
        model,
        data_dict,
        mask=None,
        chunk_size=64,
        chunk_overlap=8,
        device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Extract encoded features efficiently using chunking.
    """
    # Ensure model is in eval mode
    model.eval()
    model = model.to(device)

    # Get spatial dimensions from first excitation
    first_ex = next(iter(data_dict.values()))
    height, width = first_ex.shape[0], first_ex.shape[1]

    # Store features and shapes
    encoded_features = {}
    spatial_shapes = {}

    # Extract features for each excitation separately
    with torch.no_grad():
        for ex, data in data_dict.items():
            print(f"Extracting features for excitation {ex}...")

            # Create chunks for this excitation
            chunks_result = create_spatial_chunks(
                data.numpy(),
                mask=mask,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

            # Extract chunks and positions from the result
            chunks = chunks_result[0]  # First item is always chunks
            positions = chunks_result[1]  # Second item is always positions

            # Create feature map to store the results
            all_features = None
            all_positions = []

            # Process chunks
            for i, chunk in enumerate(chunks):
                # Convert to tensor and add batch dimension
                chunk_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(device)

                # Create input dictionary for this excitation only
                chunk_dict = {ex: chunk_tensor}

                # Extract encoded representation
                encoded = model.encode(chunk_dict)

                # Convert to numpy and store
                features = encoded.cpu().numpy()[0]  # Remove batch dimension

                # If this is the first chunk, initialize the feature array
                if all_features is None:
                    all_features = []
                    for feat_idx in range(features.shape[0]):
                        # Create empty feature map for each feature channel
                        all_features.append(np.zeros((height, width)))

                # Store features and positions
                y_start, y_end, x_start, x_end = positions[i]
                all_positions.append((y_start, y_end, x_start, x_end))

                # Extract spatial features (remove channel and emission dimensions)
                spatial_features = features.squeeze(1)  # Remove emission dimension which is 1

                # Store features in the appropriate positions
                for feat_idx in range(spatial_features.shape[0]):
                    # Handle overlapping regions by taking the mean
                    feature_chunk = spatial_features[feat_idx]
                    current = all_features[feat_idx][y_start:y_end, x_start:x_end]
                    overlap_mask = current != 0  # Areas where we already have features

                    # Set new areas directly
                    new_areas = ~overlap_mask
                    current[new_areas] = feature_chunk[new_areas]

                    # Average overlapping areas
                    if np.any(overlap_mask):
                        current[overlap_mask] = (current[overlap_mask] + feature_chunk[overlap_mask]) / 2

                    all_features[feat_idx][y_start:y_end, x_start:x_end] = current

                # Print progress
                if (i + 1) % 10 == 0 or i == len(chunks) - 1:
                    print(f"  Processed chunk {i + 1}/{len(chunks)}", end="\r")

            # Stack features into a single array and reshape
            features_array = np.stack(all_features)

            # Store features and shapes
            encoded_features[ex] = features_array
            spatial_shapes[ex] = (height, width)

            print(f"Extracted {features_array.shape[0]} features for excitation {ex}")

    return encoded_features, spatial_shapes


def optimize_kmeans_clustering(
        features,
        n_clusters=5,
        max_samples=10000,
        random_state=42,
        use_pca=True,
        n_init=10,
        max_iter=300
):
    """
    Apply K-means clustering with optimizations for large datasets.

    Args:
        features: Feature array of shape [n_features, height, width]
                or [n_samples, n_features]
        n_clusters: Number of clusters
        max_samples: Maximum number of samples to use for initial clustering
        random_state: Random seed for reproducibility
        use_pca: Whether to use PCA for dimensionality reduction
        n_init: Number of initializations to try
        max_iter: Maximum iterations for K-means

    Returns:
        Cluster labels and clustering model
    """
    print(f"Starting optimized K-means clustering with {n_clusters} clusters...")

    # Check if features need reshaping
    if len(features.shape) == 3:
        # Reshape from [n_features, height, width] to [height*width, n_features]
        n_features, height, width = features.shape
        features_reshaped = features.reshape(n_features, -1).T
    else:
        # Already in shape [n_samples, n_features]
        features_reshaped = features

    n_samples, n_features = features_reshaped.shape
    print(f"Feature matrix shape: {features_reshaped.shape}")

    # Apply PCA if requested and the number of features is large
    if use_pca and n_features > 50:
        print(f"Applying PCA to reduce dimensions from {n_features} to 50...")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(50, n_features - 1), random_state=random_state)
        features_reshaped = pca.fit_transform(features_reshaped)
        print(f"PCA reduced feature shape: {features_reshaped.shape}")
        print(f"Explained variance: {sum(pca.explained_variance_ratio_):.2f}")
    else:
        if n_features > 50:
            print(f"Keeping all {n_features} dimensions (PCA disabled)")

    # Scale data after PCA to ensure good clustering
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_reshaped)
    print(f"Scaled features - mean: {scaler.mean_.mean()}, std: {scaler.scale_.mean()}")

    # Print stats on scaled features
    print(f"Feature range after scaling: {features_scaled.min():.4f} to {features_scaled.max():.4f}")

    # Check for outliers and extreme values
    from scipy import stats
    z_scores = np.abs(stats.zscore(features_scaled))
    outliers_percent = (z_scores > 3).mean() * 100
    print(f"Outliers (|z-score| > 3): {outliers_percent:.2f}% of data points")

    # Use MiniBatchKMeans for large datasets
    if n_samples > max_samples:
        print(f"Using MiniBatchKMeans for {n_samples} samples (more than {max_samples})...")
        # Sample a subset for initial clustering
        from sklearn.cluster import MiniBatchKMeans
        indices = np.random.RandomState(random_state).choice(n_samples, max_samples, replace=False)
        sample_features = features_scaled[indices]

        # Fit MiniBatchKMeans on the sample
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=min(1000, max_samples),
            max_iter=max_iter,
            random_state=random_state,
            n_init=n_init
        )
        kmeans.fit(sample_features)

        # Now predict cluster labels for all data
        print(f"Predicting cluster labels for all {n_samples} samples...")
        labels = kmeans.predict(features_scaled)
    else:
        print(f"Using standard KMeans for {n_samples} samples...")
        from sklearn.cluster import KMeans
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=n_init,
            max_iter=max_iter
        )
        labels = kmeans.fit_predict(features_scaled)

    # Check if clustering worked properly
    unique_labels = np.unique(labels)
    print(f"Found {len(unique_labels)} unique clusters: {unique_labels}")
    cluster_sizes = np.bincount(labels)
    print(f"Cluster sizes: {cluster_sizes}")

    if len(unique_labels) < n_clusters:
        print(f"WARNING: Expected {n_clusters} clusters, but only found {len(unique_labels)}.")
        print("Trying again with different parameters...")

        # Try again with more aggressive approach
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state + 1,  # Different seed
            n_init=20,  # More initializations
            max_iter=500  # More iterations
        )
        labels = kmeans.fit_predict(features_scaled)

        # Check results again
        unique_labels = np.unique(labels)
        print(f"After retry: Found {len(unique_labels)} unique clusters: {unique_labels}")

    # Reshape labels if needed
    if len(features.shape) == 3:
        _, height, width = features.shape
        labels_reshaped = labels.reshape(height, width)
        print(f"Reshaped labels to {labels_reshaped.shape}")
        return labels_reshaped, kmeans
    else:
        return labels, kmeans

def run_pixel_wise_clustering(
        model,
        dataset,
        n_clusters=5,
        excitation_to_use=None,
        chunk_size=64,
        chunk_overlap=8,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        output_dir=None,
        calculate_metrics=True
):
    """
    Run efficient pixel-wise clustering on hyperspectral data.

    Args:
        model: Trained HyperspectralCAEWithMasking model
        dataset: MaskedHyperspectralDataset instance
        n_clusters: Number of clusters for K-means
        excitation_to_use: Specific excitation wavelength to use (if None, uses the first one)
        chunk_size: Size of spatial chunks for processing
        chunk_overlap: Overlap between adjacent chunks
        device: Device to use for computation
        output_dir: Directory to save clustering results
        calculate_metrics: Whether to calculate clustering quality metrics

    Returns:
        Dictionary with clustering results
    """
    print("Starting pixel-wise clustering...")

    # Create output directory if needed
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    # Get data
    all_data = dataset.get_all_data()

    # Use mask if available
    mask = dataset.processed_mask if hasattr(dataset, 'processed_mask') else None

    # Set model to evaluation mode
    model.eval()
    model = model.to(device)

    # Step 1: Extract encoded features
    print("Extracting encoded features...")
    encoded_features, spatial_shapes = extract_encoded_features(
        model=model,
        data_dict=all_data,
        mask=mask,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        device=device
    )

    # Determine which excitation to use for clustering
    if excitation_to_use is None:
        excitation_to_use = next(iter(encoded_features.keys()))

    if excitation_to_use not in encoded_features:
        raise ValueError(f"Excitation {excitation_to_use} not found in encoded features")

    print(f"Using excitation {excitation_to_use} for clustering")

    # Get features for the selected excitation
    features = encoded_features[excitation_to_use]

    # Step 2: Apply optimized K-means clustering
    print(f"Applying K-means clustering with {n_clusters} clusters...")
    cluster_labels, clustering_model = optimize_kmeans_clustering(
        features=features,
        n_clusters=n_clusters
    )

    # Mask the cluster labels if a mask is provided
    if mask is not None:
        # Ensure mask has the same shape as cluster labels
        if mask.shape != cluster_labels.shape:
            from scipy.ndimage import zoom
            mask_resized = zoom(mask,
                                (cluster_labels.shape[0] / mask.shape[0],
                                 cluster_labels.shape[1] / mask.shape[1]),
                                order=0)
            # Ensure binary mask
            mask_resized = (mask_resized > 0.5).astype(np.uint8)
        else:
            mask_resized = mask

        # Apply mask - set masked areas to -1 (typically used for "no cluster")
        cluster_labels[mask_resized == 0] = -1

    # Create cluster map visualization
    plt.figure(figsize=(10, 8))
    plt.imshow(cluster_labels, cmap='tab10', interpolation='nearest')
    plt.colorbar(label='Cluster ID')
    plt.title(f'Pixel-wise Clustering (Ex={excitation_to_use}nm, K={n_clusters})')
    plt.axis('off')

    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, f"cluster_map_ex{excitation_to_use}_k{n_clusters}.png"),
                    dpi=300, bbox_inches='tight')

        # Save cluster labels
        np.save(os.path.join(output_dir, f"cluster_labels_ex{excitation_to_use}_k{n_clusters}.npy"),
                cluster_labels)

    plt.close()

    print(f"Clustering complete. Found {len(np.unique(cluster_labels))} unique clusters")

    # Calculate clustering metrics if requested
    metrics = None
    if calculate_metrics:
        print("\nEvaluating clustering quality...")
        metrics = evaluate_clustering_quality(
            features=features,
            cluster_labels=cluster_labels,
            original_data=all_data,
            mask=mask,
            output_dir=output_dir
        )

        # Save metrics to file
        if output_dir is not None:
            metrics_file = os.path.join(output_dir, f"clustering_metrics_ex{excitation_to_use}_k{n_clusters}.json")
            import json
            with open(metrics_file, 'w') as f:
                # Convert numpy values to Python types
                metrics_dict = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                                for k, v in metrics.items()}
                json.dump(metrics_dict, f, indent=2)
            print(f"Clustering metrics saved to: {metrics_file}")

    # Return results
    results = {
        'cluster_labels': cluster_labels,
        'clustering_model': clustering_model,
        'excitation_used': excitation_to_use,
        'encoded_features': encoded_features,
        'spatial_shapes': spatial_shapes,
        'n_clusters': n_clusters
    }

    if metrics is not None:
        results['metrics'] = metrics

    return results


def visualize_cluster_profiles(
        cluster_results,
        dataset,
        original_data=None,
        output_dir=None
):
    """
    Visualize the spectral profiles of each cluster.

    Args:
        cluster_results: Results from run_pixel_wise_clustering
        dataset: MaskedHyperspectralDataset instance
        original_data: Optional dictionary with original data
        output_dir: Directory to save visualizations

    Returns:
        Dictionary with cluster profile statistics
    """
    # Get cluster labels and excitation
    cluster_labels = cluster_results['cluster_labels']
    excitation_used = cluster_results['excitation_used']

    # Get original data if not provided
    if original_data is None:
        original_data = dataset.get_all_data()

    # Get emission wavelengths
    emission_wavelengths = {}
    if hasattr(dataset, 'emission_wavelengths'):
        emission_wavelengths = dataset.emission_wavelengths

    # Get unique cluster IDs (excluding -1 which is used for masked areas)
    unique_clusters = sorted([c for c in np.unique(cluster_labels) if c >= 0])
    print(f"Analyzing profiles for {len(unique_clusters)} clusters")

    # Create a figure for spectral profiles
    plt.figure(figsize=(12, 8))

    # Store cluster statistics
    cluster_stats = {}

    # Get first excitation to determine marker styles
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    # Process each excitation
    for i, ex in enumerate(original_data.keys()):
        # Get data for this excitation
        data = original_data[ex].cpu().numpy()

        # Get emission wavelengths for x-axis
        if ex in emission_wavelengths:
            x_values = emission_wavelengths[ex]
        else:
            x_values = np.arange(data.shape[2])

        # Use a different marker for each excitation
        marker = markers[i % len(markers)]

        # Calculate mean spectrum for each cluster
        for cluster_id in unique_clusters:
            # Create mask for this cluster
            cluster_mask = cluster_labels == cluster_id

            # Skip if no pixels in this cluster
            if not np.any(cluster_mask):
                continue

            # Get data for this cluster
            cluster_data = data[cluster_mask]

            # Calculate mean spectrum (ignoring NaNs)
            mean_spectrum = np.nanmean(cluster_data, axis=0)

            # Calculate standard deviation (ignoring NaNs)
            std_spectrum = np.nanstd(cluster_data, axis=0)

            # Store statistics
            if cluster_id not in cluster_stats:
                cluster_stats[cluster_id] = {}

            cluster_stats[cluster_id][ex] = {
                'mean': mean_spectrum,
                'std': std_spectrum,
                'count': np.sum(cluster_mask)
            }

            # Plot mean spectrum
            label = f"Cluster {cluster_id}, Ex={ex}nm" if i == 0 else f"Ex={ex}nm"
            plt.plot(x_values, mean_spectrum, marker=marker, label=label if i == 0 else None,
                     color=plt.cm.tab10(cluster_id % 10))

            # Plot error bars for first few excitations to avoid clutter
            if i < 3:  # Limit to first 3 excitations
                plt.fill_between(
                    x_values,
                    mean_spectrum - std_spectrum,
                    mean_spectrum + std_spectrum,
                    alpha=0.2,
                    color=plt.cm.tab10(cluster_id % 10)
                )

    # Add legend for clusters
    plt.legend(loc='best')
    plt.xlabel('Emission Wavelength (nm)' if len(emission_wavelengths) > 0 else 'Emission Band Index')
    plt.ylabel('Intensity')
    plt.title(f'Spectral Profiles by Cluster (Excitation {excitation_used}nm)')
    plt.grid(True, alpha=0.3)

    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, f"cluster_profiles_ex{excitation_used}.png"),
                    dpi=300, bbox_inches='tight')

    plt.close()

    # Create a bar chart of cluster sizes
    plt.figure(figsize=(10, 6))
    cluster_sizes = [np.sum(cluster_labels == c) for c in unique_clusters]

    plt.bar(
        [f"Cluster {c}" for c in unique_clusters],
        cluster_sizes,
        color=[plt.cm.tab10(c % 10) for c in unique_clusters]
    )

    plt.xlabel('Cluster')
    plt.ylabel('Number of Pixels')
    plt.title(f'Cluster Sizes (Excitation {excitation_used}nm)')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', alpha=0.3)

    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, f"cluster_sizes_ex{excitation_used}.png"),
                    dpi=300, bbox_inches='tight')

    plt.close()

    return cluster_stats


# Add this inside the scripts/models/clustering.py file
# Modify the evaluate_clustering_quality function as follows:

def evaluate_clustering_quality(
        features,
        cluster_labels,
        original_data=None,
        mask=None,
        distance_metric='euclidean',
        output_dir=None
):
    """
    Calculate multiple metrics to evaluate clustering quality.

    Args:
        features: Feature array used for clustering
        cluster_labels: Generated cluster labels
        original_data: Optional original hyperspectral data
        mask: Optional binary mask (1=valid, 0=masked)
        distance_metric: Distance metric for silhouette calculation
        output_dir: Directory to save visualizations

    Returns:
        Dictionary of evaluation metrics
    """
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    print("Calculating clustering quality metrics...")
    print(f"Input features shape: {features.shape}")
    print(f"Input labels shape: {cluster_labels.shape}")

    # Initialize results
    metrics = {}

    # Deep inspection of input data
    if mask is not None:
        print(f"Mask shape: {mask.shape}, Valid pixels: {np.sum(mask)}")

    # Save a visualization of the raw cluster labels for debugging
    if output_dir is not None:
        plt.figure(figsize=(10, 8))
        plt.imshow(cluster_labels, cmap='tab10')
        plt.colorbar(label='Raw Cluster Labels')
        plt.title('Raw Cluster Labels (Before Evaluation)')
        plt.savefig(os.path.join(output_dir, "raw_cluster_labels.png"), dpi=300)
        plt.close()

    # Reshape features if needed
    if len(features.shape) == 3:  # [n_features, height, width]
        n_features, height, width = features.shape
        features_reshaped = features.reshape(n_features, -1).T
        print(f"Reshaped features from {features.shape} to {features_reshaped.shape}")
    else:
        features_reshaped = features
        print(f"Using original feature shape: {features_reshaped.shape}")

    if len(cluster_labels.shape) == 2:  # [height, width]
        labels_flat = cluster_labels.flatten()
    else:
        labels_flat = cluster_labels

    # Basic statistics before any filtering
    unique_before = np.unique(labels_flat)
    print(f"Original unique labels: {unique_before}")
    print(f"Number of unique labels before filtering: {len(unique_before)}")

    # Check if we have non-consecutive cluster IDs
    valid_ids = [x for x in unique_before if x >= 0]
    consecutive_ids = list(range(len(valid_ids)))
    has_gaps = set(valid_ids) != set(consecutive_ids)
    print(f"Valid cluster IDs: {valid_ids}")
    print(f"Has gaps in cluster IDs: {has_gaps}")

    # Apply mask if provided
    if mask is not None:
        if isinstance(mask, np.ndarray):
            mask_flat = mask.flatten() if len(mask.shape) == 2 else mask
            valid_indices = np.where(mask_flat > 0)[0]

            # Ensure dimensions match before applying mask
            if features_reshaped.shape[0] == len(mask_flat):
                features_reshaped = features_reshaped[valid_indices]
                print(f"Applied mask to features: {features_reshaped.shape}")
            else:
                print(f"WARNING: Feature shape {features_reshaped.shape[0]} != mask shape {len(mask_flat)}")
                print("Using unmasked features")

            # Apply mask to labels
            if len(labels_flat) == len(mask_flat):
                labels_flat = labels_flat[valid_indices]
                print(f"Applied mask to labels: {len(labels_flat)} remaining")
            else:
                print(f"WARNING: Labels shape {len(labels_flat)} != mask shape {len(mask_flat)}")
                print("Using unmasked labels")

    # Select only valid labels (>= 0) and corresponding features
    valid = labels_flat >= 0
    if np.any(valid):  # Only proceed if we have valid labels
        features_valid = features_reshaped[valid]
        labels_valid = labels_flat[valid]

        print(f"Before filtering - Unique labels: {np.unique(labels_flat)}")
        print(f"After filtering - Unique labels: {np.unique(labels_valid)}")
        print(f"Number of valid samples: {len(labels_valid)}")

        # Important: Check if labels are continuous starting from 0
        # Many clustering metrics require labels to be 0, 1, 2, ... not arbitrary values
        unique_valid_labels = np.unique(labels_valid)
        if len(unique_valid_labels) > 1 and not np.array_equal(unique_valid_labels,
                                                               np.arange(len(unique_valid_labels))):
            print("WARNING: Labels are not continuous from 0. Remapping labels for metrics calculation.")
            # Create a mapping from original labels to 0-based consecutive integers
            label_mapping = {label: i for i, label in enumerate(unique_valid_labels)}
            remapped_labels = np.array([label_mapping[label] for label in labels_valid])
            print(f"Remapped labels: {np.unique(remapped_labels)}")
            labels_valid = remapped_labels
    else:
        print("WARNING: No valid labels (>= 0) found after masking.")
        features_valid = np.array([])
        labels_valid = np.array([])

    # Check if we have enough clusters and samples for evaluation
    unique_clusters = np.unique(labels_valid)
    n_clusters = len(unique_clusters)
    n_samples = len(labels_valid)

    print(f"Final unique clusters for metric calculation: {unique_clusters}")
    print(f"Number of clusters: {n_clusters}")
    print(f"Number of samples: {n_samples}")

    # Store basic information in metrics
    metrics['n_clusters'] = n_clusters
    metrics['n_samples'] = n_samples
    metrics['unique_clusters'] = list(map(int, unique_clusters.tolist())) if len(unique_clusters) > 0 else []

    # Skip metric calculation if only one cluster or not enough samples
    if n_clusters <= 1:
        print(f"Warning: Only {n_clusters} cluster found after processing. Cannot calculate clustering metrics.")
        # Save a visualization of the filtered labels for debugging
        if output_dir is not None and len(cluster_labels.shape) == 2:
            plt.figure(figsize=(10, 8))
            plt.imshow(cluster_labels, cmap='tab10')
            plt.colorbar(label='Final Cluster Labels')
            plt.title('Final Cluster Labels (After Filtering)')
            plt.savefig(os.path.join(output_dir, "filtered_cluster_labels.png"), dpi=300)
            plt.close()
        return metrics

    if n_samples < n_clusters * 2:
        print(f"Warning: Not enough samples ({n_samples}) for {n_clusters} clusters. Need at least {n_clusters * 2}.")
        return metrics

    try:
        # Calculate internal metrics
        metrics['silhouette_score'] = silhouette_score(features_valid, labels_valid, metric=distance_metric)
        metrics['davies_bouldin_score'] = davies_bouldin_score(features_valid, labels_valid)
        metrics['calinski_harabasz_score'] = calinski_harabasz_score(features_valid, labels_valid)

        print(f"Silhouette Score: {metrics['silhouette_score']:.4f} (higher is better)")
        print(f"Davies-Bouldin Index: {metrics['davies_bouldin_score']:.4f} (lower is better)")
        print(f"Calinski-Harabasz Index: {metrics['calinski_harabasz_score']:.4f} (higher is better)")
    except Exception as e:
        print(f"Error calculating internal metrics: {str(e)}")
        import traceback
        traceback.print_exc()

    # Calculate spatial coherence if we have 2D labels
    if len(cluster_labels.shape) == 2:
        try:
            # Create a neighbor count matrix (how many neighbors have same label)
            from scipy import ndimage
            spatial_coherence = np.zeros_like(cluster_labels, dtype=float)

            for cluster_id in unique_clusters:
                # Create binary mask for this cluster
                cluster_mask = (cluster_labels == cluster_id).astype(np.int32)

                # Count neighbors with same label (3x3 kernel minus center)
                kernel = np.ones((3, 3), dtype=np.int32)
                kernel[1, 1] = 0  # Remove center
                neighbor_count = ndimage.convolve(cluster_mask, kernel, mode='constant', cval=0)

                # Max possible neighbors is 8 (3x3 kernel minus center)
                neighbor_ratio = neighbor_count / 8.0

                # Set spatial coherence for this cluster
                spatial_coherence[cluster_labels == cluster_id] = neighbor_ratio[cluster_labels == cluster_id]

            # Apply mask if provided
            if mask is not None:
                spatial_coherence = spatial_coherence * mask

            # Calculate average spatial coherence
            metrics['spatial_coherence'] = float(np.mean(spatial_coherence[cluster_labels >= 0]))
            print(f"Spatial Coherence: {metrics['spatial_coherence']:.4f} (higher is better)")

            # Create visualization of spatial coherence
            if output_dir is not None:
                plt.figure(figsize=(10, 8))
                plt.imshow(spatial_coherence, cmap='viridis', vmin=0, vmax=1)
                plt.colorbar(label='Spatial Coherence')
                plt.title('Spatial Coherence Map')
                plt.axis('off')
                plt.savefig(os.path.join(output_dir, "spatial_coherence_map.png"), dpi=300, bbox_inches='tight')
                plt.close()
        except Exception as e:
            print(f"Error calculating spatial coherence: {str(e)}")
            import traceback
            traceback.print_exc()

    # Return metrics
    return metrics


def run_4d_pixel_wise_clustering(
        model,
        dataset,
        n_clusters=5,
        chunk_size=64,
        chunk_overlap=8,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        output_dir=None,
        calculate_metrics=True,
        use_pca=True
):
    """
    Run efficient pixel-wise clustering on 4D hyperspectral data, using features from all excitations.

    Args:
        model: Trained HyperspectralCAEWithMasking model
        dataset: MaskedHyperspectralDataset instance
        n_clusters: Number of clusters for K-means
        chunk_size: Size of spatial chunks for processing
        chunk_overlap: Overlap between adjacent chunks
        device: Device to use for computation
        output_dir: Directory to save clustering results
        calculate_metrics: Whether to calculate clustering quality metrics
        use_pca: Whether to use PCA for dimensionality reduction (default: True)

    Returns:
        Dictionary with clustering results
    """
    print("Starting 4D pixel-wise clustering...")

    # Create output directory if needed
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    # Get data
    all_data = dataset.get_all_data()

    # Use mask if available
    mask = dataset.processed_mask if hasattr(dataset, 'processed_mask') else None

    # Set model to evaluation mode
    model.eval()
    model = model.to(device)

    # Step 1: Extract encoded features for each excitation
    print("Extracting encoded features...")
    encoded_features, spatial_shapes = extract_encoded_features(
        model=model,
        data_dict=all_data,
        mask=mask,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        device=device
    )

    # Step 2: Combine features from all excitations
    print("Combining features from all excitations for 4D analysis...")

    # Get a list of all excitation wavelengths
    excitation_wavelengths = list(encoded_features.keys())
    print(f"Using features from {len(excitation_wavelengths)} excitations: {excitation_wavelengths}")

    # Get dimension information
    first_ex = excitation_wavelengths[0]
    n_features, height, width = encoded_features[first_ex].shape

    # Concatenate features from all excitations
    concatenated_features = []
    for ex in excitation_wavelengths:
        # Reshape to [n_features, height*width]
        reshaped = encoded_features[ex].reshape(n_features, -1)
        concatenated_features.append(reshaped)

    # Stack along the feature dimension
    # Result shape: [n_features * n_excitations, height*width]
    combined_features = np.vstack(concatenated_features)

    # Reshape to [n_features * n_excitations, height, width] for clustering
    combined_features = combined_features.reshape(-1, height, width)

    print(f"Combined feature shape: {combined_features.shape}")

    # Step 3: Apply optimized K-means clustering on the combined features
    print(f"Applying K-means clustering with {n_clusters} clusters on 4D features...")
    cluster_labels, clustering_model = optimize_kmeans_clustering(
        features=combined_features,
        n_clusters=n_clusters,
        use_pca=use_pca  # Pass the parameter
    )

    # Mask the cluster labels if a mask is provided
    if mask is not None:
        # Ensure mask has the same shape as cluster labels
        if mask.shape != cluster_labels.shape:
            from scipy.ndimage import zoom
            mask_resized = zoom(mask,
                                (cluster_labels.shape[0] / mask.shape[0],
                                 cluster_labels.shape[1] / mask.shape[1]),
                                order=0)
            # Ensure binary mask
            mask_resized = (mask_resized > 0.5).astype(np.uint8)
        else:
            mask_resized = mask

        # Apply mask - set masked areas to -1 (typically used for "no cluster")
        cluster_labels[mask_resized == 0] = -1

    # Create cluster map visualization
    plt.figure(figsize=(10, 8))
    plt.imshow(cluster_labels, cmap='tab10', interpolation='nearest')
    plt.colorbar(label='Cluster ID')
    plt.title(f'4D Pixel-wise Clustering (K={n_clusters}, using {len(excitation_wavelengths)} excitations)')
    plt.axis('off')

    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, f"4d_cluster_map_k{n_clusters}.png"),
                    dpi=300, bbox_inches='tight')

        # Save cluster labels
        np.save(os.path.join(output_dir, f"4d_cluster_labels_k{n_clusters}.npy"),
                cluster_labels)

    plt.close()

    print(f"Clustering complete. Found {len(np.unique(cluster_labels))} unique clusters")

    # Calculate clustering metrics if requested
    metrics = None
    if calculate_metrics:
        print("\nEvaluating clustering quality...")
        metrics = evaluate_clustering_quality(
            features=combined_features,
            cluster_labels=cluster_labels,
            original_data=all_data,
            mask=mask,
            output_dir=output_dir
        )

        # Save metrics to file
        if output_dir is not None:
            metrics_file = os.path.join(output_dir, f"4d_clustering_metrics_k{n_clusters}.json")
            import json
            with open(metrics_file, 'w') as f:
                # Convert numpy values to Python types
                metrics_dict = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                                for k, v in metrics.items()}
                json.dump(metrics_dict, f, indent=2)
            print(f"Clustering metrics saved to: {metrics_file}")

    # Return results
    results = {
        'cluster_labels': cluster_labels,
        'clustering_model': clustering_model,
        'excitations_used': excitation_wavelengths,
        'encoded_features': encoded_features,
        'spatial_shapes': spatial_shapes,
        'n_clusters': n_clusters
    }

    if metrics is not None:
        results['metrics'] = metrics

    return results


def visualize_4d_cluster_profiles(
        cluster_results,
        dataset,
        original_data=None,
        output_dir=None
):
    """
    Visualize the spectral profiles of each cluster for 4D data.

    Args:
        cluster_results: Results from run_4d_pixel_wise_clustering
        dataset: MaskedHyperspectralDataset instance
        original_data: Optional dictionary with original data
        output_dir: Directory to save visualizations

    Returns:
        Dictionary with cluster profile statistics
    """
    # Get cluster labels and excitations
    cluster_labels = cluster_results['cluster_labels']
    excitations_used = cluster_results['excitations_used']

    print(f"Analyzing profiles for clusters using {len(excitations_used)} excitation wavelengths")

    # Get original data if not provided
    if original_data is None:
        original_data = dataset.get_all_data()

    # Get emission wavelengths
    emission_wavelengths = {}
    if hasattr(dataset, 'emission_wavelengths'):
        emission_wavelengths = dataset.emission_wavelengths

    # Get unique cluster IDs (excluding -1 which is used for masked areas)
    unique_clusters = sorted([c for c in np.unique(cluster_labels) if c >= 0])
    print(f"Analyzing profiles for {len(unique_clusters)} clusters")

    # Store cluster statistics
    cluster_stats = {}

    # Process each excitation and create individual plots
    for ex in excitations_used:
        # Create a figure for spectral profiles for this excitation
        plt.figure(figsize=(12, 8))

        # Get data for this excitation
        data = original_data[ex].cpu().numpy()

        # Get emission wavelengths for x-axis
        if ex in emission_wavelengths:
            x_values = emission_wavelengths[ex]
        else:
            x_values = np.arange(data.shape[2])

        # Calculate mean spectrum for each cluster
        for cluster_id in unique_clusters:
            # Create mask for this cluster
            cluster_mask = cluster_labels == cluster_id

            # Skip if no pixels in this cluster
            if not np.any(cluster_mask):
                continue

            # Get data for this cluster
            cluster_data = data[cluster_mask]

            # Calculate mean spectrum (ignoring NaNs)
            mean_spectrum = np.nanmean(cluster_data, axis=0)

            # Calculate standard deviation (ignoring NaNs)
            std_spectrum = np.nanstd(cluster_data, axis=0)

            # Store statistics
            if cluster_id not in cluster_stats:
                cluster_stats[cluster_id] = {}

            cluster_stats[cluster_id][ex] = {
                'mean': mean_spectrum,
                'std': std_spectrum,
                'count': np.sum(cluster_mask)
            }

            # Plot mean spectrum
            plt.plot(x_values, mean_spectrum, marker='o',
                     label=f"Cluster {cluster_id}",
                     color=plt.cm.tab10(cluster_id % 10))

            # Plot error bands
            plt.fill_between(
                x_values,
                mean_spectrum - std_spectrum,
                mean_spectrum + std_spectrum,
                alpha=0.2,
                color=plt.cm.tab10(cluster_id % 10)
            )

        # Add legend, labels, etc.
        plt.legend(loc='best')
        plt.xlabel('Emission Wavelength (nm)' if len(emission_wavelengths) > 0 else 'Emission Band Index')
        plt.ylabel('Intensity')
        plt.title(f'Spectral Profiles by Cluster (Excitation {ex}nm)')
        plt.grid(True, alpha=0.3)

        if output_dir is not None:
            plt.savefig(os.path.join(output_dir, f"cluster_profiles_ex{ex}.png"),
                        dpi=300, bbox_inches='tight')

        plt.close()

    # Create a combined visualization showing cluster profiles across excitations
    # This is a more complex visualization showing the 4D nature of the data
    plt.figure(figsize=(14, 10))

    # For each cluster, create a subplot showing mean intensity across excitations
    rows = int(np.ceil(len(unique_clusters) / 2))
    for i, cluster_id in enumerate(unique_clusters):
        plt.subplot(rows, 2, i + 1)

        # Collect mean intensities across all excitations
        ex_values = []
        mean_intensities = []
        std_intensities = []

        for ex in excitations_used:
            if ex in cluster_stats[cluster_id]:
                ex_values.append(ex)
                # Take mean across emission bands
                mean_intensities.append(np.mean(cluster_stats[cluster_id][ex]['mean']))
                std_intensities.append(np.mean(cluster_stats[cluster_id][ex]['std']))

        # Plot mean intensity vs excitation wavelength
        plt.errorbar(ex_values, mean_intensities, yerr=std_intensities,
                     marker='o', linestyle='-', capsize=3,
                     color=plt.cm.tab10(cluster_id % 10))

        plt.title(f'Cluster {cluster_id}')
        plt.xlabel('Excitation Wavelength (nm)')
        plt.ylabel('Mean Intensity')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, "4d_cluster_excitation_profiles.png"),
                    dpi=300, bbox_inches='tight')

    plt.close()

    # Create a bar chart of cluster sizes
    plt.figure(figsize=(10, 6))
    cluster_sizes = [np.sum(cluster_labels == c) for c in unique_clusters]

    plt.bar(
        [f"Cluster {c}" for c in unique_clusters],
        cluster_sizes,
        color=[plt.cm.tab10(c % 10) for c in unique_clusters]
    )

    plt.xlabel('Cluster')
    plt.ylabel('Number of Pixels')
    plt.title(f'Cluster Sizes (4D Clustering)')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', alpha=0.3)

    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, f"4d_cluster_sizes.png"),
                    dpi=300, bbox_inches='tight')

    plt.close()

    return cluster_stats

def create_vibrant_colors(n_colors, random_seed=42):
    """
    Create a list of vibrant, distinct colors with a neutral background color.

    Args:
        n_colors: Number of colors to generate
        random_seed: Random seed for reproducibility

    Returns:
        List of RGBA colors
    """
    import numpy as np
    from matplotlib import colormaps

    np.random.seed(random_seed)

    # Create a set of vibrant colors - starting with a NEUTRAL background color
    vibrant_colors = [
        # First color is a subtle light gray for background (cluster ID -1)
        [0.92, 0.92, 0.92, 1.0],  # Light gray background

        # Vibrant colors for actual clusters
        [1.0, 0.0, 0.0, 1.0],  # Bright red
        [0.0, 0.8, 0.0, 1.0],  # Bright green
        [0.0, 0.0, 1.0, 1.0],  # Bright blue
        [1.0, 1.0, 0.0, 1.0],  # Yellow
        [1.0, 0.0, 1.0, 1.0],  # Magenta
        [0.0, 1.0, 1.0, 1.0],  # Cyan
        [1.0, 0.5, 0.0, 1.0],  # Orange
        [0.5, 0.0, 1.0, 1.0],  # Purple
        [0.0, 0.6, 0.3, 1.0],  # Green-teal
        [0.8, 0.4, 0.0, 1.0],  # Brown
        # Additional colors if needed
        [0.8, 0.6, 0.7, 1.0],  # Pink
        [0.6, 0.8, 0.3, 1.0],  # Lime
        [0.0, 0.4, 0.8, 1.0],  # Sky blue
        [0.7, 0.0, 0.2, 1.0],  # Burgundy
        [0.4, 0.2, 0.6, 1.0],  # Indigo
        [0.0, 0.5, 0.5, 1.0],  # Teal
        [0.9, 0.7, 0.1, 1.0],  # Gold
        [0.5, 0.5, 0.1, 1.0],  # Olive
        [0.3, 0.6, 0.0, 1.0],  # Green
        [0.2, 0.0, 0.4, 1.0],  # Navy
    ]

    # If we need more colors, add from a rainbow colormap
    if n_colors > len(vibrant_colors):
        # Use hsv colormap for vibrant additional colors
        hsv_map = colormaps['hsv']
        additional_colors = [
            list(hsv_map(i / (n_colors - len(vibrant_colors))))
            for i in range(n_colors - len(vibrant_colors))
        ]
        vibrant_colors.extend(additional_colors)

    # Shuffle the colors but keep the first one (background) fixed
    first_color = vibrant_colors[0]  # Preserve the gray background
    remaining_colors = vibrant_colors[1:]
    np.random.shuffle(remaining_colors)

    return [first_color] + remaining_colors[:n_colors - 1]

def run_4d_pixel_wise_clustering_setup_color(
        model,
        dataset,
        n_clusters=5,
        chunk_size=64,
        chunk_overlap=8,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        output_dir=None,
        calculate_metrics=True,
        use_pca=True,
        custom_colors=None
):
    """
    Run efficient pixel-wise clustering on 4D hyperspectral data, using features from all excitations.
    """
    print("Starting 4D pixel-wise clustering...")

    # Create output directory if needed
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    # Get data
    all_data = dataset.get_all_data()

    # Use mask if available
    mask = dataset.processed_mask if hasattr(dataset, 'processed_mask') else None

    # Set model to evaluation mode
    model.eval()
    model = model.to(device)

    # Step 1: Extract encoded features
    print("Extracting encoded features...")
    encoded_features, spatial_shapes = extract_encoded_features(
        model=model,
        data_dict=all_data,
        mask=mask,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        device=device
    )

    # Step 2: Combine features from all excitations
    print("Combining features from all excitations for 4D analysis...")

    # Get a list of all excitation wavelengths
    excitation_wavelengths = list(encoded_features.keys())
    print(f"Using features from {len(excitation_wavelengths)} excitations: {excitation_wavelengths}")

    # Get dimension information
    first_ex = excitation_wavelengths[0]
    n_features, height, width = encoded_features[first_ex].shape

    # Concatenate features from all excitations
    concatenated_features = []
    for ex in excitation_wavelengths:
        # Reshape to [n_features, height*width]
        reshaped = encoded_features[ex].reshape(n_features, -1)
        concatenated_features.append(reshaped)

    # Stack along the feature dimension
    # Result shape: [n_features * n_excitations, height*width]
    combined_features = np.vstack(concatenated_features)

    # Reshape to [n_features * n_excitations, height, width] for clustering
    combined_features = combined_features.reshape(-1, height, width)

    print(f"Combined feature shape: {combined_features.shape}")

    # Step 3: Apply optimized K-means clustering on the combined features
    print(f"Applying K-means clustering with {n_clusters} clusters on 4D features...")
    cluster_labels, clustering_model = optimize_kmeans_clustering(
        features=combined_features,
        n_clusters=n_clusters,
        use_pca=use_pca
    )

    # Mask the cluster labels if a mask is provided
    if mask is not None:
        # Ensure mask has the same shape as cluster labels
        if mask.shape != cluster_labels.shape:
            from scipy.ndimage import zoom
            mask_resized = zoom(mask,
                                (cluster_labels.shape[0] / mask.shape[0],
                                 cluster_labels.shape[1] / mask.shape[1]),
                                order=0)
            # Ensure binary mask
            mask_resized = (mask_resized > 0.5).astype(np.uint8)
        else:
            mask_resized = mask

        # Apply mask - set masked areas to -1 (typically used for "no cluster")
        cluster_labels[mask_resized == 0] = -1

    # Generate vibrant color mapping
    if custom_colors is None:
        # Create vibrant color palette
        vibrant_colors = create_vibrant_colors(n_clusters + 1)  # +1 for background
        color_mapping = {-1: vibrant_colors[0]}  # Background
        for i in range(n_clusters):
            color_mapping[i] = vibrant_colors[i + 1]
    else:
        # Use provided custom colors
        color_mapping = custom_colors

    # Create cluster map visualization with the consistent colors
    plt.figure(figsize=(16, 14))

    # Create a colored image using the color mapping
    cluster_image = np.zeros((*cluster_labels.shape, 3))
    for cluster_id in np.unique(cluster_labels):
        mask = cluster_labels == cluster_id
        if cluster_id >= 0:
            cluster_image[mask] = color_mapping[cluster_id][:3]  # Use RGB without alpha
        else:
            # Use dark blue for background (-1)
            cluster_image[mask] = color_mapping.get(-1, [0, 0, 0.5])[:3]

    plt.imshow(cluster_image)
    plt.title(f'4D Pixel-wise Clustering (K={n_clusters}, using {len(excitation_wavelengths)} excitations)', fontsize=24)
    plt.axis('off')

    if output_dir is not None:
        # Save without colorbar
        plt.savefig(os.path.join(output_dir, f"4d_cluster_map_k{n_clusters}.png"),
                    dpi=300, bbox_inches='tight')

        # Save cluster labels
        np.save(os.path.join(output_dir, f"4d_cluster_labels_k{n_clusters}.npy"),
                cluster_labels)

    plt.close()

    print(f"Clustering complete. Found {len(np.unique(cluster_labels[cluster_labels >= 0]))} unique clusters")

    # Calculate clustering metrics if requested
    metrics = None
    if calculate_metrics:
        print("\nEvaluating clustering quality...")
        metrics = evaluate_clustering_quality(
            features=combined_features,
            cluster_labels=cluster_labels,
            original_data=all_data,
            mask=mask,
            output_dir=output_dir
        )

        # Save metrics to file
        if output_dir is not None:
            metrics_file = os.path.join(output_dir, f"4d_clustering_metrics_k{n_clusters}.json")
            import json
            with open(metrics_file, 'w') as f:
                # Convert numpy values to Python types
                metrics_dict = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                                for k, v in metrics.items()}
                json.dump(metrics_dict, f, indent=2)
            print(f"Clustering metrics saved to: {metrics_file}")

    # Return results with color mapping included
    results = {
        'cluster_labels': cluster_labels,
        'clustering_model': clustering_model,
        'excitations_used': excitation_wavelengths,
        'encoded_features': encoded_features,
        'spatial_shapes': spatial_shapes,
        'n_clusters': n_clusters,
        'color_mapping': color_mapping  # Include the color mapping in results
    }

    if metrics is not None:
        results['metrics'] = metrics

    return results