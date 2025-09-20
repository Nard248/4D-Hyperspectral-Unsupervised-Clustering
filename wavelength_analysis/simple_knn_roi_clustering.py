"""
SIMPLE KNN ROI CLUSTERING - STANDALONE SCRIPT
Works with spectral data only, no spatial coordinates in clustering.
Uses ROI regions to select reference pixels for KNN classification.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Add project paths
base_dir = Path(r"C:\Users\meloy\PycharmProjects\Capstone")
sys.path.append(str(base_dir))
sys.path.append(str(base_dir / "wavelength_analysis"))

# Import required functions
from concatenation_clustering import (
    load_masked_data,
    concatenate_hyperspectral_data
)
from ground_truth_validation import (
    extract_ground_truth_from_png,
    calculate_clustering_accuracy
)


def simple_knn_roi_clustering(data, roi_regions, n_clusters=4, verbose=True):
    """
    SIMPLE KNN clustering using ONLY SPECTRAL DATA (no x,y coordinates).
    Uses ROI regions to get reference pixels.

    Args:
        data: Hyperspectral data dictionary
        roi_regions: List of ROI dictionaries with 'coords' and 'name'
        n_clusters: Number of clusters
        verbose: Print progress

    Returns:
        labels: Cluster labels for all pixels
        metrics: Clustering metrics
        cluster_map: 2D cluster map
    """
    if verbose:
        print("=== SIMPLE KNN ROI CLUSTERING (SPECTRAL ONLY) ===")

    # Get concatenated data
    df, valid_mask, metadata = concatenate_hyperspectral_data(data, normalize=True, scale=True)

    # Get ONLY spectral features (exclude x,y)
    spectral_cols = [col for col in df.columns if col not in ['x', 'y']]
    spectral_data = df[spectral_cols].values

    if verbose:
        print(f"Spectral data shape: {spectral_data.shape} (no spatial coordinates)")

    # Get coordinate mappings for ROI selection
    x_coords = df['x'].values.astype(int)
    y_coords = df['y'].values.astype(int)

    # Extract ROI samples using SPECTRAL DATA ONLY
    roi_samples = []
    roi_labels = []

    for roi_idx, roi in enumerate(roi_regions[:n_clusters]):
        y_start, y_end, x_start, x_end = roi['coords']
        roi_pixel_count = 0

        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            if y_start <= y < y_end and x_start <= x < x_end:
                roi_samples.append(spectral_data[i])  # ONLY spectral features
                roi_labels.append(roi_idx)
                roi_pixel_count += 1

        if verbose:
            print(f"  {roi['name']}: {roi_pixel_count} pixels")

    if not roi_samples:
        if verbose:
            print("No ROI samples found!")
        return None, None, None

    # Train KNN on SPECTRAL DATA ONLY
    X_train = np.array(roi_samples)
    y_train = np.array(roi_labels)

    if verbose:
        print(f"Training KNN: {len(X_train)} samples, {X_train.shape[1]} spectral features")

    knn = KNeighborsClassifier(n_neighbors=min(5, len(X_train)), weights='distance')
    knn.fit(X_train, y_train)

    # Predict on ALL spectral data
    labels = knn.predict(spectral_data)

    # Calculate metrics on SPECTRAL DATA ONLY
    metrics = {
        'silhouette_score': silhouette_score(spectral_data, labels),
        'davies_bouldin_score': davies_bouldin_score(spectral_data, labels),
        'calinski_harabasz_score': calinski_harabasz_score(spectral_data, labels)
    }

    # Reconstruct cluster map
    height, width = metadata['original_shape'][:2]
    cluster_map = np.full((height, width), -1, dtype=int)

    # Fill in cluster labels for valid pixels
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        if 0 <= y < height and 0 <= x < width:
            cluster_map[y, x] = labels[i]

    if verbose:
        print(f"KNN completed. Unique labels: {np.unique(labels)}")

    return labels, metrics, cluster_map


def visualize_results(cluster_map, ground_truth, roi_regions, purity, save_path=None):
    """Visualize clustering results with ROI overlay."""

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Ground truth
    axes[0].imshow(ground_truth, cmap='tab10')
    axes[0].set_title('Ground Truth', fontsize=14)
    axes[0].axis('off')

    # Clustering result
    axes[1].imshow(cluster_map, cmap='tab10')
    axes[1].set_title(f'KNN ROI Clustering\nPurity: {purity:.3f}', fontsize=14)
    axes[1].axis('off')

    # ROI overlay
    axes[2].imshow(cluster_map, cmap='tab10')
    for roi in roi_regions:
        y_start, y_end, x_start, x_end = roi['coords']
        rect = patches.Rectangle(
            (x_start, y_start),
            x_end - x_start,
            y_end - y_start,
            linewidth=3,
            edgecolor=roi['color'],
            facecolor='none'
        )
        axes[2].add_patch(rect)
        axes[2].text(x_start + 5, y_start + 15, roi['name'],
                    color='white', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=roi['color'], alpha=0.8))

    axes[2].set_title('Clustering with ROI Regions', fontsize=14)
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")

    plt.show()


def main():
    """Main function to run the complete pipeline."""

    print("="*80)
    print("SIMPLE KNN ROI CLUSTERING - STANDALONE SCRIPT")
    print("="*80)

    try:
        # 1. Load data
        print("\n1. Loading data...")
        sample_name = "Lichens"
        data_path = base_dir / "data" / "processed" / sample_name / "lichens_data_masked.pkl"
        png_path = Path(r"C:\Users\meloy\Downloads\Mask_Manual.png")

        # Load hyperspectral data
        full_data = load_masked_data(data_path)
        print(f"SUCCESS: Loaded hyperspectral data with {len(full_data['excitation_wavelengths'])} excitations")

        # Crop data (same as notebook)
        start_col = 1392 - 925
        end_col = 1392

        cropped_data = {
            'excitation_wavelengths': full_data['excitation_wavelengths'],
            'metadata': full_data.get('metadata', {}),
            'data': {}
        }

        for ex in full_data['excitation_wavelengths']:
            ex_str = str(ex)
            original_cube = full_data['data'][ex_str]['cube']
            cropped_cube = original_cube[:, start_col:end_col, :]
            cropped_data['data'][ex_str] = {
                **full_data['data'][ex_str],
                'cube': cropped_cube
            }

        full_data = cropped_data
        print(f"SUCCESS: Data cropped to shape: {cropped_cube.shape}")

        # 2. Load ground truth
        print("\n2. Loading ground truth...")
        background_colors = [
            (24, 24, 24, 255),
            (168, 168, 168, 255)
        ]

        ground_truth, color_mapping, lichen_colors = extract_ground_truth_from_png(
            png_path,
            background_colors=background_colors,
            target_shape=(1040, 1392)
        )

        # Crop ground truth to match data
        ground_truth_cropped = ground_truth[:, start_col:end_col]

        # Remap ground truth labels
        ground_truth_remapped = ground_truth_cropped.copy()
        ground_truth_remapped[ground_truth_cropped == 5] = 3  # Map class 5 to class 3

        print(f"SUCCESS: Ground truth loaded and remapped. Shape: {ground_truth_remapped.shape}")
        print(f"  Classes: {np.unique(ground_truth_remapped)}")

        # 3. Define ROI regions
        print("\n3. Defining ROI regions...")
        roi_regions = [
            {'name': 'Region 1', 'coords': (175, 225, 100, 150), 'color': 'red'},
            {'name': 'Region 2', 'coords': (175, 225, 250, 300), 'color': 'blue'},
            {'name': 'Region 3', 'coords': (175, 225, 425, 475), 'color': 'green'},
            {'name': 'Region 4', 'coords': (175, 225, 650, 700), 'color': 'yellow'},
        ]

        for roi in roi_regions:
            y_start, y_end, x_start, x_end = roi['coords']
            pixel_count = (y_end - y_start) * (x_end - x_start)
            print(f"  {roi['name']}: Y[{y_start}:{y_end}], X[{x_start}:{x_end}] - {pixel_count} pixels")

        # 4. Run KNN clustering
        print("\n4. Running KNN ROI clustering...")
        labels, metrics, cluster_map = simple_knn_roi_clustering(
            full_data,
            roi_regions,
            n_clusters=4,
            verbose=True
        )

        if labels is None:
            print("FAILED: KNN clustering failed!")
            return

        # 5. Calculate accuracy
        print("\n5. Calculating accuracy...")
        valid_gt_mask = ground_truth_remapped >= 0
        gt_metrics = calculate_clustering_accuracy(
            cluster_map,
            ground_truth_remapped,
            valid_gt_mask
        )

        if gt_metrics and 'purity' in gt_metrics:
            print(f"SUCCESS!")
            print(f"  Purity: {gt_metrics['purity']:.4f}")
            print(f"  ARI: {gt_metrics['adjusted_rand_score']:.4f}")
            print(f"  NMI: {gt_metrics['normalized_mutual_info']:.4f}")
            print(f"  Silhouette: {metrics['silhouette_score']:.4f}")

            # 6. Visualize results
            print("\n6. Creating visualization...")
            save_path = base_dir / "wavelength_analysis" / "knn_roi_clustering_results.png"
            visualize_results(
                cluster_map,
                ground_truth_remapped,
                roi_regions,
                gt_metrics['purity'],
                save_path=save_path
            )

            # 7. Save results
            print("\n7. Saving results...")
            results = {
                'purity': gt_metrics['purity'],
                'ari': gt_metrics['adjusted_rand_score'],
                'nmi': gt_metrics['normalized_mutual_info'],
                'v_measure': gt_metrics['v_measure'],
                'silhouette': metrics['silhouette_score'],
                'davies_bouldin': metrics['davies_bouldin_score'],
                'calinski_harabasz': metrics['calinski_harabasz_score'],
                'n_features': len([col for col in ['ex_310_em_420'] if 'ex_' in col and 'em_' in col]) * 8,  # Approx spectral features
                'method': 'KNN_ROI_SpectralOnly'
            }

            results_path = base_dir / "wavelength_analysis" / "knn_roi_clustering_results.txt"
            with open(results_path, 'w') as f:
                f.write("KNN ROI CLUSTERING RESULTS\n")
                f.write("="*50 + "\n")
                for key, value in results.items():
                    f.write(f"{key}: {value}\n")

            print(f"SUCCESS: Results saved to: {results_path}")

        else:
            print("FAILED: Failed to calculate accuracy metrics!")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"\n{'='*80}")
    print("KNN ROI CLUSTERING COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()