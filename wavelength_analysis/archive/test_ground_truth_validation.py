"""
Test script for ground truth validation with concatenation-based clustering.
This script demonstrates how to:
1. Load masked hyperspectral data
2. Extract ground truth labels from colored PNG
3. Perform clustering with different k values
4. Validate against ground truth
5. Compare different approaches
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from ground_truth_validation import (
    extract_ground_truth_from_png,
    calculate_clustering_accuracy,
    visualize_clustering_vs_ground_truth,
    compare_multiple_clusterings_to_ground_truth
)
from concatenation_clustering import (
    load_masked_data,
    concatenate_hyperspectral_data,
    perform_clustering,
    reconstruct_cluster_map,
    compare_clustering_k_values
)


def main():
    """Main function to run ground truth validation pipeline."""
    
    # Set up paths
    base_dir = Path(r"C:\Users\meloy\PycharmProjects\Capstone")
    data_path = base_dir / "data" / "processed" / "Lichens" / "lichens_data_masked.pkl"
    png_path = Path(r"C:\Users\meloy\Downloads\Mask_Manual.png")
    output_dir = base_dir / "wavelength_analysis" / "ground_truth_validation_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Ground Truth Validation Pipeline for Concatenation Clustering")
    print("="*80)
    
    # Step 1: Extract ground truth from PNG
    print("\n1. Extracting ground truth labels from PNG...")
    background_colors = [
        (24, 24, 24, 255),      # Dark gray background
        (168, 168, 168, 255)    # Light gray background
    ]
    
    ground_truth, color_mapping, lichen_colors = extract_ground_truth_from_png(
        png_path,
        background_colors=background_colors,
        target_shape=(1040, 1392)  # Match your data dimensions
    )
    
    print(f"\nGround truth shape: {ground_truth.shape}")
    print(f"Number of lichen types (classes): {len(lichen_colors)}")
    
    # Visualize ground truth
    plt.figure(figsize=(10, 8))
    plt.imshow(ground_truth, cmap='tab10')
    plt.colorbar(label='Lichen Type')
    plt.title('Ground Truth Labels from PNG')
    plt.axis('off')
    plt.savefig(output_dir / 'ground_truth.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Step 2: Load and concatenate hyperspectral data
    print("\n2. Loading and processing hyperspectral data...")
    data_dict = load_masked_data(data_path)
    
    print("\n3. Concatenating hyperspectral features...")
    df, valid_mask, metadata = concatenate_hyperspectral_data(
        data_dict,
        normalize=True,
        scale=True
    )
    
    print(f"Concatenated data shape: {df.shape}")
    
    # Step 3: Test multiple k values and compare with ground truth
    print("\n4. Testing multiple k values with ground truth validation...")
    
    # Test k values around the number of ground truth classes
    n_true_classes = len(lichen_colors)
    k_values = list(range(max(2, n_true_classes - 2), n_true_classes + 3))
    print(f"Testing k values: {k_values}")
    
    comparison_results = compare_clustering_k_values(
        df,
        valid_mask,
        metadata,
        k_values=k_values,
        ground_truth=ground_truth,
        save_dir=output_dir
    )
    
    # Step 4: Detailed analysis of best k
    print("\n5. Analyzing best performing k value...")
    
    # Find best k based on ground truth metrics
    best_k = None
    best_purity = 0
    
    for k, results in comparison_results.items():
        if 'ground_truth_metrics' in results:
            purity = results['ground_truth_metrics'].get('purity', 0)
            if purity > best_purity:
                best_purity = purity
                best_k = k
    
    if best_k is not None:
        print(f"\nBest k={best_k} with purity={best_purity:.4f}")
        
        # Get the best clustering result
        best_results = comparison_results[best_k]
        best_cluster_map = best_results['cluster_map']
        best_gt_metrics = best_results['ground_truth_metrics']
        
        # Visualize best result vs ground truth
        print(f"\n6. Creating detailed visualization for k={best_k}...")
        visualize_clustering_vs_ground_truth(
            best_cluster_map,
            ground_truth,
            best_gt_metrics,
            color_mapping=color_mapping,
            save_path=output_dir / f'best_k{best_k}_vs_ground_truth.png'
        )
        
        # Print detailed metrics for best k
        print(f"\nDetailed metrics for k={best_k}:")
        print(f"  Purity: {best_gt_metrics.get('purity', 0):.4f}")
        print(f"  Adjusted Rand Index: {best_gt_metrics.get('adjusted_rand_score', 0):.4f}")
        print(f"  Normalized Mutual Info: {best_gt_metrics.get('normalized_mutual_info', 0):.4f}")
        print(f"  V-Measure: {best_gt_metrics.get('v_measure', 0):.4f}")
        print(f"  Homogeneity: {best_gt_metrics.get('homogeneity', 0):.4f}")
        print(f"  Completeness: {best_gt_metrics.get('completeness', 0):.4f}")
    
    # Step 5: Test with PCA for comparison
    print("\n7. Testing with PCA dimensionality reduction...")
    
    pca_results = {}
    for use_pca, n_components in [(False, None), (True, 10), (True, 20), (True, 50)]:
        if use_pca:
            method_name = f"KMeans_PCA_{n_components}"
            print(f"\nTesting {method_name}...")
        else:
            method_name = "KMeans_No_PCA"
            print(f"\nTesting {method_name}...")
        
        # Perform clustering
        labels, metrics = perform_clustering(
            df,
            n_clusters=best_k if best_k else n_true_classes,
            method='kmeans',
            use_pca=use_pca,
            n_components=n_components if use_pca else None
        )
        
        # Reconstruct cluster map
        cluster_map = reconstruct_cluster_map(labels, df, valid_mask, metadata)
        
        pca_results[method_name] = cluster_map
    
    # Compare all PCA variants
    print("\n8. Comparing different PCA configurations...")
    comparison_df = compare_multiple_clusterings_to_ground_truth(
        pca_results,
        ground_truth,
        valid_mask=valid_mask,
        save_dir=output_dir
    )
    
    # Save final summary
    print("\n9. Saving summary report...")
    
    # Create summary report
    summary = {
        'Ground Truth Info': {
            'PNG Path': str(png_path),
            'Number of Classes': len(lichen_colors),
            'Image Shape': ground_truth.shape,
            'Valid Pixels': np.sum(valid_mask)
        },
        'Data Info': {
            'Data Path': str(data_path),
            'Concatenated Features': df.shape[1] - 2,  # Exclude x, y
            'Total Samples': df.shape[0]
        },
        'Best K-Means': {
            'Optimal k': best_k,
            'Purity': best_purity,
            'ARI': best_gt_metrics.get('adjusted_rand_score', 0) if best_k else 0
        },
        'K Values Tested': k_values
    }
    
    # Save summary as text
    summary_path = output_dir / 'validation_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("Ground Truth Validation Summary\n")
        f.write("="*50 + "\n\n")
        
        for section, content in summary.items():
            f.write(f"{section}:\n")
            if isinstance(content, dict):
                for key, value in content.items():
                    f.write(f"  {key}: {value}\n")
            else:
                f.write(f"  {content}\n")
            f.write("\n")
        
        f.write("\nPCA Comparison Results:\n")
        f.write(comparison_df.to_string())
    
    print(f"Summary saved to {summary_path}")
    
    print("\n" + "="*80)
    print("Ground Truth Validation Pipeline Completed Successfully!")
    print("="*80)
    
    return ground_truth, comparison_results, comparison_df


if __name__ == "__main__":
    ground_truth, results, comparison = main()