"""
Robustness Analysis for Wavelength Selection
==============================================

This script evaluates the robustness of wavelength selection by testing many possible
combinations of n bands and comparing their performance.

Purpose: Prove that the autoencoder+perturbation method selects near-optimal combinations.

Algorithm:
1. For small n (≤10): Test ALL possible combinations (exhaustive search)
2. For large n (>10): Test random sample of combinations (e.g., 10,000)
3. For each combination:
   - Extract the selected wavelength bands
   - Run KMeans clustering
   - Compute supervised metrics (accuracy, F1, etc.)
4. Compare with autoencoder-selected combination
5. Generate distribution plots and statistics

Usage:
    python robustness_analysis.py 5          # Test all combinations of 5 bands
    python robustness_analysis.py 13 10000   # Test 10000 random combinations of 13 bands
"""

import numpy as np
import pandas as pd
import pickle
from itertools import combinations, permutations
import random
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
from tqdm import tqdm
import argparse
from scipy import stats
import sys

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# ============================================================================
# Configuration
# ============================================================================

base_dir = Path(__file__).parent.parent

DATA_PATH = base_dir / "data" / "processed" / "Lichens" / "lichens_data_masked.pkl"
MASK_PATH = Path(r"C:\Users\meloy\Downloads\Mask_Manual.png")
SAMPLE_NAME = "Lichens"
N_CLUSTERS = 4
HORIZONTAL_CROP = (467, 1392)

# ============================================================================
# Helper Functions (adapted from V2-2)
# ============================================================================

def load_data():
    """Load hyperspectral data and ground truth"""
    print("Loading data...")

    # Load hyperspectral data
    with open(DATA_PATH, 'rb') as f:
        full_data = pickle.load(f)

    # Load ground truth
    from PIL import Image
    mask_img = Image.open(MASK_PATH)
    mask_array = np.array(mask_img)

    # Resize to match data dimensions
    first_ex = full_data['excitation_wavelengths'][0]
    first_cube = full_data['data'][str(first_ex)]['cube']
    target_shape = (first_cube.shape[0], first_cube.shape[1])

    if mask_array.shape[:2] != target_shape:
        mask_img_resized = mask_img.resize((target_shape[1], target_shape[0]), Image.NEAREST)
        mask_array = np.array(mask_img_resized)

    # Extract ground truth labels
    unique_colors = {}
    for i in range(mask_array.shape[0]):
        for j in range(mask_array.shape[1]):
            color = tuple(mask_array[i, j])
            if color not in unique_colors:
                unique_colors[color] = 0
            unique_colors[color] += 1

    # Filter lichen colors (exclude background)
    lichen_colors = [(c, count) for c, count in unique_colors.items()
                     if count > 1000 and c[0] != 0]
    lichen_colors.sort(key=lambda x: x[0][0])

    # Create ground truth array
    ground_truth = np.full(target_shape, -1, dtype=np.int32)
    color_to_class = {color: idx for idx, (color, _) in enumerate(lichen_colors)}

    for i in range(mask_array.shape[0]):
        for j in range(mask_array.shape[1]):
            color = tuple(mask_array[i, j])
            if color in color_to_class:
                ground_truth[i, j] = color_to_class[color]

    # Crop data
    crop_start, crop_end = HORIZONTAL_CROP
    for ex in full_data['excitation_wavelengths']:
        ex_key = str(ex)
        full_data['data'][ex_key]['cube'] = full_data['data'][ex_key]['cube'][:, crop_start:crop_end, :]

    ground_truth = ground_truth[:, crop_start:crop_end]

    # Map to 4 classes (0, 1, 2, 5 -> 0, 1, 2, 3)
    gt_4class = ground_truth.copy()
    gt_4class[ground_truth == 5] = 3
    gt_4class[ground_truth == 3] = -1
    gt_4class[ground_truth == 4] = -1

    print(f"Data loaded: {len(full_data['excitation_wavelengths'])} excitations")
    print(f"Spatial dimensions: {ground_truth.shape}")
    print(f"Total spectral bands: 192")

    return full_data, gt_4class


def get_all_band_indices(full_data):
    """Get all (excitation_idx, emission_idx) pairs for 192 bands"""
    band_indices = []
    excitations = full_data['excitation_wavelengths']

    for ex_idx, ex in enumerate(excitations):
        ex_key = str(ex)
        n_emissions = full_data['data'][ex_key]['cube'].shape[2]
        for em_idx in range(n_emissions):
            band_indices.append((ex_idx, em_idx))

    return band_indices


def extract_combination_data(full_data, band_combination):
    """
    Extract data for a specific combination of bands.

    Args:
        full_data: Hyperspectral data dictionary
        band_combination: List of (excitation_idx, emission_idx) tuples

    Returns:
        Flattened data array (n_valid_pixels, n_bands)
    """
    excitations = full_data['excitation_wavelengths']
    first_ex = excitations[0]
    first_cube = full_data['data'][str(first_ex)]['cube']

    # Get valid pixel mask
    valid_mask = ~np.isnan(first_cube[:, :, 0])
    n_valid_pixels = np.sum(valid_mask)

    # Extract data for selected bands
    n_bands = len(band_combination)
    data_matrix = np.zeros((n_valid_pixels, n_bands))

    for band_idx, (ex_idx, em_idx) in enumerate(band_combination):
        ex = excitations[ex_idx]
        cube = full_data['data'][str(ex)]['cube']
        band_data = cube[:, :, em_idx]
        data_matrix[:, band_idx] = band_data[valid_mask]

    return data_matrix, valid_mask


def compute_metrics(full_data, ground_truth, band_combination):
    """
    Compute clustering metrics for a given band combination.

    Returns:
        Dictionary of metrics
    """
    # Extract data
    X, valid_mask = extract_combination_data(full_data, band_combination)

    # Normalize data
    X_normalized = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    # Replace any NaN/Inf with 0
    X_normalized = np.nan_to_num(X_normalized, nan=0.0, posinf=0.0, neginf=0.0)

    # Run KMeans clustering
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_normalized)

    # Map cluster labels to ground truth classes
    gt_flat = ground_truth[valid_mask]
    valid_gt_mask = gt_flat >= 0

    cluster_labels_valid = cluster_labels[valid_gt_mask]
    gt_valid = gt_flat[valid_gt_mask]

    # Find best cluster-to-class mapping
    confusion = confusion_matrix(gt_valid, cluster_labels_valid)

    # Try all permutations to find best mapping
    best_accuracy = 0
    best_mapping = None

    for perm in permutations(range(N_CLUSTERS)):
        mapped_labels = np.array([perm[label] for label in cluster_labels_valid])
        acc = accuracy_score(gt_valid, mapped_labels)
        if acc > best_accuracy:
            best_accuracy = acc
            best_mapping = perm

    # Apply best mapping
    mapped_labels = np.array([best_mapping[label] for label in cluster_labels_valid])

    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(gt_valid, mapped_labels),
        'f1_weighted': f1_score(gt_valid, mapped_labels, average='weighted', zero_division=0),
        'precision_weighted': precision_score(gt_valid, mapped_labels, average='weighted', zero_division=0),
        'recall_weighted': recall_score(gt_valid, mapped_labels, average='weighted', zero_division=0),
        'cohen_kappa': cohen_kappa_score(gt_valid, mapped_labels),
        'n_bands': len(band_combination)
    }

    return metrics


def load_autoencoder_selection(n_bands):
    """
    Load the autoencoder-selected combination for n bands from 1Dimensions results.

    Returns:
        List of (excitation_idx, emission_idx) tuples, or None if not found
    """
    results_file = f"validation_results_v2/1Dimensions/wavelength_selection_results_v2.xlsx"

    if not Path(results_file).exists():
        return None

    df = pd.read_excel(results_file)
    config_row = df[df['n_combinations_selected'] == n_bands]

    if len(config_row) == 0:
        return None

    return config_row.iloc[0]['accuracy']


# ============================================================================
# Main Robustness Analysis
# ============================================================================

def run_robustness_analysis(n_bands, max_combinations=None, output_dir="validation_results_v2/robustness"):
    """
    Run robustness analysis for n-band combinations.

    Args:
        n_bands: Number of bands to select
        max_combinations: Maximum number of combinations to test (None = all)
        output_dir: Directory to save results
    """
    print("="*80)
    print(f"ROBUSTNESS ANALYSIS FOR {n_bands} BANDS")
    print("="*80)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    full_data, ground_truth = load_data()

    # Get all possible band indices
    all_bands = get_all_band_indices(full_data)
    print(f"\nTotal available bands: {len(all_bands)}")

    # Calculate total possible combinations
    from math import comb
    total_combinations = comb(len(all_bands), n_bands)
    print(f"Total possible {n_bands}-band combinations: {total_combinations:,}")

    # Decide on sampling strategy
    if max_combinations is None or max_combinations >= total_combinations:
        # Exhaustive search
        print(f"\nRunning EXHAUSTIVE search (all {total_combinations:,} combinations)")
        do_exhaustive = True
        n_to_test = total_combinations
    else:
        # Random sampling
        print(f"\nRunning RANDOM SAMPLING ({max_combinations:,} combinations)")
        do_exhaustive = False
        n_to_test = max_combinations

    # Generate combinations to test
    print("\nGenerating combinations to test...")
    if do_exhaustive:
        combinations_to_test = list(combinations(range(len(all_bands)), n_bands))
    else:
        # Random sampling
        combinations_to_test = []
        all_indices = set()
        while len(combinations_to_test) < n_to_test:
            combo = tuple(sorted(random.sample(range(len(all_bands)), n_bands)))
            if combo not in all_indices:
                all_indices.add(combo)
                combinations_to_test.append(combo)

    print(f"Testing {len(combinations_to_test):,} combinations...")

    # Run analysis
    results = []
    start_time = time.time()

    for i, combo_indices in enumerate(tqdm(combinations_to_test, desc="Testing combinations")):
        # Convert indices to actual band selections
        band_combo = [all_bands[idx] for idx in combo_indices]

        # Compute metrics
        try:
            metrics = compute_metrics(full_data, ground_truth, band_combo)
            metrics['combo_id'] = i
            metrics['band_indices'] = combo_indices
            results.append(metrics)

            # Save intermediate results every 1000 combinations
            if (i + 1) % 1000 == 0:
                df_temp = pd.DataFrame(results)
                df_temp.to_csv(output_path / f"robustness_{n_bands}bands_temp.csv", index=False)
        except Exception as e:
            print(f"\nError processing combination {i}: {e}")
            continue

    elapsed_time = time.time() - start_time
    print(f"\nCompleted in {elapsed_time:.2f} seconds ({elapsed_time/len(combinations_to_test):.3f}s per combination)")

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    # Save complete results
    output_file = output_path / f"robustness_{n_bands}bands_results.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Generate statistics and visualizations
    generate_analysis_report(df_results, n_bands, output_path, do_exhaustive)

    return df_results


def generate_analysis_report(df_results, n_bands, output_path, is_exhaustive):
    """Generate summary statistics and visualizations"""

    print("\n" + "="*80)
    print("STATISTICAL SUMMARY")
    print("="*80)

    # Basic statistics
    acc_mean = df_results['accuracy'].mean()
    acc_std = df_results['accuracy'].std()
    acc_min = df_results['accuracy'].min()
    acc_max = df_results['accuracy'].max()
    acc_median = df_results['accuracy'].median()

    print(f"\nAccuracy Statistics ({len(df_results):,} combinations tested):")
    print(f"  Mean:   {acc_mean:.4f}")
    print(f"  Std:    {acc_std:.4f}")
    print(f"  Min:    {acc_min:.4f}")
    print(f"  Max:    {acc_max:.4f}")
    print(f"  Median: {acc_median:.4f}")

    # Percentiles
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"\nPercentiles:")
    for p in percentiles:
        val = np.percentile(df_results['accuracy'], p)
        print(f"  {p}th:  {val:.4f}")

    # Top 10 combinations
    print(f"\nTop 10 Best Combinations:")
    top_10 = df_results.nlargest(10, 'accuracy')
    for idx, row in top_10.iterrows():
        print(f"  {row['accuracy']:.4f} - Combo ID: {row['combo_id']}")

    # Load autoencoder selection performance if available
    autoencoder_acc = load_autoencoder_selection(n_bands)
    if autoencoder_acc is not None:
        print(f"\nAutoencoder Selection Accuracy: {autoencoder_acc:.4f}")

        # Calculate percentile rank
        percentile_rank = stats.percentileofscore(df_results['accuracy'], autoencoder_acc)
        print(f"Percentile Rank: {percentile_rank:.2f}th percentile")

        # How many combinations are better?
        n_better = np.sum(df_results['accuracy'] > autoencoder_acc)
        pct_better = (n_better / len(df_results)) * 100
        print(f"Combinations better than autoencoder: {n_better:,} ({pct_better:.2f}%)")

    # Create visualizations
    create_visualizations(df_results, n_bands, output_path, autoencoder_acc, is_exhaustive)

    # Save summary statistics
    summary = {
        'n_bands': n_bands,
        'n_combinations_tested': len(df_results),
        'is_exhaustive': is_exhaustive,
        'accuracy_mean': acc_mean,
        'accuracy_std': acc_std,
        'accuracy_min': acc_min,
        'accuracy_max': acc_max,
        'accuracy_median': acc_median,
        'autoencoder_accuracy': autoencoder_acc,
        'autoencoder_percentile': percentile_rank if autoencoder_acc else None,
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_path / f"robustness_{n_bands}bands_summary.csv", index=False)


def create_visualizations(df_results, n_bands, output_path, autoencoder_acc, is_exhaustive):
    """Create visualization plots"""

    print("\nGenerating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Histogram of accuracies
    ax1 = axes[0, 0]
    ax1.hist(df_results['accuracy'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.axvline(df_results['accuracy'].mean(), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {df_results["accuracy"].mean():.4f}')
    ax1.axvline(df_results['accuracy'].median(), color='green', linestyle='--',
                linewidth=2, label=f'Median: {df_results["accuracy"].median():.4f}')

    if autoencoder_acc is not None:
        ax1.axvline(autoencoder_acc, color='purple', linestyle='-',
                    linewidth=2, label=f'Autoencoder: {autoencoder_acc:.4f}')

    ax1.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title(f'Distribution of Accuracies ({n_bands} bands)\n' +
                  ('Exhaustive Search' if is_exhaustive else 'Random Sampling'),
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Cumulative distribution
    ax2 = axes[0, 1]
    sorted_acc = np.sort(df_results['accuracy'])
    cumulative = np.arange(1, len(sorted_acc) + 1) / len(sorted_acc) * 100
    ax2.plot(sorted_acc, cumulative, linewidth=2, color='blue')

    if autoencoder_acc is not None:
        percentile = stats.percentileofscore(df_results['accuracy'], autoencoder_acc)
        ax2.axvline(autoencoder_acc, color='purple', linestyle='-',
                    linewidth=2, label=f'Autoencoder ({percentile:.1f}th percentile)')
        ax2.axhline(percentile, color='purple', linestyle=':', alpha=0.5)

    ax2.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Percentage', fontsize=12, fontweight='bold')
    ax2.set_title('Cumulative Distribution Function', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Box plot with percentiles
    ax3 = axes[1, 0]
    bp = ax3.boxplot([df_results['accuracy']], vert=True, patch_artist=True,
                      widths=0.5, showmeans=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['means'][0].set_marker('D')
    bp['means'][0].set_markerfacecolor('red')

    if autoencoder_acc is not None:
        ax3.axhline(autoencoder_acc, color='purple', linestyle='-',
                    linewidth=2, label='Autoencoder')
        ax3.legend(fontsize=10)

    ax3.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax3.set_title('Accuracy Distribution (Box Plot)', fontsize=14, fontweight='bold')
    ax3.set_xticklabels([f'{n_bands} bands'])
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Top combinations comparison
    ax4 = axes[1, 1]
    top_n = min(20, len(df_results))
    top_combos = df_results.nlargest(top_n, 'accuracy')

    x_pos = np.arange(top_n)
    colors = ['purple' if autoencoder_acc and abs(acc - autoencoder_acc) < 0.001
              else 'skyblue' for acc in top_combos['accuracy']]

    ax4.bar(x_pos, top_combos['accuracy'], color=colors, edgecolor='black', alpha=0.7)

    if autoencoder_acc is not None:
        ax4.axhline(autoencoder_acc, color='red', linestyle='--',
                    linewidth=2, label='Autoencoder', alpha=0.7)

    ax4.set_xlabel('Rank', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax4.set_title(f'Top {top_n} Combinations', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path / f'robustness_{n_bands}bands_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path / f'robustness_{n_bands}bands_analysis.png'}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Robustness Analysis for Wavelength Selection')
    # parser.add_argument('n_bands', type=int, help='Number of bands to select')
    # parser.add_argument('max_combinations', type=int, nargs='?', default=None,
    #                     help='Maximum number of combinations to test (default: all if feasible)')
    # parser.add_argument('--output', type=str, default='validation_results_v2/robustness',
    #                     help='Output directory')
    #
    # args = parser.parse_args()
    n_bands = 13
    max_combinations = 10000  # MUST use sampling - exhaustive is impossible!
    output = 'validation_results_v2/robustness'
    # Run analysis
    df_results = run_robustness_analysis(n_bands, max_combinations, output)

    print("\n" + "="*80)
    print("ROBUSTNESS ANALYSIS COMPLETE")
    print("="*80)
