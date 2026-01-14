import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 11

# File paths
pca_file = r"C:\Users\meloy\PycharmProjects\Capstone\wavelength_analysis\validation_results_v2\PCA\wavelength_selection_results_v2.xlsx"
autoencoder_file = r"C:\Users\meloy\PycharmProjects\Capstone\wavelength_analysis\validation_results_v2\1Dimensions\wavelength_selection_results_v2.xlsx"

# Check if files exist
if not os.path.exists(pca_file):
    print(f"ERROR: PCA file not found: {pca_file}")
    print("Looking for latest PCA results...")
    # Try to find the latest results
    latest_dir = max([d for d in os.listdir("validation_results_v2") if d.startswith("2025")])
    pca_file = f"validation_results_v2/{latest_dir}/wavelength_selection_results_v2.xlsx"
    print(f"Using: {pca_file}")

if not os.path.exists(autoencoder_file):
    print(f"ERROR: Autoencoder file not found: {autoencoder_file}")
    exit(1)

print("Loading data...")
print(f"PCA file: {pca_file}")
print(f"Autoencoder file: {autoencoder_file}")

# Load data
df_pca = pd.read_excel(pca_file)
df_auto = pd.read_excel(autoencoder_file)

print(f"\nPCA data: {len(df_pca)} rows")
print(f"Autoencoder data: {len(df_auto)} rows")

# Filter out baseline and sort by number of bands
pca_data = df_pca[df_pca['config_name'] != 'BASELINE_FULL_DATA'].copy()
auto_data = df_auto[df_auto['config_name'] != 'BASELINE_FULL_DATA'].copy()

pca_data = pca_data.sort_values('n_combinations_selected')
auto_data = auto_data.sort_values('n_combinations_selected')

# Get baseline accuracy
baseline_accuracy = df_pca[df_pca['config_name'] == 'BASELINE_FULL_DATA']['accuracy'].values[0]
print(f"\nBaseline accuracy: {baseline_accuracy:.4f}")

# ============================================================================
# PLOT 1: PCA vs Autoencoder Comparison
# ============================================================================
print("\nCreating Plot 1: PCA vs Autoencoder Comparison...")

fig, ax = plt.subplots(figsize=(14, 6))

# Plot both methods
ax.plot(pca_data['n_combinations_selected'], pca_data['accuracy'],
        marker='o', linewidth=2, markersize=6, label='PCA Selection',
        color='#2E86AB', alpha=0.8)

ax.plot(auto_data['n_combinations_selected'], auto_data['accuracy'],
        marker='s', linewidth=2, markersize=6, label='Autoencoder + Perturbation (1D)',
        color='#A23B72', alpha=0.8)

# Add baseline reference line
ax.axhline(y=baseline_accuracy, color='red', linestyle='--',
           linewidth=2, label=f'Baseline (192 bands): {baseline_accuracy:.4f}', alpha=0.7)

# Add tolerance levels
tolerance_levels = [0.95, 0.97, 0.99]
colors_tol = ['#27AE60', '#F39C12', '#E74C3C']
for tol, color in zip(tolerance_levels, colors_tol):
    ax.axhline(y=baseline_accuracy * tol, color=color, linestyle=':',
               linewidth=1.5, alpha=0.5, label=f'{int(tol*100)}% of baseline')

# Labels and title
ax.set_xlabel('Number of Selected Wavelength Bands', fontsize=13, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
ax.set_title('PCA vs Autoencoder: Wavelength Selection Performance',
             fontsize=15, fontweight='bold', pad=20)

# Legend
ax.legend(loc='lower right', fontsize=10, framealpha=0.9)

# Grid
ax.grid(True, alpha=0.3)

# Set y-axis limits for better visibility
ax.set_ylim(0.55, 0.90)

# Store statistics for text output (don't show on plot)
max_pca_acc = pca_data['accuracy'].max()
max_auto_acc = auto_data['accuracy'].max()
max_pca_bands = pca_data[pca_data['accuracy'] == max_pca_acc]['n_combinations_selected'].values[0]
max_auto_bands = auto_data[auto_data['accuracy'] == max_auto_acc]['n_combinations_selected'].values[0]

# Print statistics as text instead of on plot
print("\n" + "="*60)
print("PLOT 1 STATISTICS:")
print("="*60)
print(f'Best PCA: {max_pca_acc:.4f} @ {int(max_pca_bands)} bands')
print(f'Best Autoencoder: {max_auto_acc:.4f} @ {int(max_auto_bands)} bands')
print(f'Difference: {abs(max_auto_acc - max_pca_acc):.4f}')
print("="*60)

plt.tight_layout()
plt.savefig('validation_results_v2/plot1_pca_vs_autoencoder_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: validation_results_v2/plot1_pca_vs_autoencoder_comparison.png")
plt.close()

# ============================================================================
# PLOT 2: Tolerance Region Analysis for 1D Case
# ============================================================================
print("\nCreating Plot 2: Tolerance Region Analysis (1D Case)...")

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Define tolerance levels
tolerance_levels = [0.95, 0.97, 0.99]
tolerance_colors = ['#27AE60', '#F39C12', '#E74C3C']
tolerance_labels = ['95% of baseline', '97% of baseline', '99% of baseline']

# Calculate minimum bands needed for each tolerance level
min_bands_required = []
data_reduction_pct = []

print(f"\nTolerance Analysis for 1D Autoencoder:")
print("="*60)

for tol, label in zip(tolerance_levels, tolerance_labels):
    threshold = baseline_accuracy * tol
    # Find configurations that meet the threshold
    meeting_threshold = auto_data[auto_data['accuracy'] >= threshold]

    if len(meeting_threshold) > 0:
        min_bands = meeting_threshold['n_combinations_selected'].min()
        min_bands_required.append(min_bands)
        reduction = (1 - min_bands / 192) * 100
        data_reduction_pct.append(reduction)

        print(f"{label}:")
        print(f"  Threshold: {threshold:.4f}")
        print(f"  Min bands required: {int(min_bands)}")
        print(f"  Data reduction: {reduction:.2f}%")
        print(f"  Configs meeting threshold: {len(meeting_threshold)}")
    else:
        print(f"{label}: No configurations meet this threshold")
        min_bands_required.append(192)  # Full data
        data_reduction_pct.append(0)

# Subplot 1: Minimum Bands Required to Reach Tolerance Levels
ax1.plot(tolerance_levels, min_bands_required, marker='o', linewidth=3,
         markersize=12, color='#2E86AB', markerfacecolor='#F39C12',
         markeredgewidth=2, markeredgecolor='#2E86AB')

# Add horizontal reference line for full data
ax1.axhline(y=192, color='red', linestyle='--', linewidth=2,
            label='Full data (192 bands)', alpha=0.7)

# Add value labels on points
for i, (tol, bands) in enumerate(zip(tolerance_levels, min_bands_required)):
    ax1.text(tol, bands + 3, f'{int(bands)} bands',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

ax1.set_xlabel('Accuracy Tolerance (fraction of baseline)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Minimum Bands Required', fontsize=13, fontweight='bold')
ax1.set_title('Minimum Bands to Reach Tolerance Levels\n(1D Autoencoder)',
              fontsize=14, fontweight='bold', pad=15)
ax1.set_xticks(tolerance_levels)
ax1.set_xticklabels([f'{int(t*100)}%' for t in tolerance_levels])
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)
ax1.set_ylim(0, 210)

# Subplot 2: Data Reduction at Tolerance Levels
ax2.plot(tolerance_levels, data_reduction_pct, marker='^', linewidth=3,
         markersize=12, color='#27AE60', markerfacecolor='#F39C12',
         markeredgewidth=2, markeredgecolor='#27AE60')

# Add value labels on points
for i, (tol, reduction) in enumerate(zip(tolerance_levels, data_reduction_pct)):
    ax2.text(tol, reduction + 0.5, f'{reduction:.1f}%',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

ax2.set_xlabel('Accuracy Tolerance (fraction of baseline)', fontsize=13, fontweight='bold')
ax2.set_ylabel('Data Reduction (%)', fontsize=13, fontweight='bold')
ax2.set_title('Data Reduction at Tolerance Levels\n(1D Autoencoder)',
              fontsize=14, fontweight='bold', pad=15)
ax2.set_xticks(tolerance_levels)
ax2.set_xticklabels([f'{int(t*100)}%' for t in tolerance_levels])
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 100)

plt.tight_layout()
plt.savefig('validation_results_v2/plot2_tolerance_region_analysis_1d.png', dpi=300, bbox_inches='tight')
print("Saved: validation_results_v2/plot2_tolerance_region_analysis_1d.png")
plt.close()

# ============================================================================
# Additional: Detailed Tolerance Region Analysis
# ============================================================================
print("\nCreating additional analysis: Tolerance Regions Distribution...")

fig, ax = plt.subplots(figsize=(14, 7))

# Plot the accuracy curve
ax.plot(auto_data['n_combinations_selected'], auto_data['accuracy'],
        marker='o', linewidth=2.5, markersize=7, color='#2E86AB',
        label='1D Autoencoder Performance', zorder=3)

# Color regions by tolerance level
baseline_line = ax.axhline(y=baseline_accuracy, color='red', linestyle='--',
                           linewidth=2, label=f'Baseline: {baseline_accuracy:.4f}',
                           alpha=0.7, zorder=2)

# Add colored regions for tolerance levels
y_bottom = 0.55
tolerance_levels_extended = [0, 0.95, 0.97, 0.99, 1.0]
region_colors = ['#E8E8E8', '#D5F4E6', '#FCE8C1', '#F9D5D3', '#FFFFFF']
region_labels = ['<95%', '95-97%', '97-99%', '99-100%', '>100%']

for i in range(len(tolerance_levels_extended) - 1):
    lower = baseline_accuracy * tolerance_levels_extended[i] if i > 0 else y_bottom
    upper = baseline_accuracy * tolerance_levels_extended[i + 1]

    ax.axhspan(lower, upper, alpha=0.3, color=region_colors[i],
               label=f'{region_labels[i]} of baseline', zorder=1)

# Count how many configurations fall in each region
print("\nDistribution of configurations by tolerance region:")
print("="*60)
for i in range(len(tolerance_levels_extended) - 1):
    if i == 0:
        lower_bound = 0
        upper_bound = baseline_accuracy * 0.95
    else:
        lower_bound = baseline_accuracy * tolerance_levels_extended[i]
        upper_bound = baseline_accuracy * tolerance_levels_extended[i + 1]

    count = len(auto_data[(auto_data['accuracy'] >= lower_bound) &
                          (auto_data['accuracy'] < upper_bound)])
    print(f"{region_labels[i]}: {count} configurations")

ax.set_xlabel('Number of Selected Wavelength Bands', fontsize=13, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
ax.set_title('Tolerance Region Distribution for 1D Autoencoder',
             fontsize=15, fontweight='bold', pad=20)
ax.legend(loc='lower right', fontsize=11, framealpha=0.95, edgecolor='black', fancybox=True)
ax.grid(True, alpha=0.3, zorder=0)

# Set y-axis limits closer to actual data range for better visibility
min_accuracy = auto_data['accuracy'].min()
ax.set_ylim(min_accuracy - 0.01, 0.90)  # Start just below minimum accuracy

plt.tight_layout()
plt.savefig('validation_results_v2/plot2b_tolerance_regions_distribution.png', dpi=300, bbox_inches='tight')
print("Saved: validation_results_v2/plot2b_tolerance_regions_distribution.png")
plt.close()

# ============================================================================
# Summary Statistics
# ============================================================================
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

print("\nPCA Method:")
print(f"  Best accuracy: {pca_data['accuracy'].max():.4f}")
print(f"  Worst accuracy: {pca_data['accuracy'].min():.4f}")
print(f"  Mean accuracy: {pca_data['accuracy'].mean():.4f}")
print(f"  Configurations tested: {len(pca_data)}")

print("\n1D Autoencoder Method:")
print(f"  Best accuracy: {auto_data['accuracy'].max():.4f}")
print(f"  Worst accuracy: {auto_data['accuracy'].min():.4f}")
print(f"  Mean accuracy: {auto_data['accuracy'].mean():.4f}")
print(f"  Configurations tested: {len(auto_data)}")

print("\nAccuracy Difference:")
print(f"  Best: Autoencoder better by {(auto_data['accuracy'].max() - pca_data['accuracy'].max()):.4f}")
print(f"  Mean: Autoencoder better by {(auto_data['accuracy'].mean() - pca_data['accuracy'].mean()):.4f}")

print("\n" + "="*60)
print("All plots created successfully!")
print("="*60)
