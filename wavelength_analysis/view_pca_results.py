import pandas as pd

df = pd.read_excel('validation_results_v2/20251029_165432/wavelength_selection_results_v2.xlsx')

# Sort by n_combinations_selected
df = df.sort_values('n_combinations_selected')

print("\nPCA Results - Accuracy vs N Bands:")
print("="*100)
result_df = df[['config_name', 'n_combinations_selected', 'accuracy', 'f1_weighted', 'selection_time']].copy()
result_df.columns = ['Config Name', 'N Bands', 'Accuracy', 'F1 Score', 'Selection Time (s)']
print(result_df.to_string(index=False))

print("\n" + "="*100)
baseline_acc = df[df["config_name"] == "BASELINE_FULL_DATA"]["accuracy"].values[0]
pca_df = df[df["config_name"] != "BASELINE_FULL_DATA"]
best_pca_acc = pca_df["accuracy"].max()
best_pca_bands = pca_df[pca_df["accuracy"] == best_pca_acc]["n_combinations_selected"].values[0]

print(f'\nBaseline Accuracy (192 bands): {baseline_acc:.4f}')
print(f'Best PCA Accuracy: {best_pca_acc:.4f} at {int(best_pca_bands)} bands')
print(f'Accuracy drop: {(baseline_acc - best_pca_acc):.4f} ({((baseline_acc - best_pca_acc)/baseline_acc * 100):.2f}%)')
print(f'Data reduction: {((192 - best_pca_bands)/192 * 100):.1f}%')

# Show trend
print("\n" + "="*100)
print("\nAccuracy Trend (PCA only):")
pca_sorted = pca_df.sort_values('n_combinations_selected')[['n_combinations_selected', 'accuracy']]
for _, row in pca_sorted.iterrows():
    n_bands = int(row['n_combinations_selected'])
    acc = row['accuracy']
    bar_length = int((acc - 0.6) * 200)  # Scale for visualization
    bar = '#' * bar_length
    print(f"{n_bands:3d} bands: {acc:.4f} {bar}")
