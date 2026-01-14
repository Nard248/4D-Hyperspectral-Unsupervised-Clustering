# Integration Instructions for Enhanced Exports

## Summary

I've created the following new modules:
1. `roi_visualization.py` - ROI overlay visualizations
2. `metrics_export.py` - Comprehensive metrics exports
3. Fixed `paper_visualizations.py` - Proper label mapping with Hungarian algorithm

The imports are already added to `WavelengthSelectionFinal.py` (line 82-85).

## Changes Needed in Experiment Loop

After line 1225 in `WavelengthSelectionFinal.py` where it says:
```python
print(f"    ✅ All experiment images saved to: experiments/{config_name}/")
```

Add the following code:

```python
        # ==================================================================
        # NEW: Export ROI overlay visualizations
        # ==================================================================
        print(f"    Creating ROI overlay visualizations...")

        # ROI overlay with clustering result
        create_roi_overlay_visualization(
            cluster_map=cluster_map,
            roi_regions=ROI_REGIONS,
            output_path=experiment_folder / f"{config_name}_roi_overlay.png",
            title=f"{config_name} - Clustering with ROI Overlay"
        )

        # ROI analysis report
        create_roi_analysis_report(
            cluster_map=cluster_map,
            roi_regions=ROI_REGIONS,
            output_path=experiment_folder / f"{config_name}_roi_analysis.png"
        )

        # ==================================================================
        # NEW: Export ground truth difference maps (with proper label mapping)
        # ==================================================================
        print(f"    Creating ground truth difference maps with label mapping...")

        gt_diff_stats = create_ground_truth_difference_maps(
            ground_truth=ground_truth,
            baseline_labels=cluster_map_full,
            optimized_labels=cluster_map,
            mask=np.ones_like(ground_truth, dtype=bool),
            config_name=config_name,
            output_dir=experiment_folder,
            baseline_purity=baseline_metrics['purity'],
            optimized_purity=gt_metrics['purity']
        )

        print(f"      Baseline errors: {gt_diff_stats['baseline_wrong']:,}")
        print(f"      Optimized errors: {gt_diff_stats['optimized_wrong']:,}")
        print(f"      Noise reduction: {gt_diff_stats['noise_reduction']:,} pixels ({gt_diff_stats['noise_reduction_pct']:.1f}%)")

        # ==================================================================
        # NEW: Export comprehensive metrics to Excel
        # ==================================================================
        print(f"    Exporting comprehensive metrics...")

        # Prepare all data for export
        config_params_export = {
            **config,  # Include all config parameters
            'data_reduction_pct': data_reduction_pct,
            'n_features': n_features,
            'n_features_baseline': n_features_full
        }

        timing_data_export = {
            'selection_time': selection_timer.elapsed,
            'clustering_time': cluster_timer.elapsed,
            'speedup_factor': speedup,
            'time_saved': baseline_timer.elapsed - cluster_timer.elapsed
        }

        # Export to Excel
        metrics_excel_path = experiment_folder / f"{config_name}_comprehensive_metrics.xlsx"
        export_experiment_metrics(
            config_params=config_params_export,
            wavelength_combinations=wavelength_combinations,
            clustering_metrics=metrics,
            ground_truth_metrics=gt_metrics,
            timing_data=timing_data_export,
            output_path=metrics_excel_path
        )

        # Also export quick CSV for review
        result_for_csv = {
            **result,  # Use the result dict we're already building
            **gt_diff_stats  # Add difference map statistics
        }
        csv_path = experiment_folder / f"{config_name}_metrics_summary.csv"
        export_experiment_csv(result_for_csv, csv_path)

        print(f"    ✅ All exports complete for: {config_name}")
```

## Changes Needed at End of Script

After the final summary (around line 1775), add:

```python
# ==================================================================
# NEW: Export comprehensive summary Excel for ALL experiments
# ==================================================================
print("\n" + "=" * 80)
print("EXPORTING COMPREHENSIVE SUMMARY")
print("=" * 80)

# Export summary of all experiments
summary_excel_path = results_dir / "ALL_EXPERIMENTS_SUMMARY.xlsx"
export_all_experiments_summary(
    all_results=results,
    output_path=summary_excel_path
)

print(f"\n✅ Comprehensive summary exported!")
print(f"   {summary_excel_path}")
print(f"   Contains: All results, timing data, top 10, wavelength info, statistics")
```

## What Gets Exported for Each Experiment

Each experiment folder (`experiments/{config_name}/`) will contain:

1. **ROI Visualizations:**
   - `{config_name}_roi_overlay.png` - Clustering with ROI boxes overlaid
   - `{config_name}_roi_analysis.png` - Per-ROI cluster distribution analysis

2. **Ground Truth Comparisons (with correct label mapping):**
   - `{config_name}_ground_truth_comparison.png` - 4-panel comparison
   - `{config_name}_GT_vs_Baseline.png` - Baseline vs GT difference
   - `{config_name}_GT_vs_Optimized.png` - Optimized vs GT difference
   - `{config_name}_noise_reduction.png` - Improvement analysis

3. **Comprehensive Metrics:**
   - `{config_name}_comprehensive_metrics.xlsx` - Multi-sheet Excel with all metrics
     - Configuration parameters
     - Selected wavelengths
     - Clustering metrics
     - Ground truth validation
     - Timing data
     - Summary
   - `{config_name}_metrics_summary.csv` - Quick CSV summary

4. **Existing Exports:**
   - Clustering result images
   - Comparison plots
   - Layer rankings
   - Misclassification maps

## Overall Summary File

At the end, `ALL_EXPERIMENTS_SUMMARY.xlsx` will contain:
- Sheet 1: All results sorted by purity
- Sheet 2: Timing comparison
- Sheet 3: Top 10 by purity
- Sheet 4: Wavelength combinations info
- Sheet 5: Statistical summary
- Sheet 6: Best configuration details

## Key Fix: Label Mapping

The `create_ground_truth_difference_maps()` function in `paper_visualizations.py` now includes proper label mapping using the Hungarian algorithm. This will correctly map cluster IDs to ground truth IDs before comparison, fixing the yellow ROI issue.

The mapping works like this:
1. Build contingency matrix showing overlap between predicted and ground truth clusters
2. Use Hungarian algorithm to find optimal 1-to-1 mapping
3. Remap predicted labels to ground truth label space
4. THEN compare for correct/incorrect classification

This ensures that even if your clustering assigns "cluster 0" to what ground truth calls "class 3", it will correctly identify that as a match.
