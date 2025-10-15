"""
Metrics Export Module
Exports detailed experiment metrics and parameters to CSV/Excel files
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime


def export_experiment_metrics(
    config_params,
    wavelength_combinations,
    clustering_metrics,
    ground_truth_metrics,
    timing_data,
    output_path
):
    """
    Export comprehensive experiment metrics to Excel file.

    Args:
        config_params: Configuration parameters dictionary
        wavelength_combinations: List of selected wavelength combinations
        clustering_metrics: Clustering quality metrics
        ground_truth_metrics: Ground truth validation metrics
        timing_data: Timing information
        output_path: Path to save Excel file
    """
    output_path = Path(output_path)

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:

        # Sheet 1: Configuration Parameters
        config_df = pd.DataFrame([{
            'Parameter': k,
            'Value': str(v)
        } for k, v in config_params.items()])
        config_df.to_excel(writer, sheet_name='Configuration', index=False)

        # Sheet 2: Selected Wavelengths
        if wavelength_combinations:
            wl_data = []
            for idx, combo in enumerate(wavelength_combinations, 1):
                wl_data.append({
                    'Rank': idx,
                    'Excitation_nm': combo.get('excitation', 'N/A'),
                    'Emission_nm': combo.get('emission', 'N/A'),
                    'Combination': combo.get('combination_name', 'N/A')
                })
            wl_df = pd.DataFrame(wl_data)
            wl_df.to_excel(writer, sheet_name='Selected_Wavelengths', index=False)

        # Sheet 3: Clustering Metrics
        clustering_data = {
            'Metric': [],
            'Value': []
        }
        for key, value in clustering_metrics.items():
            clustering_data['Metric'].append(key)
            clustering_data['Value'].append(value)
        clustering_df = pd.DataFrame(clustering_data)
        clustering_df.to_excel(writer, sheet_name='Clustering_Metrics', index=False)

        # Sheet 4: Ground Truth Validation
        gt_data = {
            'Metric': [],
            'Value': []
        }
        for key, value in ground_truth_metrics.items():
            if key != 'confusion_matrix' and key != 'cluster_to_gt_mapping':
                gt_data['Metric'].append(key)
                gt_data['Value'].append(value)
        gt_df = pd.DataFrame(gt_data)
        gt_df.to_excel(writer, sheet_name='Ground_Truth_Validation', index=False)

        # Sheet 5: Timing Data
        timing_df = pd.DataFrame([{
            'Stage': k,
            'Time_seconds': v
        } for k, v in timing_data.items()])
        timing_df.to_excel(writer, sheet_name='Timing', index=False)

        # Sheet 6: Summary
        summary_data = {
            'Category': ['Configuration', 'Performance', 'Quality', 'Efficiency'],
            'Key_Metric': [
                config_params.get('name', 'Unknown'),
                f"{wavelength_combinations.__len__() if wavelength_combinations else 0} wavelengths",
                f"Purity: {ground_truth_metrics.get('purity', 0):.4f}",
                f"Data Reduction: {config_params.get('data_reduction_pct', 0):.1f}%"
            ],
            'Details': [
                f"{config_params.get('dimension_selection_method', 'N/A')} + {config_params.get('diversity_method', 'N/A')}",
                f"Selection: {timing_data.get('selection_time', 0):.2f}s, Clustering: {timing_data.get('clustering_time', 0):.2f}s",
                f"ARI: {ground_truth_metrics.get('adjusted_rand_score', 0):.4f}, NMI: {ground_truth_metrics.get('normalized_mutual_info', 0):.4f}",
                f"Features: {clustering_metrics.get('n_features', 0)}"
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

    print(f"  Exported metrics to: {output_path.name}")
    return output_path


def export_all_experiments_summary(all_results, output_path):
    """
    Export summary of all experiments to a single Excel file.

    Args:
        all_results: List of result dictionaries from all experiments
        output_path: Path to save Excel file
    """
    output_path = Path(output_path)

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:

        # Sheet 1: Main Results Summary
        main_cols = [
            'config_name', 'dimension_method', 'perturbation_method',
            'n_important_dims', 'n_combinations_selected', 'n_features',
            'data_reduction_pct', 'purity', 'ari', 'nmi', 'v_measure',
            'homogeneity', 'completeness', 'silhouette',
            'davies_bouldin', 'calinski_harabasz'
        ]

        # Filter to only include columns that exist
        available_cols = [col for col in main_cols if col in all_results[0]]
        df_main = pd.DataFrame(all_results)[available_cols]
        df_main = df_main.sort_values('purity', ascending=False)
        df_main.to_excel(writer, sheet_name='All_Results', index=False)

        # Sheet 2: Timing Data
        if 'selection_time' in all_results[0]:
            timing_cols = ['config_name', 'selection_time', 'clustering_time',
                          'speedup_factor', 'time_saved']
            timing_available = [col for col in timing_cols if col in all_results[0]]
            df_timing = pd.DataFrame(all_results)[timing_available]
            df_timing = df_timing.sort_values('clustering_time', ascending=True)
            df_timing.to_excel(writer, sheet_name='Timing_Comparison', index=False)

        # Sheet 3: Top 10 by Purity
        top_10 = df_main.nlargest(min(10, len(df_main)), 'purity')
        top_10.to_excel(writer, sheet_name='Top_10_Purity', index=False)

        # Sheet 4: Wavelength Combinations
        if 'wavelength_combinations' in all_results[0]:
            wl_data = []
            for result in all_results:
                wl_data.append({
                    'Config': result['config_name'],
                    'N_Wavelengths': result.get('n_combinations_selected', 0),
                    'Combinations': str(result.get('wavelength_combinations', ''))[:200] + '...',
                    'Purity': result.get('purity', 0)
                })
            df_wl = pd.DataFrame(wl_data)
            df_wl = df_wl.sort_values('Purity', ascending=False)
            df_wl.to_excel(writer, sheet_name='Wavelength_Info', index=False)

        # Sheet 5: Statistical Summary
        numeric_cols = df_main.select_dtypes(include=['number']).columns
        summary_stats = df_main[numeric_cols].describe()
        summary_stats.to_excel(writer, sheet_name='Statistical_Summary')

        # Sheet 6: Best Configuration Details
        best_idx = df_main['purity'].idxmax()
        best_config = all_results[best_idx]

        best_details = {
            'Metric': list(best_config.keys()),
            'Value': [str(v) for v in best_config.values()]
        }
        df_best = pd.DataFrame(best_details)
        df_best.to_excel(writer, sheet_name='Best_Configuration', index=False)

    print(f"\n✅ Exported comprehensive summary: {output_path.name}")
    return output_path


def export_experiment_csv(result_dict, output_path):
    """
    Export single experiment results to CSV for quick review.

    Args:
        result_dict: Dictionary with all experiment results
        output_path: Path to save CSV file
    """
    output_path = Path(output_path)

    # Flatten nested dictionaries
    flat_data = {}
    for key, value in result_dict.items():
        if isinstance(value, (dict, list)):
            flat_data[key] = str(value)
        else:
            flat_data[key] = value

    df = pd.DataFrame([flat_data])
    df.to_csv(output_path, index=False)

    print(f"  Exported CSV: {output_path.name}")
    return output_path
