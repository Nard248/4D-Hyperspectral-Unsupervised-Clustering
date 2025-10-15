"""
Comprehensive logging infrastructure for paper publication
Provides detailed step-by-step logs for wavelength selection experiments
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np


class ExperimentLogger:
    """
    Comprehensive logger for wavelength selection experiments.
    Creates detailed logs that can be used directly for paper writing.
    """

    def __init__(self, base_log_dir: Path):
        """
        Initialize experiment logger

        Args:
            base_log_dir: Base directory for all logs
        """
        self.base_log_dir = Path(base_log_dir)
        self.logs_dir = self.base_log_dir / "experiment_logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Master log file
        self.master_log_path = self.logs_dir / "master_experiment_log.txt"
        self.master_log = open(self.master_log_path, 'w', encoding='utf-8')

        # Config-specific logs
        self.config_logs = {}

        # Start master log
        self._write_master_header()

    def _write_master_header(self):
        """Write header for master experiment log"""
        header = f"""
{'='*80}
WAVELENGTH SELECTION VALIDATION EXPERIMENT - MASTER LOG
{'='*80}
Experiment Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Purpose: Validate wavelength selection methods for hyperspectral data reduction
{'='*80}

"""
        self.master_log.write(header)
        self.master_log.flush()

    def log_master(self, message: str, level: str = "INFO"):
        """
        Write to master experiment log

        Args:
            message: Log message
            level: Log level (INFO, WARNING, ERROR, RESULT)
        """
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_line = f"[{timestamp}] [{level}] {message}\n"
        self.master_log.write(log_line)
        self.master_log.flush()

    def start_config_log(self, config_name: str, config_params: Dict[str, Any]):
        """
        Start a new log file for a specific configuration

        Args:
            config_name: Name of the configuration
            config_params: Configuration parameters
        """
        config_log_path = self.logs_dir / f"{config_name}_detailed_log.txt"
        config_log = open(config_log_path, 'w', encoding='utf-8')

        # Write header
        header = f"""
{'='*80}
CONFIGURATION: {config_name}
{'='*80}
Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

CONFIGURATION PARAMETERS:
{'-'*80}
"""
        config_log.write(header)

        # Write all parameters
        for key, value in sorted(config_params.items()):
            config_log.write(f"  {key:30s} = {value}\n")

        config_log.write(f"\n{'='*80}\n")
        config_log.write("EXPERIMENT EXECUTION LOG:\n")
        config_log.write(f"{'='*80}\n\n")
        config_log.flush()

        self.config_logs[config_name] = {
            'file': config_log,
            'path': config_log_path,
            'start_time': datetime.now()
        }

        self.log_master(f"Started configuration: {config_name}")

    def log_config(self, config_name: str, message: str, step: Optional[str] = None):
        """
        Write to configuration-specific log

        Args:
            config_name: Configuration name
            message: Log message
            step: Optional step identifier
        """
        if config_name not in self.config_logs:
            raise ValueError(f"Config {config_name} not initialized. Call start_config_log first.")

        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]

        if step:
            log_line = f"[{timestamp}] [STEP: {step}]\n{message}\n\n"
        else:
            log_line = f"[{timestamp}] {message}\n"

        self.config_logs[config_name]['file'].write(log_line)
        self.config_logs[config_name]['file'].flush()

    def log_dimension_selection(self, config_name: str, method: str,
                                n_dimensions: int, selection_details: Dict[str, Any]):
        """
        Log dimension selection process

        Args:
            config_name: Configuration name
            method: Selection method (variance, activation, pca)
            n_dimensions: Number of dimensions to select
            selection_details: Details about the selection process
        """
        message = f"""
STEP 1: DIMENSION SELECTION
{'-'*80}
Method: {method.upper()}
Target Dimensions: {n_dimensions}

Process:
  1. Loaded autoencoder model from checkpoint
  2. Extracted latent space representations (dimension: {selection_details.get('latent_dim', 'N/A')})
  3. Applied {method} method to identify important dimensions

Selection Criteria ({method}):
"""

        if method == 'variance':
            message += """  - Computed variance of each latent dimension across all pixels
  - Dimensions with higher variance capture more information
  - Selected top-k dimensions with highest variance
"""
        elif method == 'activation':
            message += """  - Measured average activation magnitude for each dimension
  - Higher activation indicates more important features
  - Selected top-k dimensions with highest activation
"""
        elif method == 'pca':
            message += """  - Applied PCA to latent representations
  - Selected principal components explaining most variance
  - Top-k components represent most important directions
"""

        message += f"""
Results:
  - Selected {selection_details.get('n_selected', n_dimensions)} dimensions
  - Variance explained: {selection_details.get('variance_explained', 'N/A')}
  - Cumulative importance: {selection_details.get('cumulative_importance', 'N/A')}

Mathematical Formulation:
  See mathematical_formulations.txt for detailed equations
{'-'*80}
"""
        self.log_config(config_name, message, step="DIMENSION_SELECTION")

    def log_perturbation_analysis(self, config_name: str, method: str,
                                   magnitudes: List[float], results: Dict[str, Any]):
        """
        Log perturbation analysis process

        Args:
            config_name: Configuration name
            method: Perturbation method
            magnitudes: Perturbation magnitudes tested
            results: Perturbation analysis results
        """
        message = f"""
STEP 2: PERTURBATION ANALYSIS
{'-'*80}
Method: {method.upper()}
Magnitudes: {magnitudes}
Number of Dimensions Tested: {results.get('n_dimensions', 'N/A')}

Process:
  1. For each selected dimension:
     a. Created baseline reconstruction (original latent values)
     b. Applied perturbation with each magnitude
     c. Measured reconstruction error change (MSE)
     d. Calculated influence score for dimension

  2. Perturbation Strategy ({method}):
"""

        if method == 'percentile':
            message += f"""     - Perturbed by percentile of dimension's value distribution
     - Magnitudes {magnitudes} represent percentile shifts
     - Bidirectional: both increase and decrease
"""
        elif method == 'standard_deviation':
            message += f"""     - Perturbed by standard deviations from mean
     - Magnitudes {magnitudes} represent number of std devs
     - Captures sensitivity to statistical variation
"""
        elif method == 'absolute_range':
            message += f"""     - Perturbed by absolute range percentages
     - Magnitudes {magnitudes} represent % of value range
     - Uniform perturbation magnitude
"""

        message += f"""
Results:
  - Total perturbations tested: {results.get('n_perturbations', 'N/A')}
  - Average influence score: {results.get('avg_influence', 'N/A'):.6f}
  - Max influence score: {results.get('max_influence', 'N/A'):.6f}
  - Min influence score: {results.get('min_influence', 'N/A'):.6f}

Influence Calculation:
  Influence = mean(|MSE_perturbed - MSE_baseline|) across all magnitudes
  Higher influence = dimension more important for reconstruction

Mathematical Formulation:
  See mathematical_formulations.txt for detailed equations
{'-'*80}
"""
        self.log_config(config_name, message, step="PERTURBATION_ANALYSIS")

    def log_wavelength_mapping(self, config_name: str, n_dimensions: int,
                               n_wavelengths_before: int, n_wavelengths_after: int,
                               mapping_details: Dict[str, Any]):
        """
        Log wavelength mapping process

        Args:
            config_name: Configuration name
            n_dimensions: Number of important dimensions
            n_wavelengths_before: Wavelengths before diversity filtering
            n_wavelengths_after: Wavelengths after diversity filtering
            mapping_details: Details about mapping process
        """
        message = f"""
STEP 3: WAVELENGTH MAPPING
{'-'*80}
Important Dimensions: {n_dimensions}
Initial Wavelength Candidates: {n_wavelengths_before}
Final Selected Wavelengths: {n_wavelengths_after}

Process:
  1. For each important dimension, identified wavelengths with highest contribution
  2. Analyzed decoder weights to map dimensions to spectral bands
  3. Ranked wavelengths by cumulative influence across dimensions

Mapping Strategy:
  - Used decoder layer weights to trace dimension-to-wavelength relationships
  - Weighted by dimension importance scores
  - Considered both excitation and emission wavelengths

Results (Before Diversity Filtering):
  - Total candidate wavelengths: {n_wavelengths_before}
  - Average influence score: {mapping_details.get('avg_influence_before', 'N/A')}
  - Spectral range covered: {mapping_details.get('spectral_range', 'N/A')}

{'-'*80}
"""
        self.log_config(config_name, message, step="WAVELENGTH_MAPPING")

    def log_diversity_filtering(self, config_name: str, method: str,
                                n_before: int, n_after: int,
                                diversity_params: Dict[str, Any],
                                filtering_results: Dict[str, Any]):
        """
        Log diversity filtering process

        Args:
            config_name: Configuration name
            method: Diversity method (mmr, min_distance)
            n_before: Wavelengths before filtering
            n_after: Wavelengths after filtering
            diversity_params: Diversity parameters
            filtering_results: Filtering results
        """
        message = f"""
STEP 4: DIVERSITY-BASED WAVELENGTH FILTERING
{'-'*80}
Method: {method.upper()}
Candidates Before: {n_before}
Selected After: {n_after}
Reduction: {n_before - n_after} wavelengths removed ({(1 - n_after/n_before)*100:.1f}%)

"""

        if method == 'mmr':
            lambda_val = diversity_params.get('lambda_diversity', 0.5)
            message += f"""Diversity Strategy: Maximum Marginal Relevance (MMR)
Parameters:
  - Lambda (λ): {lambda_val}
  - Balance: {"Influence-focused" if lambda_val < 0.5 else "Diversity-focused" if lambda_val > 0.5 else "Balanced"}

Process:
  1. Started with highest-influence wavelength
  2. Iteratively selected wavelengths maximizing:
     MMR = λ × Influence - (1-λ) × max(Similarity to already selected)
  3. λ controls trade-off between influence and diversity

Effect:
  - λ={lambda_val}: {int(lambda_val*100)}% weight on influence, {int((1-lambda_val)*100)}% on diversity
  - Ensures spectral coverage while maintaining high influence
"""
        elif method == 'min_distance':
            min_dist = diversity_params.get('min_distance_nm', 15.0)
            message += f"""Diversity Strategy: Minimum Distance Constraint
Parameters:
  - Minimum Distance: {min_dist} nm

Process:
  1. Sorted wavelengths by influence (descending)
  2. Selected wavelengths ensuring ≥{min_dist}nm spectral separation
  3. Greedy selection: keep if sufficiently far from all selected

Effect:
  - Enforces minimum spectral spacing
  - Prevents redundant nearby wavelengths
"""

        message += f"""
Results:
  - Final wavelength count: {n_after}
  - Average spectral spacing: {filtering_results.get('avg_spacing', 'N/A'):.2f} nm
  - Min spectral spacing: {filtering_results.get('min_spacing', 'N/A'):.2f} nm
  - Max spectral spacing: {filtering_results.get('max_spacing', 'N/A'):.2f} nm
  - Wavelength coverage: {filtering_results.get('coverage_range', 'N/A')} nm

Wavelengths Removed: {filtering_results.get('n_removed', n_before - n_after)}
Reason: {filtering_results.get('removal_reason', 'Too similar to higher-influence wavelengths')}

Mathematical Formulation:
  See mathematical_formulations.txt for detailed equations
{'-'*80}
"""
        self.log_config(config_name, message, step="DIVERSITY_FILTERING")

    def log_clustering_execution(self, config_name: str, method: str,
                                 n_features: int, n_clusters: int,
                                 execution_results: Dict[str, Any]):
        """
        Log clustering execution

        Args:
            config_name: Configuration name
            method: Clustering method (KNN, KMeans)
            n_features: Number of spectral features
            n_clusters: Number of clusters
            execution_results: Execution results
        """
        message = f"""
STEP 5: CLUSTERING EXECUTION
{'-'*80}
Method: {method}
Spectral Features: {n_features}
Number of Clusters: {n_clusters}

Process:
  1. Loaded selected wavelengths data subset
  2. Concatenated multi-excitation hyperspectral data
  3. Applied global percentile normalization
  4. Extracted ROI training samples (4 regions)
  5. Trained {method} classifier
  6. Predicted full image classification

Clustering Details:
  - Training samples: {execution_results.get('n_training_samples', 'N/A')}
  - Test samples: {execution_results.get('n_test_samples', 'N/A')}
  - Feature scaling: StandardScaler (zero mean, unit variance)
  - Classification time: {execution_results.get('clustering_time', 'N/A'):.2f}s

Results:
  - Clusters found: {execution_results.get('n_clusters_found', n_clusters)}
  - Valid pixels classified: {execution_results.get('n_classified', 'N/A')}

{'-'*80}
"""
        self.log_config(config_name, message, step="CLUSTERING_EXECUTION")

    def log_evaluation_results(self, config_name: str, metrics: Dict[str, float],
                               baseline_metrics: Dict[str, float]):
        """
        Log evaluation results and comparison with baseline

        Args:
            config_name: Configuration name
            metrics: Configuration metrics
            baseline_metrics: Baseline metrics for comparison
        """
        purity_diff = metrics['purity'] - baseline_metrics['purity']
        purity_pct = (purity_diff / baseline_metrics['purity']) * 100 if baseline_metrics['purity'] != 0 else 0

        message = f"""
STEP 6: EVALUATION RESULTS
{'-'*80}
Ground Truth Validation Metrics:

Primary Metrics:
  - Purity:          {metrics['purity']:.4f}  (Baseline: {baseline_metrics['purity']:.4f}, Δ: {purity_diff:+.4f}, {purity_pct:+.2f}%)
  - ARI:             {metrics['ari']:.4f}  (Baseline: {baseline_metrics['ari']:.4f}, Δ: {metrics['ari'] - baseline_metrics['ari']:+.4f})
  - NMI:             {metrics['nmi']:.4f}  (Baseline: {baseline_metrics['nmi']:.4f}, Δ: {metrics['nmi'] - baseline_metrics['nmi']:+.4f})

Clustering Quality Metrics:
  - Homogeneity:     {metrics.get('homogeneity', 'N/A'):.4f}
  - Completeness:    {metrics.get('completeness', 'N/A'):.4f}
  - V-Measure:       {metrics.get('v_measure', 'N/A'):.4f}

Internal Validation:
  - Silhouette:      {metrics.get('silhouette', 'N/A'):.4f}
  - Davies-Bouldin:  {metrics.get('davies_bouldin', 'N/A'):.4f}
  - Calinski-Harabasz: {metrics.get('calinski_harabasz', 'N/A'):.2f}

Performance Summary:
  - Data Reduction:  {metrics.get('data_reduction_pct', 'N/A'):.1f}%
  - Features Used:   {metrics.get('n_features', 'N/A')} / {baseline_metrics.get('n_features', 'N/A')}
  - Selection Time:  {metrics.get('selection_time', 'N/A'):.2f}s
  - Clustering Time: {metrics.get('clustering_time', 'N/A'):.2f}s
  - Speedup Factor:  {metrics.get('speedup_factor', 'N/A'):.2f}x

Interpretation:
"""

        if purity_diff > 0:
            message += f"  ✓ IMPROVED over baseline by {purity_pct:.2f}%\n"
            message += "  - Wavelength selection successfully reduced noise\n"
            message += "  - Selected wavelengths capture discriminative information\n"
        elif abs(purity_diff) < 0.01:
            message += "  ≈ COMPARABLE to baseline (within 1% margin)\n"
            message += "  - Significant data reduction with minimal quality loss\n"
            message += f"  - {metrics.get('data_reduction_pct', 'N/A'):.1f}% reduction in spectral bands\n"
        else:
            message += f"  ✗ DECREASED from baseline by {abs(purity_pct):.2f}%\n"
            message += "  - Some discriminative information lost in selection\n"
            message += "  - May indicate need for different selection strategy\n"

        message += f"\n{'-'*80}\n"

        self.log_config(config_name, message, step="EVALUATION_RESULTS")

    def finish_config_log(self, config_name: str, success: bool = True, error: str = None):
        """
        Finish and close configuration log

        Args:
            config_name: Configuration name
            success: Whether configuration completed successfully
            error: Error message if failed
        """
        if config_name not in self.config_logs:
            return

        elapsed = datetime.now() - self.config_logs[config_name]['start_time']

        footer = f"""
{'='*80}
CONFIGURATION EXECUTION {'COMPLETED' if success else 'FAILED'}
{'-'*80}
End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Duration: {elapsed.total_seconds():.2f} seconds
"""

        if not success and error:
            footer += f"\nError: {error}\n"

        footer += f"{'='*80}\n"

        self.config_logs[config_name]['file'].write(footer)
        self.config_logs[config_name]['file'].close()

        self.log_master(f"Finished configuration: {config_name} ({'SUCCESS' if success else 'FAILED'})")

    def log_experiment_summary(self, total_configs: int, successful: int,
                               failed: int, best_config: str, best_purity: float):
        """
        Log overall experiment summary

        Args:
            total_configs: Total configurations tested
            successful: Number of successful configs
            failed: Number of failed configs
            best_config: Name of best configuration
            best_purity: Best purity achieved
        """
        summary = f"""
{'='*80}
EXPERIMENT SUMMARY
{'='*80}
Configurations Tested: {total_configs}
  - Successful: {successful}
  - Failed: {failed}

Best Configuration: {best_config}
  - Purity: {best_purity:.4f}

Log Files Generated:
  - Master Log: master_experiment_log.txt
  - Config Logs: {successful} detailed logs (one per configuration)

All logs saved to: {self.logs_dir}

Experiment End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""
        self.master_log.write(summary)
        self.master_log.flush()

    def close(self):
        """Close all log files"""
        self.master_log.close()
        for config_name, log_info in self.config_logs.items():
            if not log_info['file'].closed:
                log_info['file'].close()


def export_dimension_rankings(rankings_before: List[Dict[str, Any]],
                              rankings_after: List[Dict[str, Any]],
                              config_name: str,
                              output_dir: Path):
    """
    Export dimension/wavelength rankings before and after diversity filtering

    Args:
        rankings_before: Rankings before diversity filtering
        rankings_after: Rankings after diversity filtering
        config_name: Configuration name
        output_dir: Output directory
    """
    import pandas as pd

    rankings_dir = output_dir / "dimension_rankings"
    rankings_dir.mkdir(parents=True, exist_ok=True)

    # Export before filtering
    if rankings_before:
        df_before = pd.DataFrame(rankings_before)
        before_path = rankings_dir / f"{config_name}_rankings_before_filtering.csv"
        df_before.to_csv(before_path, index=False)

    # Export after filtering
    if rankings_after:
        df_after = pd.DataFrame(rankings_after)
        after_path = rankings_dir / f"{config_name}_rankings_after_filtering.csv"
        df_after.to_csv(after_path, index=False)

        # Also export exclusion analysis
        if rankings_before and len(rankings_before) > len(rankings_after):
            selected_ids = set(r.get('wavelength_id', r.get('dimension_id')) for r in rankings_after)
            excluded = [r for r in rankings_before if r.get('wavelength_id', r.get('dimension_id')) not in selected_ids]

            df_excluded = pd.DataFrame(excluded)
            excluded_path = rankings_dir / f"{config_name}_excluded_wavelengths.csv"
            df_excluded.to_csv(excluded_path, index=False)
