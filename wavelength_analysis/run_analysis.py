"""
Main Wavelength Analysis Runner

This is the main entry point for running comprehensive wavelength analysis
on hyperspectral data. It provides a clean, organized interface for analyzing
different sample types (Lime, Kiwi, Lichens) with various configurations.

Usage:
    python run_analysis.py --sample Lime
    python run_analysis.py --sample Kiwi --config aggressive
    python run_analysis.py --sample Lichens --config high_resolution
    python run_analysis.py --all-samples
"""

import sys
import argparse
from pathlib import Path
import json
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.analyzer import WavelengthAnalyzer
from core.config import (
    AnalysisConfig, 
    LIME_CONFIG, 
    KIWI_CONFIG, 
    LICHENS_CONFIG,
    EXPERIMENTAL_CONFIGS
)


def setup_paths():
    """Ensure all necessary paths exist"""
    base_path = Path(__file__).parent.parent
    
    # Create results directories
    results_dir = Path(__file__).parent / "results"
    for sample in ["Lime", "Kiwi", "Lichens"]:
        sample_dir = results_dir / sample
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (sample_dir / "layers").mkdir(exist_ok=True)
        (sample_dir / "visualizations").mkdir(exist_ok=True)


def get_sample_config(sample_name: str, config_name: str = "default") -> AnalysisConfig:
    """
    Get configuration for a specific sample and configuration type.
    
    Args:
        sample_name: Name of the sample (Lime, Kiwi, Lichens)
        config_name: Configuration type (default, aggressive, high_resolution, pca_based)
        
    Returns:
        AnalysisConfig object
    """
    base_configs = {
        "Lime": LIME_CONFIG,
        "Kiwi": KIWI_CONFIG, 
        "Lichens": LICHENS_CONFIG
    }
    
    if config_name == "default":
        return base_configs[sample_name]
    
    # Get experimental config and update sample name
    if config_name in EXPERIMENTAL_CONFIGS:
        config = EXPERIMENTAL_CONFIGS[config_name]
        config.sample_name = sample_name
        config.__post_init__()  # Recalculate paths
        return config
    
    raise ValueError(f"Unknown config: {config_name}")


def run_single_analysis(sample_name: str, config_name: str = "default") -> Dict[str, Any]:
    """
    Run analysis for a single sample.
    
    Args:
        sample_name: Name of the sample (Lime, Kiwi, Lichens)
        config_name: Configuration type
        
    Returns:
        Dictionary with analysis results
    """
    print(f"\n{'='*80}")
    print(f"STARTING WAVELENGTH ANALYSIS: {sample_name}")
    print(f"Configuration: {config_name}")
    print(f"{'='*80}")
    
    try:
        # Get configuration
        config = get_sample_config(sample_name, config_name)
        
        # Verify data files exist
        data_path = Path(config.data_path)
        mask_path = Path(config.mask_path)
        model_path = Path(config.model_path)
        
        if not data_path.exists():
            print(f"ERROR: Data file not found: {data_path}")
            return {"status": "error", "message": f"Data file not found: {data_path}"}
        
        if not model_path.exists():
            print(f"ERROR: Model file not found: {model_path}")
            return {"status": "error", "message": f"Model file not found: {model_path}"}
        
        # Run analysis
        analyzer = WavelengthAnalyzer(config)
        results = analyzer.run_complete_analysis()
        
        print(f"\n✓ SUCCESS: Analysis completed for {sample_name}")
        print(f"✓ Results saved to: {config.output_dir}")
        
        results["status"] = "success"
        return results
        
    except Exception as e:
        print(f"\n✗ ERROR: Analysis failed for {sample_name}: {str(e)}")
        return {"status": "error", "message": str(e), "sample": sample_name}


def run_all_samples(config_name: str = "default") -> Dict[str, Any]:
    """
    Run analysis for all available samples.
    
    Args:
        config_name: Configuration type to use for all samples
        
    Returns:
        Dictionary with results for all samples
    """
    print(f"\n{'='*80}")
    print("RUNNING WAVELENGTH ANALYSIS FOR ALL SAMPLES")
    print(f"Configuration: {config_name}")
    print(f"{'='*80}")
    
    samples = ["Lime", "Kiwi", "Lichens"]
    all_results = {}
    successful_analyses = 0
    
    for sample in samples:
        print(f"\n[{samples.index(sample)+1}/{len(samples)}] Processing {sample}...")
        
        results = run_single_analysis(sample, config_name)
        all_results[sample] = results
        
        if results["status"] == "success":
            successful_analyses += 1
        else:
            print(f"⚠ Skipping {sample} due to error: {results.get('message', 'Unknown error')}")
    
    # Save combined results summary
    summary_path = Path(__file__).parent / "results" / "analysis_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            "analysis_type": "all_samples",
            "configuration": config_name,
            "total_samples": len(samples),
            "successful_analyses": successful_analyses,
            "results": all_results
        }, f, indent=2, default=str)
    
    print(f"\n{'='*80}")
    print("ALL SAMPLES ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"✓ Successful analyses: {successful_analyses}/{len(samples)}")
    print(f"✓ Summary saved to: {summary_path}")
    
    return all_results


def run_comparison_study():
    """
    Run a comparison study across different configurations for Lime sample.
    """
    print(f"\n{'='*80}")
    print("RUNNING CONFIGURATION COMPARISON STUDY")
    print(f"{'='*80}")
    
    configs_to_test = ["default", "aggressive_std", "high_resolution", "pca_based"]
    comparison_results = {}
    
    for i, config_name in enumerate(configs_to_test):
        print(f"\n[{i+1}/{len(configs_to_test)}] Testing configuration: {config_name}")
        
        # Run analysis with modified output directory
        try:
            config = get_sample_config("Lime", config_name)
            config.output_dir = f"./wavelength_analysis/results/Lime/comparison_{config_name}"
            
            analyzer = WavelengthAnalyzer(config)
            results = analyzer.run_complete_analysis()
            
            comparison_results[config_name] = {
                "status": "success",
                "max_influence": results["performance_metrics"]["max_influence_score"],
                "compression_ratio": results["performance_metrics"]["compression_ratio"],
                "n_bands": results["performance_metrics"]["bands_selected"],
                "method": f"{config.dimension_selection_method}+{config.perturbation_method}",
                "config": config.to_dict()
            }
            
        except Exception as e:
            print(f"✗ Configuration {config_name} failed: {str(e)}")
            comparison_results[config_name] = {"status": "error", "message": str(e)}
    
    # Save comparison results
    comparison_path = Path(__file__).parent / "results" / "configuration_comparison.json"
    with open(comparison_path, 'w') as f:
        json.dump(comparison_results, f, indent=2, default=str)
    
    # Print comparison summary
    print(f"\n{'='*80}")
    print("CONFIGURATION COMPARISON RESULTS")
    print(f"{'='*80}")
    
    successful_configs = {k: v for k, v in comparison_results.items() if v["status"] == "success"}
    
    if successful_configs:
        print(f"{'Config':<20} {'Method':<20} {'Max Influence':<15} {'Compression':<12}")
        print("-" * 75)
        
        # Sort by max influence score
        sorted_configs = sorted(successful_configs.items(), 
                               key=lambda x: x[1]["max_influence"], reverse=True)
        
        for config_name, results in sorted_configs:
            print(f"{config_name:<20} {results['method']:<20} "
                  f"{results['max_influence']:<15.2e} {results['compression_ratio']:<12.1f}x")
        
        best_config = sorted_configs[0]
        print(f"\n🏆 BEST CONFIGURATION: {best_config[0]}")
        print(f"   Max influence: {best_config[1]['max_influence']:.2e}")
        print(f"   Method: {best_config[1]['method']}")
    
    print(f"✓ Comparison results saved to: {comparison_path}")
    
    return comparison_results


def print_usage():
    """Print usage information"""
    print("""
Wavelength Analysis Tool - Usage Guide
=====================================

Basic Usage:
  python run_analysis.py --sample Lime              # Run standard analysis on Lime
  python run_analysis.py --sample Kiwi              # Run analysis on Kiwi  
  python run_analysis.py --sample Lichens           # Run analysis on Lichens

Advanced Usage:
  python run_analysis.py --sample Lime --config aggressive_std     # Use aggressive perturbations
  python run_analysis.py --sample Lime --config high_resolution    # High-resolution analysis
  python run_analysis.py --sample Lime --config pca_based          # PCA-based dimension selection

Batch Operations:
  python run_analysis.py --all-samples              # Run on all samples
  python run_analysis.py --comparison               # Compare different configurations

Available Configurations:
  - default: Activation-based selection with percentile perturbations
  - aggressive_std: Variance-based with large standard deviation perturbations  
  - high_resolution: Fine-grained percentile analysis with many dimensions
  - pca_based: PCA dimension selection with absolute range perturbations

Output Structure:
  wavelength_analysis/results/{Sample}/
  ├── layers/                    # TIFF files of top wavelength layers
  ├── visualizations/           # PNG plots and charts
  ├── selected_bands.json       # Machine-readable results
  ├── selected_bands.txt        # Human-readable band list
  └── analysis_config.json      # Configuration used

For more information, see the documentation in the core/ directory.
    """)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Wavelength Analysis Tool for Hyperspectral Data",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--sample", type=str, choices=["Lime", "Kiwi", "Lichens"],
                       help="Sample to analyze")
    parser.add_argument("--config", type=str, default="default",
                       choices=["default", "aggressive_std", "high_resolution", "pca_based"],
                       help="Configuration to use")
    parser.add_argument("--all-samples", action="store_true",
                       help="Run analysis on all available samples")
    parser.add_argument("--comparison", action="store_true", 
                       help="Run comparison study across configurations")
    parser.add_argument("--help-extended", action="store_true",
                       help="Show extended usage information")
    
    args = parser.parse_args()
    
    # Setup paths
    setup_paths()
    
    # Handle extended help
    if args.help_extended:
        print_usage()
        return
    
    # Handle different modes
    if args.comparison:
        run_comparison_study()
        
    elif args.all_samples:
        run_all_samples(args.config)
        
    elif args.sample:
        run_single_analysis(args.sample, args.config)
        
    else:
        print("Please specify --sample, --all-samples, or --comparison")
        print("Use --help for options or --help-extended for detailed usage guide")


if __name__ == "__main__":
    main()