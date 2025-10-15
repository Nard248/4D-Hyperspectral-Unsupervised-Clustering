"""
Simple Runner Script for Comprehensive Visualization Pipeline
=============================================================
This script can be run after the main wavelength validation pipeline
to generate all publication-ready visualizations.

Usage:
------
1. From command line:
   python run_visualizations.py <results_directory>

2. From Python/Notebook:
   from run_visualizations import generate_all_visualizations
   generate_all_visualizations(results_dir="path/to/results")
"""

import sys
from pathlib import Path
import argparse
from comprehensive_visualization_pipeline import run_comprehensive_visualization_pipeline


def find_latest_results_dir(base_dir="wavelength_analysis/validation_results"):
    """Find the most recent results directory"""
    base_path = Path(base_dir)

    if not base_path.exists():
        return None

    # Find all timestamped directories
    result_dirs = [d for d in base_path.iterdir() if d.is_dir()]

    if not result_dirs:
        return None

    # Sort by modification time, return most recent
    latest = sorted(result_dirs, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    return latest


def generate_all_visualizations(
    results_dir=None,
    hyperspectral_data_path=None,
    ground_truth_path=None,
    auto_find_latest=True
):
    """
    Convenience function to generate all visualizations.

    Parameters:
    -----------
    results_dir : str or Path, optional
        Path to results directory. If None and auto_find_latest=True,
        will find the most recent results directory.
    hyperspectral_data_path : str or Path, optional
        Path to hyperspectral data file
    ground_truth_path : str or Path, optional
        Path to ground truth file
    auto_find_latest : bool
        If True and results_dir is None, automatically find latest results

    Returns:
    --------
    output_dirs : dict
        Dictionary of output directory paths
    """

    # Find results directory
    if results_dir is None:
        if auto_find_latest:
            results_dir = find_latest_results_dir()
            if results_dir is None:
                print("❌ No results directory found!")
                print("Please run the main wavelength validation pipeline first,")
                print("or specify results_dir explicitly.")
                return None
            print(f"📁 Using latest results directory: {results_dir}")
        else:
            print("❌ Please specify results_dir")
            return None

    results_dir = Path(results_dir)

    # Verify results directory exists and has required files
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return None

    excel_path = results_dir / "wavelength_selection_results.xlsx"
    if not excel_path.exists():
        excel_files = list(results_dir.glob("*.xlsx"))
        if not excel_files:
            print(f"❌ No Excel results file found in {results_dir}")
            print("Please ensure the pipeline has completed successfully.")
            return None

    # Load optional data
    hyperspectral_data = None
    ground_truth = None

    if hyperspectral_data_path:
        import pickle
        print(f"📂 Loading hyperspectral data from: {hyperspectral_data_path}")
        with open(hyperspectral_data_path, 'rb') as f:
            hyperspectral_data = pickle.load(f)

    if ground_truth_path:
        import numpy as np

        gt_path = Path(ground_truth_path)
        print(f"📂 Loading ground truth from: {gt_path}")

        # Handle different file formats
        if gt_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
            # Load image file
            try:
                from PIL import Image
                img = Image.open(gt_path)
                ground_truth = np.array(img)
                print(f"  Loaded as image: {ground_truth.shape}")
            except ImportError:
                # Fallback to imageio or matplotlib
                try:
                    import imageio
                    ground_truth = imageio.imread(gt_path)
                    print(f"  Loaded as image: {ground_truth.shape}")
                except ImportError:
                    import matplotlib.pyplot as plt
                    ground_truth = plt.imread(gt_path)
                    print(f"  Loaded as image: {ground_truth.shape}")
        elif gt_path.suffix.lower() in ['.npy', '.npz']:
            # Load numpy file
            ground_truth = np.load(gt_path, allow_pickle=True)
            print(f"  Loaded as numpy array: {ground_truth.shape}")
        else:
            print(f"  ⚠️ Unknown file format: {gt_path.suffix}")
            print(f"  Trying np.load with allow_pickle=True...")
            ground_truth = np.load(gt_path, allow_pickle=True)

    # Run the visualization pipeline
    output_dirs = run_comprehensive_visualization_pipeline(
        results_dir=results_dir,
        hyperspectral_data=hyperspectral_data,
        ground_truth=ground_truth,
        create_all=True
    )

    return output_dirs


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive visualizations for wavelength analysis results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use latest results automatically
  python run_visualizations.py

  # Specify results directory
  python run_visualizations.py path/to/results/20250920_150313

  # Include hyperspectral data for spectral analysis
  python run_visualizations.py --data data/processed/Lichens/lichens_data_masked.pkl
        """
    )

    parser.add_argument(
        "results_dir",
        type=str,
        nargs='?',
        default=None,
        help="Path to results directory (auto-detects latest if not specified)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to hyperspectral data file"
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default=None,
        help="Path to ground truth file"
    )
    parser.add_argument(
        "--no-auto-find",
        action="store_true",
        help="Disable automatic finding of latest results"
    )

    args = parser.parse_args()

    # Run visualization pipeline
    output_dirs = generate_all_visualizations(
        results_dir=args.results_dir,
        hyperspectral_data_path=args.data,
        ground_truth_path=args.ground_truth,
        auto_find_latest=not args.no_auto_find
    )

    if output_dirs:
        print("\n✨ Visualization pipeline completed successfully!")
        return 0
    else:
        print("\n❌ Visualization pipeline failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
