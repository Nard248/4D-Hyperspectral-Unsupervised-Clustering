"""
Quick test of KNN ROI clustering functionality
"""
import sys
from pathlib import Path
import numpy as np

# Add paths
base_dir = Path(r"C:\Users\meloy\PycharmProjects\Capstone")
sys.path.append(str(base_dir))
sys.path.append(str(base_dir / "wavelength_analysis"))

from simple_knn_roi_clustering import simple_knn_roi_clustering
from concatenation_clustering import load_masked_data

def quick_test():
    print("Quick test of KNN ROI clustering...")

    try:
        # Load data
        data_path = base_dir / "data" / "processed" / "Lichens" / "lichens_data_masked.pkl"
        full_data = load_masked_data(data_path)
        print(f"Data loaded: {len(full_data['excitation_wavelengths'])} excitations")

        # Define simple ROI
        roi_regions = [
            {'name': 'Test1', 'coords': (100, 150, 100, 150), 'color': 'red'},
            {'name': 'Test2', 'coords': (200, 250, 200, 250), 'color': 'blue'},
        ]

        # Test the function
        labels, metrics, cluster_map = simple_knn_roi_clustering(
            full_data, roi_regions, n_clusters=2, verbose=True
        )

        if labels is not None:
            print(f"SUCCESS! KNN clustering worked.")
            print(f"Labels shape: {labels.shape}")
            print(f"Unique labels: {np.unique(labels)}")
            print(f"Cluster map shape: {cluster_map.shape}")
            print(f"Silhouette score: {metrics['silhouette_score']:.4f}")
            return True
        else:
            print("FAILED: No labels returned")
            return False

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_test()
    print(f"\nTest result: {'PASSED' if success else 'FAILED'}")