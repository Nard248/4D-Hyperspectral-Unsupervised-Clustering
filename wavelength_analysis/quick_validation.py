"""
Quick validation script - just test ground truth on existing results
"""
import sys
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from ground_truth_validation import extract_ground_truth_from_png, calculate_clustering_accuracy
from concatenation_clustering import load_masked_data, concatenate_hyperspectral_data

def quick_test():
    print("Quick Ground Truth Validation Test")
    print("="*50)
    
    # Extract ground truth
    png_path = r"C:\Users\meloy\Downloads\Mask_Manual.png"
    ground_truth, color_mapping, lichen_colors = extract_ground_truth_from_png(
        png_path, target_shape=(1040, 1392)
    )
    
    print(f"\n[OK] Found {len(lichen_colors)} lichen types in PNG")
    
    # Load data quickly
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / "data" / "processed" / "Lichens" / "lichens_data_masked.pkl"
    
    if data_path.exists():
        print("[OK] Found masked data file")
        data_dict = load_masked_data(data_path)
        df, valid_mask, metadata = concatenate_hyperspectral_data(data_dict, normalize=True, scale=True)
        print(f"[OK] Concatenated {df.shape[0]} pixels with {df.shape[1]-2} spectral features")
        
        # Create a simple mock clustering result for testing
        np.random.seed(42)
        mock_cluster_labels = np.random.randint(0, len(lichen_colors), size=df.shape[0])
        
        # Validate mock clustering
        metrics = calculate_clustering_accuracy(mock_cluster_labels, ground_truth, valid_mask)
        
        print("\nMock Clustering Validation (random labels):")
        print(f"  Purity: {metrics.get('purity', 0):.4f}")
        print(f"  ARI: {metrics.get('adjusted_rand_score', 0):.4f}")
        print(f"  NMI: {metrics.get('normalized_mutual_info', 0):.4f}")
        
        print("\n[OK] Ground truth validation system is working!")
        print("\nNext steps:")
        print("1. Open the concat-clustering.ipynb notebook")
        print("2. Run all cells to get real clustering results") 
        print("3. The ground truth validation cells will show actual performance")
        
    else:
        print(f"[ERROR] Data file not found at {data_path}")

if __name__ == "__main__":
    quick_test()