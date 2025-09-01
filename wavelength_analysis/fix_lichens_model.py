"""
Fix for Lichens Model Wavelength Mismatch

This script handles the wavelength mismatch issue for Lichens data by:
1. Checking if the existing model matches the data wavelengths
2. Training a new model if needed
3. Running the wavelength analysis with the correct model
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.models import MaskedHyperspectralDataset, HyperspectralCAEWithMasking
from scripts.models.dataset import load_hyperspectral_data, load_mask
from scripts.models.training import train_with_masking
from wavelength_analysis.core.analyzer import WavelengthAnalyzer
from wavelength_analysis.core.config import AnalysisConfig


def check_model_compatibility(model_path, dataset):
    """Check if saved model is compatible with dataset wavelengths"""
    if not Path(model_path).exists():
        return False, "Model file not found"
    
    try:
        # Try to load the model
        all_data = dataset.get_all_data()
        test_model = HyperspectralCAEWithMasking(
            excitations_data={ex: data.numpy() for ex, data in all_data.items()},
            k1=20, k3=20, filter_size=5
        )
        test_model.load_state_dict(torch.load(model_path, map_location='cpu'))
        return True, "Model compatible"
    except Exception as e:
        return False, str(e)


def train_lichens_model(dataset, output_dir):
    """Train a new model for Lichens data with current wavelengths"""
    print("\n" + "="*60)
    print("Training new model for Lichens data")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize model with correct wavelengths
    all_data = dataset.get_all_data()
    model = HyperspectralCAEWithMasking(
        excitations_data={ex: data.numpy() for ex, data in all_data.items()},
        k1=20,
        k3=20,
        filter_size=5,
        sparsity_target=0.1,
        sparsity_weight=1.0,
        dropout_rate=0.5
    )
    
    print(f"Model initialized with {len(dataset.excitation_wavelengths)} excitation wavelengths:")
    print(f"  {dataset.excitation_wavelengths}")
    
    # Train the model
    print("\nTraining model (this may take 10-20 minutes)...")
    model, losses = train_with_masking(
        model=model,
        dataset=dataset,
        num_epochs=30,
        learning_rate=0.001,
        chunk_size=256,
        chunk_overlap=64,
        batch_size=1,
        device=device,
        early_stopping_patience=5,
        mask=dataset.processed_mask,
        output_dir=str(output_dir)
    )
    
    # Save the trained model
    model_path = output_dir / "lichens_wavelength_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\n[OK] Model saved to: {model_path}")
    
    return model, model_path


def run_lichens_analysis():
    """Run wavelength analysis for Lichens with proper model handling"""
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / "data" / "processed" / "Lichens" / "lichens_data_masked.pkl"
    mask_path = base_dir / "data" / "processed" / "Lichens" / "lichens_mask.npy"
    
    # Check for alternative data paths
    if not data_path.exists():
        alt_path = base_dir / "data" / "processed" / "Lichens" / "Lichens_data.pkl"
        if alt_path.exists():
            data_path = alt_path
    
    print("\n" + "="*60)
    print("LICHENS WAVELENGTH ANALYSIS - MODEL FIX")
    print("="*60)
    
    # Load data
    print("\nLoading Lichens data...")
    data_dict = load_hyperspectral_data(data_path)
    mask = load_mask(mask_path) if mask_path.exists() else None
    
    # Create dataset
    dataset = MaskedHyperspectralDataset(
        data_dict=data_dict,
        mask=mask,
        normalize=True
    )
    
    print(f"Dataset created with wavelengths: {dataset.excitation_wavelengths}")
    
    # Check model compatibility
    model_dir = base_dir / "wavelength_analysis" / "models" / "Lichens"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Try different model paths
    possible_model_paths = [
        model_dir / "lichens_wavelength_model.pth",
        base_dir / "results" / "Lichens_analysis" / "model" / "best_hyperspectral_model.pth",
        base_dir / "results" / "Lichens_analysis" / "model" / "final_hyperspectral_model.pth",
    ]
    
    compatible_model_path = None
    for model_path in possible_model_paths:
        if model_path.exists():
            is_compatible, message = check_model_compatibility(model_path, dataset)
            if is_compatible:
                compatible_model_path = model_path
                print(f"[OK] Found compatible model: {model_path}")
                break
            else:
                print(f"[X] Model at {model_path} is incompatible: wavelength mismatch")
    
    # Train new model if needed
    if compatible_model_path is None:
        print("\n[WARNING] No compatible model found. Training new model...")
        model, model_path = train_lichens_model(dataset, model_dir)
        compatible_model_path = model_path
    
    # Create configuration for analysis
    config = AnalysisConfig(
        sample_name="Lichens",
        data_path=str(data_path),
        mask_path=str(mask_path),
        model_path=str(compatible_model_path),
        output_dir=str(base_dir / "wavelength_analysis" / "results" / "Lichens"),
        dimension_selection_method="activation",
        perturbation_method="percentile",
        perturbation_magnitudes=[10, 20, 30],
        n_important_dimensions=15,
        n_bands_to_select=30,
        n_layers_to_extract=10
    )
    
    # Run analysis
    print("\n" + "="*60)
    print("Running wavelength selection analysis...")
    print("="*60)
    
    try:
        analyzer = WavelengthAnalyzer(config)
        analyzer.load_data_and_model()
        results = analyzer.run_complete_analysis()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print(f"\n[OK] Results saved to: {config.output_dir}")
        
        # Print summary
        if 'selected_bands' in results and results['selected_bands']:
            print(f"\nTop 5 selected wavelength combinations:")
            for i, band in enumerate(results['selected_bands'][:5], 1):
                print(f"  {i}. Ex: {band['excitation']:.1f}nm, Em: {band['emission_wavelength']:.1f}nm"
                      f" (Influence: {band['influence']:.2f})")
        
        return results
        
    except Exception as e:
        print(f"\n[ERROR] Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run the fixed analysis for Lichens
    results = run_lichens_analysis()
    
    if results:
        print("\n[OK] Lichens wavelength analysis completed successfully!")
        print("\nYou can now run the main analysis script and it should work for all samples.")
    else:
        print("\n[ERROR] Analysis failed. Please check the error messages above.")