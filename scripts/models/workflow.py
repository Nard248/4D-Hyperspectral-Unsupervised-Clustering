"""
Complete workflow for hyperspectral data analysis.

This module provides a complete pipeline function that runs
the entire hyperspectral data analysis workflow from loading to clustering.
"""

import os
import numpy as np
import torch

from .dataset import MaskedHyperspectralDataset, load_hyperspectral_data, load_mask
from .autoencoder import HyperspectralCAEWithMasking
from .training import train_with_masking, evaluate_model_with_masking
from .clustering import run_pixel_wise_clustering, visualize_cluster_profiles
from .visualization import create_rgb_visualization, visualize_reconstruction_comparison, overlay_clusters_on_rgb

def complete_hyperspectral_workflow(
        data_path,
        mask_path=None,
        output_dir="hyperspectral_results",
        n_clusters=5,
        excitation_to_use=None,
        normalize=True,
        downscale_factor=1,
        num_epochs=50,
        learning_rate=0.001,
        chunk_size=64,
        chunk_overlap=8,
        early_stopping_patience=5,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        calculate_metrics=True
):
    """
    Run the complete hyperspectral data analysis workflow from loading to clustering.

    Args:
        data_path: Path to the hyperspectral data pickle file
        mask_path: Optional path to a binary mask file
        output_dir: Directory to save all outputs
        n_clusters: Number of clusters for the clustering step
        excitation_to_use: Specific excitation to use for clustering
        normalize: Whether to normalize the data
        downscale_factor: Factor to downscale the spatial dimensions
        num_epochs: Number of epochs for training
        learning_rate: Learning rate for training
        chunk_size: Size of spatial chunks for processing
        chunk_overlap: Overlap between adjacent chunks
        early_stopping_patience: Patience for early stopping
        device: Device to use for computation
        calculate_metrics: Whether to calculate clustering quality metrics

    Returns:
        Dictionary with results from each step
    """
    print(f"Starting complete hyperspectral analysis workflow...")
    print(f"Using device: {device}")

    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load data and mask
    print("\n--- Step 1: Loading Data ---")
    data_dict = load_hyperspectral_data(data_path)

    mask = None
    if mask_path is not None:
        mask = load_mask(mask_path)

    # Step 2: Create dataset
    print("\n--- Step 2: Creating Dataset ---")
    dataset = MaskedHyperspectralDataset(
        data_dict=data_dict,
        mask=mask,
        normalize=normalize,
        downscale_factor=downscale_factor
    )

    # Get spatial dimensions
    height, width = dataset.get_spatial_dimensions()
    print(f"Data dimensions after processing: {height}x{width}")

    # Step 3: Create model
    print("\n--- Step 3: Creating Model ---")
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

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Step 4: Train the model
    print("\n--- Step 4: Training Model ---")
    model_dir = os.path.join(output_dir, "model")

    model, losses = train_with_masking(
        model=model,
        dataset=dataset,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        batch_size=1,
        device=device,
        early_stopping_patience=early_stopping_patience,
        mask=dataset.processed_mask,
        output_dir=model_dir
    )

    # Step 5: Evaluate the model
    print("\n--- Step 5: Evaluating Model ---")
    eval_dir = os.path.join(output_dir, "evaluation")

    evaluation_results = evaluate_model_with_masking(
        model=model,
        dataset=dataset,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        device=device,
        output_dir=eval_dir
    )

    # Step 6: Generate RGB visualizations
    print("\n--- Step 6: Creating RGB Visualizations ---")
    vis_dir = os.path.join(output_dir, "visualizations")

    # Get reconstructed data
    reconstructions = evaluation_results['reconstructions']

    # Create RGB visualizations for original data
    original_rgb = create_rgb_visualization(
        data_dict=all_data,
        emission_wavelengths=dataset.emission_wavelengths,
        mask=dataset.processed_mask,
        output_dir=vis_dir
    )

    # Create RGB visualizations for reconstructed data
    recon_rgb = create_rgb_visualization(
        data_dict=reconstructions,
        emission_wavelengths=dataset.emission_wavelengths,
        mask=dataset.processed_mask,
        output_dir=vis_dir
    )

    # For each excitation, create side-by-side comparison
    for ex in all_data:
        if ex in reconstructions:
            visualize_reconstruction_comparison(
                original_data=all_data[ex],
                reconstructed_data=reconstructions[ex],
                excitation=ex,
                emission_wavelengths=dataset.emission_wavelengths.get(ex, None),
                mask=dataset.processed_mask,
                output_dir=vis_dir
            )

    # Step 7: Run clustering
    print("\n--- Step 7: Running Pixel-wise Clustering ---")
    cluster_dir = os.path.join(output_dir, "clustering")

    cluster_results = run_pixel_wise_clustering(
        model=model,
        dataset=dataset,
        n_clusters=n_clusters,
        excitation_to_use=excitation_to_use,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        device=device,
        output_dir=cluster_dir,
        calculate_metrics=calculate_metrics
    )

    # Step 8: Analyze cluster profiles
    print("\n--- Step 8: Analyzing Cluster Profiles ---")

    cluster_stats = visualize_cluster_profiles(
        cluster_results=cluster_results,
        dataset=dataset,
        original_data=all_data,
        output_dir=cluster_dir
    )

    # Step 9: Create cluster overlay on RGB
    print("\n--- Step 9: Creating Cluster Overlay ---")

    # Determine which excitation to use for RGB background
    ex_for_rgb = excitation_to_use if excitation_to_use in original_rgb else next(iter(original_rgb.keys()))

    overlay = overlay_clusters_on_rgb(
        cluster_labels=cluster_results['cluster_labels'],
        rgb_image=original_rgb[ex_for_rgb],
        mask=dataset.processed_mask,
        output_path=os.path.join(cluster_dir, "cluster_overlay.png")
    )

    # Save results
    print("\n--- Workflow Complete! ---")
    print(f"All results saved to {output_dir}")

    # Return all results
    workflow_results = {
        'dataset': dataset,
        'model': model,
        'training_losses': losses,
        'evaluation': evaluation_results,
        'original_rgb': original_rgb,
        'reconstructed_rgb': recon_rgb,
        'clustering': cluster_results,
        'cluster_stats': cluster_stats,
        'overlay': overlay
    }

    return workflow_results