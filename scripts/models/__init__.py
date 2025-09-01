"""
Hyperspectral Data Modeling Framework

This package provides a comprehensive framework for hyperspectral data processing,
including masked autoencoder training and efficient clustering.
"""

# Make key classes and functions easily importable
from .dataset import MaskedHyperspectralDataset, load_hyperspectral_data, load_mask
from .autoencoder import HyperspectralCAEWithMasking
from .training import train_with_masking, evaluate_model_with_masking, create_spatial_chunks, merge_chunk_reconstructions
from .clustering import extract_encoded_features, optimize_kmeans_clustering, run_pixel_wise_clustering, \
    visualize_cluster_profiles, evaluate_clustering_quality, run_4d_pixel_wise_clustering, \
    run_4d_pixel_wise_clustering_setup_color
from .visualization import (create_rgb_visualization, visualize_reconstruction_comparison, overlay_clusters_on_rgb,
                            overlay_clusters_with_consistent_colors, visualize_4d_cluster_profiles_consistent)
from .workflow import complete_hyperspectral_workflow
from .wavelength_selector import WavelengthSelector, select_informative_wavelengths

__all__ = [
    # Dataset classes
    'MaskedHyperspectralDataset', 'load_hyperspectral_data', 'load_mask',

    # Model classes
    'HyperspectralCAEWithMasking',

    # Training and evaluation
    'train_with_masking', 'evaluate_model_with_masking',
    'create_spatial_chunks', 'merge_chunk_reconstructions',

    # Clustering
    'extract_encoded_features', 'optimize_kmeans_clustering',
    'run_pixel_wise_clustering', 'visualize_cluster_profiles',
    'evaluate_clustering_quality',
    'run_4d_pixel_wise_clustering','run_4d_pixel_wise_clustering_setup_color',

    # Visualization
    'create_rgb_visualization', 'visualize_reconstruction_comparison',
    'overlay_clusters_on_rgb',
    'overlay_clusters_with_consistent_colors', 'visualize_4d_cluster_profiles_consistent',
    # Full workflow
    'complete_hyperspectral_workflow',

    # Wavelength selector
    'WavelengthSelector', 'select_informative_wavelengths'
]