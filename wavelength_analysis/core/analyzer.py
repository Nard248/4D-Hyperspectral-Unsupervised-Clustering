"""
Enhanced Wavelength Analyzer - Main Analysis Engine

This module provides the core wavelength analysis functionality,
combining dimension selection, perturbation analysis, and layer extraction.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import json
import tifffile
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from scripts.models import MaskedHyperspectralDataset, HyperspectralCAEWithMasking
from scripts.models.dataset import load_hyperspectral_data, load_mask

from .config import AnalysisConfig
from .visualization import WavelengthVisualizer


class WavelengthAnalyzer:
    """
    Enhanced wavelength analyzer with comprehensive functionality for
    hyperspectral data analysis and wavelength selection.
    """
    
    def __init__(self, config: AnalysisConfig):
        """
        Initialize the wavelength analyzer.
        
        Args:
            config: Analysis configuration object
        """
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # Initialize data structures
        self.dataset = None
        self.model = None
        self.baseline_latent = None
        self.baseline_reconstruction = None
        self.important_dims = None
        self.influence_matrix = None
        self.selected_bands = None
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Wavelength Analyzer initialized for {config.sample_name}")
        print(f"Using device: {self.device}")
        print(f"Output directory: {self.output_dir}")
    
    def load_data_and_model(self):
        """Load hyperspectral data and trained model with compatibility checking"""
        print("\nLoading data and model...")
        
        # Load hyperspectral data
        data_dict = load_hyperspectral_data(self.config.data_path)
        mask = load_mask(self.config.mask_path) if Path(self.config.mask_path).exists() else None
        
        # Create dataset
        self.dataset = MaskedHyperspectralDataset(
            data_dict=data_dict,
            mask=mask,
            normalize=True
        )
        
        # Load model with compatibility check
        all_data = self.dataset.get_all_data()
        self.model = HyperspectralCAEWithMasking(
            excitations_data={ex: data.numpy() for ex, data in all_data.items()},
            k1=20, k3=20, filter_size=5
        )
        
        # Try to load the model state dict
        try:
            self.model.load_state_dict(torch.load(self.config.model_path, map_location=self.device))
            self.model = self.model.to(self.device)
            print(f"Dataset loaded: {len(self.dataset.excitation_wavelengths)} excitations")
            print(f"Model loaded and moved to {self.device}")
        except FileNotFoundError as e:
            print(f"\nWarning: Model file not found: {self.config.model_path}")
            print(f"Training a new model from scratch...")
            self._train_compatible_model()
        except RuntimeError as e:
            if "Missing key(s)" in str(e) or "size mismatch" in str(e):
                print(f"\nWARNING: Model wavelengths don't match data wavelengths!")
                print(f"  Data wavelengths: {self.dataset.excitation_wavelengths}")
                print(f"  This usually means the model was trained on different wavelength data.")
                print(f"\nAttempting to train a new compatible model...")
                
                # Train a new model if there's a mismatch
                self._train_compatible_model()
            else:
                raise e
    
    def setup_baseline(self):
        """Setup baseline latent representations for analysis"""
        print("\nSetting up baseline data...")
        
        all_data = self.dataset.get_all_data()
        height, width = self.dataset.get_spatial_dimensions()
        mask = self.dataset.processed_mask
        
        # Create patches for analysis
        patch_size = self.config.patch_size
        stride = self.config.patch_stride
        
        patches_data = {}
        patch_coords = []
        
        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                if mask is not None:
                    patch_mask = mask[y:y+patch_size, x:x+patch_size]
                    valid_ratio = np.sum(patch_mask) / (patch_size * patch_size)
                    if valid_ratio <= 0.5:
                        continue
                
                patch_coords.append((y, x))
                if len(patch_coords) >= self.config.n_baseline_patches:
                    break
            if len(patch_coords) >= self.config.n_baseline_patches:
                break
        
        print(f"Selected {len(patch_coords)} patches of size {patch_size}x{patch_size}")
        
        # Extract patches for each excitation
        for ex in all_data:
            patches = []
            for y, x in patch_coords:
                patch = all_data[ex][y:y+patch_size, x:x+patch_size, :]
                patches.append(patch)
            patches_data[ex] = torch.stack(patches)
        
        # Generate baseline latent representations
        with torch.no_grad():
            patches_data_device = {ex: data.to(self.device) for ex, data in patches_data.items()}
            self.baseline_latent = self.model.encode(patches_data_device)
            self.baseline_reconstruction = self.model.decode(self.baseline_latent)
        
        print(f"Baseline latent shape: {self.baseline_latent.shape}")
    
    def select_important_dimensions(self):
        """Select most important latent dimensions"""
        print(f"\nSelecting important dimensions using {self.config.dimension_selection_method} method...")
        
        latent = self.baseline_latent
        batch_size, n_channels, n_latent, h_latent, w_latent = latent.shape
        
        # Flatten spatial dimensions for analysis
        latent_flat = latent.reshape(batch_size, -1)
        
        if self.config.dimension_selection_method == 'variance':
            # Select by variance across samples
            importance_scores = torch.var(latent_flat, dim=0)
            
        elif self.config.dimension_selection_method == 'activation':
            # Select by mean absolute activation
            importance_scores = torch.mean(torch.abs(latent_flat), dim=0)
            
        elif self.config.dimension_selection_method == 'pca':
            # Use PCA to find most important dimensions
            scaler = StandardScaler()
            latent_scaled = scaler.fit_transform(latent_flat.cpu().numpy())
            n_samples, n_features = latent_scaled.shape
            n_components = min(self.config.n_important_dimensions * 2, n_features, n_samples - 1)
            
            pca = PCA(n_components=n_components)
            pca.fit(latent_scaled)
            
            # Use PCA loadings to identify important original dimensions
            components = torch.tensor(pca.components_)
            importance_scores = torch.sum(torch.abs(components), dim=0)
        
        else:
            raise ValueError(f"Unknown dimension selection method: {self.config.dimension_selection_method}")
        
        # Convert to coordinate-based format
        coordinate_importance = []
        for i, score in enumerate(importance_scores):
            coords = np.unravel_index(i, (n_channels, n_latent, h_latent, w_latent))
            coords = tuple(int(c) for c in coords)
            coordinate_importance.append((score.item(), coords))
        
        # Sort by importance and return top N
        coordinate_importance.sort(reverse=True, key=lambda x: x[0])
        self.important_dims = coordinate_importance[:self.config.n_important_dimensions]
        
        print(f"Selected top {len(self.important_dims)} dimensions")
        print(f"Top 5 dimensions:")
        for i, (score, coords) in enumerate(self.important_dims[:5]):
            print(f"   {i+1}. {coords}: importance = {score:.6f}")
    
    def compute_influence_scores(self):
        """Compute influence scores through latent perturbation"""
        print(f"\nComputing influence scores using {self.config.perturbation_method} method...")
        
        # Initialize influence matrix
        self.influence_matrix = {}
        for ex in self.model.excitation_wavelengths:
            n_bands = len(self.dataset.emission_wavelengths.get(ex, range(self.model.emission_bands[ex])))
            self.influence_matrix[ex] = np.zeros(n_bands)
        
        # Get latent dimensions
        batch_size, n_channels, n_latent, h_latent, w_latent = self.baseline_latent.shape
        
        # Calculate statistics for perturbations
        latent_flat = self.baseline_latent.reshape(batch_size, -1)
        stats = self._calculate_latent_statistics(latent_flat)
        
        total_perturbations = 0
        magnitudes = self.config.perturbation_magnitudes
        directions = self.config.perturbation_directions
        
        print(f"Analyzing {len(self.important_dims)} dimensions with magnitudes {magnitudes}")
        
        for dim_idx, (importance_score, coords) in enumerate(self.important_dims):
            c, l, h, w = coords
            print(f"  Dimension {dim_idx+1}/{len(self.important_dims)}: {coords}, importance: {importance_score:.6f}")
            
            for magnitude in magnitudes:
                for direction in directions:
                    if direction == 'bidirectional':
                        test_directions = [-1, 1]
                    elif direction == 'positive':
                        test_directions = [1]
                    elif direction == 'negative':
                        test_directions = [-1]
                    
                    for sign in test_directions:
                        perturbation_amount = self._calculate_perturbation_amount(
                            coords, magnitude, sign, stats, latent_flat.shape
                        )
                        
                        # Apply perturbation
                        perturbed_latent = self.baseline_latent.clone()
                        perturbed_latent[:, c, l, h, w] += perturbation_amount
                        
                        # Measure influence
                        influence = self._measure_band_influence(perturbed_latent, importance_score)
                        
                        # Accumulate influence
                        weight = 1.0 if len(test_directions) == 1 else 0.5
                        for ex in influence:
                            self.influence_matrix[ex] += influence[ex] * weight
                        
                        total_perturbations += 1
        
        print(f"Completed {total_perturbations} perturbations")
        
        # Apply normalization
        if self.config.normalization_method != 'none':
            self._normalize_influences()
    
    def _calculate_latent_statistics(self, latent_flat):
        """Calculate statistics for perturbation scaling"""
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        stats = {
            'mean': torch.mean(latent_flat, dim=0),
            'std': torch.std(latent_flat, dim=0),
            'min': torch.min(latent_flat, dim=0)[0],
            'max': torch.max(latent_flat, dim=0)[0],
            'percentiles': {}
        }
        
        for p in percentiles:
            stats['percentiles'][p] = torch.quantile(latent_flat, p/100.0, dim=0)
        
        return stats
    
    def _calculate_perturbation_amount(self, coords, magnitude, sign, stats, shape):
        """Calculate perturbation amount based on method"""
        c, l, h, w = coords
        batch_size, n_features = shape
        n_channels, n_latent, h_latent, w_latent = self.baseline_latent.shape[1:]
        
        # Calculate flat index
        flat_idx = c * (n_latent * h_latent * w_latent) + l * (h_latent * w_latent) + h * w_latent + w
        
        if self.config.perturbation_method == 'percentile':
            target_percentile = 50 + sign * magnitude/2
            closest_percentile = min(stats['percentiles'].keys(), 
                                   key=lambda x: abs(x - target_percentile))
            target_value = stats['percentiles'][closest_percentile][flat_idx]
            current_mean = torch.mean(self.baseline_latent[:, c, l, h, w])
            return target_value - current_mean
            
        elif self.config.perturbation_method == 'standard_deviation':
            std_val = stats['std'][flat_idx]
            return sign * (magnitude / 100.0) * std_val
            
        elif self.config.perturbation_method == 'absolute_range':
            value_range = stats['max'][flat_idx] - stats['min'][flat_idx]
            return sign * (magnitude / 100.0) * value_range
    
    def _measure_band_influence(self, perturbed_latent, importance_weight=1.0):
        """Measure the influence of perturbation on each emission band"""
        influence = {}
        
        with torch.no_grad():
            perturbed_recon = self.model.decode(perturbed_latent)
            
            for ex in self.model.excitation_wavelengths:
                if ex in self.baseline_reconstruction:
                    baseline = self.baseline_reconstruction[ex]
                    perturbed = perturbed_recon[ex]
                    
                    # Calculate mean absolute difference across batch and spatial dimensions
                    band_differences = torch.mean(torch.abs(perturbed - baseline), dim=(0, 1, 2))
                    influence[ex] = band_differences.cpu().numpy() * importance_weight
        
        return influence
    
    def _normalize_influences(self):
        """Normalize influence scores"""
        print(f"Applying {self.config.normalization_method} normalization...")
        
        if self.config.normalization_method == 'variance':
            all_data = self.dataset.get_all_data()
            for ex in self.influence_matrix:
                if ex in all_data:
                    band_vars = np.var(all_data[ex].numpy(), axis=(0, 1))
                    band_vars[band_vars == 0] = 1  # Avoid division by zero
                    self.influence_matrix[ex] = self.influence_matrix[ex] / band_vars
                    
        elif self.config.normalization_method == 'max_per_excitation':
            for ex in self.influence_matrix:
                max_inf = np.max(self.influence_matrix[ex])
                if max_inf > 0:
                    self.influence_matrix[ex] = self.influence_matrix[ex] / max_inf
    
    def select_top_bands(self):
        """Select top wavelength combinations based on influence scores"""
        print(f"\nSelecting top {self.config.n_bands_to_select} wavelength combinations...")

        all_combinations = []

        for ex in self.influence_matrix:
            emission_wavelengths = self.dataset.emission_wavelengths.get(ex, None)

            for band_idx, influence in enumerate(self.influence_matrix[ex]):
                if emission_wavelengths and band_idx < len(emission_wavelengths):
                    em_wavelength = emission_wavelengths[band_idx]
                else:
                    em_wavelength = band_idx  # Fallback to index

                all_combinations.append({
                    'excitation': ex,
                    'emission_idx': band_idx,
                    'emission_wavelength': em_wavelength,
                    'influence': influence,
                    'rank': 0  # Will be set later
                })

        # Sort by influence and assign ranks
        all_combinations.sort(key=lambda x: x['influence'], reverse=True)
        for i, combo in enumerate(all_combinations):
            combo['rank'] = i + 1

        # Apply diversity constraint if enabled
        if self.config.use_diversity_constraint:
            print(f"Applying diversity constraint: {self.config.diversity_method}")
            if self.config.diversity_method == "mmr":
                self.selected_bands = self._select_bands_mmr(all_combinations)
            elif self.config.diversity_method == "min_distance":
                self.selected_bands = self._select_bands_min_distance(all_combinations)
            else:
                self.selected_bands = all_combinations[:self.config.n_bands_to_select]
        else:
            self.selected_bands = all_combinations[:self.config.n_bands_to_select]

        print(f"Selected {len(self.selected_bands)} bands")
        print(f"Influence range: {self.selected_bands[-1]['influence']:.2e} to {self.selected_bands[0]['influence']:.2e}")

    def _select_bands_mmr(self, all_combinations):
        """
        Maximum Marginal Relevance (MMR) selection for wavelength diversity.
        Balances influence (relevance) with spectral diversity.
        """
        print(f"  Using MMR with lambda={self.config.lambda_diversity}")

        # Get full hyperspectral data for computing spectral profiles
        all_data = self.dataset.get_all_data()

        # Build mapping from combination to spectral profile
        band_profiles = {}
        for ex in all_data:
            ex_data = all_data[ex].numpy()
            for band_idx in range(ex_data.shape[-1]):
                # Mean spectral profile across spatial locations
                profile = ex_data[:, :, band_idx].flatten()
                key = (ex, band_idx)
                band_profiles[key] = profile

        # Normalize profiles for correlation computation
        from sklearn.preprocessing import normalize
        for key in band_profiles:
            profile = band_profiles[key].reshape(1, -1)
            band_profiles[key] = normalize(profile, axis=1).flatten()

        # Start with highest influence band
        selected_bands = [all_combinations[0]]
        selected_keys = [(all_combinations[0]['excitation'], all_combinations[0]['emission_idx'])]

        print(f"  Initial selection: Ex{all_combinations[0]['excitation']:.0f} Em{all_combinations[0]['emission_wavelength']:.1f}nm")

        # Iterative MMR selection
        max_influence = all_combinations[0]['influence']

        while len(selected_bands) < self.config.n_bands_to_select:
            best_mmr_score = -np.inf
            best_combo = None
            best_key = None

            for combo in all_combinations:
                combo_key = (combo['excitation'], combo['emission_idx'])

                # Skip if already selected
                if combo_key in selected_keys:
                    continue

                # Skip if profile not available
                if combo_key not in band_profiles:
                    continue

                # Relevance: normalized influence score
                relevance = combo['influence'] / max_influence if max_influence > 0 else 0

                # Diversity: maximum similarity to any selected band
                max_similarity = 0
                for sel_key in selected_keys:
                    if sel_key in band_profiles:
                        # Cosine similarity between spectral profiles
                        similarity = np.dot(band_profiles[combo_key], band_profiles[sel_key])
                        max_similarity = max(max_similarity, abs(similarity))

                # MMR score: relevance - λ × max_similarity
                mmr_score = relevance - self.config.lambda_diversity * max_similarity

                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_combo = combo
                    best_key = combo_key

            if best_combo is not None:
                selected_bands.append(best_combo)
                selected_keys.append(best_key)

                if len(selected_bands) % 5 == 0:
                    print(f"  Selected {len(selected_bands)}/{self.config.n_bands_to_select}: "
                          f"Ex{best_combo['excitation']:.0f} Em{best_combo['emission_wavelength']:.1f}nm "
                          f"(MMR score: {best_mmr_score:.4f})")
            else:
                # No more valid candidates
                print(f"  Warning: Only found {len(selected_bands)} bands with diversity constraint")
                break

        return selected_bands

    def _select_bands_min_distance(self, all_combinations):
        """
        Minimum distance constraint selection.
        Ensures selected wavelengths are at least min_distance_nm apart.
        """
        print(f"  Using minimum distance constraint: {self.config.min_distance_nm} nm")

        selected_bands = []

        for combo in all_combinations:
            # Check if this wavelength is far enough from already-selected ones
            is_valid = True

            for selected in selected_bands:
                # Only check distance for same excitation wavelength
                if combo['excitation'] == selected['excitation']:
                    distance = abs(combo['emission_wavelength'] - selected['emission_wavelength'])
                    if distance < self.config.min_distance_nm:
                        is_valid = False
                        break

            if is_valid:
                selected_bands.append(combo)

                if len(selected_bands) % 5 == 0:
                    print(f"  Selected {len(selected_bands)}/{self.config.n_bands_to_select}: "
                          f"Ex{combo['excitation']:.0f} Em{combo['emission_wavelength']:.1f}nm")

            if len(selected_bands) >= self.config.n_bands_to_select:
                break

        if len(selected_bands) < self.config.n_bands_to_select:
            print(f"  Warning: Only found {len(selected_bands)} bands satisfying distance constraint")

        return selected_bands

    def extract_wavelength_layers(self):
        """Extract and save wavelength layers as TIFF files"""
        if not self.config.save_tiff_layers:
            return
        
        print(f"\nExtracting top {self.config.n_layers_to_extract} wavelength layers...")
        
        # Create layers subdirectory
        layers_dir = self.output_dir / "layers"
        layers_dir.mkdir(exist_ok=True)
        
        # Get full spatial data
        all_data = self.dataset.get_all_data()
        layer_info = []
        
        for i, band in enumerate(self.selected_bands[:self.config.n_layers_to_extract]):
            ex = band['excitation']
            band_idx = band['emission_idx']
            em_wavelength = band['emission_wavelength']
            influence = band['influence']
            
            print(f"  Layer {i+1}: Ex {ex}nm, Em {em_wavelength}nm, Influence: {influence:.6f}")
            
            if ex in all_data:
                # Extract the specific emission band for this excitation
                layer_data = all_data[ex][:, :, band_idx].numpy()
                
                # Handle NaN values
                layer_data = np.nan_to_num(layer_data, nan=0.0)
                
                # Normalize to 0-1 range
                layer_min = np.min(layer_data)
                layer_max = np.max(layer_data)
                if layer_max > layer_min:
                    layer_normalized = (layer_data - layer_min) / (layer_max - layer_min)
                else:
                    layer_normalized = np.zeros_like(layer_data)
                
                # Convert to 16-bit for TIFF
                layer_16bit = (layer_normalized * 65535).astype(np.uint16)
                
                # Create filename
                filename = f"layer_{i+1:02d}_ex{ex:.0f}nm_em{em_wavelength:.0f}nm_inf{influence:.6f}.tiff"
                filepath = layers_dir / filename
                
                # Save as TIFF
                tifffile.imwrite(str(filepath), layer_16bit)
                
                # Store metadata
                layer_info.append({
                    'rank': i + 1,
                    'excitation_nm': float(ex),
                    'emission_nm': float(em_wavelength),
                    'emission_band_index': int(band_idx),
                    'influence_score': float(influence),
                    'data_range_original': [float(layer_min), float(layer_max)],
                    'filename': filename
                })
        
        # Save layer metadata
        metadata_path = layers_dir / "layer_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                'layers': layer_info,
                'extraction_date': datetime.now().isoformat(),
                'config': self.config.to_dict()
            }, f, indent=2)
        
        print(f"Extracted {len(layer_info)} layers to {layers_dir}")
        return layer_info
    
    def save_results(self):
        """Save comprehensive analysis results"""
        print(f"\nSaving analysis results...")
        
        # Save configuration
        config_path = self.output_dir / "analysis_config.json"
        self.config.save(str(config_path))
        
        # Save selected bands
        bands_path = self.output_dir / "selected_bands.json"
        with open(bands_path, 'w') as f:
            # Convert to JSON-serializable format
            json_bands = []
            for band in self.selected_bands:
                json_bands.append({
                    'rank': int(band['rank']),
                    'excitation_nm': float(band['excitation']),
                    'emission_nm': float(band['emission_wavelength']),
                    'emission_band_index': int(band['emission_idx']),
                    'influence_score': float(band['influence'])
                })
            
            json.dump({
                'analysis_timestamp': datetime.now().isoformat(),
                'sample_name': self.config.sample_name,
                'method_summary': {
                    'dimension_selection': self.config.dimension_selection_method,
                    'perturbation_method': self.config.perturbation_method,
                    'normalization': self.config.normalization_method
                },
                'selected_wavelength_combinations': json_bands,
                'performance_metrics': {
                    'total_bands_available': sum(self.model.emission_bands[ex] for ex in self.model.excitation_wavelengths),
                    'bands_selected': len(self.selected_bands),
                    'compression_ratio': sum(self.model.emission_bands[ex] for ex in self.model.excitation_wavelengths) / len(self.selected_bands),
                    'max_influence_score': float(max(band['influence'] for band in self.selected_bands)),
                    'min_influence_score': float(min(band['influence'] for band in self.selected_bands))
                }
            }, f, indent=2)
        
        # Save human-readable band list
        bands_txt_path = self.output_dir / "selected_bands.txt"
        with open(bands_txt_path, 'w') as f:
            f.write(f"Wavelength Analysis Results: {self.config.sample_name}\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Method: {self.config.dimension_selection_method} + {self.config.perturbation_method}\n")
            f.write(f"Total bands selected: {len(self.selected_bands)} out of {sum(self.model.emission_bands[ex] for ex in self.model.excitation_wavelengths)}\n")
            f.write(f"Compression ratio: {sum(self.model.emission_bands[ex] for ex in self.model.excitation_wavelengths) / len(self.selected_bands):.2f}x\n\n")
            f.write(f"{'Rank':<5} {'Excitation(nm)':<15} {'Emission(nm)':<15} {'Influence':<15}\n")
            f.write("-" * 60 + "\n")
            
            for band in self.selected_bands:
                f.write(f"{band['rank']:<5} {band['excitation']:<15.1f} {band['emission_wavelength']:<15.1f} {band['influence']:<15.6f}\n")
        
        print(f"Results saved to {self.output_dir}")
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations"""
        if not self.config.save_visualizations:
            return
        
        print(f"\nGenerating visualizations...")
        
        # Create visualizations subdirectory
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # Initialize visualizer
        visualizer = WavelengthVisualizer(self.selected_bands, self.influence_matrix, viz_dir)
        
        # Generate all visualizations
        visualizer.create_influence_heatmap()
        visualizer.create_wavelength_scatter()
        visualizer.create_excitation_distribution()
        visualizer.create_summary_dashboard()
        
        print(f"Visualizations saved to {viz_dir}")
    
    def run_complete_analysis(self) -> Dict:
        """
        Run the complete wavelength analysis pipeline.
        
        Returns:
            Dictionary containing all analysis results
        """
        print("="*80)
        print(f"WAVELENGTH ANALYSIS: {self.config.sample_name}")
        print("="*80)
        
        # Execute analysis pipeline
        self.load_data_and_model()
        self.setup_baseline()
        self.select_important_dimensions()
        self.compute_influence_scores()
        self.select_top_bands()
        
        # Generate outputs
        layer_info = self.extract_wavelength_layers()
        self.save_results()
        self.generate_visualizations()
        
        # Compile results summary
        results = {
            'config': self.config.to_dict(),
            'selected_bands': self.selected_bands,
            'layer_info': layer_info,
            'performance_metrics': {
                'total_bands_available': sum(self.model.emission_bands[ex] for ex in self.model.excitation_wavelengths),
                'bands_selected': len(self.selected_bands),
                'compression_ratio': sum(self.model.emission_bands[ex] for ex in self.model.excitation_wavelengths) / len(self.selected_bands),
                'max_influence_score': float(max(band['influence'] for band in self.selected_bands)),
                'min_influence_score': float(min(band['influence'] for band in self.selected_bands))
            },
            'output_directory': str(self.output_dir)
        }
        
        print("="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print(f"Sample: {self.config.sample_name}")
        print(f"Selected {len(self.selected_bands)} wavelength combinations")
        print(f"Compression ratio: {results['performance_metrics']['compression_ratio']:.1f}x")
        print(f"Max influence score: {results['performance_metrics']['max_influence_score']:.2e}")
        print(f"Results saved to: {self.output_dir}")
        
        return results
    
    def _train_compatible_model(self):
        """Train a new model compatible with current data wavelengths"""
        from scripts.models.training import train_with_masking
        
        print("\n" + "="*60)
        print("Training new compatible model")
        print("="*60)
        
        # Create model save directory
        model_dir = self.output_dir.parent.parent / "models" / self.config.sample_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize new model
        all_data = self.dataset.get_all_data()
        self.model = HyperspectralCAEWithMasking(
            excitations_data={ex: data.numpy() for ex, data in all_data.items()},
            k1=20,
            k3=20,
            filter_size=5,
            sparsity_target=0.1,
            sparsity_weight=1.0,
            dropout_rate=0.5
        )
        
        print(f"Model initialized for wavelengths: {self.dataset.excitation_wavelengths}")
        print("Training model (this may take 10-20 minutes)...")
        
        # Train the model
        self.model, losses = train_with_masking(
            model=self.model,
            dataset=self.dataset,
            num_epochs=30,
            learning_rate=0.001,
            chunk_size=256,
            chunk_overlap=64,
            batch_size=1,
            device=self.device,
            early_stopping_patience=5,
            mask=self.dataset.processed_mask,
            output_dir=str(model_dir)
        )
        
        # Save the trained model
        new_model_path = model_dir / f"{self.config.sample_name.lower()}_wavelength_model.pth"
        torch.save(self.model.state_dict(), new_model_path)
        
        # Update config to use new model
        self.config.model_path = str(new_model_path)
        
        print(f"\nNew model trained and saved to: {new_model_path}")
        print(f"Model ready for wavelength analysis")