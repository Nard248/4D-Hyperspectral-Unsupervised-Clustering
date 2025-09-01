"""
Wavelength Selection Module - Quick Prototype
Identifies most informative excitation-emission layer combinations
through latent space perturbation analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import seaborn as sns
from tqdm import tqdm
import os


class WavelengthSelector:
    """
    Identifies the most informative wavelength combinations by analyzing
    latent space perturbations and their effects on reconstruction.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 dataset,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the wavelength selector.
        
        Args:
            model: Trained HyperspectralCAEWithMasking model
            dataset: MaskedHyperspectralDataset
            device: Computing device
        """
        self.model = model.to(device)
        self.dataset = dataset
        self.device = device
        
        # Storage for analysis results
        self.baseline_latent = None
        self.baseline_reconstruction = None
        self.latent_importance_scores = None
        self.band_influence_matrix = None
        self.selected_bands = None
        
    def step1_extract_baseline(self, n_samples: int = 1000) -> Dict:
        """
        Step 1 & 2: Extract baseline latent representations and reconstructions.
        
        Args:
            n_samples: Number of spatial patches to sample for analysis
            
        Returns:
            Dictionary containing baseline data
        """
        print("Step 1: Extracting baseline latent space and reconstructions...")
        
        self.model.eval()
        all_data = self.dataset.get_all_data()
        
        # Work with full spatial data instead of individual pixels
        height, width = self.dataset.get_spatial_dimensions()
        mask = self.dataset.processed_mask
        
        # Create patch-based sampling (using smaller patches)
        patch_size = 32  # Use 32x32 patches
        stride = 16      # Overlapping patches
        
        patches_data = {}
        patch_coords = []
        
        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                # Check if patch has enough valid pixels
                patch_mask = mask[y:y+patch_size, x:x+patch_size]
                valid_ratio = np.sum(patch_mask) / (patch_size * patch_size)
                
                if valid_ratio > 0.5:  # Only use patches with >50% valid pixels
                    patch_coords.append((y, x))
                    
                    # Limit number of patches
                    if len(patch_coords) >= min(n_samples, 50):  # Limit to reasonable number
                        break
            if len(patch_coords) >= min(n_samples, 50):
                break
        
        print(f"  Selected {len(patch_coords)} patches of size {patch_size}x{patch_size}")
        
        # Extract patches for each excitation
        for ex in all_data:
            patches = []
            for y, x in patch_coords:
                patch = all_data[ex][y:y+patch_size, x:x+patch_size, :]
                patches.append(patch)
            patches_data[ex] = torch.stack(patches)
        
        # Get baseline latent representations and reconstructions
        with torch.no_grad():
            # Move data to device
            patches_data_device = {ex: data.to(self.device) for ex, data in patches_data.items()}
            
            # Encode to latent space
            self.baseline_latent = self.model.encode(patches_data_device)
            
            # Decode back to get baseline reconstruction
            self.baseline_reconstruction = self.model.decode(self.baseline_latent)
        
        print(f"  Baseline latent shape: {self.baseline_latent.shape}")
        print(f"  Processed {len(patch_coords)} patches")
        
        return {
            'latent': self.baseline_latent,
            'reconstruction': self.baseline_reconstruction,
            'patch_coords': patch_coords,
            'patches_data': patches_data
        }
    
    def step2_score_latent_dimensions(self, method: str = 'reconstruction') -> np.ndarray:
        """
        Step 3: Score importance of each latent dimension.
        
        Args:
            method: Scoring method ('reconstruction', 'variance', 'ablation')
            
        Returns:
            Array of importance scores for each latent dimension
        """
        print(f"Step 2: Scoring latent dimensions using {method} method...")
        
        if self.baseline_latent is None:
            raise ValueError("Run step1_extract_baseline first!")
        
        n_samples, n_channels, n_latent, h_latent, w_latent = self.baseline_latent.shape
        latent_dims = n_channels * n_latent * h_latent * w_latent
        
        if method == 'reconstruction':
            # Score by reconstruction error when zeroing each dimension
            scores = []
            
            for c in range(n_channels):
                for l in range(n_latent):
                    for h in range(h_latent):
                        for w in range(w_latent):
                            # Create perturbed latent with this dimension zeroed
                            perturbed_latent = self.baseline_latent.clone()
                            perturbed_latent[:, c, l, h, w] = 0
                            
                            # Decode and measure reconstruction error
                            with torch.no_grad():
                                perturbed_recon = self.model.decode(perturbed_latent)
                            
                            # Calculate MSE increase
                            mse_increase = 0
                            for ex in self.baseline_reconstruction:
                                orig = self.baseline_reconstruction[ex]
                                pert = perturbed_recon[ex]
                                mse_increase += F.mse_loss(orig, pert).item()
                            
                            scores.append((mse_increase, (c, l, h, w)))
            
            # Sort by importance
            scores.sort(reverse=True, key=lambda x: x[0])
            self.latent_importance_scores = scores
            
        elif method == 'variance':
            # Score by variance across samples
            scores = []
            
            for c in range(n_channels):
                for l in range(n_latent):
                    for h in range(h_latent):
                        for w in range(w_latent):
                            variance = torch.var(self.baseline_latent[:, c, l, h, w]).item()
                            scores.append((variance, (c, l, h, w)))
            
            scores.sort(reverse=True, key=lambda x: x[0])
            self.latent_importance_scores = scores
            
        elif method == 'ablation':
            # Score by average activation magnitude
            scores = []
            
            for c in range(n_channels):
                for l in range(n_latent):
                    for h in range(h_latent):
                        for w in range(w_latent):
                            avg_activation = torch.mean(torch.abs(
                                self.baseline_latent[:, c, l, h, w])).item()
                            scores.append((avg_activation, (c, l, h, w)))
            
            scores.sort(reverse=True, key=lambda x: x[0])
            self.latent_importance_scores = scores
        
        print(f"  Top 5 most important latent dimensions:")
        for i in range(min(5, len(scores))):
            score, coords = scores[i]
            print(f"    Dimension {coords}: score = {score:.4f}")
        
        return scores
    
    def step3_compute_band_influence(self, 
                                     n_top_latent: int = 5,
                                     epsilon: float = 0.1) -> np.ndarray:
        """
        Step 4, 5, 6: Perturb important latent dimensions and track band changes.
        
        Args:
            n_top_latent: Number of top latent dimensions to analyze
            epsilon: Perturbation magnitude
            
        Returns:
            Influence matrix [excitation, emission_band]
        """
        print(f"Step 3: Computing band influence from top {n_top_latent} latent dimensions...")
        
        if self.latent_importance_scores is None:
            raise ValueError("Run step2_score_latent_dimensions first!")
        
        # Initialize influence matrix
        excitations = self.model.excitation_wavelengths
        influence_matrix = {}
        
        for ex in excitations:
            n_bands = self.model.emission_bands[ex]
            influence_matrix[ex] = np.zeros(n_bands)
        
        # Analyze top latent dimensions
        for i in range(min(n_top_latent, len(self.latent_importance_scores))):
            score, (c, l, h, w) = self.latent_importance_scores[i]
            
            print(f"  Analyzing latent dimension {i+1}/{n_top_latent}: {(c,l,h,w)}")
            
            # Create positive and negative perturbations
            latent_plus = self.baseline_latent.clone()
            latent_minus = self.baseline_latent.clone()
            
            # Get current value and std for scaling
            current_values = self.baseline_latent[:, c, l, h, w]
            std = torch.std(current_values).item() + 1e-8
            
            # Apply scaled perturbation
            latent_plus[:, c, l, h, w] += epsilon * std
            latent_minus[:, c, l, h, w] -= epsilon * std
            
            # Decode perturbed latents
            with torch.no_grad():
                recon_plus = self.model.decode(latent_plus)
                recon_minus = self.model.decode(latent_minus)
            
            # Compute finite difference for each excitation-emission pair
            for ex in excitations:
                # Get reconstructions
                baseline = self.baseline_reconstruction[ex]
                plus = recon_plus[ex]
                minus = recon_minus[ex]
                
                # Compute change per emission band
                delta = (plus - minus) / (2 * epsilon * std)
                
                # Average absolute change across batch and spatial dimensions  
                band_changes = torch.mean(torch.abs(delta), dim=(0, 1, 2)).cpu().numpy()
                
                # Accumulate influence
                influence_matrix[ex] += band_changes * score
        
        self.band_influence_matrix = influence_matrix
        
        return influence_matrix
    
    def step4_select_top_bands(self, 
                               n_bands: int = 10,
                               normalize: str = 'variance') -> List[Tuple[float, int]]:
        """
        Select top N most influential excitation-emission combinations.
        
        Args:
            n_bands: Number of bands to select
            normalize: Normalization method ('variance', 'max', 'none')
            
        Returns:
            List of (excitation, emission_band_idx) tuples
        """
        print(f"Step 4: Selecting top {n_bands} bands with {normalize} normalization...")
        
        if self.band_influence_matrix is None:
            raise ValueError("Run step3_compute_band_influence first!")
        
        # Normalize influence scores
        normalized_influence = {}
        
        for ex in self.band_influence_matrix:
            influences = self.band_influence_matrix[ex].copy()
            
            if normalize == 'variance':
                # Normalize by band variance in original data
                all_data = self.dataset.get_all_data()
                band_vars = np.var(all_data[ex].numpy(), axis=(0, 1))
                band_vars[band_vars == 0] = 1  # Avoid division by zero
                influences = influences / band_vars
                
            elif normalize == 'max':
                # Normalize by max influence per excitation
                max_inf = np.max(influences)
                if max_inf > 0:
                    influences = influences / max_inf
            
            normalized_influence[ex] = influences
        
        # Create ranked list of all excitation-emission combinations
        all_combinations = []
        
        for ex in normalized_influence:
            emission_wavelengths = self.dataset.emission_wavelengths.get(ex, None)
            for band_idx, influence in enumerate(normalized_influence[ex]):
                if emission_wavelengths:
                    em_wavelength = emission_wavelengths[band_idx]
                else:
                    em_wavelength = band_idx
                
                all_combinations.append({
                    'excitation': ex,
                    'emission_idx': band_idx,
                    'emission_wavelength': em_wavelength,
                    'influence': influence
                })
        
        # Sort by influence
        all_combinations.sort(key=lambda x: x['influence'], reverse=True)
        
        # Select top N
        self.selected_bands = all_combinations[:n_bands]
        
        print(f"\nTop {n_bands} selected excitation-emission combinations:")
        for i, band in enumerate(self.selected_bands):
            print(f"  {i+1}. Ex: {band['excitation']:.1f}nm, "
                  f"Em: {band['emission_wavelength']:.1f}nm, "
                  f"Influence: {band['influence']:.4f}")
        
        return self.selected_bands
    
    def visualize_results(self, output_dir: str = './wavelength_selection_results'):
        """
        Create comprehensive visualizations of the analysis results.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Latent importance scores
        if self.latent_importance_scores:
            plt.figure(figsize=(10, 6))
            scores = [s[0] for s in self.latent_importance_scores[:20]]
            plt.bar(range(len(scores)), scores)
            plt.xlabel('Latent Dimension Rank')
            plt.ylabel('Importance Score')
            plt.title('Top 20 Most Important Latent Dimensions')
            plt.savefig(os.path.join(output_dir, 'latent_importance.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        # 2. Band influence heatmap
        if self.band_influence_matrix:
            # Create matrix for heatmap
            excitations = sorted(self.band_influence_matrix.keys())
            max_bands = max(len(self.band_influence_matrix[ex]) for ex in excitations)
            
            influence_array = np.zeros((len(excitations), max_bands))
            for i, ex in enumerate(excitations):
                influences = self.band_influence_matrix[ex]
                influence_array[i, :len(influences)] = influences
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(influence_array, 
                       xticklabels=range(max_bands),
                       yticklabels=[f"{ex:.0f}" for ex in excitations],
                       cmap='YlOrRd',
                       cbar_kws={'label': 'Influence Score'})
            plt.xlabel('Emission Band Index')
            plt.ylabel('Excitation Wavelength (nm)')
            plt.title('Band Influence Matrix')
            plt.savefig(os.path.join(output_dir, 'band_influence_heatmap.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        # 3. Selected bands visualization
        if self.selected_bands:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Scatter plot of selected bands
            ex_values = [b['excitation'] for b in self.selected_bands]
            em_values = [b['emission_wavelength'] for b in self.selected_bands]
            influences = [b['influence'] for b in self.selected_bands]
            
            scatter = ax1.scatter(ex_values, em_values, c=influences, 
                                 s=100, cmap='viridis', edgecolors='black', linewidth=1)
            ax1.set_xlabel('Excitation Wavelength (nm)')
            ax1.set_ylabel('Emission Wavelength (nm)')
            ax1.set_title(f'Top {len(self.selected_bands)} Selected Bands')
            plt.colorbar(scatter, ax=ax1, label='Influence Score')
            
            # Bar plot of influences
            ax2.bar(range(len(self.selected_bands)), influences)
            ax2.set_xlabel('Band Rank')
            ax2.set_ylabel('Influence Score')
            ax2.set_title('Influence Scores of Selected Bands')
            ax2.set_xticks(range(0, len(self.selected_bands), max(1, len(self.selected_bands)//10)))
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'selected_bands.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"\nVisualizations saved to {output_dir}")
    
    def validate_selection(self, retrain_model: bool = False):
        """
        Validate the selected bands by comparing clustering metrics.
        
        Args:
            retrain_model: Whether to retrain a model with selected bands only
            
        Returns:
            Validation metrics
        """
        print("\nValidating selected bands...")
        
        if self.selected_bands is None:
            raise ValueError("Run step4_select_top_bands first!")
        
        # Create mask for selected bands
        selected_mask = {}
        for ex in self.model.excitation_wavelengths:
            n_bands = self.model.emission_bands[ex]
            mask = np.zeros(n_bands, dtype=bool)
            
            for band in self.selected_bands:
                if band['excitation'] == ex:
                    mask[band['emission_idx']] = True
            
            selected_mask[ex] = mask
        
        # Count coverage
        total_selected = sum(band['excitation'] in selected_mask and 
                           selected_mask[band['excitation']][band['emission_idx']] 
                           for band in self.selected_bands)
        
        total_bands = sum(self.model.emission_bands[ex] for ex in self.model.excitation_wavelengths)
        
        print(f"  Selected {total_selected} bands out of {total_bands} total")
        print(f"  Compression ratio: {total_bands/total_selected:.2f}x")
        
        # Distribution across excitations
        excitation_counts = {}
        for band in self.selected_bands:
            ex = band['excitation']
            excitation_counts[ex] = excitation_counts.get(ex, 0) + 1
        
        print("\n  Distribution across excitations:")
        for ex in sorted(excitation_counts.keys()):
            print(f"    {ex:.1f}nm: {excitation_counts[ex]} bands")
        
        return {
            'selected_mask': selected_mask,
            'n_selected': total_selected,
            'n_total': total_bands,
            'compression_ratio': total_bands / total_selected,
            'excitation_distribution': excitation_counts
        }
    
    def run_complete_analysis(self, 
                             n_samples: int = 1000,
                             n_top_latent: int = 5,
                             n_bands_to_select: int = 20,
                             scoring_method: str = 'reconstruction',
                             normalize: str = 'variance',
                             epsilon: float = 0.1,
                             output_dir: str = './wavelength_selection_results'):
        """
        Run the complete wavelength selection pipeline.
        
        Args:
            n_samples: Number of pixels to sample
            n_top_latent: Number of top latent dimensions to analyze
            n_bands_to_select: Number of bands to select
            scoring_method: Method for scoring latent dimensions
            normalize: Normalization method for band influences
            epsilon: Perturbation magnitude
            output_dir: Directory for saving results
            
        Returns:
            Dictionary with all analysis results
        """
        print("="*60)
        print("Running Complete Wavelength Selection Analysis")
        print("="*60)
        
        # Step 1 & 2: Extract baseline
        baseline_data = self.step1_extract_baseline(n_samples)
        
        # Step 3: Score latent dimensions
        latent_scores = self.step2_score_latent_dimensions(scoring_method)
        
        # Step 4, 5, 6: Compute band influence
        influence_matrix = self.step3_compute_band_influence(n_top_latent, epsilon)
        
        # Select top bands
        selected_bands = self.step4_select_top_bands(n_bands_to_select, normalize)
        
        # Visualize results
        self.visualize_results(output_dir)
        
        # Validate selection
        validation_results = self.validate_selection()
        
        print("\n" + "="*60)
        print("Analysis Complete!")
        print("="*60)
        
        return {
            'baseline_data': baseline_data,
            'latent_scores': latent_scores,
            'influence_matrix': influence_matrix,
            'selected_bands': selected_bands,
            'validation': validation_results
        }


# Utility function for easy integration
def select_informative_wavelengths(model, dataset, config=None):
    """
    Simple wrapper function for wavelength selection.
    
    Args:
        model: Trained autoencoder model
        dataset: Hyperspectral dataset
        config: Optional configuration dictionary
        
    Returns:
        WavelengthSelector object with results
    """
    if config is None:
        config = {
            'n_samples': 1000,
            'n_top_latent': 5,
            'n_bands_to_select': 20,
            'scoring_method': 'reconstruction',
            'normalize': 'variance',
            'epsilon': 0.1
        }
    
    selector = WavelengthSelector(model, dataset)
    results = selector.run_complete_analysis(**config)
    
    return selector
