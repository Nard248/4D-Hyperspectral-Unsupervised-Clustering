import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple, Optional, Union
import json
from itertools import combinations
from scipy import linalg
from sklearn.decomposition import PCA, NMF
import os


class HyperspectralMCR:
    """
    Multivariate Curve Resolution for hyperspectral data.

    This class implements MCR-ALS (Alternating Least Squares) for analyzing
    hyperspectral data with single or multiple excitation-emission cubes.
    """

    def __init__(self,
                 n_components: int = 3,
                 max_iter: int = 100,
                 tol: float = 1e-6,
                 initialization: str = 'pca',
                 non_negativity: bool = True,
                 normalization: bool = True,
                 verbose: bool = True):
        """
        Initialize the MCR analyzer.

        Args:
            n_components: Number of components to extract
            max_iter: Maximum number of iterations for ALS
            tol: Convergence tolerance
            initialization: Method for initial guess ('pca', 'nmf', or 'random')
            non_negativity: Whether to apply non-negativity constraint
            normalization: Whether to normalize spectral profiles
            verbose: Whether to print progress information
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.initialization = initialization
        self.non_negativity = non_negativity
        self.normalization = normalization
        self.verbose = verbose

        # Results storage
        self.C = None  # Concentration profiles
        self.S = None  # Spectral profiles
        self.excitations_used = None  # List of excitations used
        self.explained_variance = None  # Explained variance
        self.reconstruction_error = None  # Reconstruction error

    def _unfold_cube(self, cube: np.ndarray) -> np.ndarray:
        """
        Unfold a 3D data cube to a 2D matrix.

        Args:
            cube: Hyperspectral data cube of shape [height, width, bands]
                 or [bands, height, width]

        Returns:
            Unfolded 2D matrix of shape [height*width, bands]
        """
        # Determine which dimension is likely the spectral dimension
        if len(cube.shape) == 3:
            if cube.shape[0] < cube.shape[1] and cube.shape[0] < cube.shape[2]:
                # First dimension is smallest, likely spectral bands - (bands, height, width)
                bands_dim = 0
                cube_t = np.transpose(cube, (1, 2, 0))  # -> [height, width, bands]
            elif cube.shape[2] < cube.shape[0] and cube.shape[2] < cube.shape[1]:
                # Last dimension is smallest, likely spectral bands - (height, width, bands)
                bands_dim = 2
                cube_t = cube
            else:
                # Hard to tell, assume standard (bands, height, width)
                bands_dim = 0
                cube_t = np.transpose(cube, (1, 2, 0))  # -> [height, width, bands]
        else:
            raise ValueError(f"Expected 3D array, got shape {cube.shape}")

        # Unfold to 2D
        height, width, bands = cube_t.shape
        return cube_t.reshape(height * width, bands)

    def _fold_to_maps(self, C: np.ndarray, spatial_shape: Tuple[int, int]) -> np.ndarray:
        """
        Fold concentration profiles back to spatial maps.

        Args:
            C: Concentration profiles of shape [height*width, n_components]
            spatial_shape: Tuple of (height, width)

        Returns:
            Spatial maps of shape [height, width, n_components]
        """
        height, width = spatial_shape
        return C.reshape(height, width, self.n_components)

    def _initial_guess(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate initial guess for C and S.

        Args:
            X: Data matrix of shape [samples, variables]

        Returns:
            C_init: Initial concentration profiles
            S_init: Initial spectral profiles
        """
        n_samples, n_variables = X.shape

        if self.initialization == 'pca':
            # Use PCA for initial guess
            pca = PCA(n_components=self.n_components)
            C_init = pca.fit_transform(X)
            S_init = pca.components_

            # Ensure non-negativity if requested
            if self.non_negativity:
                C_init = np.abs(C_init)
                S_init = np.abs(S_init)

        elif self.initialization == 'nmf':
            # Use NMF (ensures non-negativity)
            nmf = NMF(n_components=self.n_components, init='random', random_state=0)
            C_init = nmf.fit_transform(np.abs(X))  # NMF requires non-negative input
            S_init = nmf.components_

        elif self.initialization == 'random':
            # Random initialization
            C_init = np.random.rand(n_samples, self.n_components)
            S_init = np.random.rand(self.n_components, n_variables)

        else:
            raise ValueError(f"Unknown initialization method: {self.initialization}")

        return C_init, S_init

    def _als_step(self, X: np.ndarray, C: np.ndarray, S: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform one iteration of Alternating Least Squares.

        Args:
            X: Data matrix of shape [samples, variables]
            C: Concentration profiles of shape [samples, n_components]
            S: Spectral profiles of shape [n_components, variables]

        Returns:
            Updated C and S matrices
        """
        # Update C (fix S, solve for C)
        C_new = np.linalg.lstsq(S.T, X.T, rcond=None)[0].T

        # Apply non-negativity constraint if requested
        if self.non_negativity:
            C_new = np.maximum(C_new, 0)

        # Update S (fix C, solve for S)
        S_new = np.linalg.lstsq(C_new, X, rcond=None)[0]

        # Apply non-negativity constraint if requested
        if self.non_negativity:
            S_new = np.maximum(S_new, 0)

        # Normalize spectral profiles if requested
        if self.normalization:
            norms = np.linalg.norm(S_new, axis=1, keepdims=True)
            S_new = S_new / norms
            C_new = C_new * norms.T

        return C_new, S_new

    def fit_single_cube(self,
                        cube: np.ndarray,
                        excitation: Union[float, str] = None) -> Dict:
        """
        Fit MCR model to a single hyperspectral cube.

        Args:
            cube: Hyperspectral data cube
            excitation: Optional excitation wavelength for reference

        Returns:
            Dictionary with results
        """
        # Get spatial dimensions
        if cube.shape[0] < cube.shape[1] and cube.shape[0] < cube.shape[2]:
            # (bands, height, width)
            height, width = cube.shape[1], cube.shape[2]
        else:
            # (height, width, bands) or unknown
            height, width = cube.shape[0], cube.shape[1]

        # Unfold the cube to a 2D matrix
        X = self._unfold_cube(cube)

        # Handle NaN values
        mask = ~np.isnan(X)
        X_filled = np.copy(X)
        X_filled[~mask] = 0  # Fill NaNs with zeros for computation

        # Get initial guess
        C_init, S_init = self._initial_guess(X_filled)

        # Run ALS optimization
        C = C_init
        S = S_init
        prev_error = float('inf')

        for iteration in range(self.max_iter):
            # Perform ALS step
            C, S = self._als_step(X_filled, C, S)

            # Calculate reconstruction error (only on non-NaN values)
            X_reconstructed = C @ S
            error = np.mean((X_filled[mask] - X_reconstructed[mask]) ** 2)

            if self.verbose and (iteration % 10 == 0 or iteration == self.max_iter - 1):
                print(f"Iteration {iteration}: Reconstruction error = {error:.6f}")

            # Check convergence
            if np.abs(error - prev_error) < self.tol:
                if self.verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break

            prev_error = error

        # Calculate explained variance
        total_variance = np.var(X_filled[mask])
        residual_variance = np.var((X_filled - X_reconstructed)[mask])
        explained_variance = 1 - (residual_variance / total_variance)

        # Store results
        self.C = C
        self.S = S
        self.excitations_used = [excitation] if excitation is not None else ["unknown"]
        self.explained_variance = explained_variance
        self.reconstruction_error = error

        # Create spatial maps
        concentration_maps = self._fold_to_maps(C, (height, width))

        # Return results
        results = {
            'concentration_maps': concentration_maps,  # Shape: [height, width, n_components]
            'spectral_profiles': S,  # Shape: [n_components, n_wavelengths]
            'explained_variance': explained_variance,
            'reconstruction_error': error,
            'excitations_used': self.excitations_used,
            'n_components': self.n_components
        }

        return results

    def fit_multiple_cubes(self,
                           cubes: Dict[Union[float, str], np.ndarray]) -> Dict:
        """
        Fit MCR model to multiple hyperspectral cubes simultaneously.

        Args:
            cubes: Dictionary mapping excitation wavelengths to data cubes

        Returns:
            Dictionary with results
        """
        if not cubes:
            raise ValueError("No cubes provided")

        # Store excitations used
        self.excitations_used = list(cubes.keys())

        # Process each cube
        unfolded_data = {}
        spatial_shapes = {}
        combined_X = []
        total_variables = 0

        for ex, cube in cubes.items():
            # Get spatial dimensions
            if cube.shape[0] < cube.shape[1] and cube.shape[0] < cube.shape[2]:
                # (bands, height, width)
                height, width = cube.shape[1], cube.shape[2]
            else:
                # (height, width, bands) or unknown
                height, width = cube.shape[0], cube.shape[1]

            spatial_shapes[ex] = (height, width)

            # Unfold the cube
            X = self._unfold_cube(cube)
            unfolded_data[ex] = X

            # Handle NaN values and add to combined data
            mask = ~np.isnan(X)
            X_filled = np.copy(X)
            X_filled[~mask] = 0  # Fill NaNs with zeros

            combined_X.append(X_filled)
            total_variables += X.shape[1]

        # Concatenate all data along the wavelength dimension
        X_combined = np.hstack(combined_X)

        # Get initial guess
        C_init, S_init_combined = self._initial_guess(X_combined)

        # Split S_init into components for each excitation
        S_init = {}
        start_idx = 0
        for ex, X in unfolded_data.items():
            n_vars = X.shape[1]
            S_init[ex] = S_init_combined[:, start_idx:start_idx + n_vars]
            start_idx += n_vars

        # Run ALS optimization
        C = C_init
        S = S_init
        prev_error = float('inf')

        for iteration in range(self.max_iter):
            # Update C (fix all S, solve for C)
            X_reconstructed_parts = []
            for ex, S_ex in S.items():
                X_reconstructed_parts.append(C @ S_ex)

            X_reconstructed = np.hstack(X_reconstructed_parts)
            error = np.mean((X_combined - X_reconstructed) ** 2)

            # Update all S (fix C, solve for each S separately)
            for ex, X in unfolded_data.items():
                # Handle NaN values
                mask = ~np.isnan(X)
                X_filled = np.copy(X)
                X_filled[~mask] = 0

                # Update S for this excitation
                S_ex = np.linalg.lstsq(C, X_filled, rcond=None)[0]

                # Apply non-negativity constraint if requested
                if self.non_negativity:
                    S_ex = np.maximum(S_ex, 0)

                S[ex] = S_ex

            # Update C using all S combined
            S_combined = np.hstack([S[ex] for ex in unfolded_data.keys()])
            C = np.linalg.lstsq(S_combined.T, X_combined.T, rcond=None)[0].T

            # Apply non-negativity constraint if requested
            if self.non_negativity:
                C = np.maximum(C, 0)

            # Normalize spectral profiles if requested
            if self.normalization:
                for ex in S:
                    norms = np.linalg.norm(S[ex], axis=1, keepdims=True)
                    S[ex] = S[ex] / norms
                    C = C * norms.T

            if self.verbose and (iteration % 10 == 0 or iteration == self.max_iter - 1):
                print(f"Iteration {iteration}: Reconstruction error = {error:.6f}")

            # Check convergence
            if np.abs(error - prev_error) < self.tol:
                if self.verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break

            prev_error = error

        # Calculate explained variance
        total_variance = np.var(X_combined)
        residual_variance = np.var(X_combined - X_reconstructed)
        explained_variance = 1 - (residual_variance / total_variance)

        # Store results
        self.C = C
        self.S = S
        self.explained_variance = explained_variance
        self.reconstruction_error = error

        # Create spatial maps
        first_ex = next(iter(spatial_shapes))
        concentration_maps = self._fold_to_maps(C, spatial_shapes[first_ex])

        # Return results
        results = {
            'concentration_maps': concentration_maps,  # Shape: [height, width, n_components]
            'spectral_profiles': S,  # Dictionary mapping excitations to spectral profiles
            'explained_variance': explained_variance,
            'reconstruction_error': error,
            'excitations_used': self.excitations_used,
            'n_components': self.n_components
        }

        return results

    def visualize_results(self,
                          results: Dict,
                          output_dir: Optional[str] = None,
                          filename_prefix: str = "mcr",
                          show_plots: bool = True) -> None:
        """
        Visualize MCR results.

        Args:
            results: Results dictionary from fit methods
            output_dir: Optional directory to save visualizations
            filename_prefix: Prefix for saved files
            show_plots: Whether to display plots
        """
        # Create output directory if needed
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        # Extract data
        concentration_maps = results['concentration_maps']
        spectral_profiles = results['spectral_profiles']
        excitations_used = results['excitations_used']
        n_components = results['n_components']

        # Create colormap for visualization
        colors = plt.cm.tab10(np.arange(n_components) % 10)

        # 1. Visualize concentration maps
        fig, axes = plt.subplots(1, n_components, figsize=(n_components * 4, 4))
        if n_components == 1:
            axes = [axes]

        for i in range(n_components):
            im = axes[i].imshow(concentration_maps[:, :, i], cmap='viridis')
            axes[i].set_title(f"Component {i + 1}")
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

        plt.tight_layout()
        if output_dir:
            plt.savefig(os.path.join(output_dir, f"{filename_prefix}_concentration_maps.png"), dpi=300)

        if show_plots:
            plt.show()
        else:
            plt.close()

        # 2. Visualize spectral profiles
        if isinstance(spectral_profiles, dict):
            # Multiple excitations
            fig, axes = plt.subplots(len(excitations_used), 1, figsize=(8, len(excitations_used) * 3))
            if len(excitations_used) == 1:
                axes = [axes]

            for i, ex in enumerate(excitations_used):
                for j in range(n_components):
                    axes[i].plot(spectral_profiles[ex][j], color=colors[j], linewidth=2,
                                 label=f"Component {j + 1}")
                axes[i].set_title(f"Spectral Profiles (Excitation: {ex})")
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)

        else:
            # Single excitation
            fig, ax = plt.subplots(figsize=(8, 5))
            for i in range(n_components):
                ax.plot(spectral_profiles[i], color=colors[i], linewidth=2,
                        label=f"Component {i + 1}")
            ax.set_title("Spectral Profiles")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if output_dir:
            plt.savefig(os.path.join(output_dir, f"{filename_prefix}_spectral_profiles.png"), dpi=300)

        if show_plots:
            plt.show()
        else:
            plt.close()

        # 3. Create RGB overlay of components
        height, width, _ = concentration_maps.shape
        overlay = np.zeros((height, width, 3))

        # Normalize each component for visualization
        normalized_maps = np.zeros_like(concentration_maps)
        for i in range(n_components):
            component = concentration_maps[:, :, i]
            if np.max(component) > 0:
                normalized_maps[:, :, i] = component / np.max(component)

        # Create RGB overlay (use first 3 components or fewer)
        n_rgb = min(3, n_components)
        for i in range(n_rgb):
            for j in range(3):
                if i == j:
                    overlay[:, :, j] += normalized_maps[:, :, i]

        # Clip to [0, 1]
        overlay = np.clip(overlay, 0, 1)

        # Visualize overlay
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(overlay)
        ax.set_title("RGB Component Overlay")
        ax.axis('off')

        plt.tight_layout()
        if output_dir:
            plt.savefig(os.path.join(output_dir, f"{filename_prefix}_rgb_overlay.png"), dpi=300)

        if show_plots:
            plt.show()
        else:
            plt.close()

        # 4. Print metrics
        if self.verbose:
            print("\nMCR Analysis Results:")
            print(f"Number of components: {n_components}")
            print(f"Excitations used: {excitations_used}")
            print(f"Explained variance: {results['explained_variance']:.4f}")
            print(f"Reconstruction error: {results['reconstruction_error']:.6f}")

class HyperspectralMCRCombinations(HyperspectralMCR):
    """
    Extended MCR class for analyzing different combinations of excitation-emission cubes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.combination_results = {}  # Store results for each combination

    def generate_combinations(self,
                              excitation_list: List[Union[str, float]],
                              combination_types: List[str] = ['single', 'pairs', 'triplets', 'all']) -> Dict[str, List]:
        """
        Generate different combinations of excitation wavelengths.

        Args:
            excitation_list: List of available excitation wavelengths
            combination_types: Types of combinations to generate
                              'single' - individual excitations
                              'pairs' - all pairs of excitations
                              'triplets' - all triplets of excitations
                              'all' - all excitations together
                              'custom' - user-defined combinations

        Returns:
            Dictionary mapping combination names to lists of excitations
        """
        combinations_dict = {}

        for combo_type in combination_types:
            if combo_type == 'single':
                for ex in excitation_list:
                    combinations_dict[f"single_{ex}"] = [ex]

            elif combo_type == 'pairs':
                for pair in combinations(excitation_list, 2):
                    combo_name = f"pair_{'_'.join(map(str, pair))}"
                    combinations_dict[combo_name] = list(pair)

            elif combo_type == 'triplets':
                for triplet in combinations(excitation_list, 3):
                    combo_name = f"triplet_{'_'.join(map(str, triplet))}"
                    combinations_dict[combo_name] = list(triplet)

            elif combo_type == 'all':
                combo_name = f"all_{'_'.join(map(str, excitation_list))}"
                combinations_dict[combo_name] = excitation_list

        return combinations_dict

    def add_custom_combinations(self,
                                combinations_dict: Dict[str, List],
                                custom_combinations: Dict[str, List[Union[str, float]]]) -> Dict[str, List]:
        """
        Add custom user-defined combinations.

        Args:
            combinations_dict: Existing combinations dictionary
            custom_combinations: Dictionary mapping custom names to excitation lists

        Returns:
            Updated combinations dictionary
        """
        for name, excitations in custom_combinations.items():
            combinations_dict[f"custom_{name}"] = excitations

        return combinations_dict

    def analyze_combinations(self,
                             data_dict: Dict,
                             combinations_dict: Dict[str, List],
                             output_dir: str = 'results/mcr_combinations',
                             save_individual_results: bool = True) -> Dict:
        """
        Run MCR analysis on all specified combinations.

        Args:
            data_dict: Your hyperspectral data dictionary
            combinations_dict: Dictionary mapping combination names to excitation lists
            output_dir: Output directory for results
            save_individual_results: Whether to save individual combination results

        Returns:
            Dictionary containing all combination results
        """
        os.makedirs(output_dir, exist_ok=True)

        # Initialize results storage
        all_results = {}
        summary_metrics = []

        print(f"Analyzing {len(combinations_dict)} combinations...")

        for combo_name, excitations in combinations_dict.items():
            print(f"\n--- Analyzing combination: {combo_name} ---")
            print(f"Excitations: {excitations}")

            # Extract cubes for this combination
            cubes = {}
            for ex in excitations:
                ex_key = str(ex)  # Ensure string key
                if ex_key in data_dict['data']:
                    cubes[ex_key] = data_dict['data'][ex_key]['cube']
                else:
                    print(f"Warning: Excitation {ex} not found in data")

            if not cubes:
                print(f"No valid excitations found for combination {combo_name}, skipping")
                continue

            # Run MCR analysis
            try:
                if len(cubes) == 1:
                    # Single cube analysis
                    ex_key = next(iter(cubes))
                    results = self.fit_single_cube(cubes[ex_key], excitation=ex_key)
                else:
                    # Multiple cube analysis
                    results = self.fit_multiple_cubes(cubes)

                # Add combination info to results
                results['combination_name'] = combo_name
                results['excitations_in_combination'] = excitations

                # Store results
                all_results[combo_name] = results

                # Collect summary metrics
                summary_metrics.append({
                    'combination_name': combo_name,
                    'excitations': excitations,
                    'n_excitations': len(excitations),
                    'n_components': results['n_components'],
                    'explained_variance': results['explained_variance'],
                    'reconstruction_error': results['reconstruction_error']
                })

                # Save individual results if requested
                if save_individual_results:
                    combo_dir = os.path.join(output_dir, combo_name)
                    os.makedirs(combo_dir, exist_ok=True)

                    # Visualize results
                    self.visualize_results(
                        results,
                        output_dir=combo_dir,
                        filename_prefix=combo_name,
                        show_plots=False
                    )

                    # Save detailed results as JSON
                    self._save_combination_details(results, combo_dir)

                print(f"✓ Completed: Explained variance = {results['explained_variance']:.4f}")

            except Exception as e:
                print(f"✗ Error analyzing combination {combo_name}: {str(e)}")
                continue

        # Create comprehensive summary
        self._create_combination_summary(summary_metrics, all_results, output_dir)

        # Store results in class
        self.combination_results = all_results

        return all_results

    def _save_combination_details(self, results: Dict, output_dir: str):
        """Save detailed results for a combination."""
        # Prepare data for JSON serialization
        json_data = {
            'combination_name': results['combination_name'],
            'excitations_in_combination': results['excitations_in_combination'],
            'n_components': results['n_components'],
            'explained_variance': float(results['explained_variance']),
            'reconstruction_error': float(results['reconstruction_error']),
            'excitations_used': results['excitations_used']
        }

        # Save concentration map statistics
        conc_maps = results['concentration_maps']
        json_data['concentration_stats'] = {
            'shape': list(conc_maps.shape),
            'component_means': [float(np.mean(conc_maps[:, :, i])) for i in range(conc_maps.shape[2])],
            'component_stds': [float(np.std(conc_maps[:, :, i])) for i in range(conc_maps.shape[2])],
            'component_maxes': [float(np.max(conc_maps[:, :, i])) for i in range(conc_maps.shape[2])]
        }

        # Save to JSON file
        with open(os.path.join(output_dir, 'combination_details.json'), 'w') as f:
            json.dump(json_data, f, indent=2)

    def _create_combination_summary(self, summary_metrics: List[Dict], all_results: Dict, output_dir: str):
        """Create comprehensive summary visualizations and reports."""

        # 1. Summary table
        summary_df_data = []
        for metric in summary_metrics:
            summary_df_data.append([
                metric['combination_name'],
                ', '.join(map(str, metric['excitations'])),
                metric['n_excitations'],
                metric['n_components'],
                f"{metric['explained_variance']:.4f}",
                f"{metric['reconstruction_error']:.6f}"
            ])

        # Save summary table
        with open(os.path.join(output_dir, 'combination_summary.txt'), 'w') as f:
            f.write("MCR Combination Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(
                f"{'Combination':<25} {'Excitations':<20} {'N_Ex':<5} {'N_Comp':<7} {'Expl_Var':<10} {'Recon_Err':<12}\n")
            f.write("-" * 90 + "\n")

            for row in summary_df_data:
                f.write(f"{row[0]:<25} {row[1]:<20} {row[2]:<5} {row[3]:<7} {row[4]:<10} {row[5]:<12}\n")

        # 2. Visualization: Explained variance comparison
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Sort by explained variance for better visualization
        sorted_metrics = sorted(summary_metrics, key=lambda x: x['explained_variance'], reverse=True)

        names = [m['combination_name'] for m in sorted_metrics]
        explained_vars = [m['explained_variance'] for m in sorted_metrics]
        errors = [m['reconstruction_error'] for m in sorted_metrics]
        n_excitations = [m['n_excitations'] for m in sorted_metrics]

        # Color code by number of excitations
        colors = plt.cm.viridis(np.array(n_excitations) / max(n_excitations))

        # Explained variance plot
        bars1 = ax1.bar(range(len(names)), explained_vars, color=colors)
        ax1.set_title('Explained Variance by Combination')
        ax1.set_ylabel('Explained Variance')
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars1, explained_vars)):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=8)

        # Reconstruction error plot
        bars2 = ax2.bar(range(len(names)), errors, color=colors)
        ax2.set_title('Reconstruction Error by Combination')
        ax2.set_ylabel('Reconstruction Error')
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'combination_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Heatmap of explained variance vs number of excitations
        self._create_excitation_heatmap(summary_metrics, output_dir)

        # 4. Best combinations report
        self._create_best_combinations_report(sorted_metrics, output_dir)

    def _create_excitation_heatmap(self, summary_metrics: List[Dict], output_dir: str):
        """Create heatmap showing performance vs number of excitations."""

        # Group by number of excitations
        excitation_groups = {}
        for metric in summary_metrics:
            n_ex = metric['n_excitations']
            if n_ex not in excitation_groups:
                excitation_groups[n_ex] = []
            excitation_groups[n_ex].append(metric['explained_variance'])

        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 6))

        n_excitations = sorted(excitation_groups.keys())
        avg_performance = [np.mean(excitation_groups[n]) for n in n_excitations]
        std_performance = [np.std(excitation_groups[n]) for n in n_excitations]

        ax.errorbar(n_excitations, avg_performance, yerr=std_performance,
                    marker='o', capsize=5, capthick=2, linewidth=2)
        ax.set_xlabel('Number of Excitations in Combination')
        ax.set_ylabel('Average Explained Variance')
        ax.set_title('MCR Performance vs Number of Excitations')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'excitation_performance.png'), dpi=300)
        plt.close()

    def _create_best_combinations_report(self, sorted_metrics: List[Dict], output_dir: str):
        """Create a report of the best performing combinations."""

        with open(os.path.join(output_dir, 'best_combinations_report.txt'), 'w') as f:
            f.write("Best Performing MCR Combinations\n")
            f.write("=" * 40 + "\n\n")

            # Top 5 overall
            f.write("Top 5 Combinations (by Explained Variance):\n")
            f.write("-" * 40 + "\n")
            for i, metric in enumerate(sorted_metrics[:5]):
                f.write(f"{i + 1}. {metric['combination_name']}\n")
                f.write(f"   Excitations: {metric['excitations']}\n")
                f.write(f"   Explained Variance: {metric['explained_variance']:.4f}\n")
                f.write(f"   Reconstruction Error: {metric['reconstruction_error']:.6f}\n\n")

            # Best by number of excitations
            excitation_groups = {}
            for metric in sorted_metrics:
                n_ex = metric['n_excitations']
                if n_ex not in excitation_groups:
                    excitation_groups[n_ex] = []
                excitation_groups[n_ex].append(metric)

            f.write("\nBest Combination by Number of Excitations:\n")
            f.write("-" * 40 + "\n")
            for n_ex in sorted(excitation_groups.keys()):
                best = max(excitation_groups[n_ex], key=lambda x: x['explained_variance'])
                f.write(f"Best with {n_ex} excitation(s): {best['combination_name']}\n")
                f.write(f"   Excitations: {best['excitations']}\n")
                f.write(f"   Explained Variance: {best['explained_variance']:.4f}\n\n")

    def compare_combinations(self,
                             combinations_to_compare: List[str],
                             output_dir: str = 'results/mcr_comparison') -> None:
        """
        Create detailed comparison between specific combinations.

        Args:
            combinations_to_compare: List of combination names to compare
            output_dir: Output directory for comparison results
        """
        os.makedirs(output_dir, exist_ok=True)

        if not self.combination_results:
            print("No combination results available. Run analyze_combinations first.")
            return

        # Filter results for requested combinations
        selected_results = {name: self.combination_results[name]
                            for name in combinations_to_compare
                            if name in self.combination_results}

        if not selected_results:
            print("None of the requested combinations found in results.")
            return

        # Create side-by-side comparison
        n_combinations = len(selected_results)
        n_components = next(iter(selected_results.values()))['n_components']

        # 1. Concentration maps comparison
        fig, axes = plt.subplots(n_combinations, n_components,
                                 figsize=(n_components * 4, n_combinations * 3))
        if n_combinations == 1:
            axes = axes.reshape(1, -1)
        if n_components == 1:
            axes = axes.reshape(-1, 1)

        for i, (combo_name, results) in enumerate(selected_results.items()):
            conc_maps = results['concentration_maps']
            for j in range(n_components):
                ax = axes[i, j]
                im = ax.imshow(conc_maps[:, :, j], cmap='viridis')
                ax.set_title(f'{combo_name}\nComponent {j + 1}')
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'concentration_comparison.png'), dpi=300)
        plt.close()

        # 2. Spectral profiles comparison
        fig, axes = plt.subplots(n_components, 1, figsize=(12, n_components * 3))
        if n_components == 1:
            axes = [axes]

        colors = plt.cm.tab10(np.arange(n_combinations))

        for j in range(n_components):
            for i, (combo_name, results) in enumerate(selected_results.items()):
                spectral_profiles = results['spectral_profiles']

                if isinstance(spectral_profiles, dict):
                    # Multiple excitations - plot each excitation
                    for k, ex in enumerate(results['excitations_used']):
                        linestyle = '-' if k == 0 else '--'
                        axes[j].plot(spectral_profiles[ex][j],
                                     color=colors[i], linestyle=linestyle,
                                     label=f'{combo_name}_Ex{ex}' if k == 0 else f'Ex{ex}',
                                     alpha=0.8)
                else:
                    # Single excitation
                    axes[j].plot(spectral_profiles[j], color=colors[i],
                                 label=combo_name, linewidth=2)

            axes[j].set_title(f'Component {j + 1} Spectral Profiles')
            axes[j].legend()
            axes[j].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'spectral_comparison.png'), dpi=300)
        plt.close()

        print(f"Comparison results saved to {output_dir}")


import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
import os
from typing import Dict, List, Tuple, Optional, Union


class RobustHyperspectralMCR:
    """
    Robust MCR implementation with improved numerical stability.
    """

    def __init__(self,
                 n_components: int = 3,
                 max_iter: int = 100,
                 tol: float = 1e-6,
                 initialization: str = 'pca',
                 non_negativity: bool = True,
                 normalization: bool = True,
                 data_preprocessing: str = 'standard',  # 'standard', 'minmax', 'none'
                 regularization: float = 1e-6,  # Add regularization
                 verbose: bool = True):
        """
        Initialize the robust MCR analyzer.
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.initialization = initialization
        self.non_negativity = non_negativity
        self.normalization = normalization
        self.data_preprocessing = data_preprocessing
        self.regularization = regularization
        self.verbose = verbose

        # Results storage
        self.C = None
        self.S = None
        self.excitations_used = None
        self.explained_variance = None
        self.reconstruction_error = None
        self.scaler = None

    def _preprocess_data(self, X: np.ndarray) -> np.ndarray:
        """
        Robust data preprocessing to handle numerical issues.

        Args:
            X: Input data matrix [samples, variables]

        Returns:
            Preprocessed data matrix
        """
        # Handle NaN and infinite values
        X_clean = np.copy(X)

        # Replace NaN with median of non-NaN values
        nan_mask = np.isnan(X_clean)
        if np.any(nan_mask):
            if self.verbose:
                print(f"Found {np.sum(nan_mask)} NaN values, replacing with median")
            for j in range(X_clean.shape[1]):
                col = X_clean[:, j]
                if np.any(~np.isnan(col)):
                    median_val = np.nanmedian(col)
                    X_clean[nan_mask[:, j], j] = median_val
                else:
                    X_clean[:, j] = 0  # If all NaN, set to zero

        # Replace infinite values
        inf_mask = np.isinf(X_clean)
        if np.any(inf_mask):
            if self.verbose:
                print(f"Found {np.sum(inf_mask)} infinite values, replacing with finite range")
            X_clean[inf_mask] = 0

        # Check for zero variance columns
        col_std = np.std(X_clean, axis=0)
        zero_var_cols = col_std < 1e-12
        if np.any(zero_var_cols):
            if self.verbose:
                print(f"Found {np.sum(zero_var_cols)} zero-variance columns")
            # Add small noise to zero-variance columns
            X_clean[:, zero_var_cols] += np.random.normal(0, 1e-6,
                                                          (X_clean.shape[0], np.sum(zero_var_cols)))

        # Data preprocessing
        if self.data_preprocessing == 'standard':
            self.scaler = StandardScaler()
            X_processed = self.scaler.fit_transform(X_clean)
        elif self.data_preprocessing == 'minmax':
            self.scaler = MinMaxScaler()
            X_processed = self.scaler.fit_transform(X_clean)
        else:
            X_processed = X_clean
            self.scaler = None

        # Ensure finite values
        if not np.all(np.isfinite(X_processed)):
            if self.verbose:
                print("Warning: Non-finite values after preprocessing, clipping to finite range")
            X_processed = np.clip(X_processed, -1e6, 1e6)

        return X_processed

    def _robust_initial_guess(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate robust initial guess for C and S.
        """
        n_samples, n_variables = X.shape

        # Check matrix conditioning
        try:
            cond_num = np.linalg.cond(X)
            if self.verbose:
                print(f"Data matrix condition number: {cond_num:.2e}")
        except:
            cond_num = np.inf

        if cond_num > 1e12:
            if self.verbose:
                print("Warning: Poorly conditioned data matrix, using regularization")

        # Try different initialization methods with fallbacks
        initialization_methods = [self.initialization, 'pca', 'nmf', 'random']

        for method in initialization_methods:
            try:
                if method == 'pca':
                    # Use PCA with SVD solver for better numerical stability
                    pca = PCA(n_components=self.n_components, svd_solver='auto')
                    C_init = pca.fit_transform(X)
                    S_init = pca.components_

                    # Check for valid results
                    if np.all(np.isfinite(C_init)) and np.all(np.isfinite(S_init)):
                        if self.verbose:
                            print(
                                f"Successful PCA initialization (explained variance: {pca.explained_variance_ratio_.sum():.4f})")
                        break

                elif method == 'nmf':
                    # Use NMF with robust settings
                    nmf = NMF(n_components=self.n_components,
                              init='nndsvda',  # More robust initialization
                              solver='mu',  # Multiplicative update solver
                              max_iter=200,
                              random_state=42,
                              alpha=0.1)  # Add regularization

                    # NMF requires non-negative input
                    X_pos = X - np.min(X) + 1e-6
                    C_init = nmf.fit_transform(X_pos)
                    S_init = nmf.components_

                    if np.all(np.isfinite(C_init)) and np.all(np.isfinite(S_init)):
                        if self.verbose:
                            print(
                                f"Successful NMF initialization (reconstruction error: {nmf.reconstruction_err_:.4f})")
                        break

                elif method == 'random':
                    # Random initialization with proper scaling
                    np.random.seed(42)
                    data_std = np.std(X)
                    C_init = np.random.normal(0, data_std / 10, (n_samples, self.n_components))
                    S_init = np.random.normal(0, data_std / 10, (self.n_components, n_variables))

                    if self.verbose:
                        print("Using random initialization")
                    break

            except Exception as e:
                if self.verbose:
                    print(f"Initialization method '{method}' failed: {str(e)}")
                continue

        # Ensure non-negativity if requested
        if self.non_negativity:
            C_init = np.abs(C_init)
            S_init = np.abs(S_init)

        # Add small regularization to avoid zero values
        C_init += self.regularization
        S_init += self.regularization

        return C_init, S_init

    def _robust_lstsq(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Robust least squares solver with multiple fallback methods.
        """
        methods = ['lstsq', 'pinv', 'ridge']

        for method in methods:
            try:
                if method == 'lstsq':
                    # Standard least squares with increased rcond
                    result = np.linalg.lstsq(A, b, rcond=1e-6)[0]

                elif method == 'pinv':
                    # Pseudo-inverse method
                    result = np.linalg.pinv(A, rcond=1e-6) @ b

                elif method == 'ridge':
                    # Ridge regression (L2 regularization)
                    A_reg = A.T @ A + self.regularization * np.eye(A.shape[1])
                    result = np.linalg.solve(A_reg, A.T @ b)

                # Check for valid result
                if np.all(np.isfinite(result)):
                    return result

            except Exception as e:
                if self.verbose:
                    print(f"Least squares method '{method}' failed: {str(e)}")
                continue

        # Last resort: return zeros with small noise
        if self.verbose:
            print("Warning: All least squares methods failed, using fallback solution")
        return np.random.normal(0, 1e-6, A.shape[1]) if len(b.shape) == 1 else np.random.normal(0, 1e-6, (A.shape[1],
                                                                                                          b.shape[1]))

    def _robust_als_step(self, X: np.ndarray, C: np.ndarray, S: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Robust ALS step with error handling.
        """
        # Update C (fix S, solve for C)
        try:
            C_new = self._robust_lstsq(S.T, X.T).T
        except:
            C_new = np.copy(C)

        # Apply non-negativity constraint if requested
        if self.non_negativity:
            C_new = np.maximum(C_new, self.regularization)

        # Update S (fix C, solve for S)
        try:
            S_new = self._robust_lstsq(C_new, X)
        except:
            S_new = np.copy(S)

        # Apply non-negativity constraint if requested
        if self.non_negativity:
            S_new = np.maximum(S_new, self.regularization)

        # Robust normalization
        if self.normalization:
            try:
                norms = np.linalg.norm(S_new, axis=1, keepdims=True)
                # Avoid division by zero
                norms = np.maximum(norms, 1e-12)
                S_new = S_new / norms
                C_new = C_new * norms.T
            except:
                if self.verbose:
                    print("Warning: Normalization failed, skipping this step")

        # Ensure finite values
        C_new = np.nan_to_num(C_new, nan=self.regularization, posinf=1e6, neginf=-1e6)
        S_new = np.nan_to_num(S_new, nan=self.regularization, posinf=1e6, neginf=-1e6)

        return C_new, S_new

    def _calculate_metrics(self, X: np.ndarray, C: np.ndarray, S: np.ndarray) -> Tuple[float, float]:
        """
        Robust calculation of metrics.
        """
        try:
            X_reconstructed = C @ S

            # Ensure finite values
            if not np.all(np.isfinite(X_reconstructed)):
                X_reconstructed = np.nan_to_num(X_reconstructed)

            # Calculate reconstruction error
            diff = X - X_reconstructed
            error = np.mean(diff ** 2)

            # Calculate explained variance
            total_variance = np.var(X)
            residual_variance = np.var(diff)

            if total_variance > 1e-12:
                explained_variance = 1 - (residual_variance / total_variance)
            else:
                explained_variance = 0.0

            # Ensure valid metrics
            error = float(error) if np.isfinite(error) else float('inf')
            explained_variance = float(explained_variance) if np.isfinite(explained_variance) else 0.0

            return explained_variance, error

        except Exception as e:
            if self.verbose:
                print(f"Error calculating metrics: {str(e)}")
            return 0.0, float('inf')

    def fit_single_cube(self, cube: np.ndarray, excitation: Union[float, str] = None) -> Dict:
        """
        Robust fit for single cube.
        """
        if self.verbose:
            print(f"Fitting MCR to single cube with shape {cube.shape}")

        # Get spatial dimensions
        if cube.shape[0] < cube.shape[1] and cube.shape[0] < cube.shape[2]:
            height, width = cube.shape[1], cube.shape[2]
        else:
            height, width = cube.shape[0], cube.shape[1]

        # Unfold the cube
        X = self._unfold_cube(cube)

        if self.verbose:
            print(f"Unfolded data shape: {X.shape}")
            print(f"Data range: [{np.nanmin(X):.6f}, {np.nanmax(X):.6f}]")
            print(f"Data mean: {np.nanmean(X):.6f}, std: {np.nanstd(X):.6f}")

        # Preprocess data
        X_processed = self._preprocess_data(X)

        if self.verbose:
            print(f"Processed data range: [{np.min(X_processed):.6f}, {np.max(X_processed):.6f}]")

        # Get robust initial guess
        C, S = self._robust_initial_guess(X_processed)

        if self.verbose:
            print(f"Initial C shape: {C.shape}, S shape: {S.shape}")

        # Run robust ALS optimization
        prev_error = float('inf')

        for iteration in range(self.max_iter):
            # Perform robust ALS step
            C, S = self._robust_als_step(X_processed, C, S)

            # Calculate metrics
            explained_variance, error = self._calculate_metrics(X_processed, C, S)

            if self.verbose and (iteration % 10 == 0 or iteration == self.max_iter - 1):
                print(
                    f"Iteration {iteration}: Reconstruction error = {error:.6f}, Explained variance = {explained_variance:.6f}")

            # Check convergence
            if np.abs(error - prev_error) < self.tol:
                if self.verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break

            prev_error = error

        # Store results
        self.C = C
        self.S = S
        self.excitations_used = [excitation] if excitation is not None else ["unknown"]
        self.explained_variance = explained_variance
        self.reconstruction_error = error

        # Create spatial maps
        concentration_maps = self._fold_to_maps(C, (height, width))

        # Return results
        results = {
            'concentration_maps': concentration_maps,
            'spectral_profiles': S,
            'explained_variance': explained_variance,
            'reconstruction_error': error,
            'excitations_used': self.excitations_used,
            'n_components': self.n_components
        }

        return results

    def fit_multiple_cubes(self, cubes: Dict[Union[float, str], np.ndarray]) -> Dict:
        """
        Robust fit for multiple cubes with improved data handling.
        """
        if not cubes:
            raise ValueError("No cubes provided")

        if self.verbose:
            print(f"Fitting MCR to {len(cubes)} cubes")
            for ex, cube in cubes.items():
                print(f"  Excitation {ex}: shape {cube.shape}")

        self.excitations_used = list(cubes.keys())

        # Process each cube with robust handling
        unfolded_data = {}
        spatial_shapes = {}
        data_info = {}

        for ex, cube in cubes.items():
            # Get spatial dimensions
            if cube.shape[0] < cube.shape[1] and cube.shape[0] < cube.shape[2]:
                height, width = cube.shape[1], cube.shape[2]
            else:
                height, width = cube.shape[0], cube.shape[1]

            spatial_shapes[ex] = (height, width)

            # Unfold the cube
            X = self._unfold_cube(cube)
            unfolded_data[ex] = X

            # Store data statistics
            data_info[ex] = {
                'shape': X.shape,
                'min': np.nanmin(X),
                'max': np.nanmax(X),
                'mean': np.nanmean(X),
                'std': np.nanstd(X),
                'nan_count': np.sum(np.isnan(X))
            }

            if self.verbose:
                print(f"  Excitation {ex} data: {data_info[ex]}")

        # Concatenate all data with robust handling
        combined_X = []
        for ex in self.excitations_used:
            X = unfolded_data[ex]
            combined_X.append(X)

        X_combined = np.hstack(combined_X)

        if self.verbose:
            print(f"Combined data shape: {X_combined.shape}")
            print(f"Combined data range: [{np.nanmin(X_combined):.6f}, {np.nanmax(X_combined):.6f}]")

        # Preprocess combined data
        X_processed = self._preprocess_data(X_combined)

        if self.verbose:
            print(f"Processed combined data range: [{np.min(X_processed):.6f}, {np.max(X_processed):.6f}]")

        # Get robust initial guess
        C, S_combined = self._robust_initial_guess(X_processed)

        # Split S_combined into components for each excitation
        S = {}
        start_idx = 0
        for ex in self.excitations_used:
            n_vars = unfolded_data[ex].shape[1]
            S[ex] = S_combined[:, start_idx:start_idx + n_vars]
            start_idx += n_vars

        if self.verbose:
            print(f"Initial C shape: {C.shape}")
            for ex, s in S.items():
                print(f"Initial S[{ex}] shape: {s.shape}")

        # Run robust ALS optimization
        prev_error = float('inf')

        for iteration in range(self.max_iter):
            # Update all S matrices
            for ex in self.excitations_used:
                X_ex = self._preprocess_data(unfolded_data[ex])
                S[ex] = self._robust_lstsq(C, X_ex)

                if self.non_negativity:
                    S[ex] = np.maximum(S[ex], self.regularization)

            # Combine all S matrices
            S_combined = np.hstack([S[ex] for ex in self.excitations_used])

            # Update C
            C = self._robust_lstsq(S_combined.T, X_processed.T).T

            if self.non_negativity:
                C = np.maximum(C, self.regularization)

            # Normalize if requested
            if self.normalization:
                try:
                    for ex in S:
                        norms = np.linalg.norm(S[ex], axis=1, keepdims=True)
                        norms = np.maximum(norms, 1e-12)
                        S[ex] = S[ex] / norms
                        C = C * norms.T
                except:
                    pass

            # Calculate metrics
            explained_variance, error = self._calculate_metrics(X_processed, C, S_combined)

            if self.verbose and (iteration % 10 == 0 or iteration == self.max_iter - 1):
                print(
                    f"Iteration {iteration}: Reconstruction error = {error:.6f}, Explained variance = {explained_variance:.6f}")

            # Check convergence
            if np.abs(error - prev_error) < self.tol:
                if self.verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break

            prev_error = error

        # Store results
        self.C = C
        self.S = S
        self.explained_variance = explained_variance
        self.reconstruction_error = error

        # Create spatial maps
        first_ex = next(iter(spatial_shapes))
        concentration_maps = self._fold_to_maps(C, spatial_shapes[first_ex])

        results = {
            'concentration_maps': concentration_maps,
            'spectral_profiles': S,
            'explained_variance': explained_variance,
            'reconstruction_error': error,
            'excitations_used': self.excitations_used,
            'n_components': self.n_components,
            'data_info': data_info
        }

        return results

    # Include the other methods from the original class (_unfold_cube, _fold_to_maps, visualize_results, etc.)
    def _unfold_cube(self, cube: np.ndarray) -> np.ndarray:
        """Unfold a 3D data cube to a 2D matrix."""
        if len(cube.shape) == 3:
            if cube.shape[0] < cube.shape[1] and cube.shape[0] < cube.shape[2]:
                cube_t = np.transpose(cube, (1, 2, 0))
            elif cube.shape[2] < cube.shape[0] and cube.shape[2] < cube.shape[1]:
                cube_t = cube
            else:
                cube_t = np.transpose(cube, (1, 2, 0))
        else:
            raise ValueError(f"Expected 3D array, got shape {cube.shape}")

        height, width, bands = cube_t.shape
        return cube_t.reshape(height * width, bands)

    def _fold_to_maps(self, C: np.ndarray, spatial_shape: Tuple[int, int]) -> np.ndarray:
        """Fold concentration profiles back to spatial maps."""
        height, width = spatial_shape
        return C.reshape(height, width, self.n_components)

    def visualize_results(self, results: Dict, output_dir: Optional[str] = None,
                          filename_prefix: str = "mcr", show_plots: bool = True) -> None:
        """Visualize MCR results with enhanced error checking."""
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        concentration_maps = results['concentration_maps']
        spectral_profiles = results['spectral_profiles']
        excitations_used = results['excitations_used']
        n_components = results['n_components']

        # Check for valid data
        if np.all(np.abs(concentration_maps) < 1e-10):
            print("Warning: All concentration maps are near zero - results may not be meaningful")

        # Enhanced concentration maps visualization
        fig, axes = plt.subplots(1, n_components, figsize=(n_components * 4, 4))
        if n_components == 1:
            axes = [axes]

        for i in range(n_components):
            component_map = concentration_maps[:, :, i]

            # Use robust colormap limits
            vmin, vmax = np.percentile(component_map[np.isfinite(component_map)], [1, 99])
            if vmax - vmin < 1e-10:  # Nearly constant
                vmin, vmax = np.min(component_map), np.max(component_map)
                if vmax - vmin < 1e-10:
                    vmin, vmax = -0.1, 0.1

            im = axes[i].imshow(component_map, cmap='viridis', vmin=vmin, vmax=vmax)
            axes[i].set_title(f"Component {i + 1}\n(Range: {vmin:.4f} to {vmax:.4f})")
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

        plt.tight_layout()
        if output_dir:
            plt.savefig(os.path.join(output_dir, f"{filename_prefix}_concentration_maps.png"), dpi=300)

        if show_plots:
            plt.show()
        else:
            plt.close()

        # Add diagnostic information
        if 'data_info' in results:
            print("\nData Diagnostics:")
            for ex, info in results['data_info'].items():
                print(f"Excitation {ex}:")
                print(f"  Shape: {info['shape']}")
                print(f"  Range: [{info['min']:.6f}, {info['max']:.6f}]")
                print(f"  Mean±Std: {info['mean']:.6f}±{info['std']:.6f}")
                if info['nan_count'] > 0:
                    print(f"  NaN values: {info['nan_count']}")


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
import warnings
import os
from typing import Dict, List, Tuple, Optional, Union


class UltraRobustMCR:
    """
    Ultra-robust MCR implementation using Non-Negative Least Squares and SVD.
    """

    def __init__(self,
                 n_components: int = 3,
                 max_iter: int = 50,
                 tol: float = 1e-4,  # Relaxed tolerance
                 method: str = 'nnls',  # 'nnls', 'svd', 'regularized'
                 alpha: float = 0.01,  # Regularization parameter
                 verbose: bool = True):
        """
        Initialize ultra-robust MCR.

        Args:
            n_components: Number of components
            max_iter: Maximum iterations
            tol: Tolerance for convergence
            method: Optimization method ('nnls', 'svd', 'regularized')
            alpha: Regularization strength
            verbose: Print progress
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.method = method
        self.alpha = alpha
        self.verbose = verbose

        # Results
        self.C = None
        self.S = None
        self.explained_variance = None
        self.reconstruction_error = None

    def _safe_svd(self, X: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Safe SVD with multiple fallback methods."""
        methods = ['svd', 'truncated_svd', 'pca']

        for method in methods:
            try:
                if method == 'svd':
                    U, s, Vt = np.linalg.svd(X, full_matrices=False)
                    # Keep only n_components
                    U = U[:, :n_components]
                    s = s[:n_components]
                    Vt = Vt[:n_components, :]

                elif method == 'truncated_svd':
                    svd = TruncatedSVD(n_components=n_components, random_state=42)
                    U = svd.fit_transform(X)
                    s = svd.singular_values_
                    Vt = svd.components_

                elif method == 'pca':
                    pca = PCA(n_components=n_components, random_state=42)
                    U = pca.fit_transform(X)
                    s = np.sqrt(pca.explained_variance_)
                    Vt = pca.components_

                # Validate results
                if np.all(np.isfinite(U)) and np.all(np.isfinite(s)) and np.all(np.isfinite(Vt)):
                    if self.verbose:
                        print(f"Successful SVD using method: {method}")
                    return U, s, Vt

            except Exception as e:
                if self.verbose:
                    print(f"SVD method {method} failed: {str(e)}")
                continue

        # Last resort
        if self.verbose:
            print("All SVD methods failed, using random initialization")
        U = np.random.randn(X.shape[0], n_components) * 0.01
        s = np.ones(n_components)
        Vt = np.random.randn(n_components, X.shape[1]) * 0.01
        return U, s, Vt

    def _robust_nnls(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Robust Non-Negative Least Squares."""
        if len(b.shape) == 1:
            # Single vector
            try:
                x, _ = nnls(A, b)
                return x
            except:
                # Fallback to regularized solution
                return self._regularized_solve(A, b.reshape(-1, 1)).flatten()
        else:
            # Multiple vectors
            solutions = []
            for i in range(b.shape[1]):
                try:
                    x, _ = nnls(A, b[:, i])
                    solutions.append(x)
                except:
                    # Fallback
                    x = self._regularized_solve(A, b[:, i:i + 1]).flatten()
                    solutions.append(x)
            return np.column_stack(solutions)

    def _regularized_solve(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Regularized least squares solution."""
        try:
            # Ridge regression
            AtA = A.T @ A
            regularized = AtA + self.alpha * np.eye(AtA.shape[0])
            return np.linalg.solve(regularized, A.T @ b)
        except:
            # Use pseudo-inverse as last resort
            return np.linalg.pinv(A, rcond=1e-3) @ b

    def _safe_normalize_rows(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Safely normalize rows and return scaling factors."""
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        # Avoid division by zero
        safe_norms = np.where(norms < 1e-12, 1.0, norms)
        normalized = matrix / safe_norms
        return normalized, norms.flatten()

    def _calculate_safe_metrics(self, X: np.ndarray, X_recon: np.ndarray) -> Tuple[float, float]:
        """Calculate metrics safely."""
        try:
            # Mask for finite values
            finite_mask = np.isfinite(X) & np.isfinite(X_recon)

            if np.sum(finite_mask) == 0:
                return 0.0, float('inf')

            X_finite = X[finite_mask]
            X_recon_finite = X_recon[finite_mask]

            # Reconstruction error
            mse = np.mean((X_finite - X_recon_finite) ** 2)

            # Explained variance
            total_var = np.var(X_finite)
            if total_var > 1e-12:
                residual_var = np.var(X_finite - X_recon_finite)
                explained_var = 1 - (residual_var / total_var)
            else:
                explained_var = 0.0

            return max(0.0, min(1.0, explained_var)), max(0.0, mse)

        except:
            return 0.0, float('inf')

    def fit_single_cube(self, cube: np.ndarray, excitation: Union[float, str] = None) -> Dict:
        """Fit MCR to single cube with ultra-robust methods."""
        if self.verbose:
            print(f"Fitting ultra-robust MCR to cube shape: {cube.shape}")

        # Unfold cube
        X = self._unfold_cube(cube)
        return self._fit_data_matrix(X, excitation)

    def fit_multiple_cubes(self, cubes: Dict[Union[float, str], np.ndarray]) -> Dict:
        """Fit MCR to multiple cubes."""
        if self.verbose:
            print(f"Fitting ultra-robust MCR to {len(cubes)} cubes")

        # Combine all cubes
        combined_data = []
        excitations = list(cubes.keys())

        for ex in excitations:
            cube = cubes[ex]
            X = self._unfold_cube(cube)
            combined_data.append(X)

        # Concatenate along spectral dimension
        X_combined = np.hstack(combined_data)

        if self.verbose:
            print(f"Combined data shape: {X_combined.shape}")

        return self._fit_data_matrix(X_combined, excitations)

    def _fit_data_matrix(self, X: np.ndarray, excitations: Union[str, List]) -> Dict:
        """Core MCR fitting routine."""
        original_shape = X.shape

        if self.verbose:
            print(f"Data matrix shape: {X.shape}")
            print(f"Data range: [{np.nanmin(X):.6f}, {np.nanmax(X):.6f}]")
            print(f"NaN count: {np.sum(np.isnan(X))}")
            print(f"Inf count: {np.sum(np.isinf(X))}")

        # Handle problematic values
        X_clean = np.copy(X)

        # Replace NaN and Inf
        nan_mask = ~np.isfinite(X_clean)
        if np.any(nan_mask):
            X_clean[nan_mask] = 0
            if self.verbose:
                print(f"Replaced {np.sum(nan_mask)} non-finite values with zeros")

        # Check if data is essentially zero
        data_magnitude = np.abs(X_clean).max()
        if data_magnitude < 1e-10:
            if self.verbose:
                print("Warning: Data is essentially zero")
            return self._create_zero_results(original_shape, excitations)

        # Scale data to reasonable range
        scale_factor = 1.0 / data_magnitude
        X_scaled = X_clean * scale_factor

        if self.verbose:
            print(f"Scaled data range: [{X_scaled.min():.6f}, {X_scaled.max():.6f}]")

        # Initial decomposition using SVD
        try:
            U, s, Vt = self._safe_svd(X_scaled, self.n_components)

            # Initialize C and S
            C = U * s  # Concentration profiles
            S = Vt  # Spectral profiles

            if self.verbose:
                print(f"Initial decomposition successful")
                print(f"Singular values: {s}")

        except Exception as e:
            if self.verbose:
                print(f"Initial decomposition failed: {str(e)}")
            return self._create_zero_results(original_shape, excitations)

        # Iterative refinement
        best_error = float('inf')
        best_C, best_S = C.copy(), S.copy()
        no_improvement = 0

        for iteration in range(self.max_iter):
            try:
                # Update C (fix S)
                if self.method == 'nnls':
                    C_new = self._robust_nnls(S.T, X_scaled.T).T
                else:
                    C_new = self._regularized_solve(S.T, X_scaled.T).T

                # Ensure non-negative
                C_new = np.maximum(C_new, 0)

                # Update S (fix C)
                if self.method == 'nnls':
                    S_new = self._robust_nnls(C_new, X_scaled)
                else:
                    S_new = self._regularized_solve(C_new, X_scaled)

                # Ensure non-negative
                S_new = np.maximum(S_new, 0)

                # Normalize to prevent scaling issues
                if np.any(S_new > 0):
                    S_new, norms = self._safe_normalize_rows(S_new)
                    C_new = C_new * norms

                # Check for numerical issues
                if not (np.all(np.isfinite(C_new)) and np.all(np.isfinite(S_new))):
                    if self.verbose:
                        print(f"Iteration {iteration}: Non-finite values detected, stopping")
                    break

                # Calculate reconstruction
                X_recon = C_new @ S_new

                # Calculate error
                explained_var, mse = self._calculate_safe_metrics(X_scaled, X_recon)

                if self.verbose and (iteration % 10 == 0 or iteration < 5):
                    print(f"Iteration {iteration}: MSE = {mse:.6f}, Explained variance = {explained_var:.6f}")

                # Check for improvement
                if mse < best_error:
                    best_error = mse
                    best_C, best_S = C_new.copy(), S_new.copy()
                    no_improvement = 0
                else:
                    no_improvement += 1

                # Check convergence
                if no_improvement >= 10 or mse < self.tol:
                    if self.verbose:
                        print(f"Converged at iteration {iteration}")
                    break

                # Prevent explosion
                if mse > 100 or not np.isfinite(mse):
                    if self.verbose:
                        print(f"Iteration {iteration}: Error explosion detected, reverting to best solution")
                    break

                C, S = C_new, S_new

            except Exception as e:
                if self.verbose:
                    print(f"Iteration {iteration} failed: {str(e)}, using best solution so far")
                break

        # Use best solution
        C, S = best_C, best_S

        # Scale back results
        C = C / scale_factor

        # Final metrics
        X_final_recon = C @ S
        explained_var, mse = self._calculate_safe_metrics(X_clean, X_final_recon)

        if self.verbose:
            print(f"Final results:")
            print(f"  Explained variance: {explained_var:.6f}")
            print(f"  Reconstruction error: {mse:.6f}")
            print(f"  C range: [{C.min():.6f}, {C.max():.6f}]")
            print(f"  S range: [{S.min():.6f}, {S.max():.6f}]")

        # Store results
        self.C = C
        self.S = S
        self.explained_variance = explained_var
        self.reconstruction_error = mse

        # Create spatial maps
        if len(original_shape) == 2:
            # For combined data, need to determine spatial shape
            spatial_shape = (int(np.sqrt(original_shape[0])), int(np.sqrt(original_shape[0])))
            if spatial_shape[0] * spatial_shape[1] != original_shape[0]:
                spatial_shape = (original_shape[0], 1)  # Linear arrangement
        else:
            spatial_shape = (original_shape[0], 1)

        concentration_maps = self._fold_to_maps(C, spatial_shape)

        return {
            'concentration_maps': concentration_maps,
            'spectral_profiles': S,
            'explained_variance': explained_var,
            'reconstruction_error': mse,
            'excitations_used': excitations if isinstance(excitations, list) else [excitations],
            'n_components': self.n_components
        }

    def _create_zero_results(self, shape: Tuple, excitations: Union[str, List]) -> Dict:
        """Create zero results when fitting fails."""
        spatial_shape = (int(np.sqrt(shape[0])), int(np.sqrt(shape[0])))
        if spatial_shape[0] * spatial_shape[1] != shape[0]:
            spatial_shape = (shape[0], 1)

        return {
            'concentration_maps': np.zeros((*spatial_shape, self.n_components)),
            'spectral_profiles': np.zeros((self.n_components, shape[1])),
            'explained_variance': 0.0,
            'reconstruction_error': float('inf'),
            'excitations_used': excitations if isinstance(excitations, list) else [excitations],
            'n_components': self.n_components
        }

    def _unfold_cube(self, cube: np.ndarray) -> np.ndarray:
        """Unfold cube to matrix."""
        if len(cube.shape) == 3:
            if cube.shape[0] < cube.shape[1] and cube.shape[0] < cube.shape[2]:
                cube_t = np.transpose(cube, (1, 2, 0))
            elif cube.shape[2] < cube.shape[0] and cube.shape[2] < cube.shape[1]:
                cube_t = cube
            else:
                cube_t = np.transpose(cube, (1, 2, 0))
        else:
            raise ValueError(f"Expected 3D array, got shape {cube.shape}")

        height, width, bands = cube_t.shape
        return cube_t.reshape(height * width, bands)

    def _fold_to_maps(self, C: np.ndarray, spatial_shape: Tuple[int, int]) -> np.ndarray:
        """Fold concentration profiles back to spatial maps."""
        height, width = spatial_shape
        if height * width == C.shape[0]:
            return C.reshape(height, width, self.n_components)
        else:
            # If shapes don't match, create a square-ish arrangement
            side = int(np.sqrt(C.shape[0]))
            return C[:side * side].reshape(side, side, self.n_components)

    def visualize_results(self, results: Dict, output_dir: Optional[str] = None,
                          filename_prefix: str = "mcr", show_plots: bool = True) -> None:
        """Enhanced visualization with better handling of problematic data."""
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        concentration_maps = results['concentration_maps']
        spectral_profiles = results['spectral_profiles']
        n_components = results['n_components']

        # Enhanced concentration maps
        fig, axes = plt.subplots(1, n_components, figsize=(n_components * 4, 4))
        if n_components == 1:
            axes = [axes]

        for i in range(n_components):
            component_map = concentration_maps[:, :, i]

            # Calculate robust limits
            finite_vals = component_map[np.isfinite(component_map)]
            if len(finite_vals) > 0:
                vmin, vmax = np.percentile(finite_vals, [5, 95])
                if vmax <= vmin:
                    vmin, vmax = finite_vals.min(), finite_vals.max()
                if vmax <= vmin:
                    vmin, vmax = -1, 1
            else:
                vmin, vmax = -1, 1

            im = axes[i].imshow(component_map, cmap='viridis', vmin=vmin, vmax=vmax)
            axes[i].set_title(f"Component {i + 1}\n(Range: {vmin:.4f} to {vmax:.4f})")
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

        plt.suptitle(f"MCR Results - Explained Variance: {results['explained_variance']:.4f}")
        plt.tight_layout()

        if output_dir:
            plt.savefig(os.path.join(output_dir, f"{filename_prefix}_concentration_maps.png"), dpi=300)

        if show_plots:
            plt.show()
        else:
            plt.close()


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from sklearn.decomposition import PCA, TruncatedSVD
import warnings
import os
from typing import Dict, List, Tuple, Optional, Union


class FixedUltraRobustMCR:
    """
    Fixed MCR implementation with proper spatial reconstruction and scaling.
    """

    def __init__(self,
                 n_components: int = 3,
                 max_iter: int = 30,
                 tol: float = 1e-4,
                 method: str = 'nnls',
                 alpha: float = 0.01,
                 normalize_components: bool = True,
                 verbose: bool = True):
        """Initialize fixed MCR."""
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.method = method
        self.alpha = alpha
        self.normalize_components = normalize_components
        self.verbose = verbose

        # Results and metadata
        self.C = None
        self.S = None
        self.explained_variance = None
        self.reconstruction_error = None
        self.spatial_shape = None
        self.original_data_info = None

    def _analyze_cube_structure(self, cube: np.ndarray) -> Dict:
        """Analyze cube structure to determine proper unfolding."""
        if self.verbose:
            print(f"Analyzing cube structure: {cube.shape}")

        # Determine which dimension is spectral
        if len(cube.shape) == 3:
            dims = cube.shape

            # The spectral dimension is usually the smallest
            if dims[0] <= dims[1] and dims[0] <= dims[2]:
                # (bands, height, width)
                bands_dim = 0
                spatial_dims = (dims[1], dims[2])
                format_type = "bands_first"
            elif dims[2] <= dims[0] and dims[2] <= dims[1]:
                # (height, width, bands)
                bands_dim = 2
                spatial_dims = (dims[0], dims[1])
                format_type = "bands_last"
            else:
                # Ambiguous - assume (height, width, bands)
                bands_dim = 2
                spatial_dims = (dims[0], dims[1])
                format_type = "assumed_bands_last"

            info = {
                'original_shape': dims,
                'bands_dim': bands_dim,
                'spatial_shape': spatial_dims,
                'n_bands': dims[bands_dim],
                'format_type': format_type,
                'n_spatial_pixels': spatial_dims[0] * spatial_dims[1]
            }

            if self.verbose:
                print(f"  Format: {format_type}")
                print(f"  Spatial dimensions: {spatial_dims}")
                print(f"  Spectral bands: {dims[bands_dim]}")
                print(f"  Total pixels: {info['n_spatial_pixels']}")

            return info
        else:
            raise ValueError(f"Expected 3D cube, got {len(cube.shape)}D")

    def _unfold_cube_proper(self, cube: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Properly unfold cube with detailed tracking."""
        info = self._analyze_cube_structure(cube)

        if info['format_type'] in ['bands_first', 'assumed_bands_last']:
            if info['bands_dim'] == 0:
                # (bands, height, width) -> (height, width, bands)
                cube_reoriented = np.transpose(cube, (1, 2, 0))
            else:
                # Already (height, width, bands)
                cube_reoriented = cube
        else:
            cube_reoriented = cube

        height, width, bands = cube_reoriented.shape

        # Unfold to (pixels, bands)
        X = cube_reoriented.reshape(height * width, bands)

        # Store spatial information
        info['unfolded_shape'] = X.shape
        info['reoriented_shape'] = cube_reoriented.shape

        if self.verbose:
            print(f"  Unfolded to: {X.shape}")
            print(f"  Data range: [{np.nanmin(X):.6f}, {np.nanmax(X):.6f}]")

        return X, info

    def _fold_to_spatial_maps(self, C: np.ndarray, spatial_info: Dict) -> np.ndarray:
        """Properly fold concentration profiles back to spatial maps."""
        height, width = spatial_info['spatial_shape']
        n_pixels_expected = height * width

        if C.shape[0] != n_pixels_expected:
            if self.verbose:
                print(f"Warning: Concentration matrix has {C.shape[0]} rows, expected {n_pixels_expected}")
            # Try to reshape anyway
            n_pixels_actual = C.shape[0]
            side = int(np.sqrt(n_pixels_actual))
            if side * side == n_pixels_actual:
                height, width = side, side
                if self.verbose:
                    print(f"  Reshaped to square: {height}x{width}")
            else:
                # Linear arrangement
                height, width = n_pixels_actual, 1
                if self.verbose:
                    print(f"  Using linear arrangement: {height}x{width}")

        # Reshape to spatial maps
        concentration_maps = C.reshape(height, width, self.n_components)

        if self.verbose:
            print(f"Concentration maps shape: {concentration_maps.shape}")
            for i in range(self.n_components):
                comp_map = concentration_maps[:, :, i]
                print(f"  Component {i + 1}: range [{comp_map.min():.6f}, {comp_map.max():.6f}]")

        return concentration_maps

    def _safe_svd_initialization(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Safe SVD-based initialization."""
        try:
            # Use TruncatedSVD for better numerical stability
            svd = TruncatedSVD(n_components=self.n_components, random_state=42)
            C_init = svd.fit_transform(X)
            S_init = svd.components_

            if self.verbose:
                print(f"SVD initialization successful")
                print(f"  Explained variance ratio: {svd.explained_variance_ratio_}")
                print(f"  Total explained variance: {svd.explained_variance_ratio_.sum():.4f}")

            return C_init, S_init

        except Exception as e:
            if self.verbose:
                print(f"SVD initialization failed: {str(e)}, using PCA fallback")

            try:
                pca = PCA(n_components=self.n_components, random_state=42)
                C_init = pca.fit_transform(X)
                S_init = pca.components_

                # Ensure non-negative
                C_init = np.abs(C_init)
                S_init = np.abs(S_init)

                return C_init, S_init

            except Exception as e2:
                if self.verbose:
                    print(f"PCA fallback failed: {str(e2)}, using random initialization")

                # Random initialization as last resort
                C_init = np.abs(np.random.randn(X.shape[0], self.n_components)) * 0.1
                S_init = np.abs(np.random.randn(self.n_components, X.shape[1])) * 0.1

                return C_init, S_init

    def _normalize_and_scale_components(self, C: np.ndarray, S: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Properly normalize and scale components to comparable ranges."""
        if not self.normalize_components:
            return C, S

        C_norm = np.copy(C)
        S_norm = np.copy(S)

        for i in range(self.n_components):
            # Get component vectors
            c_i = C_norm[:, i]
            s_i = S_norm[i, :]

            # Calculate norms
            c_norm = np.linalg.norm(c_i)
            s_norm = np.linalg.norm(s_i)

            if c_norm > 1e-12 and s_norm > 1e-12:
                # Normalize S to unit norm and scale C accordingly
                s_scale = s_norm
                S_norm[i, :] = s_i / s_scale
                C_norm[:, i] = c_i * s_scale

                # Further scale to reasonable range [0, 1]
                c_max = np.max(C_norm[:, i])
                if c_max > 1e-12:
                    scale = 1.0 / c_max
                    C_norm[:, i] *= scale
                    S_norm[i, :] /= scale

        if self.verbose:
            print("Component normalization completed:")
            for i in range(self.n_components):
                c_range = [C_norm[:, i].min(), C_norm[:, i].max()]
                s_range = [S_norm[i, :].min(), S_norm[i, :].max()]
                print(f"  Component {i + 1}: C range {c_range}, S range {s_range}")

        return C_norm, S_norm

    def _robust_nnls_solve(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Robust NNLS with fallbacks."""
        if len(b.shape) == 1:
            try:
                x, _ = nnls(A, b)
                return x
            except:
                # Fallback to regularized positive solution
                AtA = A.T @ A + self.alpha * np.eye(A.shape[1])
                x = np.linalg.solve(AtA, A.T @ b)
                return np.maximum(x, 0)
        else:
            # Multiple right-hand sides
            solutions = []
            for i in range(b.shape[1]):
                try:
                    x, _ = nnls(A, b[:, i])
                    solutions.append(x)
                except:
                    AtA = A.T @ A + self.alpha * np.eye(A.shape[1])
                    x = np.linalg.solve(AtA, A.T @ b[:, i])
                    solutions.append(np.maximum(x, 0))
            return np.column_stack(solutions)

    def fit_single_cube(self, cube: np.ndarray, excitation: Union[float, str] = None) -> Dict:
        """Fit MCR to single cube with proper spatial handling."""
        if self.verbose:
            print(f"\n=== Fitting MCR to single cube ===")
            print(f"Excitation: {excitation}")

        # Properly unfold the cube
        X, spatial_info = self._unfold_cube_proper(cube)
        self.spatial_shape = spatial_info['spatial_shape']

        # Store original data info
        self.original_data_info = {
            'cube_shape': cube.shape,
            'spatial_info': spatial_info,
            'data_range': [np.nanmin(X), np.nanmax(X)],
            'nan_count': np.sum(np.isnan(X))
        }

        # Clean data
        X_clean = self._clean_data(X)

        # Fit the model
        results = self._fit_mcr_model(X_clean, spatial_info)
        results['excitations_used'] = [excitation] if excitation else ['unknown']

        return results

    def fit_multiple_cubes(self, cubes: Dict[Union[float, str], np.ndarray]) -> Dict:
        """Fit MCR to multiple cubes."""
        if self.verbose:
            print(f"\n=== Fitting MCR to {len(cubes)} cubes ===")
            print(f"Excitations: {list(cubes.keys())}")

        # Process each cube and combine
        cube_data = []
        total_bands = 0
        spatial_info = None

        for ex, cube in cubes.items():
            X, info = self._unfold_cube_proper(cube)
            cube_data.append(X)
            total_bands += X.shape[1]

            if spatial_info is None:
                spatial_info = info
                self.spatial_shape = info['spatial_shape']

        # Combine all spectral data
        X_combined = np.hstack(cube_data)

        if self.verbose:
            print(f"Combined data shape: {X_combined.shape}")
            print(f"Total spectral bands: {total_bands}")

        # Store original data info
        self.original_data_info = {
            'cubes_info': {ex: cube.shape for ex, cube in cubes.items()},
            'spatial_info': spatial_info,
            'combined_shape': X_combined.shape,
            'data_range': [np.nanmin(X_combined), np.nanmax(X_combined)],
            'nan_count': np.sum(np.isnan(X_combined))
        }

        # Clean data
        X_clean = self._clean_data(X_combined)

        # Fit the model
        results = self._fit_mcr_model(X_clean, spatial_info)
        results['excitations_used'] = list(cubes.keys())

        return results

    def _clean_data(self, X: np.ndarray) -> np.ndarray:
        """Clean and preprocess data."""
        X_clean = np.copy(X)

        # Handle NaN and infinite values
        finite_mask = np.isfinite(X_clean)
        if not np.all(finite_mask):
            if self.verbose:
                print(f"Replacing {np.sum(~finite_mask)} non-finite values")
            X_clean[~finite_mask] = 0

        # Check for zero variance columns
        col_var = np.var(X_clean, axis=0)
        zero_var_cols = col_var < 1e-12
        if np.any(zero_var_cols):
            if self.verbose:
                print(f"Adding noise to {np.sum(zero_var_cols)} zero-variance columns")
            X_clean[:, zero_var_cols] += np.random.normal(0, 1e-6,
                                                          (X_clean.shape[0], np.sum(zero_var_cols)))

        return X_clean

    def _fit_mcr_model(self, X: np.ndarray, spatial_info: Dict) -> Dict:
        """Core MCR fitting with proper scaling."""
        if self.verbose:
            print(f"\nFitting MCR model:")
            print(f"  Data matrix: {X.shape}")
            print(f"  Components: {self.n_components}")

        # Initialize with SVD
        C, S = self._safe_svd_initialization(X)

        # Ensure non-negative
        C = np.abs(C)
        S = np.abs(S)

        # ALS iterations
        best_error = float('inf')
        best_C, best_S = C.copy(), S.copy()

        for iteration in range(self.max_iter):
            try:
                # Update C (fix S)
                C_new = self._robust_nnls_solve(S.T, X.T).T
                C_new = np.maximum(C_new, 1e-12)  # Ensure positive

                # Update S (fix C)
                S_new = self._robust_nnls_solve(C_new, X)
                S_new = np.maximum(S_new, 1e-12)  # Ensure positive

                # Check for convergence
                X_recon = C_new @ S_new
                mse = np.mean((X - X_recon) ** 2)

                if mse < best_error:
                    best_error = mse
                    best_C, best_S = C_new.copy(), S_new.copy()

                if self.verbose and (iteration % 10 == 0 or iteration < 5):
                    print(f"  Iteration {iteration}: MSE = {mse:.6f}")

                # Check convergence
                if iteration > 0 and abs(mse - prev_mse) < self.tol:
                    if self.verbose:
                        print(f"  Converged at iteration {iteration}")
                    break

                prev_mse = mse
                C, S = C_new, S_new

            except Exception as e:
                if self.verbose:
                    print(f"  Iteration {iteration} failed: {str(e)}")
                break

        # Use best solution and normalize
        C, S = self._normalize_and_scale_components(best_C, best_S)

        # Calculate final metrics
        X_recon = C @ S
        mse = np.mean((X - X_recon) ** 2)
        total_var = np.var(X)
        explained_var = 1 - (np.var(X - X_recon) / total_var) if total_var > 1e-12 else 0

        # Create spatial maps
        concentration_maps = self._fold_to_spatial_maps(C, spatial_info)

        if self.verbose:
            print(f"\nFinal results:")
            print(f"  Explained variance: {explained_var:.4f}")
            print(f"  Reconstruction error: {mse:.6f}")
            print(f"  Concentration maps shape: {concentration_maps.shape}")

        # Store results
        self.C = C
        self.S = S
        self.explained_variance = explained_var
        self.reconstruction_error = mse

        return {
            'concentration_maps': concentration_maps,
            'spectral_profiles': S,
            'explained_variance': explained_var,
            'reconstruction_error': mse,
            'n_components': self.n_components,
            'spatial_info': spatial_info
        }

    def visualize_results(self, results: Dict, output_dir: Optional[str] = None,
                          filename_prefix: str = "mcr", show_plots: bool = True) -> None:
        """Enhanced visualization with proper scaling."""
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        concentration_maps = results['concentration_maps']
        spectral_profiles = results['spectral_profiles']
        n_components = results['n_components']

        # 1. Concentration maps with proper scaling
        fig, axes = plt.subplots(2, n_components, figsize=(n_components * 4, 8))
        if n_components == 1:
            axes = axes.reshape(-1, 1)

        # Top row: Individual component maps
        for i in range(n_components):
            component_map = concentration_maps[:, :, i]

            # Calculate robust limits
            finite_vals = component_map[np.isfinite(component_map)]
            if len(finite_vals) > 10:
                vmin, vmax = np.percentile(finite_vals, [2, 98])
                if vmax <= vmin:
                    vmin, vmax = finite_vals.min(), finite_vals.max()
            else:
                vmin, vmax = component_map.min(), component_map.max()

            if vmax <= vmin:
                vmin, vmax = 0, 1

            im = axes[0, i].imshow(component_map, cmap='viridis', vmin=vmin, vmax=vmax)
            axes[0, i].set_title(f"Component {i + 1}\n(Range: {vmin:.3f} to {vmax:.3f})")
            axes[0, i].axis('off')
            plt.colorbar(im, ax=axes[0, i], fraction=0.046, pad=0.04)

        # Bottom row: RGB composite and statistics
        if n_components >= 3:
            # RGB composite using first 3 components
            rgb_composite = np.zeros((*concentration_maps.shape[:2], 3))
            for i in range(3):
                comp = concentration_maps[:, :, i]
                comp_norm = (comp - comp.min()) / (comp.max() - comp.min() + 1e-12)
                rgb_composite[:, :, i] = comp_norm

            axes[1, 0].imshow(rgb_composite)
            axes[1, 0].set_title('RGB Composite\n(Components 1-3)')
            axes[1, 0].axis('off')

            # Component statistics
            for i in range(1, n_components):
                if i < n_components:
                    axes[1, i].bar(range(n_components),
                                   [concentration_maps[:, :, j].mean() for j in range(n_components)],
                                   color=plt.cm.viridis(np.linspace(0, 1, n_components)))
                    axes[1, i].set_title('Component Means')
                    axes[1, i].set_xlabel('Component')
                    axes[1, i].set_ylabel('Mean Concentration')
                else:
                    axes[1, i].axis('off')
        else:
            # For fewer components, show histograms
            for i in range(n_components):
                component_map = concentration_maps[:, :, i]
                axes[1, i].hist(component_map.flatten(), bins=50, alpha=0.7,
                                color=plt.cm.viridis(i / n_components))
                axes[1, i].set_title(f'Component {i + 1} Histogram')
                axes[1, i].set_xlabel('Concentration')
                axes[1, i].set_ylabel('Frequency')

        plt.suptitle(f"MCR Results - Explained Variance: {results['explained_variance']:.4f}")
        plt.tight_layout()

        if output_dir:
            plt.savefig(os.path.join(output_dir, f"{filename_prefix}_concentration_analysis.png"),
                        dpi=300, bbox_inches='tight')

        if show_plots:
            plt.show()
        else:
            plt.close()

        # 2. Spectral profiles
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = plt.cm.tab10(np.arange(n_components))

        for i in range(n_components):
            ax.plot(spectral_profiles[i], color=colors[i],
                    label=f'Component {i + 1}', linewidth=2)

        ax.set_xlabel('Spectral Channel')
        ax.set_ylabel('Intensity')
        ax.set_title('MCR Spectral Profiles')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if output_dir:
            plt.savefig(os.path.join(output_dir, f"{filename_prefix}_spectral_profiles.png"),
                        dpi=300, bbox_inches='tight')

        if show_plots:
            plt.show()
        else:
            plt.close()

        # Print detailed diagnostics
        if self.verbose:
            print(f"\n=== MCR Results Summary ===")
            print(f"Explained variance: {results['explained_variance']:.4f}")
            print(f"Reconstruction error: {results['reconstruction_error']:.6f}")
            print(f"Concentration maps shape: {concentration_maps.shape}")
            print(f"Component value ranges:")
            for i in range(n_components):
                comp_map = concentration_maps[:, :, i]
                print(f"  Component {i + 1}: [{comp_map.min():.6f}, {comp_map.max():.6f}] "
                      f"(mean: {comp_map.mean():.6f}, std: {comp_map.std():.6f})")