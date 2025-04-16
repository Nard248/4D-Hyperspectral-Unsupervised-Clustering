import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os
import pandas as pd
from tensorly.decomposition import parafac
import tensorly as tl
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches


class DimensionalityReductionAnalyzer:
    """Class for analyzing hyperspectral data using dimensionality reduction techniques"""

    def __init__(self, h5_file_path):
        """
        Initialize with the path to the HDF5 file

        Args:
            h5_file_path: Path to the HDF5 file containing hyperspectral data
        """
        self.h5_file_path = h5_file_path
        self.data_dict = {}
        self.excitation_wavelengths = []
        self.emission_wavelengths = {}
        self.unfolded_data = None
        self.pca_model = None
        self.pca_components = None
        self.parafac_factors = None

        # Load the data
        self.load_data()

    def load_data(self):
        """Load the hyperspectral data from the HDF5 file"""
        with h5py.File(self.h5_file_path, 'r') as f:
            # Read each excitation group
            for group_name in f.keys():
                if group_name.startswith('excitation_'):
                    # Extract excitation wavelength from group name
                    excitation_wavelength = int(group_name.split('_')[1])
                    self.excitation_wavelengths.append(excitation_wavelength)

                    group = f[group_name]
                    excitation_data = {
                        'wavelengths': group['wavelengths'][:],
                        'average_cube': None
                    }

                    # Store emission wavelengths for this excitation
                    self.emission_wavelengths[excitation_wavelength] = group['wavelengths'][:]

                    # Load the average cube if available
                    if 'average_cube' in group:
                        excitation_data['average_cube'] = group['average_cube'][:]

                    self.data_dict[excitation_wavelength] = excitation_data

            # Sort excitation wavelengths
            self.excitation_wavelengths.sort()

            # Check if mask is present
            self.mask = None
            if 'mask' in f:
                self.mask = f['mask']['data'][:]
                print(f"Mask found with {np.sum(self.mask)} masked pixels")

    def prepare_data_for_pca(self):
        """
        Prepare the data for PCA analysis by unfolding the 4D data into a 2D matrix

        The data is restructured as:
        Rows = Pixels (spatial positions)
        Columns = All excitation-emission combinations (flattened spectral dimension)

        Returns:
            Unfolded data matrix, column labels (ex_em combinations)
        """
        # First, check if we have all the necessary data
        if not self.data_dict:
            raise ValueError("No data loaded. Please call load_data() first.")

        # Count total number of emission wavelengths across all excitations
        total_features = sum(len(self.emission_wavelengths[ex]) for ex in self.excitation_wavelengths)

        # Get spatial dimensions from the first data cube
        first_ex = self.excitation_wavelengths[0]
        if 'average_cube' not in self.data_dict[first_ex] or self.data_dict[first_ex]['average_cube'] is None:
            raise ValueError(f"No average cube found for excitation {first_ex} nm")

        height, width = self.data_dict[first_ex]['average_cube'].shape[1:]
        total_pixels = height * width

        # Initialize the unfolded data matrix
        # Rows = pixels, Columns = excitation-emission combinations
        unfolded_data = np.zeros((total_pixels, total_features))

        # Keep track of column labels (ex_em combinations)
        col_labels = []

        # Current column index
        col_idx = 0

        # For each excitation wavelength
        for ex_wave in self.excitation_wavelengths:
            # Get emission wavelengths for this excitation
            em_waves = self.emission_wavelengths[ex_wave]

            # Get the data cube for this excitation
            cube = self.data_dict[ex_wave]['average_cube']

            # For each emission wavelength
            for band_idx, em_wave in enumerate(em_waves):
                # Get the 2D image for this excitation-emission combination
                img = cube[band_idx]

                # Flatten the image into a column vector
                unfolded_data[:, col_idx] = img.flatten()

                # Store the label for this column
                col_labels.append((ex_wave, em_wave))

                # Increment column index
                col_idx += 1

        # Apply mask if available
        if self.mask is not None:
            # Flatten the mask
            mask_flat = self.mask.flatten()

            # Keep only the masked pixels
            unfolded_data = unfolded_data[mask_flat == 1, :]

        # Store the unfolded data
        self.unfolded_data = unfolded_data
        self.col_labels = col_labels

        print(f"Unfolded data shape: {unfolded_data.shape}")
        return unfolded_data, col_labels

    def run_pca(self, n_components=10, scale_data=True):
        """
        Perform PCA on the unfolded data

        Args:
            n_components: Number of principal components to extract
            scale_data: Whether to standardize the data before PCA

        Returns:
            PCA model, transformed data
        """
        # Prepare the data if not already done
        if self.unfolded_data is None:
            self.prepare_data_for_pca()

        # Standardize the data if requested
        if scale_data:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(self.unfolded_data)
        else:
            scaled_data = self.unfolded_data

        # Run PCA
        pca = PCA(n_components=n_components)
        transformed_data = pca.fit_transform(scaled_data)

        # Store the PCA model and components
        self.pca_model = pca
        self.pca_components = pca.components_
        self.pca_transformed = transformed_data

        # Calculate explained variance ratio
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)

        print(f"PCA completed with {n_components} components")
        print(f"Explained variance ratio: {explained_variance}")
        print(f"Cumulative explained variance: {cumulative_variance}")

        return pca, transformed_data

    def plot_pca_explained_variance(self, output_dir='analysis_results'):
        """
        Plot the explained variance ratio and cumulative explained variance for PCA

        Args:
            output_dir: Directory to save the plot
        """
        if self.pca_model is None:
            raise ValueError("No PCA model available. Please call run_pca() first.")

        # Create the plot
        plt.figure(figsize=(12, 6))

        # Plot explained variance ratio
        plt.bar(range(1, len(self.pca_model.explained_variance_ratio_) + 1),
                self.pca_model.explained_variance_ratio_, alpha=0.7, label='Explained Variance')

        # Plot cumulative explained variance
        plt.step(range(1, len(self.pca_model.explained_variance_ratio_) + 1),
                 np.cumsum(self.pca_model.explained_variance_ratio_), where='mid',
                 label='Cumulative Explained Variance')

        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('PCA Explained Variance')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Mark 90% and 95% thresholds
        cumulative = np.cumsum(self.pca_model.explained_variance_ratio_)
        n_90 = np.argmax(cumulative >= 0.9) + 1
        n_95 = np.argmax(cumulative >= 0.95) + 1

        plt.axhline(y=0.9, color='r', linestyle='--', alpha=0.5)
        plt.axhline(y=0.95, color='g', linestyle='--', alpha=0.5)

        plt.text(n_90 + 0.5, 0.9, f'{n_90} components for 90%',
                 verticalalignment='bottom', horizontalalignment='left')
        plt.text(n_95 + 0.5, 0.95, f'{n_95} components for 95%',
                 verticalalignment='bottom', horizontalalignment='left')

        # Save the plot
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, 'pca_explained_variance.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()

        print(f"PCA explained variance plot saved to {plot_path}")
        return plot_path

    def identify_important_wavelengths_from_pca(self, n_components=3, n_wavelengths=5):
        """
        Identify the most important excitation-emission combinations from PCA loadings

        Args:
            n_components: Number of top principal components to consider
            n_wavelengths: Number of top wavelengths to identify per component

        Returns:
            DataFrame with important wavelengths
        """
        if self.pca_model is None:
            raise ValueError("No PCA model available. Please call run_pca() first.")

        # Limit to the requested number of components
        components = self.pca_components[:n_components]

        results = []

        # For each component
        for comp_idx, component in enumerate(components):
            # Take absolute values to consider both positive and negative loadings
            abs_component = np.abs(component)

            # Find the indices of the top n_wavelengths
            top_indices = np.argsort(abs_component)[-n_wavelengths:][::-1]

            # For each top index
            for rank, idx in enumerate(top_indices):
                # Get the corresponding excitation-emission combination
                ex_wave, em_wave = self.col_labels[idx]

                # Store the result
                results.append({
                    'Component': comp_idx + 1,
                    'Rank': rank + 1,
                    'Excitation': ex_wave,
                    'Emission': em_wave,
                    'Loading': component[idx],
                    'AbsLoading': abs_component[idx]
                })

        # Convert to DataFrame
        important_wavelengths = pd.DataFrame(results)
        return important_wavelengths

    def plot_pca_loadings(self, component=0, output_dir='analysis_results'):
        """
        Plot the loadings of a specific principal component as a heat map

        Args:
            component: Index of the component to plot (0-based)
            output_dir: Directory to save the plot
        """
        if self.pca_model is None:
            raise ValueError("No PCA model available. Please call run_pca() first.")

        # Check if the requested component is available
        if component >= len(self.pca_components):
            raise ValueError(
                f"Component {component} not available. Only {len(self.pca_components)} components were extracted.")

        # Get the loadings for the requested component
        loadings = self.pca_components[component]

        # Create a dictionary to store loadings by excitation-emission
        loading_dict = {}

        # For each loading
        for i, (ex_wave, em_wave) in enumerate(self.col_labels):
            loading_dict[(ex_wave, em_wave)] = loadings[i]

        # Create a DataFrame for the heat map
        excitations = sorted(self.excitation_wavelengths)

        # Find the union of all emission wavelengths
        all_emissions = set()
        for ex_wave in excitations:
            all_emissions.update(self.emission_wavelengths[ex_wave])
        all_emissions = sorted(all_emissions)

        # Create a 2D array for the heat map
        heatmap_data = np.zeros((len(all_emissions), len(excitations)))
        heatmap_data.fill(np.nan)  # Fill with NaN for missing values

        # Fill the heat map data
        for i, em_wave in enumerate(all_emissions):
            for j, ex_wave in enumerate(excitations):
                if (ex_wave, em_wave) in loading_dict:
                    heatmap_data[i, j] = loading_dict[(ex_wave, em_wave)]

        # Create the heat map
        plt.figure(figsize=(12, 10))

        # Use diverging colormap centered at zero
        cmap = sns.diverging_palette(240, 10, as_cmap=True)

        sns.heatmap(heatmap_data, cmap=cmap, center=0,
                    xticklabels=excitations,
                    yticklabels=all_emissions[::10])  # Show every 10th emission wavelength for readability

        plt.title(f'PCA Component {component + 1} Loadings')
        plt.xlabel('Excitation Wavelength (nm)')
        plt.ylabel('Emission Wavelength (nm)')

        # Save the plot
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, f'pca_loadings_component_{component + 1}.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()

        print(f"PCA loadings plot saved to {plot_path}")
        return plot_path

    def visualize_pca_in_rgb(self, output_dir='analysis_results'):
        """
        Visualize the first three principal components as an RGB image

        Args:
            output_dir: Directory to save the plot
        """
        if self.pca_model is None or self.pca_transformed is None:
            raise ValueError("No PCA results available. Please call run_pca() first.")

        # Get the first three PCA components
        if self.pca_transformed.shape[1] < 3:
            raise ValueError("Need at least 3 PCA components for RGB visualization")

        # Get spatial dimensions from the first data cube
        first_ex = self.excitation_wavelengths[0]
        height, width = self.data_dict[first_ex]['average_cube'].shape[1:]

        # Reshape PCA components to image dimensions
        rgb_components = []

        for i in range(3):
            # Extract the component
            component = self.pca_transformed[:, i]

            # Normalize to [0, 1] for RGB
            component_min = component.min()
            component_max = component.max()
            normalized = (component - component_min) / (component_max - component_min)

            # Reshape to image dimensions
            if self.mask is not None:
                # Initialize with zeros
                img = np.zeros((height * width))
                # Fill in the masked pixels
                mask_flat = self.mask.flatten()
                img[mask_flat == 1] = normalized
                # Reshape to image dimensions
                img = img.reshape(height, width)
            else:
                img = normalized.reshape(height, width)

            rgb_components.append(img)

        # Stack the components to create an RGB image
        rgb_image = np.stack(rgb_components, axis=2)

        # Create the plot
        plt.figure(figsize=(10, 8))
        plt.imshow(rgb_image)
        plt.title('PCA Components as RGB (Red=PC1, Green=PC2, Blue=PC3)')
        plt.axis('off')

        # Create legend to explain the colors
        cmap = ListedColormap(['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'white'])
        legend_elements = [
            mpatches.Patch(color='red', label='PC1 High'),
            mpatches.Patch(color='green', label='PC2 High'),
            mpatches.Patch(color='blue', label='PC3 High'),
            mpatches.Patch(color='yellow', label='PC1+PC2 High'),
            mpatches.Patch(color='cyan', label='PC2+PC3 High'),
            mpatches.Patch(color='magenta', label='PC1+PC3 High'),
            mpatches.Patch(color='white', label='All High')
        ]
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

        # Save the plot
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, 'pca_rgb_visualization.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"PCA RGB visualization saved to {plot_path}")
        return plot_path

    def prepare_data_for_parafac(self):
        """
        Prepare the data for PARAFAC analysis by constructing a 4D tensor

        The tensor has dimensions:
        - Spatial rows
        - Spatial columns
        - Excitation wavelengths
        - Emission wavelengths

        Since emission wavelengths can vary by excitation, we need to interpolate
        to a common grid for consistent tensor dimensions.

        Returns:
            4D tensor, excitation wavelengths, common emission wavelengths
        """
        # First, check if we have all the necessary data
        if not self.data_dict:
            raise ValueError("No data loaded. Please call load_data() first.")

        # Get spatial dimensions from the first data cube
        first_ex = self.excitation_wavelengths[0]
        if 'average_cube' not in self.data_dict[first_ex] or self.data_dict[first_ex]['average_cube'] is None:
            raise ValueError(f"No average cube found for excitation {first_ex} nm")

        height, width = self.data_dict[first_ex]['average_cube'].shape[1:]

        # Create a common grid for emission wavelengths
        # Find the min and max emission wavelengths across all excitations
        min_em = min(self.emission_wavelengths[ex].min() for ex in self.excitation_wavelengths)
        max_em = max(self.emission_wavelengths[ex].max() for ex in self.excitation_wavelengths)

        # Create a common grid with a reasonable number of points
        n_emission_points = 100  # We can adjust this based on the original resolution
        common_emission = np.linspace(min_em, max_em, n_emission_points)

        # Initialize the 4D tensor
        # Dimensions: spatial_rows x spatial_cols x excitation x emission
        tensor_4d = np.zeros((height, width, len(self.excitation_wavelengths), n_emission_points))

        # If there's a mask, prepare a tensor with only masked pixels to save memory
        if self.mask is not None:
            # Count masked pixels
            n_masked_pixels = np.sum(self.mask)

            # Create a tensor with only masked pixels
            # Dimensions: masked_pixels x excitation x emission
            masked_tensor = np.zeros((n_masked_pixels, len(self.excitation_wavelengths), n_emission_points))

            # Get indices of masked pixels
            mask_indices = np.where(self.mask == 1)
            mask_coords = list(zip(mask_indices[0], mask_indices[1]))

            # For each excitation wavelength
            for ex_idx, ex_wave in enumerate(self.excitation_wavelengths):
                # Get emission wavelengths and data for this excitation
                em_waves = self.emission_wavelengths[ex_wave]
                cube = self.data_dict[ex_wave]['average_cube']

                # For each masked pixel
                for pixel_idx, (row, col) in enumerate(mask_coords):
                    # Extract the spectrum for this pixel
                    pixel_spectrum = np.array([cube[band_idx, row, col] for band_idx in range(len(em_waves))])

                    # Interpolate to the common emission grid
                    # Handle cases where the pixel has NaN or Inf values
                    if np.any(~np.isfinite(pixel_spectrum)):
                        # Skip this pixel or fill with zeros
                        continue

                    # Interpolate
                    from scipy.interpolate import interp1d
                    try:
                        interp_func = interp1d(em_waves, pixel_spectrum, bounds_error=False, fill_value=0)
                        interp_spectrum = interp_func(common_emission)

                        # Store in the masked tensor
                        masked_tensor[pixel_idx, ex_idx, :] = interp_spectrum
                    except:
                        # Skip if interpolation fails
                        continue

            # Store the masked tensor
            self.masked_tensor = masked_tensor
            self.mask_coords = mask_coords
            print(f"Created masked tensor of shape {masked_tensor.shape}")

            return masked_tensor, self.excitation_wavelengths, common_emission

        else:
            # If no mask, process the full tensor
            # For each excitation wavelength
            for ex_idx, ex_wave in enumerate(self.excitation_wavelengths):
                # Get emission wavelengths and data for this excitation
                em_waves = self.emission_wavelengths[ex_wave]
                cube = self.data_dict[ex_wave]['average_cube']

                # For each pixel
                for row in range(height):
                    for col in range(width):
                        # Extract the spectrum for this pixel
                        pixel_spectrum = np.array([cube[band_idx, row, col] for band_idx in range(len(em_waves))])

                        # Skip if the spectrum has non-finite values
                        if np.any(~np.isfinite(pixel_spectrum)):
                            continue

                        # Interpolate to the common emission grid
                        from scipy.interpolate import interp1d
                        try:
                            interp_func = interp1d(em_waves, pixel_spectrum, bounds_error=False, fill_value=0)
                            interp_spectrum = interp_func(common_emission)

                            # Store in the tensor
                            tensor_4d[row, col, ex_idx, :] = interp_spectrum
                        except:
                            # Skip if interpolation fails
                            continue

            # Store the tensor
            self.tensor_4d = tensor_4d
            print(f"Created 4D tensor of shape {tensor_4d.shape}")

            return tensor_4d, self.excitation_wavelengths, common_emission

    def run_parafac(self, n_components=3, random_state=0):
        """
        Perform PARAFAC decomposition on the 4D tensor

        Args:
            n_components: Number of components to extract
            random_state: Random seed for initialization

        Returns:
            PARAFAC factors
        """
        # Prepare the data if not already done
        if not hasattr(self, 'tensor_4d') and not hasattr(self, 'masked_tensor'):
            tensor, excitations, emissions = self.prepare_data_for_parafac()
        else:
            tensor = self.masked_tensor if hasattr(self, 'masked_tensor') else self.tensor_4d
            excitations = self.excitation_wavelengths

            # Estimate emissions from the tensor shape
            if hasattr(self, 'masked_tensor'):
                n_emission_points = self.masked_tensor.shape[2]
            else:
                n_emission_points = self.tensor_4d.shape[3]

            min_em = min(self.emission_wavelengths[ex].min() for ex in self.excitation_wavelengths)
            max_em = max(self.emission_wavelengths[ex].max() for ex in self.excitation_wavelengths)
            emissions = np.linspace(min_em, max_em, n_emission_points)

        # Make sure all values are finite before PARAFAC
        tensor = np.nan_to_num(tensor)

        # Ensure the tensor is in the format expected by TensorLy
        tensor = tl.tensor(tensor)

        # Run PARAFAC
        try:
            factors = parafac(tensor, rank=n_components, init='random', random_state=random_state)
            self.parafac_factors = factors
            self.parafac_emissions = emissions

            print(f"PARAFAC completed with {n_components} components")
            return factors
        except Exception as e:
            print(f"Error running PARAFAC: {e}")
            # If full PARAFAC fails, try a simpler decomposition on a subset
            if hasattr(self, 'masked_tensor') and self.masked_tensor.shape[0] > 1000:
                print("Trying PARAFAC on a subset of pixels...")
                # Take a random subset of pixels
                np.random.seed(random_state)
                subset_idx = np.random.choice(self.masked_tensor.shape[0], size=1000, replace=False)
                subset_tensor = self.masked_tensor[subset_idx, :, :]

                try:
                    subset_factors = parafac(tl.tensor(subset_tensor), rank=n_components, init='random',
                                             random_state=random_state)
                    self.parafac_factors = subset_factors
                    self.parafac_emissions = emissions
                    print("PARAFAC on subset completed successfully")
                    return subset_factors
                except Exception as e2:
                    print(f"PARAFAC on subset also failed: {e2}")
                    return None

    def plot_parafac_components(self, output_dir='analysis_results'):
        """
        Plot the PARAFAC components

        Args:
            output_dir: Directory to save the plot
        """
        if self.parafac_factors is None:
            raise ValueError("No PARAFAC factors available. Please call run_parafac() first.")

        # Extract factors
        factors = self.parafac_factors

        # In TensorLy, CP tensor has two components:
        # factors[0] = weights for each component
        # factors[1] = list of factor matrices (one per mode/dimension)
        weights = factors[0]  # Weights for each component
        all_factors = factors[1]  # List of factor matrices for each mode

        if hasattr(self, 'masked_tensor'):
            # For masked tensor (masked_pixels x excitation x emission)
            pixel_factors = all_factors[0]  # Factor for masked pixels
            excitation_factors = all_factors[1]  # Factor for excitation
            emission_factors = all_factors[2]  # Factor for emission
        else:
            # For full tensor (spatial_rows x spatial_cols x excitation x emission)
            # TensorLy flattens the first two dimensions
            # We need to reshape the pixel factors
            spatial_factors = all_factors[0]  # Factor for spatial (flattened rows x cols)
            excitation_factors = all_factors[1]  # Factor for excitation
            emission_factors = all_factors[2]  # Factor for emission

    def identify_important_wavelengths_from_parafac(self, n_wavelengths=5):
        """
        Identify the most important excitation-emission combinations from PARAFAC factors

        Args:
            n_wavelengths: Number of top wavelengths to identify per component

        Returns:
            DataFrame with important wavelengths
        """
        if self.parafac_factors is None:
            raise ValueError("No PARAFAC factors available. Please call run_parafac() first.")

        # Extract factors
        factors = self.parafac_factors

        # Get excitation and emission factors from the decomposition
        # factors[1] contains the list of all factor matrices
        all_factors = factors[1]
        excitation_factors = all_factors[1]  # Factor for excitation
        emission_factors = all_factors[2]  # Factor for emission

        results = []

        # For each component
        for comp_idx in range(excitation_factors.shape[1]):
            # Find top excitation wavelengths
            ex_factor = excitation_factors[:, comp_idx]
            top_ex_indices = np.argsort(ex_factor)[-n_wavelengths:][::-1]

            # Find top emission wavelengths
            em_factor = emission_factors[:, comp_idx]
            top_em_indices = np.argsort(em_factor)[-n_wavelengths:][::-1]

            # For all combinations of top excitation and emission
            for ex_rank, ex_idx in enumerate(top_ex_indices):
                if ex_idx >= len(self.excitation_wavelengths):
                    # Skip if index is out of bounds
                    continue

                ex_wave = self.excitation_wavelengths[ex_idx]
                ex_loading = ex_factor[ex_idx]

                for em_rank, em_idx in enumerate(top_em_indices):
                    if em_idx >= len(self.parafac_emissions):
                        # Skip if index is out of bounds
                        continue

                    em_wave = self.parafac_emissions[em_idx]
                    em_loading = em_factor[em_idx]

                    # Calculate a combined score (product of loadings)
                    combined_score = ex_loading * em_loading

                    # Store the result
                    results.append({
                        'Component': comp_idx + 1,
                        'ExRank': ex_rank + 1,
                        'EmRank': em_rank + 1,
                        'Excitation': ex_wave,
                        'Emission': em_wave,
                        'ExLoading': ex_loading,
                        'EmLoading': em_loading,
                        'CombinedScore': combined_score
                    })

        # Check if we have any results
        if not results:
            print("Warning: No valid excitation-emission combinations found for PARAFAC factors")
            # Return an empty DataFrame with the expected columns
            return pd.DataFrame(columns=['Component', 'ExRank', 'EmRank', 'Excitation',
                                         'Emission', 'ExLoading', 'EmLoading', 'CombinedScore'])

        # Convert to DataFrame and sort by component and combined score
        important_wavelengths = pd.DataFrame(results)
        important_wavelengths = important_wavelengths.sort_values(['Component', 'CombinedScore'],
                                                                  ascending=[True, False])

        return important_wavelengths
    def plot_parafac_eem_heatmap(self, component=0, output_dir='analysis_results'):
        """
        Plot the Excitation-Emission Matrix for a specific PARAFAC component

        Args:
            component: Index of the component to plot (0-based)
            output_dir: Directory to save the plot
        """
        if self.parafac_factors is None:
            raise ValueError("No PARAFAC factors available. Please call run_parafac() first.")

        # Extract factors
        factors = self.parafac_factors
        all_factors = factors[1]  # List of factor matrices

        # Check if the requested component is available
        if component >= all_factors[1].shape[1]:
            raise ValueError(
                f"Component {component} not available. Only {all_factors[1].shape[1]} components were extracted.")

        # Get excitation and emission factors for the component
        excitation_factor = all_factors[1][:, component]
        emission_factor = all_factors[2][:, component]

def run_dimensionality_reduction_analysis(h5_file_path, output_dir='analysis_results', n_pca_components=10, n_parafac_components=10):
    """
    Run the complete dimensionality reduction analysis workflow

    Args:
        h5_file_path: Path to the HDF5 file containing hyperspectral data
        output_dir: Directory to save results
        n_pca_components: Number of PCA components to extract
        n_parafac_components: Number of PARAFAC components to extract

    Returns:
        Dictionary with results
    """
    # Create the analyzer
    analyzer = DimensionalityReductionAnalyzer(h5_file_path)

    # Run PCA
    analyzer.prepare_data_for_pca()
    pca, transformed_data = analyzer.run_pca(n_components=n_pca_components)

    # Identify important wavelengths from PCA
    important_pca_wavelengths = analyzer.identify_important_wavelengths_from_pca(n_components=3, n_wavelengths=5)
    print("\nTop excitation-emission combinations from PCA:")
    print(important_pca_wavelengths)

    # Visualize PCA results
    pca_variance_plot = analyzer.plot_pca_explained_variance(output_dir)
    pca_loadings_plots = [analyzer.plot_pca_loadings(component=i, output_dir=output_dir) for i in range(3)]
    pca_rgb_plot = analyzer.visualize_pca_in_rgb(output_dir)

    # Run PARAFAC
    parafac_results = {}
    try:
        analyzer.prepare_data_for_parafac()
        factors = analyzer.run_parafac(n_components=n_parafac_components)

        if factors is not None:
            # Identify important wavelengths from PARAFAC
            try:
                important_parafac_wavelengths = analyzer.identify_important_wavelengths_from_parafac(n_wavelengths=5)
                print("\nTop excitation-emission combinations from PARAFAC:")

                if len(important_parafac_wavelengths) > 0:
                    print(important_parafac_wavelengths.head(min(15, len(important_parafac_wavelengths))))
                    parafac_results['important_parafac_wavelengths'] = important_parafac_wavelengths
                else:
                    print("No significant PARAFAC wavelength combinations found.")
            except Exception as e:
                print(f"Error extracting important wavelengths from PARAFAC: {e}")

            # Visualize PARAFAC results
            try:
                parafac_plots = analyzer.plot_parafac_components(output_dir)
                parafac_results['parafac_plots'] = parafac_plots
            except Exception as e:
                print(f"Error plotting PARAFAC components: {e}")

            try:
                parafac_eem_plots = [analyzer.plot_parafac_eem_heatmap(component=i, output_dir=output_dir)
                                   for i in range(min(3, n_parafac_components))]
                parafac_results['parafac_eem_plots'] = parafac_eem_plots
            except Exception as e:
                print(f"Error plotting PARAFAC EEM heatmaps: {e}")

    except Exception as e:
        print(f"Error in PARAFAC analysis: {e}")
        print("Returning PCA results only")

    # Return results
    results = {
        'pca_model': pca,
        'important_pca_wavelengths': important_pca_wavelengths,
        'pca_variance_plot': pca_variance_plot,
        'pca_loadings_plots': pca_loadings_plots,
        'pca_rgb_plot': pca_rgb_plot
    }

    # Add PARAFAC results if available
    results.update(parafac_results)

    return results

if __name__ == "__main__":
    # Example usage:
    h5_file_path = "../kiwi_hyperspectral_4d_data_with_mask.h5"
    output_dir = "dimensionality_reduction_results"
    results = run_dimensionality_reduction_analysis(h5_file_path, output_dir)