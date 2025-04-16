import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import seaborn as sns
from sklearn.manifold import TSNE


class SpectralAutoencoder(nn.Module):
    """
    Neural network autoencoder for hyperspectral data

    This autoencoder learns a compressed representation of excitation-emission spectra,
    which can be used to identify the most informative wavelength combinations.
    """

    def __init__(self, input_dim, hidden_dim=64, latent_dim=10, dropout=0.2):
        """
        Initialize the autoencoder with specified dimensions

        Args:
            input_dim: Dimensionality of input spectra (total number of excitation-emission combinations)
            hidden_dim: Size of hidden layer
            latent_dim: Size of the latent (bottleneck) representation
            dropout: Dropout probability for regularization
        """
        super(SpectralAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, latent_dim),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )

        self.latent_dim = latent_dim
        self.input_dim = input_dim

    def forward(self, x):
        """
        Forward pass through the autoencoder

        Args:
            x: Input data tensor

        Returns:
            Tuple of (reconstructed output, latent representation)
        """
        # Encode
        z = self.encoder(x)

        # Decode
        reconstructed = self.decoder(z)

        return reconstructed, z

    def encode(self, x):
        """
        Encode input data to the latent space

        Args:
            x: Input data tensor

        Returns:
            Latent representation
        """
        return self.encoder(x)


class DeepLearningAnalyzerWithCutoff:
    """Class for analyzing hyperspectral data using deep learning methods with spectral cutoff"""

    def __init__(self, h5_file_path, cutoff_offset=20):
        """
        Initialize with the path to the HDF5 file and cutoff offset

        Args:
            h5_file_path: Path to the HDF5 file containing hyperspectral data
            cutoff_offset: Offset in nm to add to excitation wavelength for cutoff (default: 20nm)
        """
        self.h5_file_path = h5_file_path
        self.cutoff_offset = cutoff_offset
        self.data_dict = {}
        self.filtered_data_dict = {}  # Will hold data after cutoff
        self.excitation_wavelengths = []
        self.emission_wavelengths = {}
        self.filtered_emission_wavelengths = {}  # Will hold filtered wavelengths
        self.unfolded_data = None
        self.col_labels = None
        self.autoencoder = None
        self.latent_features = None
        self.cutoff_indices = {}  # Will store cutoff indices for each excitation

        # Load the data
        self.load_data()

        # Apply spectral cutoff filtering
        self.apply_spectral_cutoff()

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

    def apply_spectral_cutoff(self):
        """
        Apply spectral cutoff to remove excitation artifacts.
        Filters emission spectra to only include wavelengths >= (excitation + cutoff_offset)
        """
        for excitation in self.excitation_wavelengths:
            data = self.data_dict[excitation]
            emission_waves = data['wavelengths']

            # Calculate cutoff wavelength
            cutoff_wavelength = excitation + self.cutoff_offset

            # Find cutoff index (first wavelength >= cutoff_wavelength)
            cutoff_idx = np.argmax(emission_waves >= cutoff_wavelength)

            # Store cutoff index for this excitation
            self.cutoff_indices[excitation] = cutoff_idx

            # Filter emission wavelengths
            filtered_waves = emission_waves[cutoff_idx:]
            self.filtered_emission_wavelengths[excitation] = filtered_waves

            # Create filtered data structure
            filtered_data = {
                'wavelengths': filtered_waves,
                'average_cube': None
            }

            # Filter average cube if available
            if data['average_cube'] is not None:
                filtered_data['average_cube'] = data['average_cube'][cutoff_idx:, :, :]

            # Store filtered data
            self.filtered_data_dict[excitation] = filtered_data

            print(f"Excitation {excitation}nm: Cutoff at {cutoff_wavelength}nm (removed {cutoff_idx} bands)")

    def visualize_spectral_cutoff(self, output_dir='dl_analysis_results'):
        """
        Create visualizations showing the effect of spectral cutoff

        Args:
            output_dir: Directory to save analysis results
        """
        os.makedirs(output_dir, exist_ok=True)

        for excitation in self.excitation_wavelengths:
            data = self.data_dict[excitation]
            filtered_data = self.filtered_data_dict[excitation]

            # Get wavelengths
            full_waves = data['wavelengths']
            filtered_waves = filtered_data['wavelengths']

            # Get average cube data if available
            if 'average_cube' in data and data['average_cube'] is not None:
                avg_cube = data['average_cube']
                avg_filtered_cube = filtered_data['average_cube']

                # Calculate band means
                if self.mask is not None and np.any(self.mask):
                    # Calculate with mask
                    full_band_means = np.array([np.mean(avg_cube[b, self.mask == 1])
                                                for b in range(avg_cube.shape[0])])
                    full_band_stds = np.array([np.std(avg_cube[b, self.mask == 1])
                                               for b in range(avg_cube.shape[0])])

                    filtered_band_means = np.array([np.mean(avg_filtered_cube[b, self.mask == 1])
                                                    for b in range(avg_filtered_cube.shape[0])])
                    filtered_band_stds = np.array([np.std(avg_filtered_cube[b, self.mask == 1])
                                                   for b in range(avg_filtered_cube.shape[0])])
                else:
                    # Calculate without mask
                    full_band_means = np.mean(avg_cube, axis=(1, 2))
                    full_band_stds = np.std(avg_cube, axis=(1, 2))

                    filtered_band_means = np.mean(avg_filtered_cube, axis=(1, 2))
                    filtered_band_stds = np.std(avg_filtered_cube, axis=(1, 2))

                # Create figure showing cutoff
                plt.figure(figsize=(12, 8))

                # Plot full spectrum with cutoff region highlighted
                plt.plot(full_waves, full_band_means, 'b-', label='Full Spectrum')
                plt.fill_between(full_waves, full_band_means - full_band_stds,
                                 full_band_means + full_band_stds, color='b', alpha=0.2)

                # Highlight cutoff region
                cutoff_wavelength = excitation + self.cutoff_offset
                plt.axvspan(excitation - 5, cutoff_wavelength, color='r', alpha=0.2,
                            label=f'Cutoff Region (Ex: {excitation}nm + {self.cutoff_offset}nm)')
                plt.axvline(x=cutoff_wavelength, color='r', linestyle='--')

                # Plot filtered spectrum
                plt.plot(filtered_waves, filtered_band_means, 'g-', linewidth=2, label='Filtered Spectrum')
                plt.fill_between(filtered_waves, filtered_band_means - filtered_band_stds,
                                 filtered_band_means + filtered_band_stds, color='g', alpha=0.2)

                plt.xlabel('Emission Wavelength (nm)')
                plt.ylabel('Mean Signal')
                title = f'Spectral Cutoff Effect - Excitation {excitation}nm'
                if self.mask is not None and np.any(self.mask):
                    title += ' (Masked Region Only)'
                plt.title(title)
                plt.legend()
                plt.grid(True, alpha=0.3)

                # Highlight excitation wavelength
                plt.axvline(x=excitation, color='orange', linestyle='-', label='Excitation Wavelength')

                # Save plot
                plt.savefig(os.path.join(output_dir, f'spectral_cutoff_ex{excitation}nm.png'), dpi=150)
                plt.close()

                print(f"Generated spectral cutoff visualization for excitation {excitation}nm")

    def prepare_data_for_autoencoder(self):
        """
        Prepare the data for autoencoder analysis by unfolding the filtered 4D data into a 2D matrix

        The data is restructured as:
        Rows = Pixels (spatial positions)
        Columns = Filtered excitation-emission combinations (wavelengths after cutoff)

        Returns:
            Unfolded data matrix, column labels (ex_em combinations)
        """
        # First, check if we have all the necessary data
        if not self.filtered_data_dict:
            raise ValueError("No filtered data available. Please call apply_spectral_cutoff() first.")

        # Count total number of emission wavelengths across all excitations (after cutoff)
        total_features = sum(len(self.filtered_emission_wavelengths[ex]) for ex in self.excitation_wavelengths)

        # Get spatial dimensions from the first data cube
        first_ex = self.excitation_wavelengths[0]
        if 'average_cube' not in self.filtered_data_dict[first_ex] or self.filtered_data_dict[first_ex][
            'average_cube'] is None:
            raise ValueError(f"No average cube found for excitation {first_ex} nm")

        height, width = self.filtered_data_dict[first_ex]['average_cube'].shape[1:]
        total_pixels = height * width

        # Initialize the unfolded data matrix
        # Rows = pixels, Columns = excitation-emission combinations (after cutoff)
        unfolded_data = np.zeros((total_pixels, total_features))

        # Keep track of column labels (ex_em combinations)
        col_labels = []

        # Current column index
        col_idx = 0

        # For each excitation wavelength
        for ex_wave in self.excitation_wavelengths:
            # Get filtered emission wavelengths for this excitation
            em_waves = self.filtered_emission_wavelengths[ex_wave]

            # Get the filtered data cube for this excitation
            cube = self.filtered_data_dict[ex_wave]['average_cube']

            # For each emission wavelength (after cutoff)
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

        print(f"Unfolded data shape after spectral cutoff: {unfolded_data.shape}")
        print(f"Total excitation-emission combinations after cutoff: {len(col_labels)}")
        return unfolded_data, col_labels

    def train_autoencoder(self, latent_dim=10, hidden_dim=128, batch_size=64, epochs=100, learning_rate=0.001):
        """
        Train an autoencoder on the filtered hyperspectral data

        Args:
            latent_dim: Dimensionality of the latent space (bottleneck)
            hidden_dim: Size of hidden layers
            batch_size: Batch size for training
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer

        Returns:
            Trained autoencoder model
        """
        # Prepare data if not already done
        if self.unfolded_data is None:
            self.prepare_data_for_autoencoder()

        # Check if CUDA is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.unfolded_data)
        self.scaler = scaler

        # Convert to PyTorch tensors
        data_tensor = torch.tensor(scaled_data, dtype=torch.float32)

        # Create dataset and dataloader
        dataset = TensorDataset(data_tensor, data_tensor)  # Input = Target for autoencoder
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Initialize the autoencoder
        input_dim = self.unfolded_data.shape[1]
        model = SpectralAutoencoder(input_dim, hidden_dim, latent_dim)
        model.to(device)

        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        model.train()
        losses = []

        print(f"Starting autoencoder training for {epochs} epochs")
        for epoch in range(epochs):
            epoch_loss = 0

            for batch_idx, (data, _) in enumerate(dataloader):
                data = data.to(device)

                # Forward pass
                reconstructed, _ = model(data)
                loss = criterion(reconstructed, data)

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # Average loss for this epoch
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)

            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

        # Store the model and training history
        self.autoencoder = model
        self.training_losses = losses

        # Generate latent features for all data
        model.eval()
        with torch.no_grad():
            data_tensor = data_tensor.to(device)
            latent_features = model.encode(data_tensor).cpu().numpy()

        self.latent_features = latent_features

        return model, losses

    def plot_training_history(self, output_dir='dl_analysis_results'):
        """
        Plot the training loss history of the autoencoder

        Args:
            output_dir: Directory to save the plot

        Returns:
            Path to the saved plot
        """
        if not hasattr(self, 'training_losses'):
            raise ValueError("No training history available. Please call train_autoencoder() first.")

        plt.figure(figsize=(10, 6))
        plt.plot(self.training_losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Autoencoder Training Loss (With Spectral Cutoff)')
        plt.grid(True, alpha=0.3)

        # Save the plot
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, 'autoencoder_training_loss.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()

        return plot_path

    def visualize_latent_space(self, output_dir='dl_analysis_results'):
        """
        Visualize the latent space of the autoencoder using t-SNE

        Args:
            output_dir: Directory to save the plot

        Returns:
            Path to the saved plot
        """
        if self.latent_features is None:
            raise ValueError("No latent features available. Please call train_autoencoder() first.")

        # Apply t-SNE to the latent representations
        # Using a smaller perplexity if few points
        perplexity = min(30, max(5, self.latent_features.shape[0] // 10))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)

        # If many points, sample to speed up t-SNE
        if self.latent_features.shape[0] > 5000:
            indices = np.random.choice(self.latent_features.shape[0], 5000, replace=False)
            tsne_results = tsne.fit_transform(self.latent_features[indices])
        else:
            tsne_results = tsne.fit_transform(self.latent_features)

        # Create the plot
        plt.figure(figsize=(12, 10))
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.5)
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title('t-SNE Visualization of Autoencoder Latent Space (With Spectral Cutoff)')

        # Save the plot
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, 'autoencoder_latent_space.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()

        return plot_path

    def analyze_latent_dimensions(self, output_dir='dl_analysis_results'):
        """
        Analyze the latent dimensions of the autoencoder to identify important features

        Args:
            output_dir: Directory to save the plots

        Returns:
            DataFrame with important wavelengths for each latent dimension
        """
        try:
            if self.autoencoder is None:
                print("No autoencoder available. Please call train_autoencoder() first.")
                return pd.DataFrame(columns=['LatentDimension', 'Rank', 'Excitation', 'Emission', 'ImportanceScore'])

            # Make sure output directory exists
            os.makedirs(output_dir, exist_ok=True)

            # Store the results for each latent dimension
            results = []

            try:
                # Extract the weights from the first layer of the decoder - safely
                decoder_weights = self.autoencoder.decoder[0].weight.detach().cpu().numpy()
                latent_dim = decoder_weights.shape[1]
            except (IndexError, AttributeError) as e:
                print(f"Error accessing decoder weights: {e}")
                # Create a simple importance score based on reconstructed data
                if hasattr(self, 'pca_transformed'):
                    print("Using PCA components as fallback for latent analysis")
                    # Use PCA as a fallback
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=min(8, self.unfolded_data.shape[1]))
                    pca_result = pca.fit_transform(self.scaler.transform(self.unfolded_data))

                    # Get the most important features based on PCA loadings
                    for latent_idx in range(min(3, pca.components_.shape[0])):
                        component = pca.components_[latent_idx]
                        top_indices = np.argsort(np.abs(component))[-10:][::-1]

                        for rank, idx in enumerate(top_indices):
                            # Get the corresponding excitation-emission combination
                            ex_wave, em_wave = self.col_labels[idx]

                            # Store the result
                            results.append({
                                'LatentDimension': latent_idx + 1,
                                'Rank': rank + 1,
                                'Excitation': ex_wave,
                                'Emission': em_wave,
                                'ImportanceScore': abs(component[idx])
                            })

                    importance_df = pd.DataFrame(results)
                    return importance_df
                else:
                    # Create a very simple result - top 10 highest variance features
                    print("Using variance as fallback for latent analysis")
                    feature_variance = np.var(self.unfolded_data, axis=0)
                    top_indices = np.argsort(feature_variance)[-30:][::-1]

                    for latent_idx in range(min(3, len(top_indices) // 10)):
                        subset_indices = top_indices[latent_idx * 10:(latent_idx + 1) * 10]
                        for rank, idx in enumerate(subset_indices):
                            # Get the corresponding excitation-emission combination
                            ex_wave, em_wave = self.col_labels[idx]

                            # Store the result
                            results.append({
                                'LatentDimension': latent_idx + 1,
                                'Rank': rank + 1,
                                'Excitation': ex_wave,
                                'Emission': em_wave,
                                'ImportanceScore': feature_variance[idx]
                            })

                    importance_df = pd.DataFrame(results)
                    return importance_df

            # For each latent dimension
            for latent_idx in range(latent_dim):
                try:
                    # Use a direct method to assess importance:
                    # Calculate the importance of each input feature to each latent dimension
                    input_dim = self.unfolded_data.shape[1]
                    importance_scores = np.zeros(input_dim)

                    # Basic approach: use the encoder's ability to reconstruct data
                    # More important features will contribute more to the reconstruction
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                    # Get the standardized data
                    data = torch.tensor(self.scaler.transform(self.unfolded_data), dtype=torch.float32).to(device)

                    # Forward pass to get reconstructed data
                    self.autoencoder.eval()
                    with torch.no_grad():
                        reconstructed, latent = self.autoencoder(data)
                        reconstructed = reconstructed.cpu().numpy()

                    # For each feature, calculate its contribution to the latent space
                    # This is a simplification - we're using the reconstruction error here
                    original_data = self.scaler.transform(self.unfolded_data)
                    feature_errors = np.mean((original_data - reconstructed) ** 2, axis=0)

                    # Invert errors (lower error = more important)
                    importance_scores = 1.0 / (feature_errors + 1e-10)  # Add small constant to avoid division by zero

                    # Find the top input features for this latent dimension
                    top_indices = np.argsort(importance_scores)[-10:][::-1]

                    # For each top feature
                    for rank, idx in enumerate(top_indices):
                        # Get the corresponding excitation-emission combination
                        ex_wave, em_wave = self.col_labels[idx]

                        # Store the result
                        results.append({
                            'LatentDimension': latent_idx + 1,
                            'Rank': rank + 1,
                            'Excitation': ex_wave,
                            'Emission': em_wave,
                            'ImportanceScore': importance_scores[idx]
                        })

                except Exception as e:
                    print(f"Error analyzing latent dimension {latent_idx}: {e}")
                    continue

            # Convert to DataFrame
            importance_df = pd.DataFrame(results)

            if len(importance_df) == 0:
                # If no results, return empty DataFrame with correct columns
                return pd.DataFrame(columns=['LatentDimension', 'Rank', 'Excitation', 'Emission', 'ImportanceScore'])

            # Create visualizations only if we have results
            if len(importance_df) > 0:
                try:
                    # Create a heatmap of importance scores for each latent dimension
                    pivot_df = importance_df.pivot_table(
                        values='ImportanceScore',
                        index='LatentDimension',
                        columns=['Excitation', 'Emission'],
                        aggfunc='first'
                    )

                    # Fill NaN values with 0
                    pivot_df = pivot_df.fillna(0)

                    # Create the heatmap
                    plt.figure(figsize=(14, 10))
                    sns.heatmap(pivot_df, cmap='viridis')
                    plt.title(
                        'Importance of Excitation-Emission Combinations for Latent Dimensions (With Spectral Cutoff)')
                    plt.xlabel('(Excitation, Emission) Combinations')
                    plt.ylabel('Latent Dimension')

                    # Save the plot
                    heatmap_path = os.path.join(output_dir, 'latent_dimension_importance.png')
                    plt.savefig(heatmap_path, dpi=150)
                    plt.close()
                except Exception as e:
                    print(f"Error creating latent dimension heatmap: {e}")

            return importance_df

        except Exception as e:
            print(f"Error in latent dimension analysis: {e}")
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=['LatentDimension', 'Rank', 'Excitation', 'Emission', 'ImportanceScore'])

    def generate_reconstruction_error_by_feature(self, output_dir='dl_analysis_results'):
        """
        Analyze reconstruction error by feature to identify the most informative features

        Args:
            output_dir: Directory to save the plots

        Returns:
            DataFrame with reconstruction error for each feature
        """
        if self.autoencoder is None:
            raise ValueError("No autoencoder available. Please call train_autoencoder() first.")

        # Check if CUDA is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get the standardized data
        data_tensor = torch.tensor(self.scaler.transform(self.unfolded_data), dtype=torch.float32)
        data_tensor = data_tensor.to(device)

        # Put model in evaluation mode
        self.autoencoder.eval()

        # Compute the full reconstruction
        with torch.no_grad():
            full_reconstruction, _ = self.autoencoder(data_tensor)
            full_reconstruction = full_reconstruction.cpu().numpy()

        # Compute per-feature reconstruction error
        original_data = self.scaler.transform(self.unfolded_data)
        mse_per_feature = np.mean((original_data - full_reconstruction) ** 2, axis=0)

        # Create a DataFrame with results
        error_df = pd.DataFrame([
            {'Excitation': ex, 'Emission': em, 'ReconstructionError': error}
            for (ex, em), error in zip(self.col_labels, mse_per_feature)
        ])

        # Sort by reconstruction error (higher error = more difficult to reconstruct = potentially more unique information)
        error_df = error_df.sort_values('ReconstructionError', ascending=False)

        # Create a visualization of reconstruction error
        plt.figure(figsize=(12, 10))

        # Create a scatter plot with point size representing error
        scatter = plt.scatter(
            error_df['Emission'],
            error_df['Excitation'],
            s=error_df['ReconstructionError'] * 1000,  # Scale for visibility
            alpha=0.7
        )

        # Label the top 10 points
        for i, row in error_df.head(10).iterrows():
            plt.text(
                row['Emission'],
                row['Excitation'],
                f"#{i + 1}",
                fontsize=10,
                fontweight='bold',
                ha='center',
                va='center'
            )

        plt.xlabel('Emission Wavelength (nm)')
        plt.ylabel('Excitation Wavelength (nm)')
        plt.title('Reconstruction Error by Excitation-Emission Combination (With Spectral Cutoff)')
        plt.grid(True, alpha=0.3)

        # Add diagonal line showing the cutoff boundary
        cutoff_x = [min(error_df['Emission']), max(error_df['Emission'])]
        max_ex = max(error_df['Excitation'])
        cutoff_y = [cutoff_x[0] - self.cutoff_offset, cutoff_x[1] - self.cutoff_offset]
        cutoff_y = [min(max_ex, y) for y in cutoff_y]  # Cap at max excitation

        plt.plot(cutoff_x, cutoff_y, 'r--', linewidth=2,
                 label=f'Spectral Cutoff Boundary (Ex + {self.cutoff_offset}nm)')
        plt.legend()

        # Save the plot
        os.makedirs(output_dir, exist_ok=True)
        error_plot_path = os.path.join(output_dir, 'reconstruction_error_by_feature.png')
        plt.savefig(error_plot_path, dpi=150)
        plt.close()

        # Now examine the reconstruction error pattern
        # Create a matrix of reconstruction errors by excitation-emission
        ex_wavelengths = sorted(set(ex for ex, _ in self.col_labels))
        em_wavelengths = sorted(set(em for _, em in self.col_labels))

        error_matrix = np.zeros((len(ex_wavelengths), len(em_wavelengths)))
        error_matrix.fill(np.nan)  # Fill with NaN for missing values

        for i, ((ex, em), error) in enumerate(zip(self.col_labels, mse_per_feature)):
            # Find indices in the matrix
            ex_idx = ex_wavelengths.index(ex)
            em_idx = em_wavelengths.index(em)

            # Fill the matrix
            error_matrix[ex_idx, em_idx] = error

        # Create a heatmap
        plt.figure(figsize=(14, 10))
        mask = np.isnan(error_matrix)  # Mask for NaN values

        # Plot heatmap with mask for NaN values
        sns.heatmap(
            error_matrix,
            cmap='rocket_r',  # Reverse rocket colormap (lower values in blue)
            xticklabels=em_wavelengths[::10],  # Show every 10th wavelength for readability
            yticklabels=ex_wavelengths,
            mask=mask  # Only show values that are not NaN
        )
        plt.title('Reconstruction Error by Excitation-Emission Wavelength (With Spectral Cutoff)')
        plt.xlabel('Emission Wavelength (nm)')
        plt.ylabel('Excitation Wavelength (nm)')

        # Add diagonal line showing the cutoff boundary approximately
        plt.plot([0, len(em_wavelengths)], [len(ex_wavelengths), 0], 'r--', linewidth=2,
                 label=f'Spectral Cutoff Boundary (Ex + {self.cutoff_offset}nm)')
        plt.legend(loc='upper right')

        # Save the plot
        error_heatmap_path = os.path.join(output_dir, 'reconstruction_error_heatmap.png')
        plt.savefig(error_heatmap_path, dpi=150)
        plt.close()

        return error_df

    def identify_important_wavelengths(self, method='reconstruction_error', n_wavelengths=10):
        """
        Identify the most important excitation-emission combinations based on autoencoder analysis

        Args:
            method: Method to use ('reconstruction_error' or 'latent_importance')
            n_wavelengths: Number of wavelength combinations to identify

        Returns:
            DataFrame with important wavelengths
        """
        try:
            if method == 'reconstruction_error':
                # Higher reconstruction error means the feature is harder to compress
                # which could indicate it contains unique information
                error_df = self.generate_reconstruction_error_by_feature()

                # Check if error_df is valid before proceeding
                if error_df is None or len(error_df) == 0:
                    print("Warning: Reconstruction error calculation failed, returning empty DataFrame")
                    return pd.DataFrame(columns=['Excitation', 'Emission', 'ReconstructionError'])

                important_df = error_df.head(n_wavelengths)

            elif method == 'latent_importance':
                # Look at the features that have highest importance across latent dimensions
                importance_df = self.analyze_latent_dimensions()

                # Check if importance_df is valid before proceeding
                if importance_df is None or len(importance_df) == 0:
                    print("Warning: Latent dimension analysis failed, returning empty DataFrame")
                    return pd.DataFrame(columns=['Excitation', 'Emission', 'ImportanceScore'])

                # Aggregate importance scores across latent dimensions without using groupby
                # Manual aggregation to avoid potential groupby issues
                agg_data = {}
                for _, row in importance_df.iterrows():
                    key = (row['Excitation'], row['Emission'])
                    if key not in agg_data:
                        agg_data[key] = 0
                    agg_data[key] += row['ImportanceScore']

                # Convert the aggregated data to DataFrame
                agg_list = []
                for (ex, em), score in agg_data.items():
                    agg_list.append({
                        'Excitation': ex,
                        'Emission': em,
                        'ImportanceScore': score
                    })

                agg_df = pd.DataFrame(agg_list)

                # Sort by total importance
                if len(agg_df) > 0:
                    important_df = agg_df.sort_values('ImportanceScore', ascending=False).head(n_wavelengths)
                else:
                    important_df = agg_df  # Empty DataFrame

            else:
                raise ValueError(f"Unknown method: {method}")

            return important_df

        except Exception as e:
            print(f"Error in identify_important_wavelengths: {e}")
            # Return an empty DataFrame with the correct columns
            if method == 'reconstruction_error':
                return pd.DataFrame(columns=['Excitation', 'Emission', 'ReconstructionError'])
            else:
                return pd.DataFrame(columns=['Excitation', 'Emission', 'ImportanceScore'])


def run_deep_learning_analysis(h5_file_path, output_dir='dl_analysis_results', latent_dim=10, epochs=100,
                               cutoff_offset=20):
    """
    Run the complete deep learning analysis workflow with spectral cutoff

    Args:
        h5_file_path: Path to the HDF5 file containing hyperspectral data
        output_dir: Directory to save results
        latent_dim: Dimensionality of the autoencoder latent space
        epochs: Number of training epochs
        cutoff_offset: Spectral cutoff offset in nm (default: 20nm)

    Returns:
        Dictionary with results
    """
    # Create the analyzer
    analyzer = DeepLearningAnalyzerWithCutoff(h5_file_path, cutoff_offset=cutoff_offset)

    # Visualize the spectral cutoff effect
    analyzer.visualize_spectral_cutoff(output_dir)

    # Prepare data
    analyzer.prepare_data_for_autoencoder()

    # Train the autoencoder
    try:
        # Try with CPU training (could take a while but should work on most systems)
        model, losses = analyzer.train_autoencoder(
            latent_dim=latent_dim,
            hidden_dim=128,
            batch_size=64,
            epochs=epochs,
            learning_rate=0.001
        )

        # Check if the training was successful by examining the loss
        initial_loss = losses[0]
        final_loss = losses[-1]
        improvement = (initial_loss - final_loss) / initial_loss

        if improvement < 0.1:
            print("Warning: Autoencoder training may not have converged properly.")
            print(f"Initial loss: {initial_loss:.4f}, Final loss: {final_loss:.4f}")
            print(f"Improvement: {improvement:.2%}")

        # Initialize result dictionary
        results = {
            'autoencoder_model': model,
            'training_losses': losses
        }

        # Plot training history
        try:
            loss_plot = analyzer.plot_training_history(output_dir)
            results['loss_plot'] = loss_plot
        except Exception as e:
            print(f"Error generating training history plot: {e}")

        # Visualize latent space
        try:
            latent_plot = analyzer.visualize_latent_space(output_dir)
            results['latent_plot'] = latent_plot
        except Exception as e:
            print(f"Error visualizing latent space: {e}")

        # Analyze latent dimensions with error handling
        try:
            importance_df = analyzer.analyze_latent_dimensions(output_dir)
            results['importance_df'] = importance_df
        except Exception as e:
            print(f"Error analyzing latent dimensions: {e}")
            importance_df = None
            results['importance_df'] = None

        # Generate reconstruction error analysis with error handling
        try:
            error_df = analyzer.generate_reconstruction_error_by_feature(output_dir)
            results['error_df'] = error_df
        except Exception as e:
            print(f"Error generating reconstruction error analysis: {e}")
            error_df = None
            results['error_df'] = None

        # Identify important wavelengths using both methods with error handling
        try:
            recon_error_wavelengths = analyzer.identify_important_wavelengths(
                method='reconstruction_error',
                n_wavelengths=10
            )
            results['recon_error_wavelengths'] = recon_error_wavelengths

            print("\nTop wavelengths by reconstruction error (after spectral cutoff):")
            print(recon_error_wavelengths)
        except Exception as e:
            print(f"Error identifying wavelengths by reconstruction error: {e}")
            results['recon_error_wavelengths'] = pd.DataFrame()

        try:
            latent_importance_wavelengths = analyzer.identify_important_wavelengths(
                method='latent_importance',
                n_wavelengths=10
            )
            results['latent_importance_wavelengths'] = latent_importance_wavelengths

            print("\nTop wavelengths by latent importance (after spectral cutoff):")
            print(latent_importance_wavelengths)
        except Exception as e:
            print(f"Error identifying wavelengths by latent importance: {e}")
            results['latent_importance_wavelengths'] = pd.DataFrame()

        return results

    except Exception as e:
        print(f"Error during autoencoder training: {e}")
        print("Try reducing the model size or number of epochs.")

        # Return minimal results
        return {
            'error': str(e),
            'recon_error_wavelengths': pd.DataFrame(),
            'latent_importance_wavelengths': pd.DataFrame()
        }


if __name__ == "__main__":
    # Example usage:
    h5_file_path = "../kiwi_hyperspectral_4d_data_with_mask.h5"
    output_dir = "dl_analysis_results_with_cutoff_2"
    results = run_deep_learning_analysis(
        h5_file_path,
        output_dir,
        latent_dim=12,
        epochs=10,
        cutoff_offset=20
    )