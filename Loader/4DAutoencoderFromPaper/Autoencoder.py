import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pickle
from pathlib import Path
import os
from typing import Dict, List, Tuple, Optional, Union


class HyperspectralCAE(nn.Module):
    """
    A 4D convolutional autoencoder for hyperspectral data aggregation.
    Adapted from the architecture described in the paper "A Convolutional Autoencoder
    for Multi-Subject fMRI Data Aggregation".

    This model takes hyperspectral data from multiple excitation wavelengths and finds
    shared patterns across them while preserving spatial locality.
    """

    def __init__(
            self,
            num_excitations: int,
            input_height: int,
            input_width: int,
            num_emission_bands: int,
            k1: int = 20,  # Number of filters in first layer
            k3: int = 20,  # Number of filters in third layer
            filter_size: int = 5,
            sparsity_target: float = 0.75,
            sparsity_weight: float = 1.0,
            dropout_rate: float = 0.5
    ):
        """
        Initialize the Hyperspectral Convolutional Autoencoder.

        Args:
            num_excitations: Number of excitation wavelengths
            input_height: Height of input images
            input_width: Width of input images
            num_emission_bands: Number of emission wavelength bands
            k1: Number of filters in first layer
            k3: Number of filters in shared feature maps
            filter_size: Size of convolutional filters
            sparsity_target: Target sparsity for regularization
            sparsity_weight: Weight of sparsity regularization
            dropout_rate: Dropout probability
        """
        super(HyperspectralCAE, self).__init__()

        self.num_excitations = num_excitations
        self.input_height = input_height
        self.input_width = input_width
        self.num_emission_bands = num_emission_bands
        self.filter_size = filter_size
        self.k1 = k1
        self.k3 = k3
        self.sparsity_target = sparsity_target
        self.sparsity_weight = sparsity_weight
        self.dropout_rate = dropout_rate

        padding = filter_size // 2

        # Encoder
        # First layer: Excitation-specific 3D convolutions
        # Instead of making separate convolution layers for each excitation wavelength,
        # we use grouped convolutions where each group corresponds to an excitation wavelength
        self.enc_conv1 = nn.Conv3d(
            in_channels=num_excitations,
            out_channels=num_excitations * k1,
            kernel_size=(filter_size, filter_size, min(5, num_emission_bands)),
            padding=(padding, padding, min(5, num_emission_bands) // 2),
            groups=num_excitations  # Each group processes one excitation wavelength
        )

        # Second layer: Average pooling across excitation dimension
        # This is handled in forward() with reshaping and mean operations

        # Third layer: Convolution on shared feature maps
        self.enc_conv3 = nn.Conv3d(
            in_channels=k1,
            out_channels=k3,
            kernel_size=(filter_size, filter_size, 1),
            padding=(padding, padding, 0)
        )

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

        # Decoder
        # Reverse the encoding process
        self.dec_conv1 = nn.Conv3d(
            in_channels=k3,
            out_channels=k1,
            kernel_size=(filter_size, filter_size, 1),
            padding=(padding, padding, 0)
        )

        # Final layer: Reconstruct each excitation wavelength
        self.dec_conv2 = nn.Conv3d(
            in_channels=k1,
            out_channels=num_excitations,
            kernel_size=(filter_size, filter_size, min(5, num_emission_bands)),
            padding=(padding, padding, min(5, num_emission_bands) // 2),
            groups=num_excitations  # Each group reconstructs one excitation wavelength
        )

    def encode(self, x):
        """
        Encode the input data to the shared feature representation.

        Args:
            x: Input tensor of shape (batch_size, num_excitations, height, width, emission_bands)

        Returns:
            Shared feature maps
        """
        # Ensure input has right shape
        batch_size, num_excitations, height, width, emission_bands = x.shape
        assert num_excitations == self.num_excitations, "Input has wrong number of excitation wavelengths"

        # First layer: Excitation-specific 3D convolutions
        # Reshape to work with Conv3D (which expects [batch, channels, depth, height, width])
        x = x.permute(0, 1, 4, 2, 3)  # [batch, excitations, emission_bands, height, width]

        # Apply first convolution - produces excitation-specific feature maps
        x = self.enc_conv1(x)  # [batch, num_excitations*k1, emission_bands', height', width']
        x = F.tanh(x)

        # Reshape to separate excitation and filter dimensions
        x = x.view(batch_size, self.num_excitations, self.k1, x.shape[2], x.shape[3], x.shape[4])

        # Second layer: Average pooling across excitations
        x = torch.mean(x, dim=1)  # [batch, k1, emission_bands', height', width']

        # Third layer: Convolution on shared feature maps
        x = self.enc_conv3(x)  # [batch, k3, emission_bands', height', width']
        x = F.tanh(x)

        # Apply dropout
        x = self.dropout(x)

        return x

    def decode(self, x):
        """
        Decode from the shared feature representation back to reconstructed input.

        Args:
            x: Shared feature maps from the encoder

        Returns:
            Reconstructed data for each excitation wavelength
        """
        batch_size = x.shape[0]

        # First decoding layer
        x = self.dec_conv1(x)  # [batch, k1, emission_bands', height', width']
        x = F.tanh(x)

        # Expand to create copies for each excitation wavelength
        x = x.unsqueeze(1).expand(-1, self.num_excitations, -1, -1, -1, -1)
        x = x.reshape(batch_size, self.num_excitations * self.k1, *x.shape[3:])

        # Final decoding layer to reconstruct each excitation wavelength
        x = self.dec_conv2(x)  # [batch, num_excitations, emission_bands, height, width]

        # Reshape back to original format
        x = x.permute(0, 1, 3, 4, 2)  # [batch, excitations, height, width, emission_bands]

        return x

    def forward(self, x):
        """
        Forward pass through the autoencoder.

        Args:
            x: Input tensor of shape (batch_size, num_excitations, height, width, emission_bands)

        Returns:
            Reconstructed data
        """
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded

    def compute_sparsity_loss(self, encoded):
        """
        Compute the sparsity regularization loss (KL divergence).

        Args:
            encoded: Encoded representation

        Returns:
            KL divergence loss
        """
        # Convert tanh activation to [0,1] range for sparsity computation
        rho_hat = 0.5 * (encoded + 1)

        # Average activation for each filter
        rho_hat = rho_hat.mean(dim=(0, 2, 3, 4))

        # Compute KL divergence
        rho = torch.tensor(self.sparsity_target).to(encoded.device)
        kl_loss = rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))

        return kl_loss.sum()


class HyperspectralDataset(Dataset):
    """
    Dataset for loading hyperspectral data for the autoencoder.
    """

    def __init__(self, data_dict: Dict, excitation_wavelengths: List[float] = None,
                 sample_size: Optional[int] = None, seed: int = 42):
        """
        Initialize the dataset from the hyperspectral data dictionary.

        Args:
            data_dict: Dictionary containing hyperspectral data (output from HyperspectralDataLoader)
            excitation_wavelengths: List of excitation wavelengths to use
            sample_size: Number of random pixels to sample (for large datasets)
            seed: Random seed for sampling
        """
        self.data_dict = data_dict

        # If no excitation wavelengths are specified, use all available
        if excitation_wavelengths is None:
            self.excitation_wavelengths = [
                float(ex) for ex in data_dict['excitation_wavelengths']
            ]
        else:
            self.excitation_wavelengths = excitation_wavelengths

        # Get the dimensions of the data
        first_ex = str(self.excitation_wavelengths[0])
        self.cube_shape = data_dict['data'][first_ex]['cube'].shape
        self.height, self.width = self.cube_shape[0], self.cube_shape[1]

        # Prepare the spatial coordinates for sampling
        y_coords, x_coords = np.mgrid[0:self.height, 0:self.width]
        self.x_coords = x_coords.flatten()
        self.y_coords = y_coords.flatten()

        # If sample_size is provided, take a random sample of pixels
        if sample_size is not None and sample_size < len(self.x_coords):
            np.random.seed(seed)
            indices = np.random.choice(len(self.x_coords), sample_size, replace=False)
            self.x_coords = self.x_coords[indices]
            self.y_coords = self.y_coords[indices]

        self.num_samples = len(self.x_coords)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Hyperspectral data for the pixel at (x_coords[idx], y_coords[idx])
            across all excitation wavelengths
        """
        x, y = self.x_coords[idx], self.y_coords[idx]

        # Extract data for each excitation wavelength
        data = []
        for ex in self.excitation_wavelengths:
            ex_str = str(ex)
            if ex_str in self.data_dict['data']:
                cube = self.data_dict['data'][ex_str]['cube']
                # Get the spectrum for this pixel
                spectrum = cube[y, x, :]
                data.append(spectrum)
            else:
                # If this excitation wavelength is not available, use zeros
                num_bands = self.data_dict['data'][str(self.excitation_wavelengths[0])]['cube'].shape[2]
                data.append(np.zeros(num_bands))

        # Convert to tensor
        data = torch.tensor(data, dtype=torch.float32)

        return data


def train_hyperspectral_cae(
        model,
        dataloader,
        num_epochs=100,
        learning_rate=0.001,
        device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Train the hyperspectral convolutional autoencoder.

    Args:
        model: HyperspectralCAE model
        dataloader: DataLoader for the hyperspectral data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for the optimizer
        device: Device to use for training

    Returns:
        Trained model and training losses
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Track losses
    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)

            # Forward pass
            output = model(batch)

            # Compute reconstruction loss
            recon_loss = F.mse_loss(output, batch)

            # Compute sparsity loss
            encoded = model.encode(batch)
            sparsity_loss = model.compute_sparsity_loss(encoded)

            # Total loss
            loss = recon_loss + model.sparsity_weight * sparsity_loss

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Record average loss for this epoch
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')

    return model, losses


def visualize_reconstruction(model, dataset, indices, excitation_idx=0,
                             device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Visualize the original and reconstructed data for selected samples.

    Args:
        model: Trained HyperspectralCAE model
        dataset: HyperspectralDataset
        indices: Indices of samples to visualize
        excitation_idx: Index of the excitation wavelength to visualize
        device: Device to use for inference
    """
    model.eval()

    fig, axes = plt.subplots(len(indices), 2, figsize=(10, 5 * len(indices)))
    if len(indices) == 1:
        axes = axes.reshape(1, -1)

    for i, idx in enumerate(indices):
        # Get a sample
        sample = dataset[idx].unsqueeze(0).to(device)

        # Get reconstruction
        with torch.no_grad():
            reconstruction = model(sample)

        # Convert to numpy for plotting
        original = sample.cpu().numpy()[0, excitation_idx]
        recon = reconstruction.cpu().numpy()[0, excitation_idx]

        # Plot original
        axes[i, 0].plot(original)
        axes[i, 0].set_title(f'Original - Ex={dataset.excitation_wavelengths[excitation_idx]}')
        axes[i, 0].set_xlabel('Emission Band Index')
        axes[i, 0].set_ylabel('Intensity')

        # Plot reconstruction
        axes[i, 1].plot(recon)
        axes[i, 1].set_title(f'Reconstruction - Ex={dataset.excitation_wavelengths[excitation_idx]}')
        axes[i, 1].set_xlabel('Emission Band Index')
        axes[i, 1].set_ylabel('Intensity')

    plt.tight_layout()
    return fig


def visualize_feature_maps(model, dataset, idx, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Visualize the feature maps of the autoencoder.

    Args:
        model: Trained HyperspectralCAE model
        dataset: HyperspectralDataset
        idx: Index of the sample to visualize
        device: Device to use for inference
    """
    model.eval()

    # Get a sample
    sample = dataset[idx].unsqueeze(0).to(device)

    # Get encoded representation
    with torch.no_grad():
        encoded = model.encode(sample)

    # Convert to numpy for plotting
    encoded_np = encoded.cpu().numpy()[0]

    # Plot feature maps
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    axes = axes.flatten()

    for i in range(min(20, encoded_np.shape[0])):
        feature_map = encoded_np[i, 0]  # Take the first slice along depth dimension
        axes[i].imshow(feature_map, cmap='viridis')
        axes[i].set_title(f'Feature {i + 1}')
        axes[i].axis('off')

    plt.tight_layout()
    return fig


def load_hyperspectral_data(data_path):
    """
    Load hyperspectral data from a pickle file.

    Args:
        data_path: Path to the pickle file containing hyperspectral data

    Returns:
        Loaded data dictionary
    """
    with open(data_path, 'rb') as f:
        data_dict = pickle.load(f)
    return data_dict


# Example usage
if __name__ == "__main__":
    # Load hyperspectral data
    data_path = "../Data/Kiwi Experiment/pickles/KiwiData.pkl"
    data_dict = load_hyperspectral_data(data_path)

    # Create dataset
    dataset = HyperspectralDataset(data_dict, sample_size=10000)

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Create model
    model = HyperspectralCAE(
        num_excitations=len(dataset.excitation_wavelengths),
        input_height=dataset.height,
        input_width=dataset.width,
        num_emission_bands=dataset.data_dict['data'][str(dataset.excitation_wavelengths[0])]['cube'].shape[2],
        k1=20,
        k3=20,
        filter_size=5,
        sparsity_target=0.75,
        sparsity_weight=1.0,
        dropout_rate=0.5
    )

    # Train model
    model, losses = train_hyperspectral_cae(model, dataloader, num_epochs=100)

    # Visualize reconstructions
    fig1 = visualize_reconstruction(model, dataset, [0, 1, 2])

    # Visualize feature maps
    fig2 = visualize_feature_maps(model, dataset, 0)

    # Save model
    torch.save(model.state_dict(), "hyperspectral_cae_model.pth")