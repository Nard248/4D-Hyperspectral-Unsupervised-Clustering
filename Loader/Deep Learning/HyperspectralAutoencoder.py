import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic convolutional block with batch normalization"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class UpConvBlock(nn.Module):
    """Upsampling block for decoder"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, scale_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.conv = ConvBlock(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        return self.conv(self.upsample(x))


class HyperspectralAutoencoder(nn.Module):
    """
    Convolutional autoencoder for hyperspectral data.

    Uses a combination of spatial and spectral convolutions to efficiently
    process hyperspectral data. The model encodes the data into a latent
    space and then reconstructs it.
    """

    def __init__(self, n_excitations, n_emissions, spatial_size=(256, 256),
                 latent_dim=128, n_clusters=10):
        super().__init__()

        self.n_excitations = n_excitations
        self.n_emissions = n_emissions
        self.spatial_size = spatial_size
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters

        # Calculate number of channels for flattened excitation-emission dimensions
        self.n_channels = n_excitations * n_emissions

        # Encoder pathway
        self.encoder = nn.Sequential(
            # First layer: [batch, n_channels, height, width]
            ConvBlock(self.n_channels, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Second layer: [batch, 64, height/2, width/2]
            ConvBlock(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Third layer: [batch, 128, height/4, width/4]
            ConvBlock(128, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Fourth layer: [batch, 256, height/8, width/8]
            ConvBlock(256, 512),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Calculate dimensionality after encoder
        self.encoded_h = spatial_size[0] // 16
        self.encoded_w = spatial_size[1] // 16

        # Latent representation
        self.flatten = nn.Flatten()
        self.fc_encode = nn.Linear(512 * self.encoded_h * self.encoded_w, latent_dim)

        # Clustering head
        self.cluster_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_clusters)
        )

        # Decoder pathway
        self.fc_decode = nn.Linear(latent_dim, 512 * self.encoded_h * self.encoded_w)
        self.unflatten = nn.Unflatten(1, (512, self.encoded_h, self.encoded_w))

        self.decoder = nn.Sequential(
            # First decoder layer: [batch, 512, height/16, width/16]
            UpConvBlock(512, 256, scale_factor=2),

            # Second decoder layer: [batch, 256, height/8, width/8]
            UpConvBlock(256, 128, scale_factor=2),

            # Third decoder layer: [batch, 128, height/4, width/4]
            UpConvBlock(128, 64, scale_factor=2),

            # Final decoder layer: [batch, 64, height/2, width/2]
            UpConvBlock(64, self.n_channels, scale_factor=2),

            # Output activation
            nn.Sigmoid()  # Normalize output to [0,1]
        )

    def forward(self, x):
        """
        Forward pass through the autoencoder.

        Args:
            x: Input tensor [batch, n_excitations, n_emissions, height, width]

        Returns:
            Tuple of (reconstructed, latent_features, cluster_logits)
        """
        batch_size = x.shape[0]

        # Save original shape for later - ADD THIS LINE
        original_shape = x.shape

        # Reshape input to merge excitation and emission dimensions
        if len(original_shape) == 5:  # [batch, n_excitations, n_emissions, height, width]
            x = x.reshape(batch_size, self.n_channels, self.spatial_size[0], self.spatial_size[1])

        # Encode
        encoded = self.encoder(x)

        # Compress to latent space
        flattened = self.flatten(encoded)
        latent = self.fc_encode(flattened)

        # Get cluster assignments
        cluster_logits = self.cluster_head(latent)

        # Decode from latent space
        decoded_flat = self.fc_decode(latent)
        decoded_3d = self.unflatten(decoded_flat)

        # Reconstruct
        reconstructed = self.decoder(decoded_3d)

        # Reshape back to original dimensions
        if len(original_shape) == 5:  # If input was 5D, make output 5D as well
            reconstructed = reconstructed.reshape(
                batch_size, self.n_excitations, self.n_emissions,
                self.spatial_size[0], self.spatial_size[1]
            )

        return reconstructed, latent, cluster_logits

class PatchedHyperspectralAutoencoder(nn.Module):
    """
    Convolutional autoencoder that processes patches of hyperspectral data.
    Similar to the main autoencoder but designed for smaller patch sizes.
    """

    def __init__(self, n_excitations, n_emissions, patch_size=64,
                 latent_dim=128, n_clusters=10):
        super().__init__()

        self.n_excitations = n_excitations
        self.n_emissions = n_emissions
        self.patch_size = patch_size
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters

        # Calculate number of channels for flattened excitation-emission dimensions
        self.n_channels = n_excitations * n_emissions

        # Encoder pathway (smaller for patches)
        self.encoder = nn.Sequential(
            # First layer: [batch, n_channels, patch_size, patch_size]
            ConvBlock(self.n_channels, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Second layer: [batch, 64, patch_size/2, patch_size/2]
            ConvBlock(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Third layer: [batch, 128, patch_size/4, patch_size/4]
            ConvBlock(128, 256),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Calculate dimensionality after encoder
        self.encoded_size = patch_size // 8

        # Latent representation
        self.flatten = nn.Flatten()
        self.fc_encode = nn.Linear(256 * self.encoded_size * self.encoded_size, latent_dim)

        # Clustering head
        self.cluster_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_clusters)
        )

        # Decoder pathway
        self.fc_decode = nn.Linear(latent_dim, 256 * self.encoded_size * self.encoded_size)
        self.unflatten = nn.Unflatten(1, (256, self.encoded_size, self.encoded_size))

        self.decoder = nn.Sequential(
            # First decoder layer: [batch, 256, patch_size/8, patch_size/8]
            UpConvBlock(256, 128, scale_factor=2),

            # Second decoder layer: [batch, 128, patch_size/4, patch_size/4]
            UpConvBlock(128, 64, scale_factor=2),

            # Final decoder layer: [batch, 64, patch_size/2, patch_size/2]
            UpConvBlock(64, self.n_channels, scale_factor=2),

            # Output activation
            nn.Sigmoid()  # Normalize output to [0,1]
        )

    def forward(self, x):
        """
        Forward pass through the autoencoder.
        """
        batch_size = x.shape[0]

        # Save original shape for later
        original_shape = x.shape

        # Reshape input to merge excitation and emission dimensions
        if len(original_shape) == 5:  # [batch, n_excitations, n_emissions, patch_size, patch_size]
            x = x.reshape(batch_size, self.n_channels, self.patch_size, self.patch_size)

        # Encode
        encoded = self.encoder(x)

        # Compress to latent space
        flattened = self.flatten(encoded)
        latent = self.fc_encode(flattened)

        # Get cluster assignments
        cluster_logits = self.cluster_head(latent)

        # Decode from latent space
        decoded_flat = self.fc_decode(latent)
        decoded_3d = self.unflatten(decoded_flat)

        # Reconstruct
        reconstructed = self.decoder(decoded_3d)

        # THIS IS THE CRITICAL FIX: Reshape back to original 5D structure
        if len(original_shape) == 5:
            reconstructed = reconstructed.reshape(
                batch_size,
                original_shape[1],  # n_excitations
                original_shape[2],  # n_emissions
                original_shape[3],  # patch_size
                original_shape[4]  # patch_size
            )

        return reconstructed, latent, cluster_logits