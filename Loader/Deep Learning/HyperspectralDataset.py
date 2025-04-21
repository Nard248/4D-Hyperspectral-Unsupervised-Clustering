import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
from pathlib import Path


class HyperspectralDataset(Dataset):
    """
    Dataset for handling hyperspectral data from pickle files.
    Prepares data for convolutional autoencoder models.
    """

    def __init__(self, pickle_file, transform=None, sample_size=None, patch_size=None):
        """
        Initialize the dataset.

        Args:
            pickle_file: Path to the pickle file
            transform: Optional transform to apply
            sample_size: Number of random pixels to sample (None = use all)
            patch_size: Size of patches to extract (None = use whole images)
        """
        self.pickle_file = Path(pickle_file)
        self.transform = transform
        self.sample_size = sample_size
        self.patch_size = patch_size

        # Load the pickle file
        print(f"Loading data from {self.pickle_file}...")
        with open(self.pickle_file, 'rb') as f:
            self.data_dict = pickle.load(f)

        # Get excitation wavelengths
        self.excitation_wavelengths = sorted([float(ex) for ex in self.data_dict['data'].keys()])
        print(f"Found {len(self.excitation_wavelengths)} excitation wavelengths")

        # Get dimensions from first excitation to determine tensor shape
        first_ex = str(self.excitation_wavelengths[0])
        self.height, self.width, _ = self.data_dict['data'][first_ex]['cube'].shape
        print(f"Image dimensions: {self.height} x {self.width}")

        # Create 4D tensor representation
        self._create_4d_tensor()

        # For patch-based sampling
        if patch_size is not None:
            self._create_patches()

    def _create_4d_tensor(self):
        """Create a 4D tensor from the data: [n_excitations, emission_bands, height, width]"""
        # Find max number of emission wavelengths (they may vary per excitation)
        self.max_emissions = 0
        self.n_excitations = len(self.excitation_wavelengths)

        for ex in self.excitation_wavelengths:
            ex_str = str(ex)
            n_em = len(self.data_dict['data'][ex_str]['wavelengths'])
            self.max_emissions = max(self.max_emissions, n_em)

        print(f"Maximum emission bands: {self.max_emissions}")

        # Store wavelength values for reference
        self.ex_values = []
        self.em_values = [[] for _ in range(self.n_excitations)]

        # Avoid creating the full tensor for very large datasets to save memory
        if self.sample_size is None and self.patch_size is None:
            self.tensor = np.zeros((self.n_excitations, self.max_emissions, self.height, self.width))

            # Fill tensor with data
            for ex_idx, ex in enumerate(self.excitation_wavelengths):
                ex_str = str(ex)
                self.ex_values.append(ex)

                # Get data for this excitation
                cube = self.data_dict['data'][ex_str]['cube']
                wavelengths = self.data_dict['data'][ex_str]['wavelengths']
                self.em_values[ex_idx] = wavelengths

                # Fill tensor (handle different numbers of emission wavelengths)
                n_em = len(wavelengths)
                # Transpose to [emissions, height, width] for PyTorch convention
                transposed_cube = np.transpose(cube, (2, 0, 1))
                self.tensor[ex_idx, :n_em, :, :] = transposed_cube

    def _create_patches(self):
        """Create a list of patches for patch-based training"""
        self.patches = []
        self.patch_positions = []

        # Calculate number of patches
        n_patches_h = (self.height - self.patch_size) + 1
        n_patches_w = (self.width - self.patch_size) + 1

        # Extract patches
        for i in range(0, n_patches_h, max(1, self.patch_size // 2)):  # 50% overlap
            for j in range(0, n_patches_w, max(1, self.patch_size // 2)):
                # Record patch position
                self.patch_positions.append((i, j))

        print(f"Created {len(self.patch_positions)} patches of size {self.patch_size}x{self.patch_size}")

    def __len__(self):
        """Return the number of samples in the dataset"""
        if self.patch_size is not None:
            return len(self.patch_positions)
        elif self.sample_size is not None:
            return self.sample_size
        else:
            return 1  # One full hyperspectral image

    def _get_patch(self, idx):
        """Extract a specific patch"""
        i, j = self.patch_positions[idx]

        # Create patch tensor
        patch = np.zeros((self.n_excitations, self.max_emissions, self.patch_size, self.patch_size))

        # Fill patch with data
        for ex_idx, ex in enumerate(self.excitation_wavelengths):
            ex_str = str(ex)
            cube = self.data_dict['data'][ex_str]['cube']
            wavelengths = self.data_dict['data'][ex_str]['wavelengths']
            n_em = len(wavelengths)

            # Extract patch for all emission bands
            patch_data = cube[i:i + self.patch_size, j:j + self.patch_size, :]

            # Transpose to [emissions, height, width] for PyTorch convention
            patch_data = np.transpose(patch_data, (2, 0, 1))

            # Store in patch tensor
            patch[ex_idx, :n_em, :, :] = patch_data

        return patch

    def _get_random_samples(self, idx):
        """Get random pixel samples from the data"""
        # Create a random seed based on idx for reproducibility
        np.random.seed(idx)

        # Select random pixels
        y_indices = np.random.randint(0, self.height, size=self.sample_size)
        x_indices = np.random.randint(0, self.width, size=self.sample_size)

        # Extract spectral signatures for these pixels
        samples = np.zeros((self.sample_size, self.n_excitations, self.max_emissions))

        for ex_idx, ex in enumerate(self.excitation_wavelengths):
            ex_str = str(ex)
            cube = self.data_dict['data'][ex_str]['cube']
            wavelengths = self.data_dict['data'][ex_str]['wavelengths']
            n_em = len(wavelengths)

            # Extract data for all pixels
            for i, (y, x) in enumerate(zip(y_indices, x_indices)):
                samples[i, ex_idx, :n_em] = cube[y, x, :]

        return samples

    def _get_full_image(self):
        """Get the full hyperspectral image tensor"""
        if hasattr(self, 'tensor'):
            return self.tensor

        # If tensor wasn't pre-created, create it now
        tensor = np.zeros((self.n_excitations, self.max_emissions, self.height, self.width))

        # Fill tensor with data
        for ex_idx, ex in enumerate(self.excitation_wavelengths):
            ex_str = str(ex)

            # Get data for this excitation
            cube = self.data_dict['data'][ex_str]['cube']
            wavelengths = self.data_dict['data'][ex_str]['wavelengths']

            # Fill tensor (handle different numbers of emission wavelengths)
            n_em = len(wavelengths)
            # Transpose to [emissions, height, width] for PyTorch convention
            transposed_cube = np.transpose(cube, (2, 0, 1))
            tensor[ex_idx, :n_em, :, :] = transposed_cube

        return tensor

    def __getitem__(self, idx):
        """Get a specific sample from the dataset"""
        if self.patch_size is not None:
            # Get a specific patch
            data = self._get_patch(idx)
        elif self.sample_size is not None:
            # Get random pixel samples
            data = self._get_random_samples(idx)
        else:
            # Get the full image
            data = self._get_full_image()

        # Convert to tensor
        data_tensor = torch.tensor(data, dtype=torch.float32)

        # Apply transform if provided
        if self.transform:
            data_tensor = self.transform(data_tensor)

        return data_tensor

    def get_excitation_values(self):
        """Get the actual excitation wavelength values"""
        return self.ex_values

    def get_emission_values(self):
        """Get the actual emission wavelength values"""
        return self.em_values