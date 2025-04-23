"""
Hyperspectral Convolutional Autoencoder Training

This module provides functions for training and evaluating hyperspectral
convolutional autoencoders, including memory-efficient processing with chunking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Union

from hyperspectral_dataset import create_spatial_chunks, merge_chunk_reconstructions


def train_variable_cae(
    model,
    dataset,
    num_epochs=50,
    learning_rate=0.001,
    chunk_size=64,
    chunk_overlap=8,
    batch_size=1,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    early_stopping_patience=None,
    scheduler_patience=5
):
    """
    Train the hyperspectral convolutional autoencoder with variable emission bands.

    Args:
        model: HyperspectralCAEVariable model
        dataset: HyperspectralDataset with variable bands
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate for the optimizer
        chunk_size: Size of spatial chunks for processing
        chunk_overlap: Overlap between adjacent chunks
        batch_size: Batch size for training
        device: Device to use for training
        early_stopping_patience: Number of epochs with no improvement before stopping
        scheduler_patience: Number of epochs with no improvement before reducing learning rate

    Returns:
        Trained model and training losses
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=scheduler_patience, verbose=True
    )

    # Get all processed data
    all_data = dataset.get_all_data()

    # Get spatial dimensions
    height, width = dataset.get_spatial_dimensions()

    # Track losses
    train_losses = []
    best_loss = float('inf')
    best_epoch = 0

    # Early stopping counter
    no_improvement_count = 0

    # Create spatial chunks for each excitation wavelength
    print("Creating spatial chunks for each excitation wavelength...")
    chunks_dict = {}
    positions_dict = {}

    for ex, data in all_data.items():
        # Generate chunks for this excitation
        data_np = data.numpy()
        chunks, positions = create_spatial_chunks(data_np, chunk_size, chunk_overlap)
        chunks_dict[ex] = chunks
        positions_dict[ex] = positions

    # Check if we have any valid chunks
    if not chunks_dict or not next(iter(chunks_dict.values())):
        raise ValueError("No valid chunks found in the dataset")

    # Get number of chunks (should be same for all excitations)
    num_chunks = len(next(iter(chunks_dict.values())))

    print(f"Created {num_chunks} chunks for each excitation")

    # Create batches of chunks
    batches = []
    for i in range(0, num_chunks, batch_size):
        batch = {}
        for ex in chunks_dict:
            # Get chunks for this batch
            batch_chunks = chunks_dict[ex][i:i+batch_size]
            if batch_chunks:  # Only add if we have chunks for this batch
                # Convert to tensor with batch dimension
                batch[ex] = torch.tensor(np.stack(batch_chunks), dtype=torch.float32).to(device)
        batches.append(batch)

    print(f"Starting training for {num_epochs} epochs with {len(batches)} batches...")

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_sparsity_loss = 0.0

        # Train on each batch
        for i, batch in enumerate(batches):
            # Forward pass
            output = model(batch)

            # Compute reconstruction loss
            recon_loss = 0
            num_valid = 0
            for ex in batch:
                if ex in output:
                    # Make sure input and output have the same shape
                    if batch[ex].shape == output[ex].shape:
                        recon_loss += F.mse_loss(output[ex], batch[ex])
                        num_valid += 1
                    else:
                        print(f"Warning: Shape mismatch for excitation {ex}. Input: {batch[ex].shape}, Output: {output[ex].shape}")

            if num_valid > 0:
                recon_loss /= num_valid
            else:
                print("Warning: No valid excitations found for loss calculation in this batch")
                continue  # Skip this batch

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
            epoch_recon_loss += recon_loss.item()
            epoch_sparsity_loss += sparsity_loss.item()

            # Print progress
            if (i + 1) % 5 == 0 or i == len(batches) - 1:
                print(f"  Processed batch {i+1}/{len(batches)}", end="\r")

        # Record average loss for this epoch
        avg_loss = epoch_loss / len(batches)
        avg_recon_loss = epoch_recon_loss / len(batches)
        avg_sparsity_loss = epoch_sparsity_loss / len(batches)
        train_losses.append(avg_loss)

        # Update learning rate scheduler
        scheduler.step(avg_loss)

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f} "
              f"(Recon: {avg_recon_loss:.4f}, Sparsity: {avg_sparsity_loss:.4f}), "
              f"Time: {epoch_time:.2f}s")

        # Check if this is the best epoch so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            no_improvement_count = 0
            # Save best model
            torch.save(model.state_dict(), "best_hyperspectral_model.pth")
            print(f"  New best model saved (loss: {best_loss:.4f})")
        else:
            no_improvement_count += 1
            print(f"  No improvement for {no_improvement_count} epochs (best: {best_loss:.4f} at epoch {best_epoch+1})")

        # Early stopping
        if early_stopping_patience is not None and no_improvement_count >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Load the best model
    model.load_state_dict(torch.load("best_hyperspectral_model.pth"))

    print(f"Training completed. Best loss: {best_loss:.4f} at epoch {best_epoch+1}")
    return model, train_losses


def evaluate_model(
    model,
    dataset,
    chunk_size=64,
    chunk_overlap=8,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Evaluate the trained model by reconstructing data and calculating metrics.

    Args:
        model: Trained HyperspectralCAEVariable model
        dataset: HyperspectralDataset with test data
        chunk_size: Size of spatial chunks for processing
        chunk_overlap: Overlap between adjacent chunks
        device: Device to use for evaluation

    Returns:
        Dictionary with evaluation metrics and reconstructions
    """
    model = model.to(device)
    model.eval()

    # Get all data
    all_data = dataset.get_all_data()

    # Get spatial dimensions
    height, width = dataset.get_spatial_dimensions()

    # Store results
    results = {
        'metrics': {},
        'reconstructions': {}
    }

    print("Evaluating model on test data...")
    with torch.no_grad():
        overall_mse = 0.0
        overall_mae = 0.0
        num_excitations = 0

        for ex in all_data:
            data = all_data[ex]

            # Create chunks for this excitation
            chunks, positions = create_spatial_chunks(data.numpy(), chunk_size, chunk_overlap)

            # Process chunks
            reconstructed_chunks = []
            for i, chunk in enumerate(chunks):
                # Convert to tensor and add batch dimension
                chunk_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(device)

                # Create input dictionary for this excitation only
                chunk_dict = {ex: chunk_tensor}

                # Generate reconstruction
                output = model(chunk_dict)

                # Add to reconstructed chunks
                if ex in output:
                    reconstructed_chunks.append(output[ex])

                # Print progress
                if (i + 1) % 10 == 0 or i == len(chunks) - 1:
                    print(f"  Processed chunk {i+1}/{len(chunks)} for excitation {ex}", end="\r")

            # Skip this excitation if no valid reconstructions
            if not reconstructed_chunks:
                print(f"Warning: No valid reconstructions for excitation {ex}")
                continue

            # Merge chunks
            full_reconstruction = merge_chunk_reconstructions(
                reconstructed_chunks, positions, height, width
            )

            # Remove batch dimension
            full_reconstruction = full_reconstruction[0]

            # Store reconstruction
            results['reconstructions'][ex] = full_reconstruction

            # Calculate metrics
            mse = F.mse_loss(full_reconstruction, data.to(device)).item()
            mae = torch.mean(torch.abs(full_reconstruction - data.to(device))).item()

            # Calculate PSNR (Peak Signal-to-Noise Ratio)
            psnr = 10 * torch.log10(1.0 / torch.tensor(mse)).item()

            # Store metrics for this excitation
            results['metrics'][ex] = {
                'mse': mse,
                'mae': mae,
                'psnr': psnr
            }

            # Update overall metrics
            overall_mse += mse
            overall_mae += mae
            num_excitations += 1

            print(f"Excitation {ex}nm - MSE: {mse:.4f}, MAE: {mae:.4f}, PSNR: {psnr:.2f} dB")

        # Calculate overall metrics
        if num_excitations > 0:
            results['metrics']['overall'] = {
                'mse': overall_mse / num_excitations,
                'mae': overall_mae / num_excitations,
                'psnr': 10 * torch.log10(1.0 / torch.tensor(overall_mse / num_excitations)).item()
            }

            print(f"Overall - MSE: {results['metrics']['overall']['mse']:.4f}, "
                  f"MAE: {results['metrics']['overall']['mae']:.4f}, "
                  f"PSNR: {results['metrics']['overall']['psnr']:.2f} dB")

    return results