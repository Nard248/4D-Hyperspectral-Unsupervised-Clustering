"""
Training module for hyperspectral autoencoder models.

This module provides functions for efficient training and evaluation
of hyperspectral autoencoder models with support for chunking and masking.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn.functional as F

from .autoencoder import HyperspectralCAEWithMasking

def create_spatial_chunks(data_tensor, mask=None, chunk_size=128, chunk_overlap=16):
    """
    Split a large spatial hyperspectral tensor into overlapping chunks.

    Args:
        data_tensor: Input tensor of shape [height, width, emission_bands]
        mask: Optional binary mask of shape [height, width]
        chunk_size: Size of each spatial chunk
        chunk_overlap: Overlap between adjacent chunks

    Returns:
        List of chunk tensors, their positions, and optionally mask chunks
    """
    # Determine input shape
    if len(data_tensor.shape) == 4:  # [num_excitations, height, width, emission_bands]
        height, width = data_tensor.shape[1], data_tensor.shape[2]
    else:  # [height, width, emission_bands]
        height, width = data_tensor.shape[0], data_tensor.shape[1]

    # Calculate stride
    stride = chunk_size - chunk_overlap

    # Calculate number of chunks in each dimension
    num_chunks_y = max(1, (height - chunk_overlap) // stride)
    num_chunks_x = max(1, (width - chunk_overlap) // stride)

    # Adjust to ensure we cover the entire image
    if stride * num_chunks_y + chunk_overlap < height:
        num_chunks_y += 1
    if stride * num_chunks_x + chunk_overlap < width:
        num_chunks_x += 1

    # Create lists to store chunks and their positions
    chunks = []
    positions = []
    mask_chunks = [] if mask is not None else None

    # Extract chunks
    for i in range(num_chunks_y):
        for j in range(num_chunks_x):
            # Calculate start and end positions
            y_start = i * stride
            x_start = j * stride
            y_end = min(y_start + chunk_size, height)
            x_end = min(x_start + chunk_size, width)

            # Handle edge cases by adjusting start positions
            if y_end == height:
                y_start = max(0, height - chunk_size)
            if x_end == width:
                x_start = max(0, width - chunk_size)

            # Extract chunk based on input shape
            if len(data_tensor.shape) == 4:  # [num_excitations, height, width, emission_bands]
                chunk = data_tensor[:, y_start:y_end, x_start:x_end, :]
            else:  # [height, width, emission_bands]
                chunk = data_tensor[y_start:y_end, x_start:x_end, :]

            # Extract mask chunk if provided
            if mask is not None:
                mask_chunk = mask[y_start:y_end, x_start:x_end]
                mask_chunks.append(mask_chunk)

            # Add to lists
            chunks.append(chunk)
            positions.append((y_start, y_end, x_start, x_end))

    if mask is not None:
        return chunks, positions, mask_chunks
    else:
        return chunks, positions


def merge_chunk_reconstructions(chunks, positions, full_height, full_width):
    """
    Merge the reconstructed chunks back into a full image.
    For overlapping regions, take the average of the reconstructions.

    Args:
        chunks: List of reconstructed chunk tensors
        positions: List of positions (y_start, y_end, x_start, x_end) for each chunk
        full_height: Height of the full reconstructed image
        full_width: Width of the full reconstructed image

    Returns:
        Merged full reconstruction
    """
    # Determine shape from the first chunk
    first_chunk = chunks[0]

    if len(first_chunk.shape) == 4:  # [batch, height, width, emission_bands]
        batch_size, _, _, num_bands = first_chunk.shape
        merged = torch.zeros((batch_size, full_height, full_width, num_bands),
                             device=first_chunk.device)
        weights = torch.zeros((batch_size, full_height, full_width, num_bands),
                              device=first_chunk.device)
    elif len(first_chunk.shape) == 5:  # [batch, num_excitations, height, width, emission_bands]
        batch_size, num_excitations, _, _, num_bands = first_chunk.shape
        merged = torch.zeros((batch_size, num_excitations, full_height, full_width, num_bands),
                             device=first_chunk.device)
        weights = torch.zeros((batch_size, num_excitations, full_height, full_width, num_bands),
                              device=first_chunk.device)
    else:
        raise ValueError(f"Unexpected chunk shape: {first_chunk.shape}")

    # Merge chunks
    for chunk, (y_start, y_end, x_start, x_end) in zip(chunks, positions):
        if len(chunk.shape) == 4:  # [batch, height, width, emission_bands]
            merged[:, y_start:y_end, x_start:x_end, :] += chunk
            weights[:, y_start:y_end, x_start:x_end, :] += 1
        else:  # [batch, num_excitations, height, width, emission_bands]
            merged[:, :, y_start:y_end, x_start:x_end, :] += chunk
            weights[:, :, y_start:y_end, x_start:x_end, :] += 1

    # Average overlapping regions
    merged = merged / torch.clamp(weights, min=1.0)

    return merged


def train_with_masking(
        model,
        dataset,
        num_epochs=50,
        learning_rate=0.001,
        chunk_size=64,
        chunk_overlap=8,
        batch_size=1,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        early_stopping_patience=None,
        scheduler_patience=5,
        mask=None,
        output_dir="model_output",
        verbose=True
):
    """
    Train the hyperspectral autoencoder with masked loss.

    Args:
        model: HyperspectralCAEWithMasking model
        dataset: MaskedHyperspectralDataset with data
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate for the optimizer
        chunk_size: Size of spatial chunks for processing
        chunk_overlap: Overlap between adjacent chunks
        batch_size: Batch size for training
        device: Device to use for training
        early_stopping_patience: Number of epochs with no improvement before stopping
        scheduler_patience: Number of epochs with no improvement before reducing learning rate
        mask: Optional binary mask (1=valid, 0=masked)
        output_dir: Directory to save model outputs
        verbose: Whether to print detailed progress

    Returns:
        Trained model and training losses
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Move model to device
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=scheduler_patience, verbose=True
    )

    # Get all processed data
    all_data = dataset.get_all_data()
    all_valid_masks = dataset.get_all_valid_masks()

    # Get spatial dimensions
    height, width = dataset.get_spatial_dimensions()

    # Process the mask if provided, otherwise use the dataset's processed mask
    if mask is None:
        mask = dataset.processed_mask

    mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device)

    # Track losses
    train_losses = []
    best_loss = float('inf')
    best_epoch = 0

    # Early stopping counter
    no_improvement_count = 0

    # Create spatial chunks for each excitation wavelength
    if verbose:
        print("Creating spatial chunks for each excitation wavelength...")

    chunks_dict = {}
    positions_dict = {}
    mask_chunks = []

    # Create chunks of the spatial mask
    mask_chunks, mask_positions = create_spatial_chunks(
        mask[..., np.newaxis],  # Add dummy dimension for compatibility
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )[0:2]  # Only get chunks and positions

    # Remove dummy dimension
    mask_chunks = [chunk[..., 0] for chunk in mask_chunks]

    # Create chunks for each excitation wavelength
    for ex in all_data:
        # Get data and convert to numpy
        data_np = all_data[ex].numpy()

        # Generate chunks for this excitation
        chunks_result = create_spatial_chunks(
            data_np,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = chunks_result[0]
        positions = chunks_result[1]

        chunks_dict[ex] = chunks
        positions_dict[ex] = positions

    # Check if we have any valid chunks
    if not chunks_dict or not next(iter(chunks_dict.values())):
        raise ValueError("No valid chunks found in the dataset")

    # Get number of chunks (should be same for all excitations)
    num_chunks = len(next(iter(chunks_dict.values())))

    if verbose:
        print(f"Created {num_chunks} chunks for each excitation")

    # Create batches of chunks
    batches = []
    mask_batches = []

    for i in range(0, num_chunks, batch_size):
        # Data batch
        batch = {}
        for ex in chunks_dict:
            # Get chunks for this batch
            batch_chunks = chunks_dict[ex][i:i + batch_size]
            if batch_chunks:  # Only add if we have chunks for this batch
                # Convert to tensor with batch dimension
                batch[ex] = torch.tensor(np.stack(batch_chunks), dtype=torch.float32).to(device)
        batches.append(batch)

        # Mask batch
        mask_batch_chunks = mask_chunks[i:i + batch_size]
        if mask_batch_chunks:
            mask_batches.append(torch.tensor(np.stack(mask_batch_chunks), dtype=torch.float32).to(device))
        else:
            mask_batches.append(None)

    if verbose:
        print(f"Starting training for {num_epochs} epochs with {len(batches)} batches...")

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_sparsity_loss = 0.0

        # Train on each batch
        for i, (batch, mask_batch) in enumerate(zip(batches, mask_batches)):
            # Forward pass
            output = model(batch)

            # Compute masked reconstruction loss
            recon_loss = model.compute_masked_loss(
                output_dict=output,
                target_dict=batch,
                spatial_mask=mask_batch
            )

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
            if verbose and ((i + 1) % 5 == 0 or i == len(batches) - 1):
                print(f"  Processed batch {i + 1}/{len(batches)}", end="\r")

        # Record average loss for this epoch
        avg_loss = epoch_loss / len(batches)
        avg_recon_loss = epoch_recon_loss / len(batches)
        avg_sparsity_loss = epoch_sparsity_loss / len(batches)
        train_losses.append(avg_loss)

        # Update learning rate scheduler
        scheduler.step(avg_loss)

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f} "
              f"(Recon: {avg_recon_loss:.4f}, Sparsity: {avg_sparsity_loss:.4f}), "
              f"Time: {epoch_time:.2f}s")

        # Check if this is the best epoch so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            no_improvement_count = 0

            # Save best model
            best_model_path = os.path.join(output_dir, "best_hyperspectral_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best model saved to {best_model_path} (loss: {best_loss:.4f})")

            # Save training curves
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, marker='o')
            plt.title('Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
            plt.yscale('log')  # Use log scale to better visualize loss decrease
            curves_path = os.path.join(output_dir, "training_curves.png")
            plt.savefig(curves_path)
            plt.close()
        else:
            no_improvement_count += 1
            print(
                f"  No improvement for {no_improvement_count} epochs (best: {best_loss:.4f} at epoch {best_epoch + 1})")

        # Early stopping
        if early_stopping_patience is not None and no_improvement_count >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    # Save final model
    final_model_path = os.path.join(output_dir, "final_hyperspectral_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

    # Load the best model
    model.load_state_dict(torch.load(best_model_path))

    # Save loss values
    np.save(os.path.join(output_dir, "training_losses.npy"), np.array(train_losses))

    # Save final training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    curves_path = os.path.join(output_dir, "final_training_curves.png")
    plt.savefig(curves_path)
    plt.close()

    print(f"Training completed. Best loss: {best_loss:.4f} at epoch {best_epoch + 1}")
    return model, train_losses


def evaluate_model_with_masking(
        model,
        dataset,
        chunk_size=64,
        chunk_overlap=8,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        output_dir=None
):
    """
    Evaluate the trained model with masking and calculate metrics only on valid pixels.

    Args:
        model: Trained HyperspectralCAEWithMasking model
        dataset: MaskedHyperspectralDataset with test data
        chunk_size: Size of spatial chunks for processing
        chunk_overlap: Overlap between adjacent chunks
        device: Device to use for evaluation
        output_dir: Directory to save evaluation results

    Returns:
        Dictionary with evaluation metrics and reconstructions
    """
    model = model.to(device)
    model.eval()

    # Get all data and masks
    all_data = dataset.get_all_data()
    all_valid_masks = dataset.get_all_valid_masks()
    spatial_mask = dataset.processed_mask

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
            valid_mask = all_valid_masks[ex] if ex in all_valid_masks else None

            # Create chunks for this excitation
            chunks_result = create_spatial_chunks(data.numpy(), chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = chunks_result[0]
            positions = chunks_result[1]
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
                    print(f"  Processed chunk {i + 1}/{len(chunks)} for excitation {ex}", end="\r")

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

            # Apply masks for metric calculation
            # Find this section in the function:
            if valid_mask is not None:
                # Check dimensions of valid_mask
                if len(valid_mask.shape) == len(data.shape):
                    # Valid mask already has same dimension structure as data
                    valid_mask_expanded = valid_mask
                elif len(valid_mask.shape) == len(data.shape) - 1:
                    # Need to add one dimension to match data
                    valid_mask_expanded = valid_mask.unsqueeze(-1).expand_as(data)
                else:
                    # Handle other cases - reshape as needed
                    print(f"Warning: Valid mask shape {valid_mask.shape} doesn't match data shape {data.shape}")
                    # Try to reshape in a best-effort manner
                    if len(valid_mask.shape) == 2:  # Spatial mask [height, width]
                        valid_mask_expanded = valid_mask.unsqueeze(-1).expand(valid_mask.shape[0],
                                                                              valid_mask.shape[1],
                                                                              data.shape[2])
                    else:
                        # Last resort - just use data shape
                        valid_mask_expanded = torch.ones_like(data, device=device)

                # Move valid_mask_expanded to the specified device
                valid_mask_expanded = valid_mask_expanded.to(device)

                # Apply spatial mask if available
                if spatial_mask is not None:
                    spatial_mask_tensor = torch.tensor(spatial_mask, dtype=torch.float32, device=device)

                    # Similar dimension checking for spatial mask
                    if len(spatial_mask_tensor.shape) == len(data.shape):
                        spatial_mask_expanded = spatial_mask_tensor
                    elif len(spatial_mask_tensor.shape) == len(data.shape) - 1:
                        spatial_mask_expanded = spatial_mask_tensor.unsqueeze(-1).expand_as(data)
                    else:
                        # Handle other cases
                        if len(spatial_mask_tensor.shape) == 2:  # Spatial mask [height, width]
                            spatial_mask_expanded = spatial_mask_tensor.unsqueeze(-1).expand(
                                spatial_mask_tensor.shape[0],
                                spatial_mask_tensor.shape[1],
                                data.shape[2])
                        else:
                            spatial_mask_expanded = torch.ones_like(data, device=device)

                    # Move spatial_mask_expanded to the specified device
                    spatial_mask_expanded = spatial_mask_expanded.to(device)

                    combined_mask = valid_mask_expanded * spatial_mask_expanded
                else:
                    combined_mask = valid_mask_expanded

                # Calculate metrics only on valid pixels
                masked_squared_error = ((full_reconstruction - data.to(device)) ** 2) * combined_mask.to(device)
                masked_abs_error = torch.abs(full_reconstruction - data.to(device)) * combined_mask.to(device)

                num_valid_pixels = combined_mask.sum().item()

                if num_valid_pixels > 0:
                    mse = masked_squared_error.sum().item() / num_valid_pixels
                    mae = masked_abs_error.sum().item() / num_valid_pixels

                    # Calculate PSNR
                    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')

                    # Store metrics
                    results['metrics'][ex] = {
                        'mse': mse,
                        'mae': mae,
                        'psnr': psnr,
                        'valid_pixels': num_valid_pixels
                    }

                    # Update overall metrics
                    overall_mse += mse
                    overall_mae += mae
                    num_excitations += 1

                    print(f"Excitation {ex}nm - MSE: {mse:.4f}, MAE: {mae:.4f}, PSNR: {psnr:.2f} dB "
                          f"(on {num_valid_pixels} valid pixels)")
                else:
                    print(f"Warning: No valid pixels for excitation {ex}")
            else:
                # If no valid mask, use all pixels
                mse = F.mse_loss(full_reconstruction, data.to(device)).item()
                mae = torch.mean(torch.abs(full_reconstruction - data.to(device))).item()
                psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')

                results['metrics'][ex] = {
                    'mse': mse,
                    'mae': mae,
                    'psnr': psnr
                }

                overall_mse += mse
                overall_mae += mae
                num_excitations += 1

                print(f"Excitation {ex}nm - MSE: {mse:.4f}, MAE: {mae:.4f}, PSNR: {psnr:.2f} dB")

        # Calculate overall metrics
        if num_excitations > 0:
            results['metrics']['overall'] = {
                'mse': overall_mse / num_excitations,
                'mae': overall_mae / num_excitations,
                'psnr': 10 * np.log10(1.0 / (overall_mse / num_excitations)) if overall_mse > 0 else float('inf')
            }

            print(f"Overall - MSE: {results['metrics']['overall']['mse']:.4f}, "
                  f"MAE: {results['metrics']['overall']['mae']:.4f}, "
                  f"PSNR: {results['metrics']['overall']['psnr']:.2f} dB")

    # Save evaluation results if output directory provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

        # Save metrics to CSV
        import csv
        metrics_file = os.path.join(output_dir, "evaluation_metrics.csv")
        with open(metrics_file, 'w', newline='') as csvfile:
            fieldnames = ['excitation', 'mse', 'mae', 'psnr', 'valid_pixels']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for ex, metrics in results['metrics'].items():
                if ex != 'overall':
                    row = {'excitation': ex}
                    for metric, value in metrics.items():
                        row[metric] = value
                    writer.writerow(row)

            # Write overall metrics
            if 'overall' in results['metrics']:
                row = {'excitation': 'overall'}
                for metric, value in results['metrics']['overall'].items():
                    row[metric] = value
                writer.writerow(row)

        print(f"Evaluation metrics saved to {metrics_file}")

    return results
