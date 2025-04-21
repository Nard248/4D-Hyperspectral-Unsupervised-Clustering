import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import argparse
from pathlib import Path
import pickle
import time

# Import our modules
from HyperspectralDataset import HyperspectralDataset
from HyperspectralAutoencoder import HyperspectralAutoencoder, PatchedHyperspectralAutoencoder
from training_utils import train_autoencoder, plot_training_history, evaluate_clustering
from visualization_utils import create_summary_visualization


def main(args):
    """Main function to run the hyperspectral clustering pipeline"""
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    model_dir = os.path.join(args.output_dir, 'models')
    results_dir = os.path.join(args.output_dir, 'results')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Step 1: Load the data
    print(f"Loading data from {args.data_path}...")
    dataset = HyperspectralDataset(
        args.data_path,
        patch_size=args.patch_size if args.use_patches else None,
        sample_size=args.sample_size if not args.use_patches and args.sample_size > 0 else None
    )

    # Get data dimensions
    n_excitations = len(dataset.excitation_wavelengths)
    n_emissions = dataset.max_emissions
    height, width = dataset.height, dataset.width

    print(f"Dataset created with dimensions: {height}x{width}, {n_excitations} excitations, {n_emissions} emissions")

    # Split dataset into train and validation
    if args.val_split > 0:
        # For patch-based training, split the patches
        if args.use_patches:
            train_size = int((1 - args.val_split) * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

            print(f"Split into {train_size} training patches and {val_size} validation patches")
        else:
            # For whole-image training, we don't split
            train_dataset, val_dataset = dataset, None
            print("Using whole image - no validation split applied")
    else:
        train_dataset, val_dataset = dataset, None

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True if args.use_patches else False,
        num_workers=args.num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    ) if val_dataset is not None else None

    # Step 2: Create the model
    print("Creating model...")

    if args.use_patches:
        model = PatchedHyperspectralAutoencoder(
            n_excitations=n_excitations,
            n_emissions=n_emissions,
            patch_size=args.patch_size,
            latent_dim=args.latent_dim,
            n_clusters=args.n_clusters
        )
        print(f"Created patched autoencoder with {args.patch_size}x{args.patch_size} patches")
    else:
        # For whole image or sample-based training
        model = HyperspectralAutoencoder(
            n_excitations=n_excitations,
            n_emissions=n_emissions,
            spatial_size=(height, width),
            latent_dim=args.latent_dim,
            n_clusters=args.n_clusters
        )
        print(f"Created autoencoder for {height}x{width} spatial dimensions")

    # Print model summary
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

    # Load checkpoint if specified
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    # Step 3: Train the model if not skipping training
    if not args.skip_training:
        print("Starting model training...")
        model, history = train_autoencoder(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            n_epochs=args.epochs,
            n_clusters=args.n_clusters,
            learning_rate=args.learning_rate,
            device=args.device,
            model_save_path=model_dir,
            update_interval=args.update_interval,
            patience=args.patience
        )

        # Plot training history
        history_path = os.path.join(results_dir, 'training_history.png')
        plot_training_history(history, save_path=history_path)
    else:
        print("Skipping training, using loaded model...")

    # Step 4: Evaluate and cluster
    print("Evaluating clustering performance...")
    model.eval()

    # For evaluation, we want to use the whole dataset if possible
    eval_dataset = dataset
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,  # Process as a single batch for full image
        shuffle=False,
        num_workers=args.num_workers
    )

    # Perform clustering
    result = evaluate_clustering(
        model=model,
        dataloader=eval_loader,
        n_clusters=args.n_clusters,
        device=args.device
    )

    # Get original and reconstructed data for visualization
    print("Getting reconstructed data for visualization...")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    with torch.no_grad():
        for batch in eval_loader:
            batch = batch.to(device)
            reconstructed, _, _ = model(batch)
            original_data = batch.cpu().numpy()
            reconstructed_data = reconstructed.cpu().numpy()
            break  # Just need one batch for visualization

    # Step 5: Visualize results
    print("Creating visualizations...")
    paths = create_summary_visualization(
        result=result,
        original_data=original_data,
        reconstructed_data=reconstructed_data,
        data_dict=dataset.data_dict,
        height=height,
        width=width,
        n_clusters=args.n_clusters,
        save_dir=results_dir,
        prefix='cluster_'
    )

    print("Process complete!")
    print(f"Results saved to {results_dir}")

    return result, paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperspectral Clustering with Convolutional Autoencoders")

    # Data parameters
    parser.add_argument("--data_path", type=str, required=True, help="Path to data pickle file")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory for results")
    parser.add_argument("--use_patches", action="store_true", help="Use patch-based training instead of whole image")
    parser.add_argument("--patch_size", type=int, default=64, help="Size of patches to extract")
    parser.add_argument("--sample_size", type=int, default=0, help="Number of random pixels to sample (0=use all)")

    # Model parameters
    parser.add_argument("--latent_dim", type=int, default=128, help="Dimensionality of latent space")
    parser.add_argument("--n_clusters", type=int, default=10, help="Number of clusters")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--update_interval", type=int, default=10, help="Update clustering centroids every N epochs")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to checkpoint to resume from")
    parser.add_argument("--skip_training", action="store_true", help="Skip training and use loaded model")

    # Miscellaneous
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")

    args = parser.parse_args()
    main(args)