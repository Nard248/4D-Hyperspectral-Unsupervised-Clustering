import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import os
from pathlib import Path


class DeepClusteringLoss(nn.Module):
    """
    Combined loss function for deep clustering with reconstruction.

    Combines autoencoder reconstruction loss with clustering objectives.
    Uses k-means to initialize and update cluster centroids during training.
    """

    def __init__(self, n_clusters, alpha=1.0, beta=0.1, gamma=0.01):
        super().__init__()
        self.n_clusters = n_clusters

        # Loss weights
        self.alpha = alpha  # Reconstruction weight
        self.beta = beta  # Clustering weight
        self.gamma = gamma  # Regularization weight

        # K-means state
        self.kmeans = None
        self.centroids = None
        self.initialized = False

    def forward(self, original, reconstructed, latent_features, target_distribution=None):
        """
        Compute the loss combining reconstruction and clustering objectives.
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstructed, original)

        # Initialize centroids with k-means if not already done
        if not self.initialized or self.centroids is None:
            # Detach to avoid backprop through initialization
            features_np = latent_features.detach().cpu().numpy()

            # Handle NaN values
            if np.isnan(features_np).any():
                nan_count = np.isnan(features_np).sum()
                print(f"Warning: Found {nan_count} NaN values in features")

                # First try to filter samples with NaN
                mask = ~np.isnan(features_np).any(axis=1)
                filtered_features = features_np[mask]

                # If filtering removes ALL samples, use nan_to_num instead
                if len(filtered_features) == 0:
                    print("All samples contain NaN values - replacing NaNs with zeros")
                    features_np = np.nan_to_num(features_np, nan=0.0)
                else:
                    features_np = filtered_features
                    print(f"Using {len(features_np)} samples after filtering NaNs")

            # Additional safety check
            if len(features_np) == 0:
                print("ERROR: No valid samples for clustering! Using random initialization.")
                # Create a small set of random samples as fallback
                features_np = np.random.randn(self.n_clusters, latent_features.shape[1])

            # Initialize k-means
            self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
            self.kmeans.fit(features_np)

            # Convert centroids to tensor
            self.centroids = torch.tensor(
                self.kmeans.cluster_centers_,
                dtype=torch.float,
                device=latent_features.device
            )

            self.initialized = True

        # Compute distances to centroids
        distances = torch.cdist(latent_features, self.centroids)

        # Convert distances to probabilities (Student's t-distribution)
        q = 1.0 / (1.0 + distances ** 2)
        q = q / q.sum(dim=1, keepdim=True)

        # If target distribution is provided, use KL divergence
        if target_distribution is not None:
            # KL divergence clustering loss
            cluster_loss = F.kl_div(
                q.log(), target_distribution, reduction='batchmean'
            )
        else:
            # Basic clustering loss (minimize distance to nearest centroid)
            min_distances = torch.min(distances, dim=1)[0]
            cluster_loss = torch.mean(min_distances)

        # Regularization: feature variance to avoid trivial solutions
        feature_var = torch.var(latent_features, dim=0).mean()
        reg_loss = 1.0 / (feature_var + 1e-6)  # Encourage feature variance

        # Combined loss
        total_loss = (
                self.alpha * recon_loss +
                self.beta * cluster_loss +
                self.gamma * reg_loss
        )

        return total_loss, recon_loss, cluster_loss, reg_loss
    def update_centroids(self, latent_features):
        """
        Update centroids based on the current batch.

        Args:
            latent_features: Current batch latent features

        Returns:
            Updated centroids
        """
        # Get all features
        features_np = latent_features.detach().cpu().numpy()

        # Update centroids with new k-means
        self.kmeans.fit(features_np)
        self.centroids = torch.tensor(
            self.kmeans.cluster_centers_,
            dtype=torch.float,
            device=latent_features.device
        )

        return self.centroids

    def target_distribution(self, q):
        """
        Compute the target distribution for clustering.

        This is used in the Deep Embedded Clustering approach to
        sharpen the distribution and emphasize high confidence assignments.

        Args:
            q: Soft assignments [batch_size, n_clusters]

        Returns:
            Target distribution
        """
        # Sharpen the distribution
        weight = q ** 2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()


class EarlyStopping:
    """Early stopping to prevent overfitting"""

    def __init__(self, patience=10, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered")


def train_autoencoder(model, train_loader, val_loader=None,
                      n_epochs=100, n_clusters=10,
                      learning_rate=0.001, device='cuda',
                      model_save_path='models',
                      update_interval=10,
                      patience=10):
    """
    Train the autoencoder model.

    Args:
        model: The autoencoder model
        train_loader: DataLoader for training data
        val_loader: Optional DataLoader for validation data
        n_epochs: Number of training epochs
        n_clusters: Number of clusters
        learning_rate: Learning rate for optimizer
        device: Device to use for training
        model_save_path: Directory to save model checkpoints
        update_interval: Update cluster centroids every N epochs
        patience: Early stopping patience

    Returns:
        Trained model and training history
    """
    # Set device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Define loss and optimizer
    criterion = DeepClusteringLoss(n_clusters=n_clusters)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Early stopping
    early_stopping = EarlyStopping(patience=patience)

    # Create save directory
    os.makedirs(model_save_path, exist_ok=True)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'recon_loss': [],
        'cluster_loss': [],
        'reg_loss': []
    }

    # Track best validation loss
    best_val_loss = float('inf')

    # Training loop
    start_time = time.time()
    print(f"Starting training for {n_epochs} epochs...")

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        recon_loss_sum = 0.0
        cluster_loss_sum = 0.0
        reg_loss_sum = 0.0

        # Training
        for batch_idx, data in enumerate(train_loader):
            # Move data to device
            data = data.to(device)

            # Forward pass
            reconstructed, latent, _ = model(data)

            # Update target distribution
            if epoch >= 20 and epoch % update_interval == 0 and batch_idx == 0:
                # Compute distances to centroids
                with torch.no_grad():
                    distances = torch.cdist(latent, criterion.centroids)
                    q = 1.0 / (1.0 + distances ** 2)
                    q = q / q.sum(dim=1, keepdim=True)

                    # Compute target distribution
                    target_dist = criterion.target_distribution(q)
            else:
                target_dist = None

            # Compute loss
            loss, recon_loss, cluster_loss, reg_loss = criterion(
                data, reconstructed, latent, target_dist
            )

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update statistics
            total_loss += loss.item() * data.size(0)
            recon_loss_sum += recon_loss.item() * data.size(0)
            cluster_loss_sum += cluster_loss.item() * data.size(0)
            reg_loss_sum += reg_loss.item() * data.size(0)

        # Update centroids periodically
        if epoch % update_interval == 0 and epoch > 0:
            # Collect all latent representations
            all_features = []
            model.eval()
            with torch.no_grad():
                for data in train_loader:
                    data = data.to(device)
                    _, latent, _ = model(data)
                    all_features.append(latent)

            all_features = torch.cat(all_features, dim=0)
            criterion.update_centroids(all_features)

        # Compute epoch averages
        train_samples = len(train_loader.dataset)
        avg_loss = total_loss / train_samples
        avg_recon_loss = recon_loss_sum / train_samples
        avg_cluster_loss = cluster_loss_sum / train_samples
        avg_reg_loss = reg_loss_sum / train_samples

        # Validation
        val_loss = 0.0
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)
                    reconstructed, latent, _ = model(data)

                    # Compute loss - only reconstruction for validation
                    loss = F.mse_loss(reconstructed, data)
                    val_loss += loss.item() * data.size(0)

            val_loss /= len(val_loader.dataset)

            # Update learning rate based on validation loss
            scheduler.step(val_loss)

            # Check early stopping
            early_stopping(val_loss)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(model_save_path, 'best_model.pt'))
                print(f"Saved best model with validation loss: {val_loss:.6f}")

            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break

        # Store in history
        history['train_loss'].append(avg_loss)
        history['val_loss'].append(val_loss if val_loader is not None else None)
        history['recon_loss'].append(avg_recon_loss)
        history['cluster_loss'].append(avg_cluster_loss)
        history['reg_loss'].append(avg_reg_loss)

        # Print progress
        elapsed = time.time() - start_time
        print(f'Epoch {epoch + 1}/{n_epochs}, Time: {elapsed:.2f}s, '
              f'Loss: {avg_loss:.6f}, '
              f'Recon: {avg_recon_loss:.6f}, '
              f'Cluster: {avg_cluster_loss:.6f}, '
              f'Reg: {avg_reg_loss:.6f}' +
              (f', Val Loss: {val_loss:.6f}' if val_loader is not None else ''))

        # Save model checkpoint periodically
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'history': history
            }, os.path.join(model_save_path, f'checkpoint_{epoch + 1}.pt'))

    # Save final model
    torch.save(model.state_dict(), os.path.join(model_save_path, 'final_model.pt'))

    # Save training history
    np.save(os.path.join(model_save_path, 'training_history.npy'), history)

    # Load best model if validation was used
    if val_loader is not None:
        model.load_state_dict(torch.load(os.path.join(model_save_path, 'best_model.pt')))

    return model, history


def plot_training_history(history, save_path=None):
    """
    Plot training history.

    Args:
        history: Dictionary with training history
        save_path: Path to save plot
    """
    plt.figure(figsize=(15, 10))

    # Plot loss curves
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    if history['val_loss'][0] is not None:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot reconstruction loss
    plt.subplot(2, 2, 2)
    plt.plot(history['recon_loss'], label='Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Reconstruction Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot clustering loss
    plt.subplot(2, 2, 3)
    plt.plot(history['cluster_loss'], label='Clustering Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Clustering Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot regularization loss
    plt.subplot(2, 2, 4)
    plt.plot(history['reg_loss'], label='Regularization Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Regularization Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")

    plt.show()


def evaluate_clustering(model, dataloader, n_clusters=10, device='cuda'):
    """
    Perform clustering on model features and evaluate quality.

    Args:
        model: Trained autoencoder model
        dataloader: DataLoader with data to cluster
        n_clusters: Number of clusters
        device: Device to use

    Returns:
        Dictionary with clustering results
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Collect all latent features
    all_features = []
    all_reconstructions = []

    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            reconstructed, latent, _ = model(data)

            all_features.append(latent.cpu().numpy())
            all_reconstructions.append(reconstructed.cpu().numpy())

    # Concatenate features
    features = np.vstack(all_features)
    reconstructions = np.vstack(all_reconstructions)

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features)

    # Evaluate clustering quality
    metrics = {}

    try:
        # Silhouette Score (higher is better, range: -1 to 1)
        silhouette = silhouette_score(features, cluster_labels)
        metrics['silhouette'] = silhouette
        print(f"Silhouette Score: {silhouette:.4f}")

        # Davies-Bouldin Index (lower is better)
        db_score = davies_bouldin_score(features, cluster_labels)
        metrics['davies_bouldin'] = db_score
        print(f"Davies-Bouldin Index: {db_score:.4f}")

        # Calinski-Harabasz Index (higher is better)
        ch_score = calinski_harabasz_score(features, cluster_labels)
        metrics['calinski_harabasz'] = ch_score
        print(f"Calinski-Harabasz Index: {ch_score:.4f}")
    except Exception as e:
        print(f"Error computing clustering metrics: {e}")

    # Create result dictionary
    result = {
        'features': features,
        'reconstructions': reconstructions,
        'cluster_labels': cluster_labels,
        'kmeans': kmeans,
        'metrics': metrics
    }

    return result