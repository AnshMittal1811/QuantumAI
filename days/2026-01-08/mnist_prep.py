import torch
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
import numpy as np
import os

# January 08, 2026: MNIST Preprocessing for Quantum Circuits
# Reducing 28x28 images to small latent vectors (e.g., dim=4) using PCA.

def main():
    # Setup data directory
    data_dir = "../../data/mnist"
    os.makedirs(data_dir, exist_ok=True)

    # Download MNIST
    print("Downloading MNIST...")
    mnist_train = datasets.MNIST(data_dir, train=True, download=True,
                                 transform=transforms.Compose([transforms.ToTensor()]))

    # Flatten images: (60000, 1, 28, 28) -> (60000, 784)
    x_train = mnist_train.data.numpy().reshape(-1, 28*28) / 255.0
    y_train = mnist_train.targets.numpy()

    # Filter for only 0s and 1s to simplify early experiments
    mask = (y_train == 0) | (y_train == 1)
    x_filtered = x_train[mask]
    y_filtered = y_train[mask]

    print(f"Filtered Dataset Size: {len(x_filtered)} (0s and 1s)")

    # Reduce dimensionality using PCA to 4 dimensions (suitable for 2-4 qubits)
    print("Running PCA (784 -> 4 dimensions)...")
    pca = PCA(n_components=4)
    x_pca = pca.fit_transform(x_filtered)

    # Normalize to [-pi, pi] for angle encoding
    x_norm = 2 * np.pi * (x_pca - x_pca.min()) / (x_pca.max() - x_pca.min()) - np.pi

    print(f"Sample Latent Vector: {x_norm[0]}")
    
    # Save processed data for Day 9/10
    np.save(os.path.join(data_dir, "mnist_01_pca4.npy"), x_norm)
    np.save(os.path.join(data_dir, "mnist_01_labels.npy"), y_filtered)
    print(f"Saved processed data to {data_dir}")

if __name__ == "__main__":
    main()
