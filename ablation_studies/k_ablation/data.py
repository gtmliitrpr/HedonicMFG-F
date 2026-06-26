"""
data.py — Dataset loading and non-IID Dirichlet partitioning for FL.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
from collections import defaultdict


# ──────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────
MNIST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


# ──────────────────────────────────────────
# Dirichlet Partitioning
# ──────────────────────────────────────────
def dirichlet_partition(targets: np.ndarray, num_clients: int,
                         alpha: float, seed: int = 42) -> list:
    """
    Partition dataset indices among clients using Dirichlet distribution.

    Args:
        targets     : array of class labels for all samples
        num_clients : number of FL clients
        alpha       : Dirichlet concentration (smaller = more heterogeneous)
        seed        : random seed

    Returns:
        client_indices : list of np.arrays, one per client
    """
    np.random.seed(seed)
    num_classes = int(targets.max()) + 1
    class_indices = [np.where(targets == c)[0] for c in range(num_classes)]

    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        np.random.shuffle(class_indices[c])
        # Sample proportions from Dirichlet — standard correct implementation
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        # Convert to split points along this class's indices
        proportions = (np.cumsum(proportions) * len(class_indices[c])).astype(int)[:-1]
        splits = np.split(class_indices[c], proportions)
        for i, split in enumerate(splits):
            client_indices[i].extend(split.tolist())

    # Shuffle each client's data
    for i in range(num_clients):
        np.random.shuffle(client_indices[i])

    return [np.array(idx) for idx in client_indices]


def print_partition_stats(client_indices: list, targets: np.ndarray, num_classes: int):
    """Print data distribution stats across clients."""
    print(f"\n{'Client':<8} {'Samples':>8} {'Classes':>8}  Distribution")
    print("-" * 60)
    for i, idx in enumerate(client_indices):
        if len(idx) == 0:
            print(f"{i:<8} {'0':>8}")
            continue
        labels = targets[idx]
        dist = np.bincount(labels, minlength=num_classes)
        classes_present = (dist > 0).sum()
        dist_str = " ".join(f"{d:3d}" for d in dist)
        print(f"{i:<8} {len(idx):>8} {classes_present:>8}  [{dist_str}]")


# ──────────────────────────────────────────
# MNIST Dataset Loading
# ──────────────────────────────────────────
def load_mnist(data_dir: str = "./data") -> tuple:
    """Load full MNIST train and test sets."""
    train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=MNIST_TRANSFORM
    )
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=MNIST_TRANSFORM
    )
    return train_dataset, test_dataset


def get_mnist_client_loaders(config: dict) -> tuple:
    """
    Build per-client DataLoaders for MNIST with Dirichlet partitioning.

    Returns:
        client_train_loaders : list of DataLoader (one per client)
        client_val_loaders   : list of DataLoader (one per client)
        global_test_loader   : DataLoader for global test set
        client_data_sizes    : list of ints (training samples per client)
    """
    train_dataset, test_dataset = load_mnist(config.get("data_dir", "./data"))

    targets = np.array(train_dataset.targets)
    num_clients = config["num_clients"]
    alpha = config["dirichlet_alpha"]
    seed = config["seed"]
    batch_size = config["batch_size"]
    val_frac = config.get("val_fraction", 0.1)

    # Partition
    client_indices = dirichlet_partition(targets, num_clients, alpha, seed)

    print(f"\n[Data] MNIST | {num_clients} clients | Dirichlet α={alpha}")
    print_partition_stats(client_indices, targets, num_classes=10)

    client_train_loaders = []
    client_val_loaders = []
    client_data_sizes = []

    # Minimum samples needed: at least 1 val + batch_size train
    min_samples = batch_size + 1

    for i, indices in enumerate(client_indices):
        # Boost tiny/empty clients by sampling from global pool
        if len(indices) < min_samples:
            np.random.seed(seed + i + 1000)
            all_idx = np.arange(len(targets))
            extra = np.random.choice(all_idx, size=min_samples, replace=False)
            indices = np.union1d(indices, extra)
            print(f"  [Data] Client {i} boosted: {len(indices)} samples "
                  f"(was too small for α={alpha})")

        # Train/val split
        np.random.seed(seed + i)
        np.random.shuffle(indices)
        n_val     = max(1, int(len(indices) * val_frac))
        n_train   = len(indices) - n_val
        # Guard: ensure train set has at least batch_size samples
        if n_train < batch_size:
            n_val   = max(1, len(indices) - batch_size)
            n_train = len(indices) - n_val
        val_idx   = indices[:n_val]
        train_idx = indices[n_val:]

        train_subset = Subset(train_dataset, train_idx)
        val_subset   = Subset(train_dataset, val_idx)

        train_loader = DataLoader(
            train_subset, batch_size=min(batch_size, len(train_idx)),
            shuffle=True, num_workers=0, pin_memory=True
        )
        val_loader = DataLoader(
            val_subset, batch_size=min(batch_size * 2, len(val_idx)),
            shuffle=False, num_workers=0, pin_memory=True
        )

        client_train_loaders.append(train_loader)
        client_val_loaders.append(val_loader)
        client_data_sizes.append(len(train_idx))

    global_test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False,
        num_workers=0, pin_memory=False
    )

    total_samples = sum(client_data_sizes)
    print(f"[Data] Total training samples: {total_samples} | "
          f"Min/Max per client: {min(client_data_sizes)}/{max(client_data_sizes)}")

    return client_train_loaders, client_val_loaders, global_test_loader, client_data_sizes
