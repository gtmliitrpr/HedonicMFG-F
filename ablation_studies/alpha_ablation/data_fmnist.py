"""
data_fmnist.py — FashionMNIST dataset for FL experiments.

FashionMNIST classes:
  0: T-shirt/top  1: Trouser    2: Pullover   3: Dress     4: Coat
  5: Sandal       6: Shirt      7: Sneaker    8: Bag       9: Ankle boot

Dirichlet α=0.3 creates real skew — some clients see only
clothing, others only footwear → gradient similarity is
a strong coalition formation signal.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ──────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────
FMNIST_TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,)),
])

FMNIST_TEST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,)),
])

CLASS_NAMES = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


# ──────────────────────────────────────────
# Dirichlet Partitioning
# ──────────────────────────────────────────
def dirichlet_partition_fmnist(targets: np.ndarray, num_clients: int,
                                alpha: float, seed: int = 42) -> list:
    """Standard Dirichlet partition on class label."""
    np.random.seed(seed)
    num_classes = 10
    class_indices = [np.where(targets == c)[0].copy() for c in range(num_classes)]
    for c in range(num_classes):
        np.random.shuffle(class_indices[c])

    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(class_indices[c])).astype(int)[:-1]
        splits = np.split(class_indices[c], proportions)
        for i, split in enumerate(splits):
            client_indices[i].extend(split.tolist())

    for i in range(num_clients):
        np.random.shuffle(client_indices[i])

    return [np.array(idx) for idx in client_indices]


def print_fmnist_partition_stats(client_indices: list, targets: np.ndarray):
    """Print per-client class distribution."""
    print(f"\n{'Client':<8} {'Samples':>8} {'Classes':>8}  Dominant class")
    print("-" * 55)
    for i, idx in enumerate(client_indices):
        if len(idx) == 0:
            continue
        labels = targets[idx]
        dist   = np.bincount(labels, minlength=10)
        dom    = CLASS_NAMES[dist.argmax()]
        n_cls  = (dist > 0).sum()
        print(f"{i:<8} {len(idx):>8} {n_cls:>8}  {dom}")


# ──────────────────────────────────────────
# Main DataLoader Builder
# ──────────────────────────────────────────
def get_fmnist_client_loaders(config: dict) -> tuple:
    """
    Build per-client DataLoaders for FashionMNIST.

    Returns:
        client_train_loaders, client_val_loaders,
        global_test_loader, client_data_sizes
    """
    data_dir    = config.get("data_dir", "./data")
    num_clients = config["num_clients"]
    alpha       = config["dirichlet_alpha"]
    seed        = config["seed"]
    batch_size  = config["batch_size"]
    val_frac    = config.get("val_fraction", 0.1)
    num_workers = config.get("num_workers", 4)

    # Load datasets
    train_dataset = datasets.FashionMNIST(
        root=data_dir, train=True, download=True,
        transform=FMNIST_TRAIN_TRANSFORM)
    test_dataset = datasets.FashionMNIST(
        root=data_dir, train=False, download=True,
        transform=FMNIST_TEST_TRANSFORM)

    targets = np.array(train_dataset.targets)

    # Partition
    client_indices = dirichlet_partition_fmnist(targets, num_clients, alpha, seed)

    print(f"\n[Data] FashionMNIST | {num_clients} clients | Dirichlet alpha={alpha}")
    print_fmnist_partition_stats(client_indices, targets)

    client_train_loaders = []
    client_val_loaders   = []
    client_data_sizes    = []

    # Minimum samples needed: at least 1 val + batch_size train
    min_samples = batch_size + 1

    for i, indices in enumerate(client_indices):
        # Boost tiny/empty clients by sampling from global pool
        if len(indices) < min_samples:
            np.random.seed(seed + i + 1000)
            all_idx = np.arange(len(targets))
            extra   = np.random.choice(all_idx, size=min_samples, replace=False)
            indices = np.union1d(indices, extra)
            print(f"  [Data] Client {i} boosted: {len(indices)} samples "
                  f"(was too small for alpha={alpha})")

        np.random.seed(seed + i)
        np.random.shuffle(indices)
        n_val   = max(1, int(len(indices) * val_frac))
        n_train = len(indices) - n_val
        # Guard: ensure train set has at least batch_size samples
        if n_train < batch_size:
            n_val   = max(1, len(indices) - batch_size)
        val_idx   = indices[:n_val]
        train_idx = indices[n_val:]

        client_train_loaders.append(DataLoader(
            Subset(train_dataset, train_idx),
            batch_size=min(batch_size, len(train_idx)), shuffle=True,
            num_workers=num_workers, pin_memory=True))
        client_val_loaders.append(DataLoader(
            Subset(train_dataset, val_idx),
            batch_size=min(batch_size * 2, max(1, len(val_idx))), shuffle=False,
            num_workers=num_workers, pin_memory=True))
        client_data_sizes.append(len(train_idx))

    global_test_loader = DataLoader(
        test_dataset, batch_size=512, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    total = sum(client_data_sizes)
    print(f"[Data] Total train: {total} | Test: {len(test_dataset)} | "
          f"Min/Max per client: {min(client_data_sizes)}/{max(client_data_sizes)}")

    return client_train_loaders, client_val_loaders, global_test_loader, client_data_sizes
