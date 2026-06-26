"""
data.py — Dataset loading for MNIST and CIFAR-10 with Dirichlet non-IID partitioning.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ──────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────
MNIST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

CIFAR10_TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

CIFAR10_TEST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])


# ──────────────────────────────────────────
# Dirichlet partitioning (shared)
# ──────────────────────────────────────────
def dirichlet_partition(targets: np.ndarray, num_clients: int,
                         alpha: float, seed: int = 42) -> list:
    np.random.seed(seed)
    num_classes  = int(targets.max()) + 1
    class_indices = [np.where(targets == c)[0] for c in range(num_classes)]
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        np.random.shuffle(class_indices[c])
        props = np.random.dirichlet(np.repeat(alpha, num_clients))
        cuts  = (np.cumsum(props) * len(class_indices[c])).astype(int)[:-1]
        for i, split in enumerate(np.split(class_indices[c], cuts)):
            client_indices[i].extend(split.tolist())

    for i in range(num_clients):
        np.random.shuffle(client_indices[i])
    return [np.array(idx) for idx in client_indices]


def print_partition_stats(client_indices: list, targets: np.ndarray,
                           num_classes: int):
    print(f"\n{'Client':<8} {'Samples':>8} {'Classes':>8}  Distribution")
    print("-" * 60)
    for i, idx in enumerate(client_indices):
        if len(idx) == 0:
            print(f"{i:<8} {'0':>8}")
            continue
        labels  = targets[idx]
        dist    = np.bincount(labels, minlength=num_classes)
        present = (dist > 0).sum()
        dist_str = " ".join(f"{d:3d}" for d in dist)
        print(f"{i:<8} {len(idx):>8} {present:>8}  [{dist_str}]")


def _build_loaders(train_dataset, client_indices, batch_size,
                   val_frac, seed, pin_memory=True):
    """Build per-client train/val loaders from a partitioned dataset."""
    train_loaders, val_loaders, data_sizes = [], [], []

    for i, indices in enumerate(client_indices):
        if len(indices) == 0:
            train_loaders.append(None)
            val_loaders.append(None)
            data_sizes.append(0)
            continue

        np.random.seed(seed + i)
        np.random.shuffle(indices)
        n_val     = max(1, int(len(indices) * val_frac))
        val_idx   = indices[:n_val]
        train_idx = indices[n_val:]

        train_loaders.append(DataLoader(
            Subset(train_dataset, train_idx),
            batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=pin_memory
        ))
        val_loaders.append(DataLoader(
            Subset(train_dataset, val_idx),
            batch_size=batch_size * 2, shuffle=False,
            num_workers=0, pin_memory=pin_memory
        ))
        data_sizes.append(len(train_idx))

    return train_loaders, val_loaders, data_sizes


# ──────────────────────────────────────────
# MNIST
# ──────────────────────────────────────────
def get_mnist_client_loaders(config: dict) -> tuple:
    data_dir    = config.get("data_dir", "./data")
    num_clients = config["num_clients"]
    alpha       = config["dirichlet_alpha"]
    seed        = config["seed"]
    batch_size  = config["batch_size"]
    val_frac    = config.get("val_fraction", 0.1)

    train_ds = datasets.MNIST(data_dir, train=True,  download=True,
                               transform=MNIST_TRANSFORM)
    test_ds  = datasets.MNIST(data_dir, train=False, download=True,
                               transform=MNIST_TRANSFORM)

    targets = np.array(train_ds.targets)
    client_indices = dirichlet_partition(targets, num_clients, alpha, seed)

    print(f"\n[Data] MNIST | {num_clients} clients | Dirichlet α={alpha}")
    print_partition_stats(client_indices, targets, 10)

    train_loaders, val_loaders, data_sizes = _build_loaders(
        train_ds, client_indices, batch_size, val_frac, seed)

    global_test = DataLoader(test_ds, batch_size=256, shuffle=False,
                              num_workers=0, pin_memory=False)

    print(f"[Data] Total training samples: {sum(data_sizes)} | "
          f"Min/Max per client: {min(data_sizes)}/{max(data_sizes)}")
    return train_loaders, val_loaders, global_test, data_sizes


# ──────────────────────────────────────────
# CIFAR-10
# ──────────────────────────────────────────
def get_cifar10_client_loaders(config: dict) -> tuple:
    """
    Build per-client CIFAR-10 DataLoaders with Dirichlet partitioning.
    Training data uses augmentation (random crop + flip + colour jitter).
    Validation + test use clean normalisation only.
    """
    data_dir    = config.get("data_dir", "./data")
    num_clients = config["num_clients"]
    alpha       = config["dirichlet_alpha"]
    seed        = config["seed"]
    batch_size  = config["batch_size"]
    val_frac    = config.get("val_fraction", 0.1)

    # Train set with augmentation, val subset uses same dataset object
    # (augmentation is fine for val since it's sampled from train indices)
    train_ds_aug  = datasets.CIFAR10(data_dir, train=True, download=True,
                                      transform=CIFAR10_TRAIN_TRANSFORM)
    train_ds_clean = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=CIFAR10_TEST_TRANSFORM)
    test_ds  = datasets.CIFAR10(data_dir, train=False, download=True,
                                 transform=CIFAR10_TEST_TRANSFORM)

    targets = np.array(train_ds_aug.targets)
    client_indices = dirichlet_partition(targets, num_clients, alpha, seed)

    print(f"\n[Data] CIFAR-10 | {num_clients} clients | Dirichlet α={alpha}")
    print_partition_stats(client_indices, targets, 10)

    train_loaders, val_loaders, data_sizes = [], [], []

    for i, indices in enumerate(client_indices):
        if len(indices) == 0:
            train_loaders.append(None)
            val_loaders.append(None)
            data_sizes.append(0)
            continue

        np.random.seed(seed + i)
        np.random.shuffle(indices)
        n_val     = max(1, int(len(indices) * val_frac))
        val_idx   = indices[:n_val]
        train_idx = indices[n_val:]

        # Training: augmented dataset
        train_loaders.append(DataLoader(
            Subset(train_ds_aug, train_idx),
            batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=True
        ))
        # Validation: clean (no random augmentation for fair eval)
        val_loaders.append(DataLoader(
            Subset(train_ds_clean, val_idx),
            batch_size=batch_size * 2, shuffle=False,
            num_workers=0, pin_memory=True
        ))
        data_sizes.append(len(train_idx))

    global_test = DataLoader(test_ds, batch_size=256, shuffle=False,
                              num_workers=0, pin_memory=False)

    total = sum(data_sizes)
    print(f"[Data] Total training samples: {total} | "
          f"Min/Max per client: {min(data_sizes)}/{max(data_sizes)}")
    return train_loaders, val_loaders, global_test, data_sizes


# ──────────────────────────────────────────
# Unified loader dispatcher
# ──────────────────────────────────────────
def get_client_loaders(config: dict) -> tuple:
    ds = config.get("dataset", "mnist")
    if ds == "cifar10":
        return get_cifar10_client_loaders(config)
    return get_mnist_client_loaders(config)
