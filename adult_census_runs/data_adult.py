"""
data_adult.py — Adult Census Income dataset for FL experiments.

Key design choices for HedonicMFG advantage:
  - Dirichlet on income label (binary: <=50K / >50K) creates real client skew
  - Feature normalisation per-client (not global) → feature shift across clients
  - Categorical encoding preserved to maintain inter-client heterogeneity
  - Stratified train/val split per client to avoid empty val sets

Dataset: UCI Adult Census Income
  - 48,842 samples, 14 features, binary classification (income >50K)
  - Features: age, workclass, education, marital-status, occupation,
              relationship, race, sex, capital-gain, capital-loss,
              hours-per-week, native-country
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os
import urllib.request


# ──────────────────────────────────────────
# Download & Load
# ──────────────────────────────────────────
ADULT_URL_TRAIN = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
ADULT_URL_TEST  = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

CATEGORICAL_COLS = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country"
]

NUMERICAL_COLS = [
    "age", "fnlwgt", "education-num", "capital-gain",
    "capital-loss", "hours-per-week"
]


def download_adult(data_dir: str = "./data/adult"):
    """Download Adult Census data if not already present."""
    os.makedirs(data_dir, exist_ok=True)
    train_path = os.path.join(data_dir, "adult.data")
    test_path  = os.path.join(data_dir, "adult.test")

    if not os.path.exists(train_path):
        print("[Adult] Downloading adult.data ...")
        urllib.request.urlretrieve(ADULT_URL_TRAIN, train_path)

    if not os.path.exists(test_path):
        print("[Adult] Downloading adult.test ...")
        urllib.request.urlretrieve(ADULT_URL_TEST, test_path)

    return train_path, test_path


def load_and_preprocess_adult(data_dir: str = "./data/adult"):
    """
    Load, clean, and encode Adult Census dataset.
    Returns: X (np.ndarray), y (np.ndarray), feature_dim (int)
    """
    train_path, test_path = download_adult(data_dir)

    # Load
    df_train = pd.read_csv(train_path, header=None, names=COLUMNS,
                            na_values=" ?", skipinitialspace=True)
    df_test  = pd.read_csv(test_path,  header=None, names=COLUMNS,
                            na_values=" ?", skipinitialspace=True,
                            skiprows=1)  # test file has a header row

    # Clean income label in test set (has trailing period)
    df_test["income"] = df_test["income"].str.replace(".", "", regex=False)

    # Combine for consistent encoding
    df = pd.concat([df_train, df_test], ignore_index=True)

    # Drop fnlwgt (census weight — not a feature)
    df = df.drop(columns=["fnlwgt"])

    # Drop rows with missing values
    df = df.dropna()

    # Target: binary income label
    df["income"] = (df["income"].str.strip() == ">50K").astype(int)

    # Encode categoricals
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # Features and labels
    feature_cols = [c for c in df.columns if c != "income"]
    X = df[feature_cols].values.astype(np.float32)
    y = df["income"].values.astype(np.int64)

    print(f"[Adult] Loaded {len(X)} samples | "
          f"Features: {X.shape[1]} | "
          f"Class balance: {y.mean()*100:.1f}% positive")

    return X, y, X.shape[1]


# ──────────────────────────────────────────
# Dirichlet Partitioning (label-based)
# ──────────────────────────────────────────
def dirichlet_partition_adult(y: np.ndarray, num_clients: int,
                               alpha: float, seed: int = 42) -> list:
    """
    Partition Adult Census indices using Dirichlet on income label.
    Binary label (0/1) — creates realistic income-skewed clients.
    """
    np.random.seed(seed)
    num_classes = 2
    class_indices = [np.where(y == c)[0].copy() for c in range(num_classes)]

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


def print_adult_partition_stats(client_indices: list, y: np.ndarray):
    """Print per-client label distribution."""
    print(f"\n{'Client':<8} {'Samples':>8} {'%>50K':>7}  Skew")
    print("-" * 40)
    pos_rates = []
    for i, idx in enumerate(client_indices):
        if len(idx) == 0:
            continue
        pos_rate = y[idx].mean() * 100
        pos_rates.append(pos_rate)
        skew = "HIGH" if pos_rate > 40 or pos_rate < 15 else "mild"
        print(f"{i:<8} {len(idx):>8} {pos_rate:>6.1f}%  {skew}")
    print(f"\n  Global >50K rate: {np.mean(pos_rates):.1f}%  "
          f"Std across clients: {np.std(pos_rates):.1f}%")


# ──────────────────────────────────────────
# Client DataLoaders
# ──────────────────────────────────────────
def get_adult_client_loaders(config: dict) -> tuple:
    """
    Build per-client DataLoaders for Adult Census with Dirichlet partitioning.

    Key: each client normalises features with their OWN scaler →
    creates feature-level heterogeneity that FedBN/HedonicMFG must handle.

    Returns:
        client_train_loaders : list of DataLoader
        client_val_loaders   : list of DataLoader
        global_test_loader   : DataLoader (globally normalised)
        client_data_sizes    : list of ints
        feature_dim          : int
    """
    data_dir = config.get("data_dir", "./data/adult")
    X, y, feature_dim = load_and_preprocess_adult(data_dir)

    num_clients  = config["num_clients"]
    alpha        = config["dirichlet_alpha"]
    seed         = config["seed"]
    batch_size   = config["batch_size"]
    val_frac     = config.get("val_fraction", 0.15)

    # Hold out 15% as global test set (before partitioning)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, random_state=seed, stratify=y
    )

    # Global test set — normalise with global scaler
    global_scaler = StandardScaler()
    global_scaler.fit(X_trainval)
    X_test_norm = global_scaler.transform(X_test).astype(np.float32)

    X_test_t = torch.from_numpy(X_test_norm)
    y_test_t = torch.from_numpy(y_test)
    global_test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t),
        batch_size=512, shuffle=False
    )

    # Dirichlet partition on trainval
    client_indices = dirichlet_partition_adult(y_trainval, num_clients, alpha, seed)

    print(f"\n[Data] Adult Census | {num_clients} clients | Dirichlet α={alpha}")
    print_adult_partition_stats(client_indices, y_trainval)

    client_train_loaders = []
    client_val_loaders   = []
    client_data_sizes    = []

    for i, indices in enumerate(client_indices):
        if len(indices) < 10:
            # Too few samples — give client a tiny subset of global data
            indices = np.random.choice(len(y_trainval), 50, replace=False)

        # Per-client train/val split (stratified if possible)
        client_y = y_trainval[indices]
        unique_classes = np.unique(client_y)

        try:
            if len(unique_classes) > 1:
                tr_idx, va_idx = train_test_split(
                    indices, test_size=val_frac,
                    random_state=seed + i,
                    stratify=client_y
                )
            else:
                n_val = max(5, int(len(indices) * val_frac))
                tr_idx = indices[n_val:]
                va_idx = indices[:n_val]
        except Exception:
            n_val = max(5, int(len(indices) * val_frac))
            tr_idx = indices[n_val:]
            va_idx = indices[:n_val]

        # Per-client feature normalisation — this creates feature shift!
        client_scaler = StandardScaler()
        X_tr = client_scaler.fit_transform(X_trainval[tr_idx]).astype(np.float32)
        X_va = client_scaler.transform(X_trainval[va_idx]).astype(np.float32)

        y_tr = y_trainval[tr_idx]
        y_va = y_trainval[va_idx]

        tr_ds = TensorDataset(torch.from_numpy(X_tr),
                               torch.from_numpy(y_tr))
        va_ds = TensorDataset(torch.from_numpy(X_va),
                               torch.from_numpy(y_va))

        client_train_loaders.append(
            DataLoader(tr_ds, batch_size=batch_size,
                       shuffle=True, drop_last=False)
        )
        client_val_loaders.append(
            DataLoader(va_ds, batch_size=batch_size * 2,
                       shuffle=False)
        )
        client_data_sizes.append(len(tr_idx))

    total = sum(client_data_sizes)
    print(f"[Data] Total train samples: {total} | "
          f"Test: {len(y_test)} | "
          f"Feature dim: {feature_dim} | "
          f"Min/Max per client: {min(client_data_sizes)}/{max(client_data_sizes)}")

    return (client_train_loaders, client_val_loaders,
            global_test_loader, client_data_sizes, feature_dim)
