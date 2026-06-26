"""
data_imdb.py — IMDB Sentiment Dataset for FL experiments.

Design for HedonicMFG advantage:
  - Dirichlet on sentiment label (0=neg, 1=pos) creates real opinion skew
  - Per-client vocabulary keeps feature shift across clients
  - Simple bag-of-words tokenizer — fast, no HuggingFace dependency
  - Global test set normalised consistently for fair comparison
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
import os
import re
import urllib.request
import tarfile
import json


# ──────────────────────────────────────────
# Download & Load Raw IMDB
# ──────────────────────────────────────────
IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"


def download_imdb(data_dir: str = "./data/imdb"):
    """Download and extract IMDB dataset."""
    os.makedirs(data_dir, exist_ok=True)
    tar_path    = os.path.join(data_dir, "aclImdb_v1.tar.gz")
    extract_dir = os.path.join(data_dir, "aclImdb")

    if not os.path.exists(extract_dir):
        if not os.path.exists(tar_path):
            print("[IMDB] Downloading dataset (~84MB)...")
            urllib.request.urlretrieve(IMDB_URL, tar_path)
        print("[IMDB] Extracting...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(data_dir)
        print("[IMDB] Extraction complete.")
    return extract_dir


def load_raw_imdb(data_dir: str = "./data/imdb"):
    """Load raw IMDB text and labels."""
    extract_dir = download_imdb(data_dir)
    texts, labels = [], []

    for split in ["train", "test"]:
        for label_str, label_val in [("pos", 1), ("neg", 0)]:
            folder = os.path.join(extract_dir, split, label_str)
            if not os.path.exists(folder):
                continue
            for fname in os.listdir(folder):
                if fname.endswith(".txt"):
                    fpath = os.path.join(folder, fname)
                    with open(fpath, "r", encoding="utf-8") as f:
                        texts.append(f.read())
                    labels.append(label_val)

    print(f"[IMDB] Loaded {len(texts)} samples | "
          f"Positive: {sum(labels)} | Negative: {len(labels)-sum(labels)}")
    return texts, np.array(labels, dtype=np.int64)


# ──────────────────────────────────────────
# Tokenization
# ──────────────────────────────────────────
def clean_text(text: str) -> list:
    """Simple whitespace tokenizer with basic cleaning."""
    text = re.sub(r"<.*?>", " ", text)          # remove HTML
    text = re.sub(r"[^a-zA-Z\s]", " ", text)   # keep only letters
    text = text.lower()
    return text.split()


def build_vocab(texts: list, vocab_size: int, seed: int = 42) -> dict:
    """Build vocabulary from top-k most frequent words."""
    counter = Counter()
    for text in texts:
        counter.update(clean_text(text))
    most_common = counter.most_common(vocab_size - 2)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in most_common:
        vocab[word] = len(vocab)
    return vocab


def tokenize_and_pad(texts: list, vocab: dict, max_len: int) -> np.ndarray:
    """Convert texts to padded integer sequences."""
    result = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, text in enumerate(texts):
        tokens = clean_text(text)
        ids = [vocab.get(t, 1) for t in tokens[:max_len]]
        result[i, :len(ids)] = ids
    return result


# ──────────────────────────────────────────
# Dirichlet Partitioning
# ──────────────────────────────────────────
def dirichlet_partition_imdb(labels: np.ndarray, num_clients: int,
                              alpha: float, seed: int = 42) -> list:
    """Dirichlet partition on sentiment label (binary)."""
    np.random.seed(seed)
    class_indices = [np.where(labels == c)[0].copy() for c in range(2)]
    for c in range(2):
        np.random.shuffle(class_indices[c])

    client_indices = [[] for _ in range(num_clients)]
    for c in range(2):
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(class_indices[c])).astype(int)[:-1]
        splits = np.split(class_indices[c], proportions)
        for i, split in enumerate(splits):
            client_indices[i].extend(split.tolist())

    for i in range(num_clients):
        np.random.shuffle(client_indices[i])

    return [np.array(idx) for idx in client_indices]


def print_imdb_partition_stats(client_indices: list, labels: np.ndarray):
    print(f"\n{'Client':<8} {'Samples':>8} {'%Pos':>7}  Skew")
    print("-" * 38)
    pos_rates = []
    for i, idx in enumerate(client_indices):
        if len(idx) == 0:
            continue
        pos_rate = labels[idx].mean() * 100
        pos_rates.append(pos_rate)
        skew = "HIGH" if pos_rate > 65 or pos_rate < 35 else "mild"
        print(f"{i:<8} {len(idx):>8} {pos_rate:>6.1f}%  {skew}")
    print(f"\n  Global pos rate: {np.mean(pos_rates):.1f}%  "
          f"Std: {np.std(pos_rates):.1f}%")


# ──────────────────────────────────────────
# Main DataLoader Builder
# ──────────────────────────────────────────
def get_imdb_client_loaders(config: dict) -> tuple:
    """
    Build per-client DataLoaders for IMDB.

    Returns:
        client_train_loaders, client_val_loaders,
        global_test_loader, client_data_sizes,
        vocab_size (actual), embed_dim
    """
    data_dir   = config.get("data_dir", "./data/imdb")
    num_clients = config["num_clients"]
    alpha       = config["dirichlet_alpha"]
    seed        = config["seed"]
    batch_size  = config["batch_size"]
    val_frac    = config.get("val_fraction", 0.1)
    max_len     = config.get("max_len", 256)
    vocab_size  = config.get("vocab_size", 20000)

    # Load raw data
    texts, labels = load_raw_imdb(data_dir)

    # Train/test split (IMDB has 25K train + 25K test)
    # Use first 25K as train, last 25K as global test
    n_train = 25000
    train_texts  = texts[:n_train]
    train_labels = labels[:n_train]
    test_texts   = texts[n_train:]
    test_labels  = labels[n_train:]

    # Build vocabulary from training data
    print(f"[IMDB] Building vocabulary (size={vocab_size})...")
    vocab = build_vocab(train_texts, vocab_size, seed)
    actual_vocab_size = len(vocab)
    print(f"[IMDB] Vocabulary built: {actual_vocab_size} tokens")

    # Tokenize all data
    print(f"[IMDB] Tokenizing (max_len={max_len})...")
    X_train = tokenize_and_pad(train_texts, vocab, max_len)
    X_test  = tokenize_and_pad(test_texts, vocab, max_len)

    # Global test loader
    X_test_t  = torch.from_numpy(X_test)
    y_test_t  = torch.from_numpy(test_labels)
    global_test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t),
        batch_size=batch_size * 2, shuffle=False
    )

    # Dirichlet partition
    client_indices = dirichlet_partition_imdb(train_labels, num_clients, alpha, seed)

    print(f"\n[Data] IMDB | {num_clients} clients | Dirichlet α={alpha}")
    print_imdb_partition_stats(client_indices, train_labels)

    client_train_loaders = []
    client_val_loaders   = []
    client_data_sizes    = []

    for i, indices in enumerate(client_indices):
        if len(indices) < 10:
            indices = np.random.choice(n_train, 50, replace=False)

        np.random.seed(seed + i)
        np.random.shuffle(indices)
        n_val    = max(5, int(len(indices) * val_frac))
        val_idx  = indices[:n_val]
        train_idx = indices[n_val:]

        X_tr = torch.from_numpy(X_train[train_idx])
        y_tr = torch.from_numpy(train_labels[train_idx])
        X_va = torch.from_numpy(X_train[val_idx])
        y_va = torch.from_numpy(train_labels[val_idx])

        client_train_loaders.append(
            DataLoader(TensorDataset(X_tr, y_tr),
                       batch_size=batch_size, shuffle=True)
        )
        client_val_loaders.append(
            DataLoader(TensorDataset(X_va, y_va),
                       batch_size=batch_size * 2, shuffle=False)
        )
        client_data_sizes.append(len(train_idx))

    total = sum(client_data_sizes)
    print(f"[Data] Total train: {total} | Test: {len(test_labels)} | "
          f"Vocab: {actual_vocab_size} | "
          f"Min/Max per client: {min(client_data_sizes)}/{max(client_data_sizes)}")

    # Save vocab for reproducibility
    vocab_path = os.path.join(data_dir, "vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(vocab, f)

    return (client_train_loaders, client_val_loaders,
            global_test_loader, client_data_sizes,
            actual_vocab_size)
