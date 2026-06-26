"""
utils.py — Shared utilities: seeding, metrics, FedAvg aggregation, logging.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import copy
import os
import json
from collections import defaultdict


# ──────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(config: dict) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ──────────────────────────────────────────
# Model helpers
# ──────────────────────────────────────────
def get_model_params(model: nn.Module) -> list:
    """Return list of parameter tensors (detached copies)."""
    return [p.data.clone() for p in model.parameters()]


def set_model_params(model: nn.Module, params: list):
    """Set model parameters from a list of tensors."""
    for p, new_p in zip(model.parameters(), params):
        p.data.copy_(new_p)


def clone_model(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model)


def zero_model_params(model: nn.Module):
    for p in model.parameters():
        p.data.zero_()


# ──────────────────────────────────────────
# Federated Aggregation
# ──────────────────────────────────────────
def fedavg_aggregate(models: list, weights: list) -> list:
    """
    Weighted FedAvg aggregation.
    models  : list of nn.Module
    weights : list of floats (should sum to 1)
    Returns : list of aggregated parameter tensors
    """
    assert len(models) == len(weights), "models and weights must match"
    total = sum(weights)
    norm_weights = [w / total for w in weights]

    agg_params = [torch.zeros_like(p) for p in models[0].parameters()]
    for model, w in zip(models, norm_weights):
        for agg_p, p in zip(agg_params, model.parameters()):
            agg_p.data += w * p.data.clone()
    return agg_params


def fedavg_aggregate_params(param_list: list, weights: list) -> list:
    """
    Aggregate raw parameter lists (not models).
    param_list : list of list-of-tensors
    weights    : list of floats
    """
    total = sum(weights)
    norm_w = [w / total for w in weights]
    agg = [torch.zeros_like(p) for p in param_list[0]]
    for params, w in zip(param_list, norm_w):
        for a, p in zip(agg, params):
            a.data += w * p.data.clone()
    return agg


# ──────────────────────────────────────────
# Gradient utilities
# ──────────────────────────────────────────
def get_gradients(model: nn.Module) -> torch.Tensor:
    """Flatten all gradients into a single vector."""
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.data.clone().flatten())
    if not grads:
        return torch.zeros(1)
    return torch.cat(grads)


def cosine_similarity(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """Cosine similarity between two flat vectors."""
    if v1.norm() == 0 or v2.norm() == 0:
        return 0.0
    return torch.dot(v1, v2) / (v1.norm() * v2.norm() + 1e-10)


def pairwise_cosine_similarity(grad_vectors: list) -> np.ndarray:
    """
    Compute N×N pairwise cosine similarity matrix.
    grad_vectors : list of flat torch.Tensor
    """
    N = len(grad_vectors)
    sim_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            sim_matrix[i, j] = cosine_similarity(grad_vectors[i], grad_vectors[j]).item()
    return sim_matrix


def get_model_update(global_params: list, local_params: list) -> list:
    """Compute local update = local_params - global_params."""
    return [lp - gp for lp, gp in zip(local_params, global_params)]


# ──────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────
def evaluate_model(model: nn.Module, dataloader, device: torch.device) -> float:
    """Return accuracy on a dataloader."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            # Handle tuple output (model + head)
            if isinstance(out, tuple):
                out = out[0]
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


def evaluate_with_head(backbone: nn.Module, head: nn.Module,
                       dataloader, device: torch.device) -> float:
    """Evaluate backbone + personalized head combo."""
    backbone.eval()
    head.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            feat = backbone.get_features(x)
            out = head(feat)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


def compute_client_accuracies(model: nn.Module, client_loaders: list,
                               device: torch.device) -> list:
    return [evaluate_model(model, loader, device) for loader in client_loaders]


# ──────────────────────────────────────────
# Results tracking
# ──────────────────────────────────────────
class ResultsTracker:
    def __init__(self, algorithm_name: str):
        self.name = algorithm_name
        self.global_accs = []       # global model on test set each round
        self.avg_client_accs = []   # mean client accuracy each round
        self.round_times = []
        self.metadata = {}

    def log(self, rnd: int, global_acc: float, client_accs: list, elapsed: float = 0.0):
        self.global_accs.append(global_acc)
        self.avg_client_accs.append(np.mean(client_accs) if client_accs else 0.0)
        self.round_times.append(elapsed)

    def final_summary(self) -> dict:
        if not self.global_accs:
            return {}
        return {
            "algorithm": self.name,
            "best_global_acc": max(self.global_accs),
            "final_global_acc": self.global_accs[-1],
            "best_avg_client_acc": max(self.avg_client_accs),
            "final_avg_client_acc": self.avg_client_accs[-1],
            "avg_round_time": np.mean(self.round_times),
        }

    def to_dict(self) -> dict:
        return {
            "algorithm": self.name,
            "global_accs": self.global_accs,
            "avg_client_accs": self.avg_client_accs,
            "round_times": self.round_times,
            "metadata": self.metadata,
        }


def save_results(results: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    serializable = {}
    for k, v in results.items():
        if isinstance(v, ResultsTracker):
            serializable[k] = v.to_dict()
        else:
            serializable[k] = v
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Results saved to {path}")


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ──────────────────────────────────────────
# Printing
# ──────────────────────────────────────────
def print_round_summary(algorithm: str, rnd: int, total: int,
                         global_acc: float, avg_client_acc: float):
    bar_len = 20
    filled = int(bar_len * rnd / total)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"[{algorithm:>15}] Round {rnd:>3}/{total} [{bar}] "
          f"Global: {global_acc*100:5.2f}%  Client: {avg_client_acc*100:5.2f}%")


def print_final_table(all_results: dict):
    print("\n" + "="*72)
    print(f"{'Algorithm':<20} {'Best Global':>12} {'Final Global':>13} "
          f"{'Best Client':>12} {'Final Client':>13}")
    print("="*72)
    for name, tracker in all_results.items():
        s = tracker.final_summary()
        print(f"{s['algorithm']:<20} "
              f"{s['best_global_acc']*100:>11.2f}%  "
              f"{s['final_global_acc']*100:>12.2f}%  "
              f"{s['best_avg_client_acc']*100:>11.2f}%  "
              f"{s['final_avg_client_acc']*100:>12.2f}%")
    print("="*72)
