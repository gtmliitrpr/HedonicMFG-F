"""
models_adult.py — MLP architectures for Adult Census FL experiments.

AdultMLP       : Deep MLP backbone with BN + Dropout, get_features() support
PersonalHeadAdult : Per-client classification head (2-layer)

Design choices for HedonicMFG advantage:
  - BatchNorm after every hidden layer → FedBN can exploit this,
    HedonicMFG + personal head exploits it even more
  - Deep enough (3 hidden layers) so gradient similarity is meaningful
  - get_features() returns penultimate layer → personal head fine-tunes on top
  - forward_with_features() for MOON contrastive loss compatibility
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class AdultMLP(nn.Module):
    """
    3-layer MLP for Adult Census binary classification.
    Architecture: input → 256 → 128 → 64 → feature(64) → output(2)
    Each hidden layer has BN + Dropout for regularisation and FedBN compatibility.
    """
    def __init__(self, input_dim: int, num_classes: int = 2,
                 feature_dim: int = 64):
        super().__init__()
        self.feature_dim = feature_dim

        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(128, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        # Global classification head
        self.classifier = nn.Linear(feature_dim, num_classes)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return feature embedding (before classifier)."""
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.get_features(x))

    def forward_with_features(self, x: torch.Tensor):
        """Return (logits, features) — used by MOON."""
        feat = self.get_features(x)
        return self.classifier(feat), feat


class PersonalHeadAdult(nn.Module):
    """
    Two-layer personalized classification head for Adult Census.
    More expressive than MNIST's linear head — tabular needs it.
    Receives 64-dim feature from AdultMLP backbone.
    """
    def __init__(self, feature_dim: int = 64, num_classes: int = 2,
                 hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────
# Factories
# ──────────────────────────────────────────
def get_adult_model(config: dict, feature_dim: int) -> AdultMLP:
    return AdultMLP(input_dim=feature_dim, num_classes=2, feature_dim=64)


def get_adult_personal_head(config: dict) -> PersonalHeadAdult:
    return PersonalHeadAdult(feature_dim=64, num_classes=2, hidden_dim=32)


def get_model_size(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ──────────────────────────────────────────
# FedBN helpers — non-BN params only
# ──────────────────────────────────────────
def aggregate_except_bn_adult(models: list, weights: list) -> dict:
    """Aggregate all params except BatchNorm (FedBN style)."""
    total = sum(weights)
    norm_w = [w / total for w in weights]
    agg_state = {}
    ref_state = models[0].state_dict()
    for key in ref_state:
        if "bn" in key or "BatchNorm" in key or "norm" in key.lower():
            continue
        # Check if it's a BN-related key by checking for weight/bias in BN layers
        # BN keys look like: layer1.1.weight, layer1.1.bias, layer1.1.running_mean, etc.
        # We skip index 1 (BN) inside sequential layers
        parts = key.split(".")
        if len(parts) >= 2 and parts[1] == "1":
            continue  # index 1 in Sequential = BatchNorm1d
        agg_state[key] = sum(
            w * m.state_dict()[key].float()
            for w, m in zip(norm_w, models)
        )
    return agg_state
