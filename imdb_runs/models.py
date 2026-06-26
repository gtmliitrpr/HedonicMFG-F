"""
models.py — Model architectures for MNIST FL experiments.

MNISTNet    : CNN backbone with get_features() support for personalized heads
PersonalHead: Per-client classification head (linear layer)
MNISTNetFull: Backbone + head combined (used by baselines without personalization)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


# ──────────────────────────────────────────
# MNIST CNN Backbone
# ──────────────────────────────────────────
class MNISTNet(nn.Module):
    """
    CNN backbone for MNIST.
    Two conv layers → two FC layers → feature vector of size 128.
    Supports get_features() for personalized head experiments.
    """
    def __init__(self, num_classes: int = 10, feature_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(0.25)

        # After two 2×2 max pools on 28×28: 64 × 7 × 7
        self.fc1   = nn.Linear(64 * 7 * 7, feature_dim)
        self.drop2 = nn.Dropout(0.5)

        # Global classification head (used when no personalized head)
        self.fc2   = nn.Linear(feature_dim, num_classes)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return feature embedding before final classification layer."""
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.drop1(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.get_features(x)
        return self.fc2(feat)

    def forward_with_features(self, x: torch.Tensor):
        """Return (logits, features) tuple — used by MOON."""
        feat = self.get_features(x)
        return self.fc2(feat), feat


# ──────────────────────────────────────────
# Personalized Head
# ──────────────────────────────────────────
class PersonalHead(nn.Module):
    """
    Lightweight per-client classification head.
    Receives feature_dim-dimensional embeddings from MNISTNet backbone.
    """
    def __init__(self, feature_dim: int = 128, num_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# ──────────────────────────────────────────
# Model factory
# ──────────────────────────────────────────
def get_model(config: dict) -> MNISTNet:
    return MNISTNet(num_classes=10, feature_dim=128)


def get_personal_head(config: dict) -> PersonalHead:
    return PersonalHead(feature_dim=128, num_classes=10)


def get_model_size(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ──────────────────────────────────────────
# FedBN helper — per-client BN layers
# ──────────────────────────────────────────
def get_non_bn_params(model: nn.Module) -> list:
    """Return parameter names that are NOT BatchNorm parameters."""
    non_bn = []
    for name, _ in model.named_parameters():
        if "bn" not in name:
            non_bn.append(name)
    return non_bn


def aggregate_except_bn(models: list, weights: list) -> dict:
    """
    FedBN-style aggregation: aggregate all params EXCEPT BatchNorm.
    Returns state_dict of aggregated params (non-BN only).
    """
    total = sum(weights)
    norm_w = [w / total for w in weights]

    agg_state = {}
    ref_state = models[0].state_dict()

    for key in ref_state:
        if "bn" in key:
            continue  # skip BN params — kept local
        agg_state[key] = torch.zeros_like(ref_state[key])
        for model, w in zip(models, norm_w):
            agg_state[key] += w * model.state_dict()[key].float()

    return agg_state
