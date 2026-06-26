"""
models.py — Model architectures for MNIST FL experiments.

MNISTNet    : Stronger CNN backbone with get_features() support for personalized heads
PersonalHead: Per-client classification head (MLP head for better personalization)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


# ──────────────────────────────────────────
# MNIST CNN Backbone  (improved)
# ──────────────────────────────────────────
class MNISTNet(nn.Module):
    """
    Stronger CNN backbone for MNIST.
    Three conv layers → two FC layers → feature vector of size 256.
    Supports get_features() for personalized head experiments.
    """
    def __init__(self, num_classes: int = 10, feature_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim

        # Conv block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)

        # Conv block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        # Conv block 3 — extra depth for richer features
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        self.pool  = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(0.25)

        # After pool1: 32 × 14 × 14
        # After pool2: 64 × 7  × 7
        # After pool3: 128 × 3 × 3  (floor(7/2)=3)
        self.fc1   = nn.Linear(128 * 3 * 3, feature_dim)
        self.bn_fc = nn.BatchNorm1d(feature_dim)
        self.drop2 = nn.Dropout(0.4)

        # Global classification head (used when no personalized head)
        self.fc2   = nn.Linear(feature_dim, num_classes)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return feature embedding before final classification layer."""
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # → 32×14×14
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # → 64×7×7
        x = self.pool(F.relu(self.bn3(self.conv3(x))))   # → 128×3×3
        x = self.drop1(x)
        x = x.view(x.size(0), -1)                        # → 128*3*3=1152
        x = F.relu(self.bn_fc(self.fc1(x)))
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
# Personalized Head (improved MLP)
# ──────────────────────────────────────────
class PersonalHead(nn.Module):
    """
    Lightweight MLP per-client classification head.
    Two-layer MLP for better personalization than a single linear layer.
    """
    def __init__(self, feature_dim: int = 256, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────
# Model factory
# ──────────────────────────────────────────
def get_model(config: dict) -> MNISTNet:
    return MNISTNet(num_classes=10, feature_dim=256)


def get_personal_head(config: dict) -> PersonalHead:
    return PersonalHead(feature_dim=256, num_classes=10)


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
