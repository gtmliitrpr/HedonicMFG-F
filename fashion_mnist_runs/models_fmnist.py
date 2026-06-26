"""
models_fmnist.py — Simple CNN for FashionMNIST FL experiments.

Design philosophy — SIMPLE AND STABLE:
  - No BatchNorm: avoids FedBN having an unfair structural advantage
    and removes BN-related gradient noise during coalition formation
  - 3 conv layers with ReLU + MaxPool: enough capacity for FashionMNIST
    without overfitting small client datasets
  - Dropout for regularisation: prevents per-client overfitting
  - feature_dim=128: compact embedding space, fast personal head training
  - get_features(): penultimate layer → personalized head
  - forward_with_features(): for MOON contrastive loss

Architecture:
  Input (1×28×28)
  → Conv(1→32, 3×3) + ReLU + MaxPool(2×2)   → 32×13×13
  → Conv(32→64, 3×3) + ReLU + MaxPool(2×2)  → 64×5×5
  → Conv(64→128, 3×3) + ReLU                → 128×3×3
  → Flatten → FC(1152→256) + ReLU + Dropout
  → FC(256→128) + ReLU [= feature_dim]
  → FC(128→10) [global classifier]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FashionCNN(nn.Module):
    """
    Simple 3-layer CNN for FashionMNIST.
    No BatchNorm — clean gradients for coalition formation.
    """
    def __init__(self, num_classes: int = 10, feature_dim: int = 128,
                 dropout: float = 0.4):
        super().__init__()
        self.feature_dim = feature_dim

        # Conv block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)   # 28→14

        # Conv block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)   # 14→7

        # Conv block 3 — no pooling, extract richer features
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # 7×7 stays

        # Fully connected
        # After 3 convs + 2 pools: 128 × 7 × 7 = 6272
        self.fc1     = nn.Linear(128 * 7 * 7, 256)
        self.fc2     = nn.Linear(256, feature_dim)   # feature layer
        self.dropout = nn.Dropout(dropout)

        # Global classifier
        self.classifier = nn.Linear(feature_dim, num_classes)

        # Weight init
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 128-dim feature embedding."""
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)           # flatten
        x = self.dropout(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.get_features(x))

    def forward_with_features(self, x: torch.Tensor):
        """Return (logits, features) — used by MOON."""
        feat = self.get_features(x)
        return self.classifier(feat), feat


class PersonalHeadFMNIST(nn.Module):
    """
    Per-client personalized classification head.
    Two-layer MLP on top of FashionCNN features.
    Captures client-specific fashion preferences
    (e.g. a client with mostly footwear data).
    """
    def __init__(self, feature_dim: int = 128, num_classes: int = 10,
                 hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────
# Factories
# ──────────────────────────────────────────
def get_fmnist_model(config: dict) -> FashionCNN:
    return FashionCNN(num_classes=10, feature_dim=128, dropout=0.4)


def get_fmnist_personal_head(config: dict) -> PersonalHeadFMNIST:
    return PersonalHeadFMNIST(feature_dim=128, num_classes=10, hidden_dim=64)


def get_model_size(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
