"""
models.py — Model architectures for MNIST and CIFAR-10 FL experiments.

For MNIST  : MNISTNet  — 3-block CNN, 256-dim features
For CIFAR-10: CIFARNet — ResNet-style with residual blocks, 512-dim features

Both expose:
  get_features(x)             → feature embedding
  forward(x)                  → logits
  forward_with_features(x)    → (logits, features)   [for MOON]

PersonalHead : 2-layer MLP per-client head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


# ══════════════════════════════════════════════════════
# MNIST backbone  (unchanged from improved MNIST version)
# ══════════════════════════════════════════════════════
class MNISTNet(nn.Module):
    def __init__(self, num_classes: int = 10, feature_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.pool  = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(0.25)

        self.fc1   = nn.Linear(128 * 3 * 3, feature_dim)
        self.bn_fc = nn.BatchNorm1d(feature_dim)
        self.drop2 = nn.Dropout(0.4)
        self.fc2   = nn.Linear(feature_dim, num_classes)

    def get_features(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.drop1(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.drop2(x)
        return x

    def forward(self, x):
        return self.fc2(self.get_features(x))

    def forward_with_features(self, x):
        feat = self.get_features(x)
        return self.fc2(feat), feat


# ══════════════════════════════════════════════════════
# CIFAR-10 backbone  — ResNet-style with residual blocks
# ══════════════════════════════════════════════════════

class ResBlock(nn.Module):
    """Basic residual block: two 3×3 convs with BN + skip connection."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        self.skip = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.skip(x))


class CIFARNet(nn.Module):
    """
    ResNet-inspired backbone for CIFAR-10.
    Architecture:
      stem  : 3×3 conv → 64 ch
      stage1: 64 → 64  (2 residual blocks, stride 1)
      stage2: 64 → 128 (2 residual blocks, stride 2)
      stage3: 128→ 256 (2 residual blocks, stride 2)
      gap   : global average pool → 256-d vector
      head  : FC 256 → feature_dim → logits

    Input : 3 × 32 × 32
    Features: 512-dim (rich enough for personalized heads)
    """
    def __init__(self, num_classes: int = 10, feature_dim: int = 512):
        super().__init__()
        self.feature_dim = feature_dim

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Residual stages
        self.stage1 = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 64),
        )
        self.stage2 = nn.Sequential(
            ResBlock(64, 128, stride=2),
            ResBlock(128, 128),
        )
        self.stage3 = nn.Sequential(
            ResBlock(128, 256, stride=2),
            ResBlock(256, 256),
        )

        # Global average pool: 256 × 8 × 8 → 256
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Projection to feature space
        self.proj = nn.Sequential(
            nn.Linear(256, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        # Global classification head
        self.classifier = nn.Linear(feature_dim, num_classes)

        # Weight initialisation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_features(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.proj(x)
        return x

    def forward(self, x):
        return self.classifier(self.get_features(x))

    def forward_with_features(self, x):
        feat = self.get_features(x)
        return self.classifier(feat), feat


# ══════════════════════════════════════════════════════
# Personalized Head — shared by both datasets
# ══════════════════════════════════════════════════════
class PersonalHead(nn.Module):
    """
    2-layer MLP per-client classification head.
    feature_dim must match the backbone's feature_dim.
    """
    def __init__(self, feature_dim: int = 512, num_classes: int = 10):
        super().__init__()
        hidden = max(64, feature_dim // 4)
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ══════════════════════════════════════════════════════
# Factories — dispatch on config["dataset"]
# ══════════════════════════════════════════════════════
def get_model(config: dict):
    ds = config.get("dataset", "mnist")
    if ds == "cifar10":
        return CIFARNet(num_classes=10, feature_dim=512)
    return MNISTNet(num_classes=10, feature_dim=256)


def get_personal_head(config: dict):
    ds = config.get("dataset", "mnist")
    if ds == "cifar10":
        return PersonalHead(feature_dim=512, num_classes=10)
    return PersonalHead(feature_dim=256, num_classes=10)


def get_model_size(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════
# FedBN helpers
# ══════════════════════════════════════════════════════
def get_non_bn_params(model: nn.Module) -> list:
    return [n for n, _ in model.named_parameters() if "bn" not in n]


def aggregate_except_bn(models: list, weights: list) -> dict:
    total   = sum(weights)
    norm_w  = [w / total for w in weights]
    ref     = models[0].state_dict()
    agg     = {}
    for key in ref:
        if "bn" in key:
            continue
        agg[key] = torch.zeros_like(ref[key])
        for m, w in zip(models, norm_w):
            agg[key] += w * m.state_dict()[key].float()
    return agg
