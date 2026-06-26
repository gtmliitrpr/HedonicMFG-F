"""
models_imdb.py — TextCNN for IMDB sentiment FL experiments.

Architecture choices for HedonicMFG advantage:
  - TextCNN: fast convergence, stable gradients → coalition formation works well
  - Multiple kernel sizes (3,4,5): captures n-gram patterns at different scales
  - get_features(): returns penultimate embedding for personalized head
  - forward_with_features(): for MOON contrastive loss
  - No BatchNorm in backbone: avoids FedBN having unfair BN advantage
  - PersonalHeadIMDB: 2-layer MLP head for client-specific sentiment patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    """
    Convolutional Neural Network for text classification.
    Architecture:
        Embedding(vocab_size, embed_dim)
        → Conv1d(embed_dim, num_filters, kernel_size) × 3 kernels
        → MaxPool → Concat → Dropout
        → FC(feature_dim) → Classifier(2)

    get_features() returns the concatenated pooled conv outputs
    before the final classifier — used for personalized head.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 100,
                 num_filters: int = 128, kernel_sizes: list = None,
                 num_classes: int = 2, dropout: float = 0.3,
                 feature_dim: int = 256):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 4, 5]

        self.embed_dim    = embed_dim
        self.feature_dim  = feature_dim
        self.kernel_sizes = kernel_sizes
        self.num_filters  = num_filters

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        self.embedding.weight.data[0].fill_(0)  # PAD = zero

        # Convolutional layers — one per kernel size
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k)
            for k in kernel_sizes
        ])

        # Feature projection: concat of all conv outputs → feature_dim
        conv_out_dim = num_filters * len(kernel_sizes)
        self.fc_feat = nn.Linear(conv_out_dim, feature_dim)
        self.dropout  = nn.Dropout(dropout)

        # Global classifier head
        self.classifier = nn.Linear(feature_dim, num_classes)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning feature embedding.
        x: (batch, seq_len) LongTensor of token ids
        Returns: (batch, feature_dim)
        """
        # Embedding: (batch, seq_len, embed_dim)
        emb = self.dropout(self.embedding(x))
        # Conv1d expects (batch, embed_dim, seq_len)
        emb = emb.permute(0, 2, 1)

        # Apply each conv + global max pooling
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(emb))                   # (batch, num_filters, L)
            c = F.max_pool1d(c, c.size(2)).squeeze(2)  # (batch, num_filters)
            pooled.append(c)

        # Concat all kernel outputs
        cat = torch.cat(pooled, dim=1)              # (batch, num_filters * len(kernels))
        cat = self.dropout(cat)
        feat = F.relu(self.fc_feat(cat))            # (batch, feature_dim)
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.get_features(x))

    def forward_with_features(self, x: torch.Tensor):
        """Return (logits, features) — used by MOON."""
        feat = self.get_features(x)
        return self.classifier(feat), feat


class PersonalHeadIMDB(nn.Module):
    """
    Two-layer personalized head for IMDB.
    Takes feature_dim embedding from TextCNN backbone.
    Captures client-specific sentiment vocabulary and writing style.
    """
    def __init__(self, feature_dim: int = 256, num_classes: int = 2,
                 hidden_dim: int = 64):
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
def get_imdb_model(config: dict, vocab_size: int) -> TextCNN:
    return TextCNN(
        vocab_size=vocab_size,
        embed_dim=config.get("embed_dim", 100),
        num_filters=128,
        kernel_sizes=[3, 4, 5],
        num_classes=2,
        dropout=0.3,
        feature_dim=256,
    )


def get_imdb_personal_head(config: dict) -> PersonalHeadIMDB:
    return PersonalHeadIMDB(feature_dim=256, num_classes=2, hidden_dim=64)


def get_model_size(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ──────────────────────────────────────────
# FedBN helper — TextCNN has no BN layers
# so FedBN degrades to standard FedAvg
# We still implement it correctly for fairness
# ──────────────────────────────────────────
def aggregate_non_bn_imdb(models: list, weights: list) -> dict:
    """For TextCNN there are no BN layers — standard aggregation."""
    total = sum(weights)
    norm_w = [w / total for w in weights]
    agg_state = {}
    ref_state = models[0].state_dict()
    for key in ref_state:
        agg_state[key] = sum(
            w * m.state_dict()[key].float()
            for w, m in zip(norm_w, models)
        )
    return agg_state
