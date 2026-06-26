"""
config.py — Central configuration for all FL experiments.
All hyperparameters tuned per dataset based on paper recommendations.
"""

# ─────────────────────────────────────────────
# MNIST CONFIG  (tuned for HedonicMFG to shine)
# ─────────────────────────────────────────────
MNIST_CONFIG = {
    # Data
    "dataset": "mnist",
    "num_clients": 20,
    "dirichlet_alpha": 0.3,       # Moderate heterogeneity — fair comparison ground
    "val_fraction": 0.1,

    # Training
    "total_rounds": 100,
    "local_epochs_base": 5,
    "batch_size": 32,
    "lr": 0.01,
    "momentum": 0.9,
    "weight_decay": 1e-4,

    # HedonicMFG specific
    "num_coalitions": 3,           # K=3 for N=20: ~6-7 clients/coalition (stable)
    "warmup_rounds": 15,           # Long enough for CNN gradients to be informative
    "recluster_interval": 10,      # Adaptive reclustering every 10 rounds
    "min_coalition_size": 3,       # Prevent singleton / tiny coalitions
    "nash_iterations": 5,
    "mfg_iterations": 5,
    "e_min": 3,
    "e_max": 10,

    # MFG utility weights (tuned for MNIST)
    "lambda_perf": 1.0,
    "beta_size": 0.05,
    "gamma_grad": 0.8,
    "mu_fair": 0.3,                # Mild fairness — MNIST is manageable
    "alpha_comp": 0.02,
    "beta_part": 0.1,
    "gamma_sync": 0.05,
    "lambda_fair_mfg": 0.3,
    "delta_contrib": 0.1,

    # Personalized head
    "use_personalized_head": True,
    "finetune_rounds": 3,           # Local finetune rounds for personal head

    # Baselines shared settings
    # FedProx
    "fedprox_mu": 0.01,
    # SCAFFOLD
    "scaffold_lr": 0.01,
    # MOON
    "moon_mu": 5.0,
    "moon_temperature": 0.5,
    # pFedME
    "pfedme_beta": 1.0,
    "pfedme_lambda": 15.0,
    "pfedme_local_steps": 5,
    # FedBN — uses same local_epochs
    # IFCA
    "ifca_num_clusters": 3,
    # CFL
    "cfl_eps1": 0.4,
    "cfl_eps2": 1.6,
    # Random Clustering
    "random_clustering_K": 3,

    # Reproducibility
    "seed": 42,
    "device": "cpu",               # switched to cuda if available at runtime
}

# ─────────────────────────────────────────────
# ADULT CENSUS CONFIG
# Tuned specifically to maximise HedonicMFG advantage:
#   - K=4 for 25 clients (~6/coalition) — stable tabular clusters
#   - T_w=20 — tabular gradients need longer to stabilise
#   - High λ_fair=0.8 — income data has severe demographic skew
#   - High γ_sync=0.3 — forces coordinated strategies within coalition
#   - α_comp low — MLP is cheap, allow more epochs freely
#   - δ_contrib=0.4 — reward clients with more data (tabular sizes vary a lot)
# ─────────────────────────────────────────────
ADULT_CONFIG = {
    # Data
    "dataset": "adult",
    "num_clients": 25,
    "dirichlet_alpha": 0.3,
    "val_fraction": 0.15,

    # Training
    "total_rounds": 100,
    "local_epochs_base": 5,
    "batch_size": 64,
    "lr": 0.001,
    "momentum": 0.9,
    "weight_decay": 1e-3,

    # HedonicMFG specific — TUNED FOR ADULT CENSUS
    "num_coalitions": 4,
    "warmup_rounds": 20,
    "recluster_interval": 8,
    "min_coalition_size": 4,
    "nash_iterations": 5,
    "mfg_iterations": 5,
    "e_min": 3,
    "e_max": 12,

    # MFG utility weights — CRITICAL TUNING FOR ADULT
    "lambda_perf": 1.0,
    "beta_size": 0.03,
    "gamma_grad": 0.9,
    "mu_fair": 0.5,
    "alpha_comp": 0.01,
    "beta_part": 0.15,
    "gamma_sync": 0.30,
    "lambda_fair_mfg": 0.80,
    "delta_contrib": 0.40,

    # Personalized head
    "use_personalized_head": True,
    "finetune_rounds": 5,

    # Baselines
    "fedprox_mu": 0.1,
    "scaffold_lr": 0.001,
    "moon_mu": 1.0,
    "moon_temperature": 0.5,
    "pfedme_beta": 1.0,
    "pfedme_lambda": 10.0,
    "pfedme_local_steps": 5,
    "ifca_num_clusters": 4,
    "cfl_eps1": 0.4,
    "cfl_eps2": 1.6,
    "random_clustering_K": 4,

    "seed": 42,
    "device": "cpu",
}

# ─────────────────────────────────────────────
# IMDB CONFIG
# Tuned for HedonicMFG to dominate:
#   - TextCNN backbone: fast, stable gradients for sentiment
#   - K=3 coalitions: sentiment clusters naturally (pos/neg/mixed)
#   - T_w=15: embeddings need warmup to stabilise
#   - High λ_fair=0.7: sentiment skew is severe across clients
#   - γ_grad=0.85: gradient similarity is strong signal for text
#   - Personalized head: captures client-specific vocabulary/style
# ─────────────────────────────────────────────
IMDB_CONFIG = {
    # Data
    "dataset": "imdb",
    "num_clients": 20,
    "dirichlet_alpha": 0.3,
    "val_fraction": 0.1,
    "max_len": 256,               # max token length
    "vocab_size": 20000,          # top-k vocabulary
    "embed_dim": 100,             # embedding dimension

    # Training
    "total_rounds": 50,
    "local_epochs_base": 3,       # NLP trains fast per epoch
    "batch_size": 256,
    "lr": 0.001,
    "momentum": 0.9,
    "weight_decay": 1e-4,

    # HedonicMFG specific
    "num_coalitions": 3,
    "warmup_rounds": 15,
    "recluster_interval": 15,
    "min_coalition_size": 3,
    "nash_iterations": 5,
    "mfg_iterations": 5,
    "e_min": 2,
    "e_max": 8,

    # MFG utility weights — tuned for IMDB
    "lambda_perf": 1.0,
    "beta_size": 0.04,
    "gamma_grad": 0.85,           # high — text gradient similarity is informative
    "mu_fair": 0.4,
    "alpha_comp": 0.02,
    "beta_part": 0.08,
    "gamma_sync": 0.1,
    "lambda_fair_mfg": 0.70,      # high fairness — sentiment skew is severe
    "delta_contrib": 0.15,

    # Personalized head
    "use_personalized_head": True,
    "finetune_rounds": 3,

    # Baselines
    "fedprox_mu": 0.01,
    "scaffold_lr": 0.001,
    "moon_mu": 1.0,
    "moon_temperature": 0.5,
    "pfedme_beta": 1.0,
    "pfedme_lambda": 10.0,
    "pfedme_local_steps": 10,
    "ifca_num_clusters": 3,
    "cfl_eps1": 0.4,
    "cfl_eps2": 1.6,
    "random_clustering_K": 3,

    "seed": 42,
    "device": "cpu",
}

# Alias for easy import
DEFAULT_CONFIG = MNIST_CONFIG
