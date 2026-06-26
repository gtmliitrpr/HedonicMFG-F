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
    "total_rounds": 50,
    "local_epochs_base": 5,
    "batch_size": 128,
    "num_workers": 4,
    "lr": 0.01,                   # Increased from 0.005 for faster convergence
    "momentum": 0.9,
    "weight_decay": 1e-4,

    # HedonicMFG specific
    "num_coalitions": 3,           # K=3 for N=20: ~6-7 clients/coalition (stable)
    "warmup_rounds": 15,           # Increased from 10 — gradients more informative
    "recluster_interval": 10,      # More frequent reclustering (was 20)
    "min_coalition_size": 3,       # Prevent singleton / tiny coalitions
    "nash_iterations": 8,          # More Nash iterations for better stability
    "mfg_iterations": 8,
    "e_min": 3,
    "e_max": 15,
    "hedonic_mu_prox": 0.1,

    # MFG utility weights (tuned for MNIST)
    "lambda_perf": 1.2,            # Slightly higher performance weight
    "beta_size": 0.03,             # Lighter size penalty
    "gamma_grad": 1.0,             # Higher gradient similarity weight
    "mu_fair": 0.4,                # Slightly more fairness
    "alpha_comp": 0.015,           # Lower compute cost
    "beta_part": 0.08,
    "gamma_sync": 0.03,
    "lambda_fair_mfg": 0.4,
    "delta_contrib": 0.15,

    # Personalized head
    "use_personalized_head": True,
    "finetune_rounds": 3,          # Local finetune rounds for personal head

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
    "device": "cuda",               # switched to cuda if available at runtime
}

# Alias for easy import
DEFAULT_CONFIG = MNIST_CONFIG
