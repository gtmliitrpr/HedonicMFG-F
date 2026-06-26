"""
config.py — Central configuration for ablation study.
Only MNIST and FashionMNIST are kept. All other dataset configs removed.
"""

# ─────────────────────────────────────────────
# MNIST CONFIG
# ─────────────────────────────────────────────
MNIST_CONFIG = {
    "dataset": "mnist",
    "num_clients": 20,
    "dirichlet_alpha": 0.3,
    "val_fraction": 0.1,

    "total_rounds": 50,
    "local_epochs_base": 5,
    "batch_size": 32,
    "lr": 0.01,
    "momentum": 0.9,
    "weight_decay": 1e-4,

    # HedonicMFG specific
    "num_coalitions": 3,
    "warmup_rounds": 15,
    "recluster_interval": 10,
    "min_coalition_size": 3,
    "nash_iterations": 5,
    "mfg_iterations": 5,
    "e_min": 3,
    "e_max": 10,

    # MFG utility weights
    "lambda_perf": 1.0,
    "beta_size": 0.05,
    "gamma_grad": 0.8,
    "mu_fair": 0.3,
    "alpha_comp": 0.02,
    "beta_part": 0.1,
    "gamma_sync": 0.05,
    "lambda_fair_mfg": 0.3,
    "delta_contrib": 0.1,

    # Personalized head
    "use_personalized_head": True,
    "finetune_rounds": 3,

    # FedAvg baseline (only baseline kept)
    "fedprox_mu": 0.01,  # kept for completeness, not used

    "seed": 42,
    "device": "cuda",
}

# ─────────────────────────────────────────────
# FASHIONMNIST CONFIG
# ─────────────────────────────────────────────
FMNIST_CONFIG = {
    "dataset": "fmnist",
    "num_clients": 20,
    "dirichlet_alpha": 0.3,
    "val_fraction": 0.1,

    "total_rounds": 50,
    "local_epochs_base": 5,
    "batch_size": 256,
    "num_workers": 4,
    "lr": 0.005,
    "momentum": 0.9,
    "weight_decay": 1e-4,

    # HedonicMFG specific
    "num_coalitions": 3,
    "warmup_rounds": 15,
    "recluster_interval": 10,
    "min_coalition_size": 3,
    "nash_iterations": 5,
    "mfg_iterations": 5,
    "e_min": 3,
    "e_max": 10,

    # MFG utility weights
    "lambda_perf": 1.0,
    "beta_size": 0.04,
    "gamma_grad": 0.90,
    "mu_fair": 0.4,
    "alpha_comp": 0.02,
    "beta_part": 0.08,
    "gamma_sync": 0.05,
    "lambda_fair_mfg": 0.50,
    "delta_contrib": 0.12,

    # Personalized head
    "use_personalized_head": True,
    "finetune_rounds": 3,

    "seed": 42,
    "device": "cuda",
}

# Dirichlet alpha values for ablation study
ABLATION_ALPHAS = [0.05, 0.1, 0.3, 0.5, 1.0]
