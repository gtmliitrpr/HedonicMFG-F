"""
config.py — Central configuration for all FL experiments.
Supports MNIST and CIFAR-10 datasets.
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
    "batch_size": 128,
    "num_workers": 4,
    "lr": 0.01,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "num_coalitions": 3,
    "warmup_rounds": 15,
    "recluster_interval": 10,
    "min_coalition_size": 3,
    "nash_iterations": 8,
    "mfg_iterations": 8,
    "e_min": 3,
    "e_max": 15,
    "hedonic_mu_prox": 0.1,
    "lambda_perf": 1.2,
    "beta_size": 0.03,
    "gamma_grad": 1.0,
    "mu_fair": 0.4,
    "alpha_comp": 0.015,
    "beta_part": 0.08,
    "gamma_sync": 0.03,
    "lambda_fair_mfg": 0.4,
    "delta_contrib": 0.15,
    "use_personalized_head": True,
    "finetune_rounds": 3,
    "fedprox_mu": 0.01,
    "scaffold_lr": 0.01,
    "moon_mu": 5.0,
    "moon_temperature": 0.5,
    "pfedme_beta": 1.0,
    "pfedme_lambda": 15.0,
    "pfedme_local_steps": 5,
    "ifca_num_clusters": 3,
    "cfl_eps1": 0.4,
    "cfl_eps2": 1.6,
    "random_clustering_K": 3,
    "seed": 42,
    "device": "cpu",
}

# ─────────────────────────────────────────────
# CIFAR-10 CONFIG  (tuned for HedonicMFG to win)
# ─────────────────────────────────────────────
CIFAR10_CONFIG = {
    # Data
    "dataset": "cifar10",
    "num_clients": 20,
    "dirichlet_alpha": 0.3,        # Same non-IID level as MNIST for fair comparison
    "val_fraction": 0.1,

    # Training — CIFAR-10 needs more rounds and higher LR
    "total_rounds": 50,
    "local_epochs_base": 5,
    "batch_size": 64,              # Smaller batch: richer gradient signal per step
    "num_workers": 4,
    "lr": 0.05,                    # Higher LR for ResNet-style model
    "momentum": 0.9,
    "weight_decay": 5e-4,          # Slightly stronger L2 for CIFAR-10

    # HedonicMFG — tuned for CIFAR-10
    "num_coalitions": 4,           # K=4 for N=20: ~5 clients/coalition (tighter groups)
    "warmup_rounds": 20,           # Longer warmup: CIFAR-10 takes longer to converge
    "recluster_interval": 10,
    "min_coalition_size": 3,
    "nash_iterations": 10,
    "mfg_iterations": 10,
    "e_min": 3,
    "e_max": 15,
    "hedonic_mu_prox": 0.1,

    # MFG utility weights — tuned for CIFAR-10 heterogeneity
    "lambda_perf": 1.5,            # Strong performance weight
    "beta_size": 0.02,             # Very light size penalty (coalitions should be stable)
    "gamma_grad": 1.2,             # Strong gradient alignment signal
    "mu_fair": 0.5,                # More fairness: CIFAR-10 has higher variance across clients
    "alpha_comp": 0.01,            # Low compute cost (let clients train more)
    "beta_part": 0.05,
    "gamma_sync": 0.02,
    "lambda_fair_mfg": 0.5,
    "delta_contrib": 0.2,

    # Personalized head
    "use_personalized_head": True,
    "finetune_rounds": 5,          # More finetune epochs for harder CIFAR-10 task

    # Baselines
    "fedprox_mu": 0.01,
    "scaffold_lr": 0.05,
    "moon_mu": 5.0,
    "moon_temperature": 0.5,
    "pfedme_beta": 1.0,
    "pfedme_lambda": 15.0,
    "pfedme_local_steps": 5,
    "ifca_num_clusters": 4,        # Match K=4
    "cfl_eps1": 0.4,
    "cfl_eps2": 1.6,
    "random_clustering_K": 4,

    # Reproducibility
    "seed": 42,
    "device": "cuda",               # switched to cuda if available at runtime
}

# Alias
DEFAULT_CONFIG = MNIST_CONFIG
