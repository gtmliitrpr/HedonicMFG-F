"""
config.py — Configuration for K (num_coalitions) ablation study.
MNIST only. Sweeps K: [2, 3, 4, 5, 6, 8].
All other hyperparameters fixed — only num_coalitions changes.

Fixed settings (same as alpha ablation default):
  num_clients  = 20
  alpha        = 0.3
  total_rounds = 50
"""

MNIST_CONFIG = {
    "dataset":          "mnist",
    "num_clients":      20,        # fixed
    "dirichlet_alpha":  0.3,       # fixed
    "val_fraction":     0.1,

    # Training — all fixed
    "total_rounds":      50,
    "local_epochs_base": 5,
    "batch_size":        64,
    "num_workers":       4,
    "lr":                0.01,
    "momentum":          0.9,
    "weight_decay":      1e-4,

    # HedonicMFG — num_coalitions overridden per run, rest fixed
    "num_coalitions":    3,        # default — overridden by make_config()
    "warmup_rounds":     15,       # fixed — 30% of 50 rounds
    "recluster_interval": 10,      # fixed — 3 reclusters in 35 active rounds
    "min_coalition_size": 2,       # fixed at 2 — allows small coalitions at high K
    "nash_iterations":   5,
    "mfg_iterations":    5,
    "e_min":             3,
    "e_max":             10,

    # MFG utility weights — all fixed, same as alpha ablation
    "lambda_perf":      1.0,
    "beta_size":        0.05,
    "gamma_grad":       0.8,
    "mu_fair":          0.3,
    "alpha_comp":       0.02,
    "beta_part":        0.1,
    "gamma_sync":       0.05,
    "lambda_fair_mfg":  0.3,
    "delta_contrib":    0.1,

    # Personalized head
    "use_personalized_head": True,
    "finetune_rounds":   3,

    "seed":   42,
    "device": "cuda",
}

# K values to sweep
# K=2: undercluster baseline  K=3: default sweet spot
# K=4,5: finer granularity    K=6,8: expected degradation
ABLATION_K = [2, 3, 4, 5, 6, 8]
