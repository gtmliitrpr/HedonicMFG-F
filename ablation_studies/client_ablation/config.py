"""
config.py — Configuration for client ablation study.
FashionMNIST only. Sweeps num_clients: [10, 20, 30, 50, 75, 100].
All other hyperparameters fixed — only num_clients changes across runs.
"""

FMNIST_CONFIG = {
    "dataset":          "fmnist",
    "dirichlet_alpha":  0.3,       # fixed — same as alpha ablation default
    "val_fraction":     0.1,

    # Training
    "total_rounds":     50,
    "local_epochs_base": 5,
    "batch_size":       256,
    "num_workers":      4,
    "lr":               0.005,
    "momentum":         0.9,
    "weight_decay":     1e-4,

    # HedonicMFG — fixed across all client counts
    "num_coalitions":    3,        # K=3 fixed — only clients varies
    "warmup_rounds":     12,       # adjusted per run if needed (see runner)
    "recluster_interval": 10,
    "min_coalition_size": 3,
    "nash_iterations":   5,
    "mfg_iterations":    5,
    "e_min":             3,
    "e_max":             10,

    # MFG utility weights — same as alpha ablation
    "lambda_perf":      1.0,
    "beta_size":        0.04,
    "gamma_grad":       0.90,
    "mu_fair":          0.4,
    "alpha_comp":       0.02,
    "beta_part":        0.08,
    "gamma_sync":       0.05,
    "lambda_fair_mfg":  0.50,
    "delta_contrib":    0.12,

    # Personalized head
    "use_personalized_head": True,
    "finetune_rounds":   3,

    "seed":   42,
    "device": "cuda",
}

# Client counts to sweep
ABLATION_CLIENTS = [10, 20, 30, 50, 75, 100]
