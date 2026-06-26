"""
runner.py — Main experiment runner for FL comparison (MNIST + CIFAR-10).
Runs all 10 algorithms sequentially and generates plots + summary table.

Usage:
    # CIFAR-10 (default for this codebase)
    python runner.py
    python runner.py --dataset cifar10

    # MNIST
    python runner.py --dataset mnist

    # Quick smoke test (fewer rounds, fewer clients)
    python runner.py --quick
    python runner.py --dataset cifar10 --quick

    # Run specific algorithms only
    python runner.py --algos hedonicmfg fedavg fedbn

    # Custom settings
    python runner.py --dataset cifar10 --rounds 100 --seed 123

    # Skip plots
    python runner.py --no-plots
"""

import argparse
import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(__file__))

from config import CIFAR10_CONFIG, MNIST_CONFIG
from data import get_client_loaders
from utils import set_seed, get_device, print_final_table, save_results
from visualize import generate_all_plots

# Algorithm imports
from algorithms.fedavg             import run_fedavg
from algorithms.fedprox            import run_fedprox
from algorithms.scaffold           import run_scaffold
from algorithms.moon               import run_moon
from algorithms.fedbn              import run_fedbn
from algorithms.pfedme             import run_pfedme
from algorithms.ifca               import run_ifca
from algorithms.cfl                import run_cfl
from algorithms.random_clustering  import run_random_clustering
from algorithms.hedonic_mfg        import run_hedonic_mfg


ALGORITHM_REGISTRY = {
    "fedavg":        run_fedavg,
    "fedprox":       run_fedprox,
    "scaffold":      run_scaffold,
    "moon":          run_moon,
    "fedbn":         run_fedbn,
    "pfedme":        run_pfedme,
    "ifca":          run_ifca,
    "cfl":           run_cfl,
    "randomcluster": run_random_clustering,
    "hedonicmfg":    run_hedonic_mfg,
}

DISPLAY_NAMES = {
    "fedavg":        "FedAvg",
    "fedprox":       "FedProx",
    "scaffold":      "SCAFFOLD",
    "moon":          "MOON",
    "fedbn":         "FedBN",
    "pfedme":        "pFedME",
    "ifca":          "IFCA",
    "cfl":           "CFL",
    "randomcluster": "RandomCluster",
    "hedonicmfg":    "HedonicMFG",
}

DEFAULT_ORDER = [
    "fedavg", "fedprox", "scaffold", "moon",
    "fedbn",  "pfedme",  "ifca",     "cfl",
    "randomcluster", "hedonicmfg",
]


# ──────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="FL Experiment Runner — MNIST / CIFAR-10",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--dataset", type=str, default="cifar10",
        choices=["mnist", "cifar10"],
        help="Dataset to use (default: cifar10)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick run: 15 rounds, 10 clients, K=2 (smoke test)",
    )
    parser.add_argument(
        "--algos", nargs="+", default=None,
        help=(
            "Algorithms to run (default: all).\n"
            "Options: " + " ".join(DEFAULT_ORDER)
        ),
    )
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--rounds",  type=int, default=None,
                        help="Override total_rounds in config")
    parser.add_argument("--clients", type=int, default=None,
                        help="Override num_clients in config")
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: ./results/<dataset>)",
    )
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation")
    return parser.parse_args()


# ──────────────────────────────────────────
# Config builder
# ──────────────────────────────────────────
def build_config(args) -> dict:
    base = CIFAR10_CONFIG if args.dataset == "cifar10" else MNIST_CONFIG
    config = dict(base)
    config["seed"] = args.seed

    if args.quick:
        print(f"\n[Mode] QUICK RUN — reduced settings for smoke test")
        config["total_rounds"]        = 15
        config["num_clients"]         = 10
        config["warmup_rounds"]       = 5
        config["num_coalitions"]      = 2
        config["random_clustering_K"] = 2
        config["ifca_num_clusters"]   = 2
        config["nash_iterations"]     = 3
        config["mfg_iterations"]      = 3
        config["recluster_interval"]  = 5

    if args.rounds:
        config["total_rounds"] = args.rounds
    if args.clients:
        config["num_clients"] = args.clients

    return config


# ──────────────────────────────────────────
# Pretty config box
# ──────────────────────────────────────────
def print_config_summary(config: dict, device: torch.device):
    ds = config["dataset"].upper()
    w  = 54
    print("\n" + "╔" + "═" * w + "╗")
    print(f"║  {ds} Federated Learning Experiment" + " " * (w - len(ds) - 35) + "║")
    print("╠" + "═" * w + "╣")
    rows = [
        ("Dataset",          config["dataset"]),
        ("Clients",          config["num_clients"]),
        ("Rounds",           config["total_rounds"]),
        ("Dirichlet α",      config["dirichlet_alpha"]),
        ("Batch size",       config["batch_size"]),
        ("LR",               config["lr"]),
        ("Coalitions (K)",   config["num_coalitions"]),
        ("Warmup rounds",    config["warmup_rounds"]),
        ("Recluster int.",   config["recluster_interval"]),
        ("Personal heads",   config["use_personalized_head"]),
        ("Finetune epochs",  config.get("finetune_rounds", 3)),
        ("Device",           str(device)),
        ("Seed",             config["seed"]),
    ]
    for label, value in rows:
        line = f"║  {label:<18}: {str(value):<{w - 22}}║"
        print(line)
    print("╚" + "═" * w + "╝")


# ──────────────────────────────────────────
# Main
# ──────────────────────────────────────────
def main():
    args   = parse_args()
    config = build_config(args)

    set_seed(config["seed"])
    device = get_device(config)
    config["device"] = str(device)

    print_config_summary(config, device)

    # Output directory
    out_dir = args.output or f"./results/{config['dataset']}"
    os.makedirs(out_dir, exist_ok=True)

    # Select algorithms
    if args.algos:
        selected = [a.lower().replace("-", "").replace("_", "") for a in args.algos]
        invalid  = [a for a in selected if a not in ALGORITHM_REGISTRY]
        if invalid:
            print(f"\n[Error] Unknown algorithms: {invalid}")
            print(f"        Valid: {list(ALGORITHM_REGISTRY.keys())}")
            sys.exit(1)
    else:
        selected = list(DEFAULT_ORDER)

    print(f"\n[Runner] Algorithms: {[DISPLAY_NAMES[a] for a in selected]}")
    print(f"[Runner] Output dir: {out_dir}")

    # ── Load data (shared across all algorithms) ──────────
    print(f"\n[Data] Loading {config['dataset'].upper()} with Dirichlet α={config['dirichlet_alpha']}...")
    (client_train_loaders,
     client_val_loaders,
     global_test_loader,
     client_data_sizes) = get_client_loaders(config)
    config["client_data_sizes"] = client_data_sizes

    # ── Run algorithms ────────────────────────────────────
    all_results  = {}
    total_start  = time.time()

    for algo_key in selected:
        display    = DISPLAY_NAMES[algo_key]
        runner_fn  = ALGORITHM_REGISTRY[algo_key]
        set_seed(config["seed"])   # identical starting conditions for every algo
        t0 = time.time()

        try:
            tracker = runner_fn(
                config,
                client_train_loaders,
                client_val_loaders,
                global_test_loader,
                device,
            )
            all_results[display] = tracker
            elapsed = time.time() - t0
            print(f"  ✓ {display} completed in {elapsed / 60:.1f} min")

        except Exception as exc:
            print(f"  ✗ {display} FAILED: {exc}")
            import traceback
            traceback.print_exc()

    total_elapsed = time.time() - total_start

    # ── Final summary table ───────────────────────────────
    ds_upper = config["dataset"].upper()
    print(f"\n\n{'='*72}")
    print(f"  FINAL RESULTS — {ds_upper} (Dirichlet α={config['dirichlet_alpha']})")
    print(f"{'='*72}")
    print_final_table(all_results)
    print(f"\n  Total experiment time: {total_elapsed / 60:.1f} minutes")

    # ── HedonicMFG delta summary ──────────────────────────
    if "HedonicMFG" in all_results:
        hm   = all_results["HedonicMFG"].final_summary()
        hm_g = hm["best_global_acc"]    * 100
        hm_c = hm["best_avg_client_acc"] * 100

        print(f"\n{'─'*64}")
        print(f"  HedonicMFG vs Baselines (Best Global / Best Client Δ):")
        print(f"{'─'*64}")
        wins_g = wins_c = 0
        for name, tracker in all_results.items():
            if name == "HedonicMFG":
                continue
            s   = tracker.final_summary()
            bg  = s["best_global_acc"]    * 100
            bc  = s["best_avg_client_acc"] * 100
            dg  = hm_g - bg
            dc  = hm_c - bc
            mg  = "✓" if dg >= 0 else "✗"
            mc  = "✓" if dc >= 0 else "✗"
            wins_g += (dg >= 0)
            wins_c += (dc >= 0)
            print(f"  vs {name:<16}  "
                  f"Global: {mg} {dg:+.2f}%   "
                  f"Client: {mc} {dc:+.2f}%")

        total_baselines = len(all_results) - 1
        print(f"{'─'*64}")
        print(f"  HedonicMFG wins — Global: {wins_g}/{total_baselines}  "
              f"Client: {wins_c}/{total_baselines}")
        print(f"{'─'*64}")

    # ── Save JSON results ─────────────────────────────────
    json_path = os.path.join(out_dir, f"{config['dataset']}_results.json")
    save_results(all_results, json_path)

    # ── Generate plots ────────────────────────────────────
    if not args.no_plots and all_results:
        print("\n[Plots] Generating visualizations ...")
        generate_all_plots(all_results, config["dataset"], out_dir)
        print(
            f"\n[Done] Plots saved:\n"
            f"       {config['dataset']}_convergence.png\n"
            f"       {config['dataset']}_bar_comparison.png\n"
            f"       {config['dataset']}_improvement.png"
        )

    print(f"\n[Done] All results in: {out_dir}/")


if __name__ == "__main__":
    main()
