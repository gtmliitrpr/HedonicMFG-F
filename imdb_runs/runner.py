"""
runner.py — Main experiment runner for MNIST FL comparison.
Runs all 10 algorithms sequentially and generates plots + summary table.

Usage:
    python runner.py                        # full run (all algorithms)
    python runner.py --quick               # 20 rounds, 10 clients (smoke test)
    python runner.py --algos fedavg moon   # run specific algorithms only
    python runner.py --seed 123            # different random seed
"""

import argparse
import os
import sys
import time
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from config import MNIST_CONFIG
from data import get_mnist_client_loaders
from utils import set_seed, get_device, print_final_table, save_results
from visualize import generate_all_plots

# Algorithm imports
from algorithms.fedavg          import run_fedavg
from algorithms.fedprox         import run_fedprox
from algorithms.scaffold        import run_scaffold
from algorithms.moon            import run_moon
from algorithms.fedbn           import run_fedbn
from algorithms.pfedme          import run_pfedme
from algorithms.ifca            import run_ifca
from algorithms.cfl             import run_cfl
from algorithms.random_clustering import run_random_clustering
from algorithms.hedonic_mfg     import run_hedonic_mfg


ALGORITHM_REGISTRY = {
    "fedavg":         run_fedavg,
    "fedprox":        run_fedprox,
    "scaffold":       run_scaffold,
    "moon":           run_moon,
    "fedbn":          run_fedbn,
    "pfedme":         run_pfedme,
    "ifca":           run_ifca,
    "cfl":            run_cfl,
    "randomcluster":  run_random_clustering,
    "hedonicmfg":     run_hedonic_mfg,
}

# Display names for printing
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


def parse_args():
    parser = argparse.ArgumentParser(description="FL Experiment Runner — MNIST")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run: 20 rounds, 10 clients (smoke test)")
    parser.add_argument("--algos", nargs="+", default=None,
                        help="Algorithms to run (default: all). "
                             "Options: fedavg fedprox scaffold moon fedbn "
                             "pfedme ifca cfl randomcluster hedonicmfg")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--rounds", type=int, default=None,
                        help="Override total rounds")
    parser.add_argument("--clients", type=int, default=None,
                        help="Override number of clients")
    parser.add_argument("--output", type=str, default="./results/mnist",
                        help="Output directory for results and plots")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation")
    return parser.parse_args()


def build_config(args) -> dict:
    config = dict(MNIST_CONFIG)
    config["seed"] = args.seed

    if args.quick:
        print("\n[Mode] QUICK RUN — reduced rounds/clients for smoke test")
        config["total_rounds"]   = 20
        config["num_clients"]    = 10
        config["warmup_rounds"]  = 5
        config["num_coalitions"] = 2
        config["random_clustering_K"] = 2
        config["ifca_num_clusters"]   = 2
        config["nash_iterations"] = 3
        config["mfg_iterations"]  = 3

    if args.rounds:
        config["total_rounds"] = args.rounds
    if args.clients:
        config["num_clients"] = args.clients

    return config


def print_config_summary(config: dict, device: torch.device):
    print("\n" + "╔" + "═"*54 + "╗")
    print("║  MNIST Federated Learning Experiment               ║")
    print("╠" + "═"*54 + "╣")
    print(f"║  Clients        : {config['num_clients']:<35} ║")
    print(f"║  Rounds         : {config['total_rounds']:<35} ║")
    print(f"║  Dirichlet α    : {config['dirichlet_alpha']:<35} ║")
    print(f"║  Coalitions (K) : {config['num_coalitions']:<35} ║")
    print(f"║  Warmup rounds  : {config['warmup_rounds']:<35} ║")
    print(f"║  Recluster int. : {config['recluster_interval']:<35} ║")
    print(f"║  Personal heads : {str(config['use_personalized_head']):<35} ║")
    print(f"║  Device         : {str(device):<35} ║")
    print(f"║  Seed           : {config['seed']:<35} ║")
    print("╚" + "═"*54 + "╝")


def main():
    args = parse_args()
    config = build_config(args)

    # Device + seed
    set_seed(config["seed"])
    device = get_device(config)
    config["device"] = str(device)

    print_config_summary(config, device)

    # Select algorithms to run
    if args.algos:
        selected = [a.lower().replace("-", "") for a in args.algos]
        invalid = [a for a in selected if a not in ALGORITHM_REGISTRY]
        if invalid:
            print(f"[Error] Unknown algorithms: {invalid}")
            print(f"        Valid options: {list(ALGORITHM_REGISTRY.keys())}")
            sys.exit(1)
    else:
        # Default order: baselines first, HedonicMFG last
        selected = ["fedavg", "fedprox", "scaffold", "moon",
                    "fedbn", "pfedme", "ifca", "cfl",
                    "randomcluster", "hedonicmfg"]

    print(f"\n[Runner] Algorithms to run: {[DISPLAY_NAMES[a] for a in selected]}")

    # Load data (shared across all algorithms)
    print("\n[Data] Loading MNIST with Dirichlet partitioning...")
    (client_train_loaders,
     client_val_loaders,
     global_test_loader,
     client_data_sizes) = get_mnist_client_loaders(config)
    config["client_data_sizes"] = client_data_sizes

    # Run all algorithms
    all_results = {}
    total_start = time.time()

    for algo_key in selected:
        display = DISPLAY_NAMES[algo_key]
        runner_fn = ALGORITHM_REGISTRY[algo_key]

        set_seed(config["seed"])  # Reset seed for each algorithm (fair comparison)
        t0 = time.time()

        try:
            tracker = runner_fn(
                config,
                client_train_loaders,
                client_val_loaders,
                global_test_loader,
                device
            )
            all_results[display] = tracker
            elapsed = time.time() - t0
            print(f"  ✓ {display} completed in {elapsed/60:.1f} min")

        except Exception as e:
            print(f"  ✗ {display} FAILED: {e}")
            import traceback
            traceback.print_exc()
            continue

    total_elapsed = time.time() - total_start

    # ── Final Summary Table ────────────────────────────────
    print(f"\n\n{'='*72}")
    print(f"  FINAL RESULTS — MNIST (Dirichlet α={config['dirichlet_alpha']})")
    print(f"{'='*72}")
    print_final_table(all_results)
    print(f"\n  Total experiment time: {total_elapsed/60:.1f} minutes")

    # ── HedonicMFG improvement summary ────────────────────
    if "HedonicMFG" in all_results:
        hm = all_results["HedonicMFG"].final_summary()
        hm_g = hm["best_global_acc"] * 100
        hm_c = hm["best_avg_client_acc"] * 100

        print(f"\n{'─'*60}")
        print(f"  HedonicMFG vs Baselines (Best Global / Best Client):")
        print(f"{'─'*60}")
        for name, tracker in all_results.items():
            if name == "HedonicMFG":
                continue
            s = tracker.final_summary()
            bg = s["best_global_acc"] * 100
            bc = s["best_avg_client_acc"] * 100
            dg = hm_g - bg
            dc = hm_c - bc
            mark_g = "✓" if dg > 0 else "✗"
            mark_c = "✓" if dc > 0 else "✗"
            print(f"  vs {name:<16}  "
                  f"Global: {mark_g} {dg:+.2f}%   "
                  f"Client: {mark_c} {dc:+.2f}%")
        print(f"{'─'*60}")

    # ── Save results ───────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)
    results_path = os.path.join(args.output, "mnist_results.json")
    save_results(all_results, results_path)

    # ── Generate plots ─────────────────────────────────────
    if not args.no_plots and all_results:
        print("\n[Plots] Generating visualizations...")
        generate_all_plots(all_results, "mnist", args.output)

    print(f"\n[Done] Results saved to: {args.output}/")
    print(f"       Files: mnist_results.json | "
          f"mnist_convergence.png | mnist_bar_comparison.png | "
          f"mnist_improvement.png")


if __name__ == "__main__":
    main()
