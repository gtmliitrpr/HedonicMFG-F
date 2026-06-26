"""
runner_mnist.py — MNIST experiment runner (FedAvg vs HedonicMFG).

Each alpha run saves to its own folder so runs are independent
and can be combined later with combine_plots.py.

Usage:
    # Run one alpha at a time:
    python runner_mnist.py --alpha 0.05
    python runner_mnist.py --alpha 0.1
    python runner_mnist.py --alpha 0.3
    python runner_mnist.py --alpha 0.5
    python runner_mnist.py --alpha 1.0

    # Run only one algorithm:
    python runner_mnist.py --alpha 0.3 --algos hedonicmfg
    python runner_mnist.py --alpha 0.3 --algos fedavg

    # Quick smoke test:
    python runner_mnist.py --alpha 0.3 --quick

    # After all alphas done, combine plots:
    python combine_plots.py --datasets mnist
"""

import argparse, os, sys, time, json
import torch

sys.path.insert(0, os.path.dirname(__file__))

from config import MNIST_CONFIG
from data import get_mnist_client_loaders
from utils import set_seed, get_device, print_final_table, save_results, load_results
from visualize import generate_run_plots

from algorithms.mnist.fedavg_mnist      import run_fedavg_mnist
from algorithms.mnist.hedonic_mfg_mnist import run_hedonic_mfg_mnist


REGISTRY = {
    "fedavg":     run_fedavg_mnist,
    "hedonicmfg": run_hedonic_mfg_mnist,
}

DISPLAY = {
    "fedavg":     "FedAvg",
    "hedonicmfg": "HedonicMFG",
}


def parse_args():
    p = argparse.ArgumentParser(description="FL Runner — MNIST (per-alpha)")
    p.add_argument("--alpha",    type=float, required=True,
                   help="Dirichlet alpha value, e.g. 0.05 0.1 0.3 0.5 1.0")
    p.add_argument("--algos",    nargs="+", default=None,
                   choices=["fedavg", "hedonicmfg"],
                   help="Which algorithms to run (default: both)")
    p.add_argument("--quick",    action="store_true",
                   help="Smoke test: 5 rounds, 5 clients")
    p.add_argument("--seed",     type=int,   default=42)
    p.add_argument("--rounds",   type=int,   default=None)
    p.add_argument("--output",   type=str,   default="./results/mnist",
                   help="Base output dir. Results saved to output/alpha_X/")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def alpha_tag(alpha: float) -> str:
    """Convert alpha float to safe folder name. e.g. 0.05 -> alpha_005"""
    return "alpha_" + str(alpha).replace(".", "")


def print_config(config, device):
    print("\n" + "╔" + "═"*54 + "╗")
    print("║  MNIST Federated Learning Experiment               ║")
    print("╠" + "═"*54 + "╣")
    print(f"║  Clients        : {config['num_clients']:<35} ║")
    print(f"║  Rounds         : {config['total_rounds']:<35} ║")
    print(f"║  Dirichlet α    : {config['dirichlet_alpha']:<35} ║")
    print(f"║  Coalitions (K) : {config['num_coalitions']:<35} ║")
    print(f"║  Warmup rounds  : {config['warmup_rounds']:<35} ║")
    print(f"║  Device         : {str(device):<35} ║")
    print(f"║  Seed           : {config['seed']:<35} ║")
    print("╚" + "═"*54 + "╝")


def main():
    args   = parse_args()
    config = dict(MNIST_CONFIG)
    config["seed"]           = args.seed
    config["dirichlet_alpha"] = args.alpha

    if args.quick:
        print("\n[Mode] QUICK RUN — 5 rounds, 5 clients")
        config.update({
            "total_rounds": 5, "num_clients": 5,
            "warmup_rounds": 2, "num_coalitions": 2,
            "nash_iterations": 2, "mfg_iterations": 2,
        })
    if args.rounds:
        config["total_rounds"] = args.rounds

    set_seed(config["seed"])
    device = get_device(config)
    print_config(config, device)

    selected = ([a.lower().replace("-", "") for a in args.algos]
                if args.algos else ["fedavg", "hedonicmfg"])

    # Per-alpha output folder — runs never overwrite each other
    run_dir = os.path.join(args.output, alpha_tag(args.alpha))
    os.makedirs(run_dir, exist_ok=True)
    results_path = os.path.join(run_dir, "results.json")

    # Load existing partial results so we can skip already-done algos
    existing = {}
    if os.path.exists(results_path):
        try:
            existing = load_results(results_path)
            done = list(existing.keys())
            print(f"\n[Runner] Found existing results for: {done}")
            selected = [a for a in selected if DISPLAY[a] not in done]
            if not selected:
                print("[Runner] All requested algorithms already done. "
                      "Delete results.json to re-run.")
                return
            print(f"[Runner] Running remaining: {[DISPLAY[a] for a in selected]}")
        except Exception:
            pass

    print(f"\n[Runner] Loading MNIST data (α={args.alpha}) ...")
    (client_train_loaders, client_val_loaders,
     global_test_loader, client_data_sizes) = get_mnist_client_loaders(config)
    config["client_data_sizes"] = client_data_sizes

    all_results = {}
    t_start     = time.time()

    for algo in selected:
        set_seed(config["seed"])
        t0 = time.time()
        print(f"\n[Runner] Starting {DISPLAY[algo]} ...")
        try:
            tracker = REGISTRY[algo](
                config, client_train_loaders, client_val_loaders,
                global_test_loader, device)
            all_results[DISPLAY[algo]] = tracker
            # Merge with any existing results and save incrementally
            merged = dict(existing)
            merged.update({k: v.to_dict() for k, v in all_results.items()})
            with open(results_path, "w") as f:
                import json; json.dump(merged, f, indent=2)
            print(f"  ✓ {DISPLAY[algo]} done in {(time.time()-t0)/60:.1f} min  "
                  f"→ saved to {results_path}")
        except Exception as e:
            print(f"  ✗ {DISPLAY[algo]} FAILED: {e}")
            import traceback; traceback.print_exc()

    # Print summary
    print(f"\n{'='*65}")
    print(f"  RESULTS — MNIST  α={args.alpha}")
    print(f"{'='*65}")
    print_final_table(all_results)
    print(f"  Total time: {(time.time()-t_start)/60:.1f} min")

    if "HedonicMFG" in all_results and "FedAvg" in all_results:
        hm = all_results["HedonicMFG"].final_summary()
        fa = all_results["FedAvg"].final_summary()
        dg = (hm["best_global_acc"]     - fa["best_global_acc"])     * 100
        dc = (hm["best_avg_client_acc"] - fa["best_avg_client_acc"]) * 100
        print(f"\n  HedonicMFG vs FedAvg:")
        print(f"  Global Δ: {'✓' if dg>0 else '✗'} {dg:+.2f}%   "
              f"Client Δ: {'✓' if dc>0 else '✗'} {dc:+.2f}%")

    # Per-run convergence plot for this alpha
    if not args.no_plots and all_results:
        generate_run_plots(all_results, "mnist", args.alpha, run_dir)

    print(f"\n[Done] α={args.alpha} results → {run_dir}/")
    print(f"       Run 'python combine_plots.py --datasets mnist' when all alphas are done.")


if __name__ == "__main__":
    main()
