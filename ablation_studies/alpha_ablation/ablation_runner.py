"""
ablation_runner.py — Dirichlet α ablation study: FedAvg vs HedonicMFG
                     on MNIST and FashionMNIST.

Usage:
    python ablation_runner.py                              # full ablation
    python ablation_runner.py --quick                      # smoke test
    python ablation_runner.py --datasets mnist             # only MNIST
    python ablation_runner.py --datasets fmnist            # only FashionMNIST
    python ablation_runner.py --alphas 0.1 0.3 1.0        # custom alpha values
    python ablation_runner.py --algos hedonicmfg           # only HedonicMFG
    python ablation_runner.py --rounds 30                  # override rounds
"""

import argparse, os, sys, time, json
import torch

sys.path.insert(0, os.path.dirname(__file__))

from config import MNIST_CONFIG, FMNIST_CONFIG, ABLATION_ALPHAS
from data import get_mnist_client_loaders
from data_fmnist import get_fmnist_client_loaders
from utils import set_seed, get_device, save_results, ResultsTracker

from algorithms.mnist.fedavg_mnist         import run_fedavg_mnist
from algorithms.mnist.hedonic_mfg_mnist    import run_hedonic_mfg_mnist
from algorithms.fmnist.fedavg_fmnist       import run_fedavg_fmnist
from algorithms.fmnist.hedonic_mfg_fmnist  import run_hedonic_mfg_fmnist

from visualize import (generate_run_plots, generate_ablation_plots,
                        generate_all_plots, plot_combined_ablation)


# ──────────────────────────────────────────
# Registry
# ──────────────────────────────────────────
REGISTRY = {
    "mnist": {
        "fedavg":     run_fedavg_mnist,
        "hedonicmfg": run_hedonic_mfg_mnist,
    },
    "fmnist": {
        "fedavg":     run_fedavg_fmnist,
        "hedonicmfg": run_hedonic_mfg_fmnist,
    },
}

DATA_LOADERS = {
    "mnist":  get_mnist_client_loaders,
    "fmnist": get_fmnist_client_loaders,
}

BASE_CONFIGS = {
    "mnist":  MNIST_CONFIG,
    "fmnist": FMNIST_CONFIG,
}

DISPLAY = {
    "fedavg":     "FedAvg",
    "hedonicmfg": "HedonicMFG",
}


# ──────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Ablation Runner — Dirichlet α")
    p.add_argument("--quick",    action="store_true",
                   help="Smoke test: 10 rounds, 10 clients")
    p.add_argument("--datasets", nargs="+", default=["mnist", "fmnist"],
                   choices=["mnist", "fmnist"])
    p.add_argument("--algos",    nargs="+", default=["fedavg", "hedonicmfg"],
                   choices=["fedavg", "hedonicmfg"])
    p.add_argument("--alphas",   nargs="+", type=float,
                   default=None,
                   help="Dirichlet alpha values to sweep. "
                        "Default: [0.05, 0.1, 0.3, 0.5, 1.0]")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--rounds",   type=int, default=None,
                   help="Override total_rounds from config")
    p.add_argument("--output",   type=str, default="./results/ablation")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


# ──────────────────────────────────────────
# Pretty header
# ──────────────────────────────────────────
def print_header(datasets, alphas, algos, device, quick):
    print("\n" + "╔" + "═"*58 + "╗")
    print("║  Ablation Study: Dirichlet α Sweep                      ║")
    print("║  FedAvg vs HedonicMFG                                   ║")
    print("╠" + "═"*58 + "╣")
    print(f"║  Datasets  : {str(datasets):<44} ║")
    print(f"║  Algorithms: {str([DISPLAY[a] for a in algos]):<44} ║")
    print(f"║  α values  : {str(alphas):<44} ║")
    print(f"║  Device    : {str(device):<44} ║")
    mode = "QUICK (smoke test)" if quick else "FULL"
    print(f"║  Mode      : {mode:<44} ║")
    print("╚" + "═"*58 + "╝\n")


# ──────────────────────────────────────────
# Single (dataset, alpha) run
# ──────────────────────────────────────────
def run_one(dataset, alpha, algos, config, device, output_dir):
    """
    Run FedAvg and/or HedonicMFG on one (dataset, alpha) configuration.
    Returns dict: {"FedAvg": tracker, "HedonicMFG": tracker}
    """
    cfg = dict(config)
    cfg["dirichlet_alpha"] = alpha

    print(f"\n{'─'*60}")
    print(f"  Dataset: {dataset.upper()}  |  α = {alpha}")
    print(f"{'─'*60}")

    set_seed(cfg["seed"])
    loader_fn = DATA_LOADERS[dataset]
    (client_train_loaders, client_val_loaders,
     global_test_loader, client_data_sizes) = loader_fn(cfg)
    cfg["client_data_sizes"] = client_data_sizes

    results = {}
    for algo in algos:
        set_seed(cfg["seed"])
        t0 = time.time()
        try:
            tracker = REGISTRY[dataset][algo](
                cfg, client_train_loaders, client_val_loaders,
                global_test_loader, device)
            results[DISPLAY[algo]] = tracker
            elapsed = (time.time() - t0) / 60
            print(f"  ✓ {DISPLAY[algo]} done in {elapsed:.1f} min")
        except Exception as e:
            print(f"  ✗ {DISPLAY[algo]} FAILED: {e}")
            import traceback; traceback.print_exc()

    # Per-run incremental save
    alpha_tag   = str(alpha).replace(".", "")
    run_dir     = os.path.join(output_dir, dataset)
    os.makedirs(run_dir, exist_ok=True)
    run_path    = os.path.join(run_dir, f"run_alpha{alpha_tag}.json")
    save_results(results, run_path)

    # Per-run plots
    generate_run_plots(results, dataset, alpha, run_dir)

    return results


# ──────────────────────────────────────────
# Summary table
# ──────────────────────────────────────────
def print_ablation_table(ablation_results: dict, dataset: str, alphas: list):
    print(f"\n{'='*72}")
    print(f"  ABLATION RESULTS — {dataset.upper()}")
    print(f"{'='*72}")
    print(f"  {'α':<6} {'Algorithm':<14} "
          f"{'Best Global':>12} {'Final Global':>13} "
          f"{'Best Client':>12} {'Final Client':>13}")
    print(f"  {'-'*68}")
    for alpha in alphas:
        run = ablation_results.get(alpha, {})
        for algo_name in ["FedAvg", "HedonicMFG"]:
            tracker = run.get(algo_name)
            if tracker is None:
                continue
            s = tracker.final_summary()
            print(f"  {alpha:<6} {algo_name:<14} "
                  f"{s['best_global_acc']*100:>11.2f}%  "
                  f"{s['final_global_acc']*100:>12.2f}%  "
                  f"{s['best_avg_client_acc']*100:>11.2f}%  "
                  f"{s['final_avg_client_acc']*100:>12.2f}%")
    print(f"{'='*72}")


def print_gain_table(ablation_results: dict, dataset: str, alphas: list):
    """Print HedonicMFG vs FedAvg gain for each alpha."""
    print(f"\n  HedonicMFG Gain over FedAvg — {dataset.upper()}")
    print(f"  {'α':<6}  {'ΔGlobal':>10}  {'ΔClient':>10}")
    print(f"  {'-'*30}")
    for alpha in alphas:
        run = ablation_results.get(alpha, {})
        hm  = run.get("HedonicMFG")
        fa  = run.get("FedAvg")
        if hm is None or fa is None:
            continue
        hm_s = hm.final_summary()
        fa_s = fa.final_summary()
        dg   = (hm_s["best_global_acc"] - fa_s["best_global_acc"]) * 100
        dc   = (hm_s["best_avg_client_acc"] - fa_s["best_avg_client_acc"]) * 100
        tag_g = "✓" if dg > 0 else "✗"
        tag_c = "✓" if dc > 0 else "✗"
        print(f"  {alpha:<6}  {tag_g} {dg:>+7.2f}%   {tag_c} {dc:>+7.2f}%")
    print()


# ──────────────────────────────────────────
# Main
# ──────────────────────────────────────────
def main():
    args   = parse_args()
    alphas = args.alphas if args.alphas else ABLATION_ALPHAS

    set_seed(args.seed)
    device = get_device({})

    # Apply quick-mode overrides
    quick_overrides = {}
    if args.quick:
        print("\n[Mode] QUICK RUN — 10 rounds, 10 clients, alphas=[0.1, 0.3, 1.0]")
        quick_overrides = {
            "total_rounds": 10,
            "num_clients": 10,
            "warmup_rounds": 3,
            "num_coalitions": 2,
            "nash_iterations": 2,
            "mfg_iterations": 2,
        }
        alphas = [0.1, 0.3, 1.0]

    if args.rounds:
        quick_overrides["total_rounds"] = args.rounds

    print_header(args.datasets, alphas, args.algos, device, args.quick)

    t_total       = time.time()
    all_ablation  = {}  # {dataset: {alpha: {algo: tracker}}}
    os.makedirs(args.output, exist_ok=True)

    for dataset in args.datasets:
        config = dict(BASE_CONFIGS[dataset])
        config["seed"]   = args.seed
        config["device"] = str(device)
        config.update(quick_overrides)

        ablation_results = {}  # {alpha: {algo_display: tracker}}

        for alpha in alphas:
            results = run_one(dataset, alpha, args.algos,
                               config, device, args.output)
            ablation_results[alpha] = results

        all_ablation[dataset] = ablation_results

        # Summary tables
        print_ablation_table(ablation_results, dataset, alphas)
        print_gain_table(ablation_results, dataset, alphas)

        # Ablation plots
        if not args.no_plots:
            ds_dir = os.path.join(args.output, dataset)
            generate_ablation_plots(ablation_results, dataset, ds_dir)

        # Save full ablation JSON for this dataset
        out_path = os.path.join(args.output, dataset,
                                f"{dataset}_ablation_full.json")
        serializable = {}
        for alpha, run in ablation_results.items():
            serializable[str(alpha)] = {
                k: (v.to_dict() if hasattr(v, "to_dict") else v)
                for k, v in run.items()
            }
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"[Saved] {out_path}")

    # Combined dataset plot
    if not args.no_plots and len(all_ablation) == 2:
        plot_combined_ablation(all_ablation, args.output)

    total_min = (time.time() - t_total) / 60
    print(f"\n{'='*60}")
    print(f"  Ablation complete. Total time: {total_min:.1f} min")
    print(f"  Results → {args.output}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
