"""
runner_fmnist.py — Main experiment runner for FashionMNIST FL comparison.

Usage:
    python runner_fmnist.py                        # full run
    python runner_fmnist.py --quick               # smoke test
    python runner_fmnist.py --algos hedonicmfg fedavg
    python runner_fmnist.py --rounds 60
"""

import argparse, os, sys, time
import torch

sys.path.insert(0, os.path.dirname(__file__))

from config import FMNIST_CONFIG
from data_fmnist import get_fmnist_client_loaders
from utils import set_seed, get_device, print_final_table, save_results
from visualize import generate_all_plots

from algorithms.fmnist.fedavg_fmnist            import run_fedavg_fmnist
from algorithms.fmnist.fedprox_fmnist           import run_fedprox_fmnist
from algorithms.fmnist.scaffold_fmnist          import run_scaffold_fmnist
from algorithms.fmnist.moon_fmnist              import run_moon_fmnist
from algorithms.fmnist.fedbn_fmnist             import run_fedbn_fmnist
from algorithms.fmnist.pfedme_fmnist            import run_pfedme_fmnist
from algorithms.fmnist.ifca_fmnist              import run_ifca_fmnist
from algorithms.fmnist.cfl_fmnist               import run_cfl_fmnist
from algorithms.fmnist.random_clustering_fmnist import run_random_clustering_fmnist
from algorithms.fmnist.hedonic_mfg_fmnist       import run_hedonic_mfg_fmnist


REGISTRY = {
    "fedavg":        run_fedavg_fmnist,
    "fedprox":       run_fedprox_fmnist,
    "scaffold":      run_scaffold_fmnist,
    "moon":          run_moon_fmnist,
    "fedbn":         run_fedbn_fmnist,
    "pfedme":        run_pfedme_fmnist,
    "ifca":          run_ifca_fmnist,
    "cfl":           run_cfl_fmnist,
    "randomcluster": run_random_clustering_fmnist,
    "hedonicmfg":    run_hedonic_mfg_fmnist,
}

DISPLAY = {
    "fedavg": "FedAvg", "fedprox": "FedProx", "scaffold": "SCAFFOLD",
    "moon": "MOON", "fedbn": "FedBN", "pfedme": "pFedME",
    "ifca": "IFCA", "cfl": "CFL", "randomcluster": "RandomCluster",
    "hedonicmfg": "HedonicMFG",
}


def parse_args():
    p = argparse.ArgumentParser(description="FL Runner — FashionMNIST")
    p.add_argument("--quick",    action="store_true")
    p.add_argument("--algos",    nargs="+", default=None)
    p.add_argument("--seed",     type=int,   default=42)
    p.add_argument("--rounds",   type=int,   default=None)
    p.add_argument("--alpha",    type=float, default=None)
    p.add_argument("--output",   type=str,   default="./results/fmnist")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def print_config(config, device):
    print("\n" + "╔" + "═"*54 + "╗")
    print("║  FashionMNIST Federated Learning Experiment       ║")
    print("╠" + "═"*54 + "╣")
    print(f"║  Clients        : {config['num_clients']:<35} ║")
    print(f"║  Rounds         : {config['total_rounds']:<35} ║")
    print(f"║  Dirichlet α    : {config['dirichlet_alpha']:<35} ║")
    print(f"║  Coalitions (K) : {config['num_coalitions']:<35} ║")
    print(f"║  Warmup rounds  : {config['warmup_rounds']:<35} ║")
    print(f"║  Recluster int. : {config['recluster_interval']:<35} ║")
    print(f"║  λ_fair (MFG)   : {config['lambda_fair_mfg']:<35} ║")
    print(f"║  γ_grad         : {config['gamma_grad']:<35} ║")
    print(f"║  Personal heads : {str(config['use_personalized_head']):<35} ║")
    print(f"║  Device         : {str(device):<35} ║")
    print(f"║  Seed           : {config['seed']:<35} ║")
    print("╚" + "═"*54 + "╝")


def main():
    args   = parse_args()
    config = dict(FMNIST_CONFIG)
    config["seed"] = args.seed

    if args.quick:
        print("\n[Mode] QUICK RUN — 10 rounds, 10 clients")
        config.update({
            "total_rounds": 10, "num_clients": 10,
            "warmup_rounds": 3, "num_coalitions": 2,
            "random_clustering_K": 2, "ifca_num_clusters": 2,
            "nash_iterations": 2, "mfg_iterations": 2,
        })
    if args.rounds: config["total_rounds"] = args.rounds
    if args.alpha:  config["dirichlet_alpha"] = args.alpha

    set_seed(config["seed"])
    device = get_device(config)
    config["device"] = str(device)
    print_config(config, device)

    selected = ([a.lower().replace("-","") for a in args.algos] if args.algos else
                ["fedavg","fedprox","scaffold","moon","fedbn","pfedme",
                 "ifca","cfl","randomcluster","hedonicmfg"])

    invalid = [a for a in selected if a not in REGISTRY]
    if invalid:
        print(f"[Error] Unknown: {invalid}. Valid: {list(REGISTRY.keys())}")
        sys.exit(1)

    print(f"\n[Runner] Loading FashionMNIST data ...")
    (client_train_loaders, client_val_loaders,
     global_test_loader, client_data_sizes) = get_fmnist_client_loaders(config)
    config["client_data_sizes"] = client_data_sizes

    print(f"\n[Runner] Algorithms: {[DISPLAY[a] for a in selected]}")

    all_results  = {}
    os.makedirs(args.output, exist_ok=True)
    results_path = os.path.join(args.output, "fmnist_results.json")
    t_start      = time.time()

    for algo in selected:
        set_seed(config["seed"])
        t0 = time.time()
        try:
            tracker = REGISTRY[algo](
                config, client_train_loaders, client_val_loaders,
                global_test_loader, device)
            all_results[DISPLAY[algo]] = tracker
            save_results(all_results, results_path)  # incremental save
            print(f"  ✓ {DISPLAY[algo]} done in {(time.time()-t0)/60:.1f} min")
        except Exception as e:
            print(f"  ✗ {DISPLAY[algo]} FAILED: {e}")
            import traceback; traceback.print_exc()

    # Final summary
    print(f"\n\n{'='*72}")
    print(f"  FINAL RESULTS — FashionMNIST (α={config['dirichlet_alpha']})")
    print(f"{'='*72}")
    print_final_table(all_results)
    print(f"\n  Total time: {(time.time()-t_start)/60:.1f} min")

    if "HedonicMFG" in all_results:
        hm = all_results["HedonicMFG"].final_summary()
        hg = hm["best_global_acc"] * 100
        hc = hm["best_avg_client_acc"] * 100
        print(f"\n{'─'*60}")
        print(f"  HedonicMFG vs Baselines:")
        print(f"{'─'*60}")
        for name, tracker in all_results.items():
            if name == "HedonicMFG": continue
            s  = tracker.final_summary()
            dg = hg - s["best_global_acc"]*100
            dc = hc - s["best_avg_client_acc"]*100
            print(f"  vs {name:<16}  "
                  f"Global: {'✓' if dg>0 else '✗'} {dg:+.2f}%   "
                  f"Client: {'✓' if dc>0 else '✗'} {dc:+.2f}%")
        print(f"{'─'*60}")

    save_results(all_results, results_path)
    if not args.no_plots and all_results:
        print("\n[Plots] Generating visualizations ...")
        generate_all_plots(all_results, "fmnist", args.output)

    print(f"\n[Done] Results → {args.output}/")
    print("       fmnist_results.json | fmnist_convergence.png | "
          "fmnist_bar_comparison.png | fmnist_improvement.png")


if __name__ == "__main__":
    main()
