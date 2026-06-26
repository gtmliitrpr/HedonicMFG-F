"""
runner_imdb.py — Main experiment runner for IMDB FL comparison.

Usage:
    python runner_imdb.py                        # full run
    python runner_imdb.py --quick               # 5 rounds smoke test
    python runner_imdb.py --algos hedonicmfg fedavg
    python runner_imdb.py --rounds 50
"""

import argparse, os, sys, time
import torch

sys.path.insert(0, os.path.dirname(__file__))

from config import IMDB_CONFIG
from data_imdb import get_imdb_client_loaders
from utils import set_seed, get_device, print_final_table, save_results
from visualize import generate_all_plots

from algorithms.imdb.fedavg_imdb            import run_fedavg_imdb
from algorithms.imdb.fedprox_imdb           import run_fedprox_imdb
from algorithms.imdb.scaffold_imdb          import run_scaffold_imdb
from algorithms.imdb.moon_imdb              import run_moon_imdb
from algorithms.imdb.fedbn_imdb             import run_fedbn_imdb
from algorithms.imdb.pfedme_imdb            import run_pfedme_imdb
from algorithms.imdb.ifca_imdb              import run_ifca_imdb
from algorithms.imdb.cfl_imdb              import run_cfl_imdb
from algorithms.imdb.random_clustering_imdb import run_random_clustering_imdb
from algorithms.imdb.hedonic_mfg_imdb       import run_hedonic_mfg_imdb


REGISTRY = {
    "fedavg":        run_fedavg_imdb,
    "fedprox":       run_fedprox_imdb,
    "scaffold":      run_scaffold_imdb,
    "moon":          run_moon_imdb,
    "fedbn":         run_fedbn_imdb,
    "pfedme":        run_pfedme_imdb,
    "ifca":          run_ifca_imdb,
    "cfl":           run_cfl_imdb,
    "randomcluster": run_random_clustering_imdb,
    "hedonicmfg":    run_hedonic_mfg_imdb,
}

DISPLAY = {
    "fedavg": "FedAvg", "fedprox": "FedProx", "scaffold": "SCAFFOLD",
    "moon": "MOON", "fedbn": "FedBN", "pfedme": "pFedME",
    "ifca": "IFCA", "cfl": "CFL", "randomcluster": "RandomCluster",
    "hedonicmfg": "HedonicMFG",
}


def parse_args():
    p = argparse.ArgumentParser(description="FL Runner — IMDB")
    p.add_argument("--quick",   action="store_true")
    p.add_argument("--algos",   nargs="+", default=None)
    p.add_argument("--seed",    type=int,   default=42)
    p.add_argument("--rounds",  type=int,   default=None)
    p.add_argument("--alpha",   type=float, default=None)
    p.add_argument("--output",  type=str,   default="./results/imdb")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def print_config(config, device):
    print("\n" + "╔" + "═"*54 + "╗")
    print("║  IMDB Federated Learning Experiment               ║")
    print("╠" + "═"*54 + "╣")
    print(f"║  Clients        : {config['num_clients']:<35} ║")
    print(f"║  Rounds         : {config['total_rounds']:<35} ║")
    print(f"║  Dirichlet α    : {config['dirichlet_alpha']:<35} ║")
    print(f"║  Coalitions (K) : {config['num_coalitions']:<35} ║")
    print(f"║  Warmup rounds  : {config['warmup_rounds']:<35} ║")
    print(f"║  Recluster int. : {config['recluster_interval']:<35} ║")
    print(f"║  λ_fair (MFG)   : {config['lambda_fair_mfg']:<35} ║")
    print(f"║  Max seq len    : {config['max_len']:<35} ║")
    print(f"║  Vocab size     : {config['vocab_size']:<35} ║")
    print(f"║  Personal heads : {str(config['use_personalized_head']):<35} ║")
    print(f"║  Device         : {str(device):<35} ║")
    print(f"║  Seed           : {config['seed']:<35} ║")
    print("╚" + "═"*54 + "╝")


def main():
    args   = parse_args()
    config = dict(IMDB_CONFIG)
    config["seed"] = args.seed

    if args.quick:
        print("\n[Mode] QUICK RUN — 5 rounds, 10 clients")
        config.update({
            "total_rounds": 5, "num_clients": 10,
            "warmup_rounds": 2, "num_coalitions": 2,
            "random_clustering_K": 2, "ifca_num_clusters": 2,
            "nash_iterations": 2, "mfg_iterations": 2,
            "vocab_size": 5000, "max_len": 128,
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
        print(f"[Error] Unknown: {invalid}")
        sys.exit(1)

    print(f"\n[Runner] Loading IMDB data ...")
    (client_train_loaders, client_val_loaders,
     global_test_loader, client_data_sizes, vocab_size) = \
        get_imdb_client_loaders(config)
    config["client_data_sizes"] = client_data_sizes

    print(f"\n[Runner] Algorithms: {[DISPLAY[a] for a in selected]}")

    all_results = {}
    os.makedirs(args.output, exist_ok=True)
    results_path = os.path.join(args.output, "imdb_results.json")
    t_start = time.time()

    for algo in selected:
        set_seed(config["seed"])
        t0 = time.time()
        try:
            tracker = REGISTRY[algo](
                config, client_train_loaders, client_val_loaders,
                global_test_loader, device, vocab_size)
            all_results[DISPLAY[algo]] = tracker
            # Save incrementally after each algorithm
            save_results(all_results, results_path)
            print(f"  ✓ {DISPLAY[algo]} done in {(time.time()-t0)/60:.1f} min")
        except Exception as e:
            print(f"  ✗ {DISPLAY[algo]} FAILED: {e}")
            import traceback; traceback.print_exc()

    # Final summary
    print(f"\n\n{'='*72}")
    print(f"  FINAL RESULTS — IMDB (α={config['dirichlet_alpha']})")
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
        generate_all_plots(all_results, "imdb", args.output)

    print(f"\n[Done] Results → {args.output}/")


if __name__ == "__main__":
    main()
