"""
runner_adult.py — Main experiment runner for Adult Census FL comparison.
Runs all 10 algorithms and generates plots + summary table.

Usage:
    python runner_adult.py                        # full run (all algorithms)
    python runner_adult.py --quick                # 20 rounds, smoke test
    python runner_adult.py --algos hedonicmfg fedavg fedprox
    python runner_adult.py --seed 123
    python runner_adult.py --alpha 0.1            # override Dirichlet alpha
"""

import argparse, os, sys, time
import torch

sys.path.insert(0, os.path.dirname(__file__))

from config import ADULT_CONFIG
from data_adult import get_adult_client_loaders
from utils import set_seed, get_device, print_final_table, save_results
from visualize import generate_all_plots

from algorithms.adult.fedavg_adult            import run_fedavg_adult
from algorithms.adult.fedprox_adult           import run_fedprox_adult
from algorithms.adult.scaffold_adult          import run_scaffold_adult
from algorithms.adult.moon_adult              import run_moon_adult
from algorithms.adult.fedbn_adult             import run_fedbn_adult
from algorithms.adult.pfedme_adult            import run_pfedme_adult
from algorithms.adult.ifca_adult              import run_ifca_adult
from algorithms.adult.cfl_adult               import run_cfl_adult
from algorithms.adult.random_clustering_adult import run_random_clustering_adult
from algorithms.adult.hedonic_mfg_adult       import run_hedonic_mfg_adult


REGISTRY = {
    "fedavg":        run_fedavg_adult,
    "fedprox":       run_fedprox_adult,
    "scaffold":      run_scaffold_adult,
    "moon":          run_moon_adult,
    "fedbn":         run_fedbn_adult,
    "pfedme":        run_pfedme_adult,
    "ifca":          run_ifca_adult,
    "cfl":           run_cfl_adult,
    "randomcluster": run_random_clustering_adult,
    "hedonicmfg":    run_hedonic_mfg_adult,
}

DISPLAY = {
    "fedavg": "FedAvg", "fedprox": "FedProx", "scaffold": "SCAFFOLD",
    "moon": "MOON", "fedbn": "FedBN", "pfedme": "pFedME",
    "ifca": "IFCA", "cfl": "CFL", "randomcluster": "RandomCluster",
    "hedonicmfg": "HedonicMFG",
}


def parse_args():
    p = argparse.ArgumentParser(description="FL Experiment Runner — Adult Census")
    p.add_argument("--quick",  action="store_true", help="Smoke test: 20 rounds")
    p.add_argument("--algos",  nargs="+", default=None)
    p.add_argument("--seed",   type=int,   default=42)
    p.add_argument("--rounds", type=int,   default=None)
    p.add_argument("--alpha",  type=float, default=None, help="Dirichlet alpha override")
    p.add_argument("--output", type=str,   default="./results/adult")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def print_config(config, device):
    print("\n" + "╔" + "═"*54 + "╗")
    print("║  Adult Census Federated Learning Experiment        ║")
    print("╠" + "═"*54 + "╣")
    print(f"║  Clients        : {config['num_clients']:<35} ║")
    print(f"║  Rounds         : {config['total_rounds']:<35} ║")
    print(f"║  Dirichlet α    : {config['dirichlet_alpha']:<35} ║")
    print(f"║  Coalitions (K) : {config['num_coalitions']:<35} ║")
    print(f"║  Warmup rounds  : {config['warmup_rounds']:<35} ║")
    print(f"║  Recluster int. : {config['recluster_interval']:<35} ║")
    print(f"║  λ_fair (MFG)   : {config['lambda_fair_mfg']:<35} ║")
    print(f"║  γ_sync         : {config['gamma_sync']:<35} ║")
    print(f"║  Personal heads : {str(config['use_personalized_head']):<35} ║")
    print(f"║  Device         : {str(device):<35} ║")
    print(f"║  Seed           : {config['seed']:<35} ║")
    print("╚" + "═"*54 + "╝")


def main():
    args   = parse_args()
    config = dict(ADULT_CONFIG)
    config["seed"] = args.seed

    if args.quick:
        print("\n[Mode] QUICK RUN — 20 rounds, 10 clients")
        config.update({"total_rounds": 20, "num_clients": 10,
                        "warmup_rounds": 5, "num_coalitions": 2,
                        "random_clustering_K": 2, "ifca_num_clusters": 2,
                        "nash_iterations": 2, "mfg_iterations": 2})
    if args.rounds: config["total_rounds"] = args.rounds
    if args.alpha:  config["dirichlet_alpha"] = args.alpha

    set_seed(config["seed"])
    device = get_device(config)
    config["device"] = str(device)
    print_config(config, device)

    selected = [a.lower().replace("-","") for a in args.algos] if args.algos else \
        ["fedavg","fedprox","scaffold","moon","fedbn","pfedme",
         "ifca","cfl","randomcluster","hedonicmfg"]

    invalid = [a for a in selected if a not in REGISTRY]
    if invalid:
        print(f"[Error] Unknown: {invalid}. Valid: {list(REGISTRY.keys())}")
        sys.exit(1)

    print(f"\n[Runner] Loading Adult Census data ...")
    (client_train_loaders, client_val_loaders,
     global_test_loader, client_data_sizes, feature_dim) = \
        get_adult_client_loaders(config)
    config["client_data_sizes"] = client_data_sizes

    print(f"\n[Runner] Algorithms: {[DISPLAY[a] for a in selected]}")

    all_results = {}
    t_start = time.time()

    for algo in selected:
        set_seed(config["seed"])
        t0 = time.time()
        try:
            tracker = REGISTRY[algo](
                config, client_train_loaders, client_val_loaders,
                global_test_loader, device, feature_dim)
            all_results[DISPLAY[algo]] = tracker
            print(f"  ✓ {DISPLAY[algo]} done in {(time.time()-t0)/60:.1f} min")
        except Exception as e:
            print(f"  ✗ {DISPLAY[algo]} FAILED: {e}")
            import traceback; traceback.print_exc()

    # ── Final table ───────────────────────────────────────
    print(f"\n\n{'='*72}")
    print(f"  FINAL RESULTS — Adult Census (α={config['dirichlet_alpha']})")
    print(f"{'='*72}")
    print_final_table(all_results)
    print(f"\n  Total time: {(time.time()-t_start)/60:.1f} min")

    # ── HedonicMFG margin summary ─────────────────────────
    if "HedonicMFG" in all_results:
        hm = all_results["HedonicMFG"].final_summary()
        hg, hc = hm["best_global_acc"]*100, hm["best_avg_client_acc"]*100
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

    # ── Save + plot ───────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)
    save_results(all_results, os.path.join(args.output, "adult_results.json"))

    if not args.no_plots and all_results:
        print("\n[Plots] Generating visualizations ...")
        generate_all_plots(all_results, "adult_census", args.output)

    print(f"\n[Done] Results → {args.output}/")
    print("       adult_results.json | adult_census_convergence.png | "
          "adult_census_bar_comparison.png | adult_census_improvement.png")


if __name__ == "__main__":
    main()
