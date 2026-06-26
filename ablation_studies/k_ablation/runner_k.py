"""
runner_k.py — MNIST K (num_coalitions) ablation runner.

Runs FedAvg and HedonicMFG for a single K value.
Results saved to results/K_N/ for later combining.

Usage:
    python runner_k.py --k 2
    python runner_k.py --k 3
    python runner_k.py --k 4
    python runner_k.py --k 5
    python runner_k.py --k 6
    python runner_k.py --k 8

    # Run only one algorithm:
    python runner_k.py --k 4 --algos hedonicmfg
    python runner_k.py --k 4 --algos fedavg

    # Quick smoke test:
    python runner_k.py --k 3 --quick

    # Reuse existing K=3 result from alpha ablation:
    mkdir -p results/K_003
    cp ../ablation_study/results/mnist/alpha_03/results.json results/K_003/results.json

    # After all K values done:
    python combine_plots.py
"""

import argparse, os, sys, time, json
sys.path.insert(0, os.path.dirname(__file__))

from config import MNIST_CONFIG, ABLATION_K
from data import get_mnist_client_loaders
from utils import set_seed, get_device, print_final_table, load_results
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
    p = argparse.ArgumentParser(
        description="K ablation runner — MNIST")
    p.add_argument("--k",      type=int, required=True,
                   help=f"Number of coalitions. Suggested: {ABLATION_K}")
    p.add_argument("--algos",  nargs="+", default=None,
                   choices=["fedavg", "hedonicmfg"])
    p.add_argument("--quick",  action="store_true",
                   help="Smoke test: 5 rounds")
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--rounds", type=int, default=None)
    p.add_argument("--output", type=str, default="./results",
                   help="Base output dir. Saves to output/K_NNN/")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def k_tag(k: int) -> str:
    """Folder name for this K. e.g. 3 -> K_003"""
    return f"K_{k:03d}"


def make_config(k: int, seed: int) -> dict:
    """
    Build config for a given K.
    Only num_coalitions changes — everything else fixed.
    min_coalition_size adjusted so coalitions stay valid:
      floor(num_clients / K) - 1, capped between 2 and 5.
    """
    cfg = dict(MNIST_CONFIG)
    cfg["seed"]           = seed
    cfg["num_coalitions"] = k

    # Adjust min_coalition_size so it's always < clients_per_coalition
    # Prevents coalition formation from getting stuck at high K
    clients_per_coal = cfg["num_clients"] // k
    cfg["min_coalition_size"] = max(2, min(clients_per_coal - 1, 5))

    return cfg


def print_config(cfg, device):
    clients_per_coal = cfg["num_clients"] // cfg["num_coalitions"]
    print("\n" + "╔" + "═"*56 + "╗")
    print("║  K Ablation — MNIST                                   ║")
    print("╠" + "═"*56 + "╣")
    print(f"║  Num coalitions (K) : {cfg['num_coalitions']:<34} ║")
    print(f"║  Clients per coal.  : {clients_per_coal:<34} ║")
    print(f"║  Min coalition size : {cfg['min_coalition_size']:<34} ║")
    print(f"║  Num clients        : {cfg['num_clients']:<34} ║")
    print(f"║  Dirichlet α        : {cfg['dirichlet_alpha']:<34} ║")
    print(f"║  Total rounds       : {cfg['total_rounds']:<34} ║")
    print(f"║  Warmup rounds      : {cfg['warmup_rounds']:<34} ║")
    print(f"║  Recluster interval : {cfg['recluster_interval']:<34} ║")
    print(f"║  Device             : {str(device):<34} ║")
    print(f"║  Seed               : {cfg['seed']:<34} ║")
    print("╚" + "═"*56 + "╝")


def main():
    args = parse_args()

    if args.k not in ABLATION_K:
        print(f"[Warn] K={args.k} not in standard list {ABLATION_K}. "
              f"Continuing anyway.")

    cfg = make_config(args.k, args.seed)

    if args.quick:
        print("\n[Mode] QUICK RUN — 5 rounds")
        cfg.update({
            "total_rounds":  5,
            "warmup_rounds": 2,
            "nash_iterations": 2,
            "mfg_iterations":  2,
        })
    if args.rounds:
        cfg["total_rounds"] = args.rounds

    set_seed(cfg["seed"])
    device = get_device(cfg)
    print_config(cfg, device)

    selected = ([a.lower().replace("-", "") for a in args.algos]
                if args.algos else ["fedavg", "hedonicmfg"])

    # Per-K output folder
    run_dir      = os.path.join(args.output, k_tag(args.k))
    os.makedirs(run_dir, exist_ok=True)
    results_path = os.path.join(run_dir, "results.json")

    # Resume support — skip already completed algorithms
    existing = {}
    if os.path.exists(results_path):
        try:
            existing = load_results(results_path)
            done     = list(existing.keys())
            print(f"\n[Runner] Found existing results for: {done}")
            selected = [a for a in selected if DISPLAY[a] not in done]
            if not selected:
                print("[Runner] All algorithms done. "
                      "Delete results.json to re-run.")
                return
            print(f"[Runner] Running remaining: {[DISPLAY[a] for a in selected]}")
        except Exception:
            pass

    print(f"\n[Runner] Loading MNIST "
          f"(K={args.k}, N={cfg['num_clients']}, α={cfg['dirichlet_alpha']}) ...")
    (client_train_loaders, client_val_loaders,
     global_test_loader, client_data_sizes) = get_mnist_client_loaders(cfg)
    cfg["client_data_sizes"] = client_data_sizes

    all_results = {}
    t_start     = time.time()

    for algo in selected:
        set_seed(cfg["seed"])
        t0 = time.time()
        print(f"\n[Runner] Starting {DISPLAY[algo]} (K={args.k}) ...")
        try:
            tracker = REGISTRY[algo](
                cfg, client_train_loaders, client_val_loaders,
                global_test_loader, device)
            all_results[DISPLAY[algo]] = tracker

            # Merge with existing and save incrementally
            merged = dict(existing)
            merged.update({k: v.to_dict() for k, v in all_results.items()})
            with open(results_path, "w") as f:
                json.dump(merged, f, indent=2)
            elapsed = (time.time() - t0) / 60
            print(f"  ✓ {DISPLAY[algo]} done in {elapsed:.1f} min  "
                  f"→ {results_path}")
        except Exception as e:
            print(f"  ✗ {DISPLAY[algo]} FAILED: {e}")
            import traceback; traceback.print_exc()

    # Summary
    print(f"\n{'='*65}")
    print(f"  RESULTS — MNIST  K={args.k} coalitions")
    print(f"{'='*65}")
    print_final_table(all_results)
    print(f"  Total time: {(time.time()-t_start)/60:.1f} min")

    if "HedonicMFG" in all_results and "FedAvg" in all_results:
        hm = all_results["HedonicMFG"].final_summary()
        fa = all_results["FedAvg"].final_summary()
        dg = (hm["best_global_acc"]     - fa["best_global_acc"])     * 100
        dc = (hm["best_avg_client_acc"] - fa["best_avg_client_acc"]) * 100
        print(f"\n  HedonicMFG vs FedAvg (K={args.k}):")
        print(f"  Global Δ: {'✓' if dg>0 else '✗'} {dg:+.2f}%   "
              f"Client Δ: {'✓' if dc>0 else '✗'} {dc:+.2f}%")

    if not args.no_plots and all_results:
        generate_run_plots(all_results, "mnist",
                           label=f"K={args.k}",
                           save_dir=run_dir)

    print(f"\n[Done] K={args.k} → {run_dir}/")
    print("       Run 'python combine_plots.py' when all K values are done.")


if __name__ == "__main__":
    main()
