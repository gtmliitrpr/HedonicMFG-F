"""
runner_clients.py — FashionMNIST client count ablation runner.

Runs FedAvg and HedonicMFG for a single num_clients value.
Results saved to results/clients_N/ for later combining.

Usage:
    python runner_clients.py --clients 10
    python runner_clients.py --clients 20
    python runner_clients.py --clients 30
    python runner_clients.py --clients 50
    python runner_clients.py --clients 75
    python runner_clients.py --clients 100

    # Run only one algorithm:
    python runner_clients.py --clients 50 --algos hedonicmfg
    python runner_clients.py --clients 50 --algos fedavg

    # Quick smoke test:
    python runner_clients.py --clients 10 --quick

    # After all client counts done, combine:
    python combine_plots.py
"""

import argparse, os, sys, time, json
sys.path.insert(0, os.path.dirname(__file__))

from config import FMNIST_CONFIG, ABLATION_CLIENTS
from data_fmnist import get_fmnist_client_loaders
from utils import set_seed, get_device, print_final_table, load_results
from visualize import generate_run_plots

from algorithms.fmnist.fedavg_fmnist      import run_fedavg_fmnist
from algorithms.fmnist.hedonic_mfg_fmnist import run_hedonic_mfg_fmnist


REGISTRY = {
    "fedavg":     run_fedavg_fmnist,
    "hedonicmfg": run_hedonic_mfg_fmnist,
}
DISPLAY = {
    "fedavg":     "FedAvg",
    "hedonicmfg": "HedonicMFG",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Client ablation runner — FashionMNIST")
    p.add_argument("--clients", type=int, required=True,
                   help=f"Number of clients. Suggested: {ABLATION_CLIENTS}")
    p.add_argument("--algos",   nargs="+", default=None,
                   choices=["fedavg", "hedonicmfg"])
    p.add_argument("--quick",   action="store_true",
                   help="Smoke test: 5 rounds")
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--rounds",  type=int, default=None)
    p.add_argument("--output",  type=str, default="./results",
                   help="Base output dir. Saves to output/clients_N/")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def clients_tag(n: int) -> str:
    """Folder name for this client count. e.g. 10 -> clients_010"""
    return f"clients_{n:03d}"


def make_config(num_clients: int, seed: int) -> dict:
    """
    Build config for a given num_clients.
    Scales warmup_rounds and min_coalition_size sensibly.
    All MFG weights and training HPs stay fixed.
    """
    cfg = dict(FMNIST_CONFIG)
    cfg["seed"]        = seed
    cfg["num_clients"] = num_clients

    # Scale warmup: always ~20-25% of total rounds, min 5
    cfg["warmup_rounds"] = max(5, int(cfg["total_rounds"] * 0.22))

    # Scale min_coalition_size: floor(N/K) - 1, minimum 2
    # Prevents degenerate coalitions at very high client counts
    clients_per_coalition = num_clients // cfg["num_coalitions"]
    cfg["min_coalition_size"] = max(2, min(clients_per_coalition - 1, 8))

    return cfg


def print_config(cfg, device):
    print("\n" + "╔" + "═"*56 + "╗")
    print("║  Client Ablation — FashionMNIST                       ║")
    print("╠" + "═"*56 + "╣")
    print(f"║  Num clients    : {cfg['num_clients']:<37} ║")
    print(f"║  Coalitions (K) : {cfg['num_coalitions']:<37} ║")
    print(f"║  Min coal size  : {cfg['min_coalition_size']:<37} ║")
    print(f"║  Dirichlet α    : {cfg['dirichlet_alpha']:<37} ║")
    print(f"║  Total rounds   : {cfg['total_rounds']:<37} ║")
    print(f"║  Warmup rounds  : {cfg['warmup_rounds']:<37} ║")
    print(f"║  Recluster int. : {cfg['recluster_interval']:<37} ║")
    print(f"║  Device         : {str(device):<37} ║")
    print(f"║  Seed           : {cfg['seed']:<37} ║")
    print("╚" + "═"*56 + "╝")


def main():
    args = parse_args()

    if args.clients not in ABLATION_CLIENTS:
        print(f"[Warn] {args.clients} not in standard list {ABLATION_CLIENTS}. "
              f"Continuing anyway.")

    cfg = make_config(args.clients, args.seed)

    if args.quick:
        print("\n[Mode] QUICK RUN — 5 rounds")
        cfg.update({
            "total_rounds":  5,
            "warmup_rounds": 2,
            "num_coalitions": 2,
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

    # Per-clients output folder
    run_dir      = os.path.join(args.output, clients_tag(args.clients))
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

    print(f"\n[Runner] Loading FashionMNIST "
          f"(N={args.clients}, α={cfg['dirichlet_alpha']}) ...")
    (client_train_loaders, client_val_loaders,
     global_test_loader, client_data_sizes) = get_fmnist_client_loaders(cfg)
    cfg["client_data_sizes"] = client_data_sizes

    all_results = {}
    t_start     = time.time()

    for algo in selected:
        set_seed(cfg["seed"])
        t0 = time.time()
        print(f"\n[Runner] Starting {DISPLAY[algo]} (N={args.clients}) ...")
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
    print(f"  RESULTS — FashionMNIST  N={args.clients} clients")
    print(f"{'='*65}")
    print_final_table(all_results)
    print(f"  Total time: {(time.time()-t_start)/60:.1f} min")

    if "HedonicMFG" in all_results and "FedAvg" in all_results:
        hm = all_results["HedonicMFG"].final_summary()
        fa = all_results["FedAvg"].final_summary()
        dg = (hm["best_global_acc"]     - fa["best_global_acc"])     * 100
        dc = (hm["best_avg_client_acc"] - fa["best_avg_client_acc"]) * 100
        print(f"\n  HedonicMFG vs FedAvg (N={args.clients}):")
        print(f"  Global Δ: {'✓' if dg>0 else '✗'} {dg:+.2f}%   "
              f"Client Δ: {'✓' if dc>0 else '✗'} {dc:+.2f}%")

    if not args.no_plots and all_results:
        generate_run_plots(all_results, "fmnist",
                           label=f"N={args.clients}",
                           save_dir=run_dir)

    print(f"\n[Done] N={args.clients} → {run_dir}/")
    print("       Run 'python combine_plots.py' when all client counts are done.")


if __name__ == "__main__":
    main()
