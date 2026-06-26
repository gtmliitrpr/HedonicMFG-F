"""
combine_plots.py — Load per-alpha results and generate combined ablation plots.

Run this AFTER all individual alpha runs are complete.

Usage:
    python combine_plots.py                          # both datasets
    python combine_plots.py --datasets mnist         # MNIST only
    python combine_plots.py --datasets fmnist        # FashionMNIST only
    python combine_plots.py --datasets mnist fmnist  # both (default)
    python combine_plots.py --alphas 0.1 0.3 0.5    # custom alpha subset
    python combine_plots.py --output ./results       # custom base dir

Folder structure expected (created by runners):
    results/
      mnist/
        alpha_005/results.json
        alpha_01/results.json
        alpha_03/results.json
        alpha_05/results.json
        alpha_10/results.json
      fmnist/
        alpha_005/results.json
        ...

Plots generated:
    results/mnist/
        mnist_convergence_all_alphas.png     ← convergence per algo, one line per alpha
        mnist_accuracy_vs_alpha.png          ← best acc vs alpha (both algos)
        mnist_gain_vs_alpha.png              ← HedonicMFG - FedAvg gain per alpha
    results/fmnist/
        (same set)
    results/
        combined_summary.png                 ← 2x3 grid: both datasets side by side
"""

import argparse, os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from config import ABLATION_ALPHAS


# ──────────────────────────────────────────
# Colors / styles
# ──────────────────────────────────────────
ALGO_COLORS = {
    "HedonicMFG": "#D62728",
    "FedAvg":     "#1F77B4",
}
ALGO_LS = {
    "HedonicMFG": "-",
    "FedAvg":     "--",
}
ALGO_LW = {
    "HedonicMFG": 2.5,
    "FedAvg":     1.8,
}
ALGO_MARKER = {
    "HedonicMFG": "o",
    "FedAvg":     "s",
}

# One color per alpha value
ALPHA_COLORS = {
    0.05: "#D62728",
    0.1:  "#FF7F0E",
    0.3:  "#2CA02C",
    0.5:  "#1F77B4",
    1.0:  "#9467BD",
}


def alpha_tag(alpha: float) -> str:
    return "alpha_" + str(alpha).replace(".", "")


def smooth(values, window=5):
    if len(values) <= window:
        return np.array(values)
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(values)]


# ──────────────────────────────────────────
# Load all results for one dataset
# ──────────────────────────────────────────
def load_dataset_results(dataset: str, alphas: list, base_dir: str) -> dict:
    """
    Returns: {alpha: {"FedAvg": dict, "HedonicMFG": dict}}
    where each inner dict has keys: global_accs, avg_client_accs, round_times
    """
    ds_dir  = os.path.join(base_dir, dataset)
    results = {}
    missing = []

    for alpha in alphas:
        path = os.path.join(ds_dir, alpha_tag(alpha), "results.json")
        if not os.path.exists(path):
            missing.append(alpha)
            continue
        with open(path) as f:
            data = json.load(f)
        results[alpha] = data
        algos_found = list(data.keys())
        print(f"  [Load] {dataset} α={alpha}: found {algos_found}")

    if missing:
        print(f"  [Warn] {dataset}: missing results for α={missing}")
        print(f"         Run: python runner_{dataset}.py --alpha <value>")

    return results


# ──────────────────────────────────────────
# PLOT 1 — Convergence curves per alpha
# One figure per algorithm, one line per alpha
# ──────────────────────────────────────────
def plot_convergence_per_alpha(dataset_results: dict, dataset: str,
                                out_dir: str):
    """
    2 figures (one per algorithm), each with:
      left panel  = global test accuracy over rounds
      right panel = avg client accuracy over rounds
    One line per alpha value.
    """
    alphas  = sorted(dataset_results.keys())
    algos   = ["FedAvg", "HedonicMFG"]
    ds_label = "MNIST" if dataset == "mnist" else "FashionMNIST"

    for algo in algos:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(f"{algo} — Convergence at Different α\n{ds_label}",
                     fontsize=13, fontweight="bold", y=1.02)

        keys   = ["global_accs", "avg_client_accs"]
        titles = ["Global Test Accuracy (%)", "Avg Client Accuracy (%)"]

        for ax, key, title in zip(axes, keys, titles):
            for alpha in alphas:
                run     = dataset_results.get(alpha, {})
                tracker = run.get(algo)
                if tracker is None:
                    continue
                values   = [v * 100 for v in tracker[key]]
                smoothed = smooth(values)
                rounds   = list(range(1, len(values) + 1))
                color    = ALPHA_COLORS.get(alpha, "#333")

                ax.plot(rounds, smoothed,
                        label=f"α={alpha}",
                        color=color, linewidth=2.0, alpha=0.9)
                # Mark best point
                best_idx = int(np.argmax(smoothed))
                ax.scatter(rounds[best_idx], smoothed[best_idx],
                           color=color, s=40, zorder=5)

            ax.set_xlabel("Communication Round", fontsize=11)
            ax.set_ylabel("Accuracy (%)", fontsize=11)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(fontsize=9, loc="lower right", framealpha=0.9,
                      title="Dirichlet α")

        plt.tight_layout()
        fname = os.path.join(out_dir, f"{dataset}_{algo.lower()}_convergence_per_alpha.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Plot] {fname}")


# ──────────────────────────────────────────
# PLOT 2 — Best accuracy vs alpha
# Both algorithms on same axes
# ──────────────────────────────────────────
def plot_accuracy_vs_alpha(dataset_results: dict, dataset: str, out_dir: str):
    """
    Left: best global accuracy vs alpha for FedAvg and HedonicMFG
    Right: best avg client accuracy vs alpha
    """
    alphas   = sorted(dataset_results.keys())
    ds_label = "MNIST" if dataset == "mnist" else "FashionMNIST"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Best Accuracy vs Dirichlet α — {ds_label}",
                 fontsize=13, fontweight="bold", y=1.02)

    keys   = ["global_accs", "avg_client_accs"]
    titles = ["Best Global Accuracy (%)", "Best Avg Client Accuracy (%)"]

    for ax, key, title in zip(axes, keys, titles):
        for algo in ["FedAvg", "HedonicMFG"]:
            y_vals = []
            x_vals = []
            for alpha in alphas:
                run     = dataset_results.get(alpha, {})
                tracker = run.get(algo)
                if tracker is None:
                    continue
                vals = tracker[key]
                if vals:
                    y_vals.append(max(vals) * 100)
                    x_vals.append(alpha)

            if not x_vals:
                continue

            ax.plot(x_vals, y_vals,
                    label=algo,
                    color=ALGO_COLORS[algo],
                    linestyle=ALGO_LS[algo],
                    linewidth=ALGO_LW[algo],
                    marker=ALGO_MARKER[algo],
                    markersize=8, zorder=10 if algo == "HedonicMFG" else 2)

            # Annotate each point with value
            for x, y in zip(x_vals, y_vals):
                ax.annotate(f"{y:.1f}",
                            xy=(x, y), xytext=(0, 6),
                            textcoords="offset points",
                            ha="center", fontsize=7.5,
                            color=ALGO_COLORS[algo])

        # Shade HedonicMFG advantage region
        hm_pts = {alpha: dataset_results[alpha]["HedonicMFG"][key]
                  for alpha in alphas
                  if alpha in dataset_results and
                  "HedonicMFG" in dataset_results[alpha] and
                  dataset_results[alpha]["HedonicMFG"][key]}
        fa_pts = {alpha: dataset_results[alpha]["FedAvg"][key]
                  for alpha in alphas
                  if alpha in dataset_results and
                  "FedAvg" in dataset_results[alpha] and
                  dataset_results[alpha]["FedAvg"][key]}
        common = sorted(set(hm_pts) & set(fa_pts))
        if common:
            hm_y = [max(hm_pts[a]) * 100 for a in common]
            fa_y = [max(fa_pts[a]) * 100 for a in common]
            ax.fill_between(common, fa_y, hm_y,
                            where=[h > f for h, f in zip(hm_y, fa_y)],
                            alpha=0.10, color="#D62728",
                            label="HedonicMFG advantage")

        ax.set_xlabel("Dirichlet α  (← more heterogeneous | more IID →)",
                      fontsize=10)
        ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xscale("log")
        ax.set_xticks(alphas)
        ax.set_xticklabels([str(a) for a in alphas], fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=9, loc="lower right", framealpha=0.9)

    plt.tight_layout()
    fname = os.path.join(out_dir, f"{dataset}_accuracy_vs_alpha.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] {fname}")


# ──────────────────────────────────────────
# PLOT 3 — HedonicMFG gain over FedAvg vs alpha
# ──────────────────────────────────────────
def plot_gain_vs_alpha(dataset_results: dict, dataset: str, out_dir: str):
    """
    Bar chart of (HedonicMFG − FedAvg) at each alpha.
    Green = positive gain, Red = negative.
    """
    alphas   = sorted(dataset_results.keys())
    ds_label = "MNIST" if dataset == "mnist" else "FashionMNIST"

    gain_global = []
    gain_client = []
    valid_alphas = []

    for alpha in alphas:
        run = dataset_results.get(alpha, {})
        hm  = run.get("HedonicMFG")
        fa  = run.get("FedAvg")
        if hm is None or fa is None:
            continue
        hg = max(hm["global_accs"])     * 100 if hm["global_accs"]     else 0
        fg = max(fa["global_accs"])     * 100 if fa["global_accs"]     else 0
        hc = max(hm["avg_client_accs"]) * 100 if hm["avg_client_accs"] else 0
        fc = max(fa["avg_client_accs"]) * 100 if fa["avg_client_accs"] else 0
        gain_global.append(hg - fg)
        gain_client.append(hc - fc)
        valid_alphas.append(alpha)

    if not valid_alphas:
        print(f"[Warn] No complete pairs for gain plot ({dataset})")
        return

    x     = np.arange(len(valid_alphas))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"HedonicMFG Gain over FedAvg vs α — {ds_label}",
                 fontsize=13, fontweight="bold")

    for ax, gains, title in zip(
            axes,
            [gain_global, gain_client],
            ["Global Accuracy Gain (Δ%)", "Client Accuracy Gain (Δ%)"]):

        bar_colors = ["#2CA02C" if g >= 0 else "#D62728" for g in gains]
        bars = ax.bar(x, gains, width * 2.2,
                      color=bar_colors, alpha=0.82,
                      edgecolor="white", linewidth=0.8)

        for bar, val in zip(bars, gains):
            ypos = bar.get_height() + 0.05 if val >= 0 else bar.get_height() - 0.35
            ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                    f"{val:+.2f}%",
                    ha="center", va="bottom", fontsize=9.5, fontweight="bold")

        ax.axhline(0, color="black", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([f"α={a}" for a in valid_alphas], fontsize=10)
        ax.set_ylabel("Δ Accuracy (%)", fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Zero line label
        ax.annotate("FedAvg baseline", xy=(x[-1], 0),
                    xytext=(5, 4), textcoords="offset points",
                    fontsize=7.5, color="gray")

    plt.tight_layout()
    fname = os.path.join(out_dir, f"{dataset}_gain_vs_alpha.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] {fname}")


# ──────────────────────────────────────────
# PLOT 4 — Combined 2×3 summary (both datasets)
# ──────────────────────────────────────────
def plot_combined_summary(all_results: dict, base_dir: str):
    """
    all_results: {"mnist": dataset_results, "fmnist": dataset_results}
    Generates a 2-row × 3-col figure:
      Row 0 = MNIST,  Row 1 = FashionMNIST
      Col 0 = best global acc vs alpha
      Col 1 = best client acc vs alpha
      Col 2 = gain (HedonicMFG - FedAvg) vs alpha (bar)
    """
    datasets = [d for d in ["mnist", "fmnist"] if d in all_results]
    if not datasets:
        return
    n_rows   = len(datasets)
    ds_labels = {"mnist": "MNIST", "fmnist": "FashionMNIST"}

    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows))
    if n_rows == 1:
        axes = [axes]   # make iterable

    fig.suptitle("Ablation Study: HedonicMFG vs FedAvg across Dirichlet α",
                 fontsize=14, fontweight="bold", y=1.01)

    for row, dataset in enumerate(datasets):
        dataset_results = all_results[dataset]
        alphas          = sorted(dataset_results.keys())
        ds_label        = ds_labels[dataset]

        # ── Col 0: best global acc vs alpha ──────────────
        ax = axes[row][0]
        for algo in ["FedAvg", "HedonicMFG"]:
            xs, ys = [], []
            for alpha in alphas:
                run = dataset_results.get(alpha, {})
                t   = run.get(algo)
                if t and t["global_accs"]:
                    xs.append(alpha)
                    ys.append(max(t["global_accs"]) * 100)
            if xs:
                ax.plot(xs, ys, label=algo,
                        color=ALGO_COLORS[algo], linestyle=ALGO_LS[algo],
                        linewidth=ALGO_LW[algo], marker=ALGO_MARKER[algo],
                        markersize=7)
        ax.set_title(f"{ds_label} — Best Global Acc (%)",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Dirichlet α", fontsize=10)
        ax.set_ylabel("Accuracy (%)", fontsize=10)
        ax.set_xscale("log")
        ax.set_xticks(alphas); ax.set_xticklabels([str(a) for a in alphas], fontsize=8)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(fontsize=9); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

        # ── Col 1: best client acc vs alpha ──────────────
        ax = axes[row][1]
        for algo in ["FedAvg", "HedonicMFG"]:
            xs, ys = [], []
            for alpha in alphas:
                run = dataset_results.get(alpha, {})
                t   = run.get(algo)
                if t and t["avg_client_accs"]:
                    xs.append(alpha)
                    ys.append(max(t["avg_client_accs"]) * 100)
            if xs:
                ax.plot(xs, ys, label=algo,
                        color=ALGO_COLORS[algo], linestyle=ALGO_LS[algo],
                        linewidth=ALGO_LW[algo], marker=ALGO_MARKER[algo],
                        markersize=7)
        ax.set_title(f"{ds_label} — Best Client Acc (%)",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Dirichlet α", fontsize=10)
        ax.set_ylabel("Accuracy (%)", fontsize=10)
        ax.set_xscale("log")
        ax.set_xticks(alphas); ax.set_xticklabels([str(a) for a in alphas], fontsize=8)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(fontsize=9); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

        # ── Col 2: gain bar chart ─────────────────────────
        ax            = axes[row][2]
        gain_g, gain_c, valid_a = [], [], []
        for alpha in alphas:
            run = dataset_results.get(alpha, {})
            hm  = run.get("HedonicMFG")
            fa  = run.get("FedAvg")
            if not hm or not fa:
                continue
            hg = max(hm["global_accs"])     * 100 if hm["global_accs"]     else 0
            fg = max(fa["global_accs"])     * 100 if fa["global_accs"]     else 0
            hc = max(hm["avg_client_accs"]) * 100 if hm["avg_client_accs"] else 0
            fc = max(fa["avg_client_accs"]) * 100 if fa["avg_client_accs"] else 0
            gain_g.append(hg - fg)
            gain_c.append(hc - fc)
            valid_a.append(alpha)

        if valid_a:
            x      = np.arange(len(valid_a))
            width  = 0.32
            bc_g   = ["#2CA02C" if g >= 0 else "#D62728" for g in gain_g]
            bc_c   = ["#1F77B4" if g >= 0 else "#FF7F0E" for g in gain_c]

            bars_g = ax.bar(x - width/2, gain_g, width,
                            color=bc_g, alpha=0.82, label="Global Δ",
                            edgecolor="white")
            bars_c = ax.bar(x + width/2, gain_c, width,
                            color=bc_c, alpha=0.62, label="Client Δ",
                            edgecolor="white", hatch="//")

            for bar, val in list(zip(bars_g, gain_g)) + list(zip(bars_c, gain_c)):
                ypos = bar.get_height() + 0.03 if val >= 0 else bar.get_height() - 0.28
                ax.text(bar.get_x() + bar.get_width()/2, ypos,
                        f"{val:+.1f}", ha="center", fontsize=7, fontweight="bold")

            ax.axhline(0, color="black", linewidth=0.9)
            ax.set_xticks(x)
            ax.set_xticklabels([f"α={a}" for a in valid_a], fontsize=8)
            ax.set_ylabel("Δ Accuracy (%)", fontsize=10)

        ax.set_title(f"{ds_label} — HedonicMFG Gain over FedAvg",
                     fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fname = os.path.join(base_dir, "combined_summary.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Combined summary → {fname}")


# ──────────────────────────────────────────
# Print summary table to terminal
# ──────────────────────────────────────────
def print_summary_table(all_results: dict):
    for dataset, dataset_results in all_results.items():
        ds_label = "MNIST" if dataset == "mnist" else "FashionMNIST"
        alphas   = sorted(dataset_results.keys())

        print(f"\n{'='*75}")
        print(f"  {ds_label} — Ablation Summary")
        print(f"{'='*75}")
        print(f"  {'α':<6} {'Algorithm':<14} "
              f"{'Best Global':>12} {'Final Global':>13} "
              f"{'Best Client':>12} {'Final Client':>13}")
        print(f"  {'-'*71}")

        for alpha in alphas:
            run = dataset_results.get(alpha, {})
            for algo in ["FedAvg", "HedonicMFG"]:
                t = run.get(algo)
                if not t:
                    continue
                bg = max(t["global_accs"])     * 100 if t["global_accs"]     else 0
                fg = t["global_accs"][-1]      * 100 if t["global_accs"]     else 0
                bc = max(t["avg_client_accs"]) * 100 if t["avg_client_accs"] else 0
                fc = t["avg_client_accs"][-1]  * 100 if t["avg_client_accs"] else 0
                print(f"  {alpha:<6} {algo:<14} "
                      f"{bg:>11.2f}%  {fg:>12.2f}%  "
                      f"{bc:>11.2f}%  {fc:>12.2f}%")

        # Gain summary
        print(f"\n  HedonicMFG Gain over FedAvg:")
        print(f"  {'α':<6}  {'ΔGlobal':>10}  {'ΔClient':>10}")
        for alpha in alphas:
            run = dataset_results.get(alpha, {})
            hm  = run.get("HedonicMFG")
            fa  = run.get("FedAvg")
            if not hm or not fa:
                continue
            dg = (max(hm["global_accs"])     - max(fa["global_accs"]))     * 100
            dc = (max(hm["avg_client_accs"]) - max(fa["avg_client_accs"])) * 100
            print(f"  {alpha:<6}  "
                  f"{'✓' if dg>0 else '✗'} {dg:>+7.2f}%   "
                  f"{'✓' if dc>0 else '✗'} {dc:>+7.2f}%")

        print(f"{'='*75}")


# ──────────────────────────────────────────
# Main
# ──────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Combine per-alpha results into plots")
    p.add_argument("--datasets", nargs="+", default=["mnist", "fmnist"],
                   choices=["mnist", "fmnist"])
    p.add_argument("--alphas",   nargs="+", type=float, default=None,
                   help="Alpha subset to include (default: all found)")
    p.add_argument("--output",   type=str, default="./results",
                   help="Base results directory (same as used in runners)")
    return p.parse_args()


def main():
    args    = parse_args()
    alphas  = args.alphas if args.alphas else ABLATION_ALPHAS

    print("\n" + "╔" + "═"*52 + "╗")
    print("║  combine_plots.py — Ablation Plot Generator       ║")
    print("╠" + "═"*52 + "╣")
    print(f"║  Datasets : {str(args.datasets):<40} ║")
    print(f"║  Alphas   : {str(alphas):<40} ║")
    print(f"║  Base dir : {args.output:<40} ║")
    print("╚" + "═"*52 + "╝\n")

    all_results = {}

    for dataset in args.datasets:
        ds_dir = os.path.join(args.output, dataset)
        print(f"\n[Load] {dataset.upper()} results from {ds_dir}/")
        dataset_results = load_dataset_results(dataset, alphas, args.output)
        if not dataset_results:
            print(f"  [Skip] No results found for {dataset}.")
            continue
        all_results[dataset] = dataset_results

        # Per-dataset plots saved inside results/{dataset}/
        os.makedirs(ds_dir, exist_ok=True)
        print(f"\n[Plots] Generating {dataset.upper()} plots ...")
        plot_convergence_per_alpha(dataset_results, dataset, ds_dir)
        plot_accuracy_vs_alpha(dataset_results, dataset, ds_dir)
        plot_gain_vs_alpha(dataset_results, dataset, ds_dir)

    # Combined summary across both datasets
    if len(all_results) >= 1:
        print(f"\n[Plots] Generating combined summary ...")
        plot_combined_summary(all_results, args.output)

    # Terminal summary table
    if all_results:
        print_summary_table(all_results)

    print(f"\n[Done] All plots saved to {args.output}/")
    print("       Files generated:")
    for dataset in all_results:
        ds_dir = os.path.join(args.output, dataset)
        print(f"         {ds_dir}/{dataset}_convergence_per_alpha_*.png")
        print(f"         {ds_dir}/{dataset}_accuracy_vs_alpha.png")
        print(f"         {ds_dir}/{dataset}_gain_vs_alpha.png")
    if len(all_results) >= 1:
        print(f"         {args.output}/combined_summary.png")


if __name__ == "__main__":
    main()
