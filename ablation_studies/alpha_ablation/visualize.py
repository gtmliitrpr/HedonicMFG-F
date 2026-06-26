"""
visualize.py — Publication-quality result plots for FL ablation experiments.
Generates convergence curves, final comparison bar charts, and ablation plots.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os


# ──────────────────────────────────────────
# Color palette (colorblind-friendly)
# ──────────────────────────────────────────
COLORS = {
    "HedonicMFG": "#D62728",   # bold red — stands out
    "FedAvg":     "#1F77B4",   # blue
}

LINESTYLES = {
    "HedonicMFG": "-",
    "FedAvg":     "--",
}

LINEWIDTHS = {
    "HedonicMFG": 2.8,
    "FedAvg":     1.8,
}

# For ablation: one color per alpha value
ALPHA_COLORS = {
    0.05: "#D62728",
    0.1:  "#FF7F0E",
    0.3:  "#2CA02C",
    0.5:  "#1F77B4",
    1.0:  "#9467BD",
}


def smooth(values, window=5):
    """Simple moving average smoothing."""
    if len(values) <= window:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window // 2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')[:len(values)]


# ──────────────────────────────────────────
# Per-run convergence curves (FedAvg vs HedonicMFG)
# ──────────────────────────────────────────
def plot_convergence(results: dict, dataset: str, alpha: float, save_dir: str):
    """
    Convergence curves: global acc and avg client acc.
    results: {"FedAvg": ResultsTracker, "HedonicMFG": ResultsTracker}
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"FL Convergence — {dataset.upper()} (α={alpha})",
                 fontsize=13, fontweight="bold", y=1.02)

    titles   = ["Global Test Accuracy", "Average Client Accuracy"]
    data_keys = ["global_accs", "avg_client_accs"]

    for ax, title, key in zip(axes, titles, data_keys):
        for name, tracker in results.items():
            data   = tracker.to_dict() if hasattr(tracker, "to_dict") else tracker
            values = [v * 100 for v in data[key]]
            if not values:
                continue
            smoothed = smooth(values)
            rounds   = list(range(1, len(values) + 1))
            lw       = LINEWIDTHS.get(name, 1.8)
            ls       = LINESTYLES.get(name, "-")
            color    = COLORS.get(name, "#333333")
            zorder   = 10 if name == "HedonicMFG" else 2

            ax.plot(rounds, smoothed, label=name,
                    color=color, linestyle=ls, linewidth=lw,
                    zorder=zorder, alpha=0.92)

            if name == "HedonicMFG":
                ax.fill_between(rounds,
                                 [v - 0.3 for v in smoothed],
                                 [v + 0.3 for v in smoothed],
                                 color=color, alpha=0.1, zorder=1)

        ax.set_xlabel("Communication Round", fontsize=11)
        ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=10, loc="lower right", framealpha=0.9)
        ax.set_ylim(bottom=max(0, ax.get_ylim()[0] - 2))

    plt.tight_layout()
    alpha_tag = str(alpha).replace(".", "")
    path = os.path.join(save_dir, f"{dataset}_alpha{alpha_tag}_convergence.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Convergence saved: {path}")


# ──────────────────────────────────────────
# Bar chart: FedAvg vs HedonicMFG for one run
# ──────────────────────────────────────────
def plot_final_comparison(results: dict, dataset: str, alpha: float, save_dir: str):
    names       = list(results.keys())
    global_accs = []
    client_accs = []

    for name, tracker in results.items():
        data = tracker.to_dict() if hasattr(tracker, "to_dict") else tracker
        global_accs.append(max(data["global_accs"]) * 100 if data["global_accs"] else 0)
        client_accs.append(max(data["avg_client_accs"]) * 100 if data["avg_client_accs"] else 0)

    x     = np.arange(len(names))
    width = 0.55
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"Best Accuracy — {dataset.upper()} (α={alpha})",
                 fontsize=13, fontweight="bold")

    for ax, accs, title in zip(axes,
                                 [global_accs, client_accs],
                                 ["Best Global Accuracy (%)",
                                  "Best Avg Client Accuracy (%)"]):
        bar_colors = [COLORS.get(n, "#aaaaaa") for n in names]
        bars = ax.bar(x, accs, width, color=bar_colors, alpha=0.85,
                      edgecolor="white", linewidth=0.8)

        for bar, val in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.2,
                    f"{val:.1f}%",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

        if "HedonicMFG" in names:
            idx = names.index("HedonicMFG")
            bars[idx].set_edgecolor("#8B0000")
            bars[idx].set_linewidth(2.5)

        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=11)
        ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ymax = max(accs) if accs else 100
        ax.set_ylim(max(0, min(accs) - 5) if accs else 0, min(100, ymax + 5))

    plt.tight_layout()
    alpha_tag = str(alpha).replace(".", "")
    path = os.path.join(save_dir, f"{dataset}_alpha{alpha_tag}_bar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Bar chart saved: {path}")


# ──────────────────────────────────────────
# ABLATION PLOT 1 — Accuracy vs Alpha (line plot)
# One line per algorithm across all alpha values
# ──────────────────────────────────────────
def plot_ablation_vs_alpha(ablation_results: dict, dataset: str, save_dir: str):
    """
    ablation_results: {alpha_value: {"FedAvg": tracker, "HedonicMFG": tracker}}
    Plots best global and best client accuracy vs alpha for each algorithm.
    """
    alphas = sorted(ablation_results.keys())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Ablation: Accuracy vs Dirichlet α — {dataset.upper()}",
                 fontsize=13, fontweight="bold", y=1.02)

    titles    = ["Best Global Accuracy", "Best Avg Client Accuracy"]
    data_keys = ["global_accs", "avg_client_accs"]

    for ax, title, key in zip(axes, titles, data_keys):
        for algo_name in ["FedAvg", "HedonicMFG"]:
            y_vals = []
            for alpha in alphas:
                run = ablation_results.get(alpha, {})
                tracker = run.get(algo_name)
                if tracker is None:
                    y_vals.append(None)
                    continue
                data = tracker.to_dict() if hasattr(tracker, "to_dict") else tracker
                vals = data[key]
                y_vals.append(max(vals) * 100 if vals else None)

            valid = [(a, v) for a, v in zip(alphas, y_vals) if v is not None]
            if not valid:
                continue
            xs, ys = zip(*valid)
            lw     = LINEWIDTHS.get(algo_name, 1.8)
            ls     = LINESTYLES.get(algo_name, "-")
            color  = COLORS.get(algo_name, "#333333")

            ax.plot(xs, ys, label=algo_name,
                    color=color, linestyle=ls, linewidth=lw,
                    marker="o", markersize=7, zorder=10 if algo_name == "HedonicMFG" else 2)

            # Fill gap between algorithms
        # Shade HedonicMFG advantage region
        hm_vals = []
        fa_vals = []
        for alpha in alphas:
            run = ablation_results.get(alpha, {})
            for algo, store in [("HedonicMFG", hm_vals), ("FedAvg", fa_vals)]:
                tracker = run.get(algo)
                if tracker is not None:
                    data = tracker.to_dict() if hasattr(tracker, "to_dict") else tracker
                    vals = data[key]
                    store.append(max(vals) * 100 if vals else None)
                else:
                    store.append(None)

        valid_idx = [i for i in range(len(alphas))
                     if hm_vals[i] is not None and fa_vals[i] is not None]
        if valid_idx:
            xs_sh = [alphas[i] for i in valid_idx]
            hm_sh = [hm_vals[i] for i in valid_idx]
            fa_sh = [fa_vals[i] for i in valid_idx]
            ax.fill_between(xs_sh, fa_sh, hm_sh,
                            where=[h > f for h, f in zip(hm_sh, fa_sh)],
                            alpha=0.12, color="#D62728", label="HedonicMFG advantage")

        ax.set_xlabel("Dirichlet α (heterogeneity)", fontsize=11)
        ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xscale("log")
        ax.set_xticks(alphas)
        ax.set_xticklabels([str(a) for a in alphas], fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=9, loc="lower right", framealpha=0.9)

        # Annotation: low α = high heterogeneity
        ax.annotate("← High\nheterogeneity", xy=(alphas[0], ax.get_ylim()[0]),
                    fontsize=7.5, color="gray", ha="left")
        ax.annotate("Low\nheterogeneity →", xy=(alphas[-1], ax.get_ylim()[0]),
                    fontsize=7.5, color="gray", ha="right")

    plt.tight_layout()
    path = os.path.join(save_dir, f"{dataset}_ablation_vs_alpha.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Ablation vs alpha saved: {path}")


# ──────────────────────────────────────────
# ABLATION PLOT 2 — Gain (HedonicMFG - FedAvg) vs Alpha
# Shows where HedonicMFG advantage is largest
# ──────────────────────────────────────────
def plot_ablation_gain(ablation_results: dict, dataset: str, save_dir: str):
    """
    Bar chart of accuracy gain (HedonicMFG − FedAvg) at each alpha.
    """
    alphas = sorted(ablation_results.keys())
    gain_global = []
    gain_client = []

    for alpha in alphas:
        run = ablation_results.get(alpha, {})
        hm  = run.get("HedonicMFG")
        fa  = run.get("FedAvg")
        if hm is None or fa is None:
            gain_global.append(0.0)
            gain_client.append(0.0)
            continue
        hm_d = hm.to_dict() if hasattr(hm, "to_dict") else hm
        fa_d = fa.to_dict() if hasattr(fa, "to_dict") else fa
        hg   = max(hm_d["global_accs"]) * 100 if hm_d["global_accs"] else 0
        fg   = max(fa_d["global_accs"]) * 100 if fa_d["global_accs"] else 0
        hc   = max(hm_d["avg_client_accs"]) * 100 if hm_d["avg_client_accs"] else 0
        fc   = max(fa_d["avg_client_accs"]) * 100 if fa_d["avg_client_accs"] else 0
        gain_global.append(hg - fg)
        gain_client.append(hc - fc)

    x     = np.arange(len(alphas))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"HedonicMFG Gain over FedAvg vs α — {dataset.upper()}",
                 fontsize=13, fontweight="bold")

    for ax, gains, title in zip(axes,
                                   [gain_global, gain_client],
                                   ["Global Accuracy Gain (Δ%)",
                                    "Client Accuracy Gain (Δ%)"]):
        bar_colors = ["#2CA02C" if g >= 0 else "#D62728" for g in gains]
        bars = ax.bar(x, gains, width * 2, color=bar_colors, alpha=0.82,
                      edgecolor="white", linewidth=0.8)

        for bar, val in zip(bars, gains):
            y_pos = bar.get_height() + 0.05 if val >= 0 else bar.get_height() - 0.3
            ax.text(bar.get_x() + bar.get_width() / 2,
                    y_pos, f"{val:+.2f}%",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.axhline(0, color="black", linewidth=1.0, linestyle="-")
        ax.set_xticks(x)
        ax.set_xticklabels([f"α={a}" for a in alphas], fontsize=9)
        ax.set_ylabel("Accuracy Improvement (%)", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(save_dir, f"{dataset}_ablation_gain.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Ablation gain chart saved: {path}")


# ──────────────────────────────────────────
# ABLATION PLOT 3 — Convergence curves for all alphas (one algo)
# 2×2 or 1×n grid per algorithm showing how convergence changes with alpha
# ──────────────────────────────────────────
def plot_ablation_convergence_per_alpha(ablation_results: dict,
                                         dataset: str, save_dir: str):
    """
    For each algorithm, plot convergence curves at all alpha values.
    Separate figure per algorithm.
    """
    algos  = ["FedAvg", "HedonicMFG"]
    alphas = sorted(ablation_results.keys())

    for algo in algos:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(f"{algo} Convergence at Different α — {dataset.upper()}",
                     fontsize=13, fontweight="bold", y=1.02)

        titles    = ["Global Test Accuracy", "Average Client Accuracy"]
        data_keys = ["global_accs", "avg_client_accs"]

        for ax, title, key in zip(axes, titles, data_keys):
            for alpha in alphas:
                run     = ablation_results.get(alpha, {})
                tracker = run.get(algo)
                if tracker is None:
                    continue
                data   = tracker.to_dict() if hasattr(tracker, "to_dict") else tracker
                values = [v * 100 for v in data[key]]
                if not values:
                    continue
                smoothed = smooth(values)
                rounds   = list(range(1, len(values) + 1))
                color    = ALPHA_COLORS.get(alpha, "#333333")

                ax.plot(rounds, smoothed,
                        label=f"α={alpha}",
                        color=color, linewidth=2.0, alpha=0.88)

            ax.set_xlabel("Communication Round", fontsize=11)
            ax.set_ylabel("Accuracy (%)", fontsize=11)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(fontsize=9, loc="lower right", framealpha=0.9)

        plt.tight_layout()
        path = os.path.join(save_dir,
                            f"{dataset}_{algo.lower()}_convergence_per_alpha.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Plot] Per-alpha convergence for {algo}: {path}")


# ──────────────────────────────────────────
# COMBINED DATASET ABLATION — MNIST + FashionMNIST
# ──────────────────────────────────────────
def plot_combined_ablation(all_ablation: dict, save_dir: str):
    """
    all_ablation: {"mnist": ablation_results, "fmnist": ablation_results}
    Side-by-side summary across both datasets.
    """
    datasets = [d for d in ["mnist", "fmnist"] if d in all_ablation]
    if len(datasets) < 2:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Ablation Study: HedonicMFG vs FedAvg across Dirichlet α\n"
                 "MNIST (top) · FashionMNIST (bottom)",
                 fontsize=13, fontweight="bold")

    titles    = ["Best Global Accuracy (%)", "Best Avg Client Accuracy (%)"]
    data_keys = ["global_accs", "avg_client_accs"]

    for row, dataset in enumerate(datasets):
        ablation_results = all_ablation[dataset]
        alphas = sorted(ablation_results.keys())

        for col, (title, key) in enumerate(zip(titles, data_keys)):
            ax = axes[row][col]
            for algo_name in ["FedAvg", "HedonicMFG"]:
                y_vals = []
                for alpha in alphas:
                    run     = ablation_results.get(alpha, {})
                    tracker = run.get(algo_name)
                    if tracker is None:
                        y_vals.append(None)
                        continue
                    data = tracker.to_dict() if hasattr(tracker, "to_dict") else tracker
                    vals = data[key]
                    y_vals.append(max(vals) * 100 if vals else None)

                valid = [(a, v) for a, v in zip(alphas, y_vals) if v is not None]
                if not valid:
                    continue
                xs, ys = zip(*valid)
                ax.plot(xs, ys,
                        label=algo_name,
                        color=COLORS.get(algo_name, "#333"),
                        linestyle=LINESTYLES.get(algo_name, "-"),
                        linewidth=LINEWIDTHS.get(algo_name, 1.8),
                        marker="o", markersize=6)

            ds_label = "MNIST" if dataset == "mnist" else "FashionMNIST"
            ax.set_title(f"{ds_label} — {title}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Dirichlet α", fontsize=10)
            ax.set_ylabel("Accuracy (%)", fontsize=10)
            ax.set_xscale("log")
            ax.set_xticks(alphas)
            ax.set_xticklabels([str(a) for a in alphas], fontsize=8)
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(save_dir, "combined_ablation_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Combined ablation summary saved: {path}")


# ──────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────
def generate_run_plots(results: dict, dataset: str, alpha: float, save_dir: str):
    """Generate convergence + bar chart for a single (dataset, alpha) run."""
    os.makedirs(save_dir, exist_ok=True)
    plot_convergence(results, dataset, alpha, save_dir)
    plot_final_comparison(results, dataset, alpha, save_dir)


def generate_ablation_plots(ablation_results: dict, dataset: str, save_dir: str):
    """Generate all ablation plots for one dataset."""
    os.makedirs(save_dir, exist_ok=True)
    plot_ablation_vs_alpha(ablation_results, dataset, save_dir)
    plot_ablation_gain(ablation_results, dataset, save_dir)
    plot_ablation_convergence_per_alpha(ablation_results, dataset, save_dir)


def generate_all_plots(results: dict, dataset: str, save_dir: str):
    """Legacy helper for single-run plot generation at default alpha."""
    alpha = 0.3
    os.makedirs(save_dir, exist_ok=True)
    plot_convergence(results, dataset, alpha, save_dir)
    plot_final_comparison(results, dataset, alpha, save_dir)
    print(f"[Plots] All plots saved to {save_dir}/")
