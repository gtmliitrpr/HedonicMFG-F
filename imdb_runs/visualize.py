"""
visualize.py — Publication-quality result plots for FL experiments.
Generates convergence curves and final comparison bar charts.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os


# ──────────────────────────────────────────
# Color palette (colorblind-friendly)
# ──────────────────────────────────────────
COLORS = {
    "HedonicMFG":     "#D62728",   # bold red — stands out
    "FedAvg":         "#1F77B4",
    "FedProx":        "#FF7F0E",
    "SCAFFOLD":       "#2CA02C",
    "MOON":           "#9467BD",
    "FedBN":          "#8C564B",
    "pFedME":         "#E377C2",
    "IFCA":           "#7F7F7F",
    "CFL":            "#BCBD22",
    "RandomCluster":  "#17BECF",
}

LINESTYLES = {
    "HedonicMFG":    "-",
    "FedAvg":        "--",
    "FedProx":       "--",
    "SCAFFOLD":      "-.",
    "MOON":          "-.",
    "FedBN":         ":",
    "pFedME":        ":",
    "IFCA":          "--",
    "CFL":           "-.",
    "RandomCluster": ":",
}

LINEWIDTHS = {
    "HedonicMFG": 2.8,
}


def smooth(values, window=5):
    """Simple moving average smoothing."""
    if len(values) <= window:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window//2, window//2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')[:len(values)]


def plot_convergence(results: dict, dataset: str, save_dir: str):
    """
    Plot convergence curves for global accuracy and average client accuracy.
    Two-panel figure: left = global acc, right = avg client acc.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"FL Algorithm Comparison — {dataset.upper()}",
                 fontsize=14, fontweight="bold", y=1.02)

    titles = ["Global Test Accuracy", "Average Client Accuracy"]
    data_keys = ["global_accs", "avg_client_accs"]

    for ax, title, key in zip(axes, titles, data_keys):
        for name, tracker in results.items():
            data = tracker.to_dict() if hasattr(tracker, 'to_dict') else tracker
            values = [v * 100 for v in data[key]]
            if not values:
                continue

            smoothed = smooth(values)
            rounds = list(range(1, len(values) + 1))

            lw = LINEWIDTHS.get(name, 1.8)
            ls = LINESTYLES.get(name, "-")
            color = COLORS.get(name, "#333333")
            zorder = 10 if name == "HedonicMFG" else 2

            ax.plot(rounds, smoothed, label=name,
                    color=color, linestyle=ls, linewidth=lw,
                    zorder=zorder, alpha=0.9)

            # Highlight HedonicMFG with shaded best region
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
        ax.legend(fontsize=8, loc="lower right",
                  framealpha=0.9, ncol=2)
        ax.set_ylim(bottom=max(0, ax.get_ylim()[0] - 2))

    plt.tight_layout()
    path = os.path.join(save_dir, f"{dataset}_convergence.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Convergence curve saved: {path}")


def plot_final_comparison(results: dict, dataset: str, save_dir: str):
    """
    Bar chart comparing final best global and client accuracy across algorithms.
    HedonicMFG bar is highlighted in red.
    """
    names = list(results.keys())
    global_accs = []
    client_accs = []

    for name, tracker in results.items():
        data = tracker.to_dict() if hasattr(tracker, 'to_dict') else tracker
        global_accs.append(max(data["global_accs"]) * 100 if data["global_accs"] else 0)
        client_accs.append(max(data["avg_client_accs"]) * 100
                           if data["avg_client_accs"] else 0)

    x = np.arange(len(names))
    width = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"Best Accuracy Comparison — {dataset.upper()}",
                 fontsize=14, fontweight="bold")

    for ax, accs, title in zip(axes,
                                 [global_accs, client_accs],
                                 ["Best Global Accuracy (%)",
                                  "Best Avg Client Accuracy (%)"]):
        bar_colors = [COLORS.get(n, "#aaaaaa") for n in names]
        bars = ax.bar(x, accs, width * 2, color=bar_colors, alpha=0.85,
                      edgecolor="white", linewidth=0.8)

        # Add value labels on bars
        for bar, val in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{val:.1f}%",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold")

        # Highlight HedonicMFG with thick border
        if "HedonicMFG" in names:
            idx = names.index("HedonicMFG")
            bars[idx].set_edgecolor("#8B0000")
            bars[idx].set_linewidth(2.5)

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ymax = max(accs) if accs else 100
        ax.set_ylim(max(0, min(accs) - 5) if accs else 0, min(100, ymax + 5))

    plt.tight_layout()
    path = os.path.join(save_dir, f"{dataset}_bar_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Bar comparison saved: {path}")


def plot_improvement_over_baselines(results: dict, dataset: str, save_dir: str):
    """
    Show HedonicMFG improvement (Δ%) over each baseline.
    Horizontal bar chart — positive = HedonicMFG wins.
    """
    if "HedonicMFG" not in results:
        return

    hm_data = results["HedonicMFG"].to_dict() if hasattr(results["HedonicMFG"], 'to_dict') \
        else results["HedonicMFG"]
    hm_global = max(hm_data["global_accs"]) * 100 if hm_data["global_accs"] else 0
    hm_client = max(hm_data["avg_client_accs"]) * 100 if hm_data["avg_client_accs"] else 0

    baselines = [n for n in results if n != "HedonicMFG"]
    delta_global = []
    delta_client = []

    for name in baselines:
        data = results[name].to_dict() if hasattr(results[name], 'to_dict') \
            else results[name]
        bg = max(data["global_accs"]) * 100 if data["global_accs"] else 0
        bc = max(data["avg_client_accs"]) * 100 if data["avg_client_accs"] else 0
        delta_global.append(hm_global - bg)
        delta_client.append(hm_client - bc)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"HedonicMFG Improvement Over Baselines — {dataset.upper()}",
                 fontsize=13, fontweight="bold")

    for ax, deltas, title in zip(axes,
                                   [delta_global, delta_client],
                                   ["Global Accuracy Δ (%)", "Client Accuracy Δ (%)"]):
        y = np.arange(len(baselines))
        bar_colors = ["#2CA02C" if d >= 0 else "#D62728" for d in deltas]
        bars = ax.barh(y, deltas, color=bar_colors, alpha=0.8,
                       edgecolor="white", linewidth=0.8, height=0.6)

        for bar, val in zip(bars, deltas):
            x_pos = val + 0.1 if val >= 0 else val - 0.1
            ha = "left" if val >= 0 else "right"
            ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                    f"{val:+.1f}%", va="center", ha=ha, fontsize=9)

        ax.set_yticks(y)
        ax.set_yticklabels(baselines, fontsize=9)
        ax.axvline(0, color="black", linewidth=1.0, linestyle="-")
        ax.set_xlabel("Accuracy Improvement (%)", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(save_dir, f"{dataset}_improvement.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Improvement chart saved: {path}")


def generate_all_plots(results: dict, dataset: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    plot_convergence(results, dataset, save_dir)
    plot_final_comparison(results, dataset, save_dir)
    plot_improvement_over_baselines(results, dataset, save_dir)
    print(f"[Plots] All plots saved to {save_dir}/")
