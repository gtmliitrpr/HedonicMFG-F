"""
visualize.py — Plots for client ablation study (FashionMNIST).

Per-run:   generate_run_plots()  — convergence + bar for one N
Combined:  called from combine_plots.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os


# ──────────────────────────────────────────
# Styles
# ──────────────────────────────────────────
ALGO_COLORS = {
    "HedonicMFG": "#D62728",
    "FedAvg":     "#1F77B4",
}
ALGO_LS = {"HedonicMFG": "-",  "FedAvg": "--"}
ALGO_LW = {"HedonicMFG": 2.5,  "FedAvg": 1.8}
ALGO_MK = {"HedonicMFG": "o",  "FedAvg": "s"}

# One color per client count
CLIENT_COLORS = {
    10:  "#D62728",
    20:  "#FF7F0E",
    30:  "#2CA02C",
    50:  "#1F77B4",
    75:  "#9467BD",
    100: "#8C564B",
}


def smooth(values, window=5):
    if len(values) <= window:
        return np.array(values, dtype=float)
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(values)]


# ──────────────────────────────────────────
# Per-run convergence + bar  (called by runner)
# label = "N=20" etc.
# ──────────────────────────────────────────
def generate_run_plots(results: dict, dataset: str,
                        label: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    _plot_run_convergence(results, dataset, label, save_dir)
    _plot_run_bar(results, dataset, label, save_dir)


def _plot_run_convergence(results, dataset, label, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"FL Convergence — FashionMNIST ({label})",
                 fontsize=13, fontweight="bold", y=1.02)

    keys   = ["global_accs", "avg_client_accs"]
    titles = ["Global Test Accuracy (%)", "Avg Client Accuracy (%)"]

    for ax, key, title in zip(axes, keys, titles):
        for name, tracker in results.items():
            data   = tracker.to_dict() if hasattr(tracker, "to_dict") else tracker
            values = [v * 100 for v in data[key]]
            if not values: continue
            sm     = smooth(values)
            rounds = list(range(1, len(values) + 1))
            ax.plot(rounds, sm, label=name,
                    color=ALGO_COLORS.get(name, "#333"),
                    linestyle=ALGO_LS.get(name, "-"),
                    linewidth=ALGO_LW.get(name, 1.8), alpha=0.92)
            if name == "HedonicMFG":
                ax.fill_between(rounds,
                                [v - 0.3 for v in sm],
                                [v + 0.3 for v in sm],
                                color=ALGO_COLORS["HedonicMFG"],
                                alpha=0.10)

        ax.set_xlabel("Communication Round", fontsize=11)
        ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=10, loc="lower right", framealpha=0.9)

    plt.tight_layout()
    tag  = label.replace("=", "").replace(" ", "_")
    path = os.path.join(save_dir, f"fmnist_{tag}_convergence.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] {path}")


def _plot_run_bar(results, dataset, label, save_dir):
    names = list(results.keys())
    g_acc = []
    c_acc = []
    for name, tracker in results.items():
        data = tracker.to_dict() if hasattr(tracker, "to_dict") else tracker
        g_acc.append(max(data["global_accs"])     * 100 if data["global_accs"]     else 0)
        c_acc.append(max(data["avg_client_accs"]) * 100 if data["avg_client_accs"] else 0)

    x     = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"Best Accuracy — FashionMNIST ({label})",
                 fontsize=13, fontweight="bold")

    for ax, accs, title in zip(axes,
                                [g_acc, c_acc],
                                ["Best Global Accuracy (%)",
                                 "Best Avg Client Accuracy (%)"]):
        colors = [ALGO_COLORS.get(n, "#aaa") for n in names]
        bars   = ax.bar(x, accs, 0.55, color=colors, alpha=0.85,
                        edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.2,
                    f"{val:.1f}%", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")
        if "HedonicMFG" in names:
            bars[names.index("HedonicMFG")].set_edgecolor("#8B0000")
            bars[names.index("HedonicMFG")].set_linewidth(2.5)

        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
        ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ymax = max(accs) if accs else 100
        ax.set_ylim(max(0, min(accs) - 5) if accs else 0, min(100, ymax + 5))

    plt.tight_layout()
    tag  = label.replace("=", "").replace(" ", "_")
    path = os.path.join(save_dir, f"fmnist_{tag}_bar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] {path}")
