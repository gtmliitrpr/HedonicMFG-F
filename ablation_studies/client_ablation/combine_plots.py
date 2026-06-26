"""
combine_plots.py — Combine per-client-count results into ablation plots.

Run this AFTER all individual client runs are complete.

Usage:
    python combine_plots.py                            # all client counts
    python combine_plots.py --clients 10 20 50         # subset
    python combine_plots.py --output ./results         # custom dir

Folder structure expected:
    results/
        clients_010/results.json
        clients_020/results.json
        clients_030/results.json
        clients_050/results.json
        clients_075/results.json
        clients_100/results.json

Plots generated in results/:
    fmnist_convergence_per_N_FedAvg.png       ← one line per N
    fmnist_convergence_per_N_HedonicMFG.png   ← one line per N
    fmnist_accuracy_vs_clients.png            ← best acc vs N (both algos)
    fmnist_gain_vs_clients.png                ← HedonicMFG - FedAvg gain
    fmnist_scalability.png                    ← round time vs N (efficiency)
    fmnist_combined_summary.png               ← 2x3 master summary grid
"""

import argparse, os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from config import ABLATION_CLIENTS

# ──────────────────────────────────────────
# Styles
# ──────────────────────────────────────────
ALGO_COLORS = {"HedonicMFG": "#D62728", "FedAvg": "#1F77B4"}
ALGO_LS     = {"HedonicMFG": "-",       "FedAvg": "--"}
ALGO_LW     = {"HedonicMFG": 2.5,       "FedAvg": 1.8}
ALGO_MK     = {"HedonicMFG": "o",       "FedAvg": "s"}

CLIENT_COLORS = {
    10: "#D62728", 20: "#FF7F0E", 30: "#2CA02C",
    50: "#1F77B4", 75: "#9467BD", 100: "#8C564B",
}


def clients_tag(n: int) -> str:
    return f"clients_{n:03d}"


def smooth(values, window=5):
    if len(values) <= window:
        return np.array(values, dtype=float)
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(values)]


# ──────────────────────────────────────────
# Load all results
# ──────────────────────────────────────────
def load_all_results(client_counts: list, base_dir: str) -> dict:
    """
    Returns: {num_clients: {"FedAvg": dict, "HedonicMFG": dict}}
    """
    results = {}
    missing = []
    for N in client_counts:
        path = os.path.join(base_dir, clients_tag(N), "results.json")
        if not os.path.exists(path):
            missing.append(N)
            continue
        with open(path) as f:
            data = json.load(f)
        results[N] = data
        print(f"  [Load] N={N:>3}: found {list(data.keys())}")
    if missing:
        print(f"  [Warn] Missing results for N={missing}")
        print(f"         Run: python runner_clients.py --clients <N>")
    return results


# ──────────────────────────────────────────
# PLOT 1 — Convergence per N, one algo per figure
# ──────────────────────────────────────────
def plot_convergence_per_N(all_results: dict, out_dir: str):
    """
    Two figures — one per algorithm.
    Each has left=global acc, right=client acc.
    One line per client count N.
    """
    client_counts = sorted(all_results.keys())

    for algo in ["FedAvg", "HedonicMFG"]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f"{algo} — Convergence at Different Client Counts\nFashionMNIST",
            fontsize=13, fontweight="bold", y=1.02)

        keys   = ["global_accs", "avg_client_accs"]
        titles = ["Global Test Accuracy (%)", "Avg Client Accuracy (%)"]

        for ax, key, title in zip(axes, keys, titles):
            for N in client_counts:
                run     = all_results.get(N, {})
                tracker = run.get(algo)
                if tracker is None:
                    continue
                values   = [v * 100 for v in tracker[key]]
                smoothed = smooth(values)
                rounds   = list(range(1, len(values) + 1))
                color    = CLIENT_COLORS.get(N, "#333")

                ax.plot(rounds, smoothed,
                        label=f"N={N}", color=color,
                        linewidth=2.0, alpha=0.9)
                # Mark best point
                bi = int(np.argmax(smoothed))
                ax.scatter(rounds[bi], smoothed[bi],
                           color=color, s=45, zorder=5)

            ax.set_xlabel("Communication Round", fontsize=11)
            ax.set_ylabel("Accuracy (%)", fontsize=11)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(fontsize=9, loc="lower right",
                      framealpha=0.9, title="Num Clients")

        plt.tight_layout()
        fname = os.path.join(
            out_dir, f"fmnist_convergence_per_N_{algo.replace(' ','')}.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Plot] {fname}")


# ──────────────────────────────────────────
# PLOT 2 — Best accuracy vs num_clients
# ──────────────────────────────────────────
def plot_accuracy_vs_clients(all_results: dict, out_dir: str):
    """
    Left:  best global accuracy vs N
    Right: best client accuracy vs N
    Both algorithms on same axes.
    """
    client_counts = sorted(all_results.keys())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Best Accuracy vs Number of Clients — FashionMNIST",
                 fontsize=13, fontweight="bold", y=1.02)

    keys   = ["global_accs",     "avg_client_accs"]
    titles = ["Best Global Accuracy (%)", "Best Avg Client Accuracy (%)"]

    for ax, key, title in zip(axes, keys, titles):
        for algo in ["FedAvg", "HedonicMFG"]:
            xs, ys = [], []
            for N in client_counts:
                run     = all_results.get(N, {})
                tracker = run.get(algo)
                if tracker is None or not tracker[key]:
                    continue
                xs.append(N)
                ys.append(max(tracker[key]) * 100)

            if not xs:
                continue

            ax.plot(xs, ys,
                    label=algo,
                    color=ALGO_COLORS[algo],
                    linestyle=ALGO_LS[algo],
                    linewidth=ALGO_LW[algo],
                    marker=ALGO_MK[algo],
                    markersize=9,
                    zorder=10 if algo == "HedonicMFG" else 2)

            # Annotate each point
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.1f}",
                            xy=(x, y), xytext=(0, 7),
                            textcoords="offset points",
                            ha="center", fontsize=8,
                            color=ALGO_COLORS[algo])

        # Shade HedonicMFG advantage
        hm_map = {}
        fa_map = {}
        for N in client_counts:
            run = all_results.get(N, {})
            hm  = run.get("HedonicMFG")
            fa  = run.get("FedAvg")
            if hm and hm[key]: hm_map[N] = max(hm[key]) * 100
            if fa and fa[key]: fa_map[N] = max(fa[key]) * 100
        common = sorted(set(hm_map) & set(fa_map))
        if common:
            ax.fill_between(common,
                            [fa_map[n] for n in common],
                            [hm_map[n] for n in common],
                            where=[hm_map[n] > fa_map[n] for n in common],
                            alpha=0.10, color="#D62728",
                            label="HedonicMFG advantage")

        ax.set_xlabel("Number of Clients (N)", fontsize=11)
        ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(client_counts)
        ax.set_xticklabels([str(n) for n in client_counts], fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=9, loc="lower left", framealpha=0.9)

    plt.tight_layout()
    fname = os.path.join(out_dir, "fmnist_accuracy_vs_clients.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] {fname}")


# ──────────────────────────────────────────
# PLOT 3 — HedonicMFG gain over FedAvg vs N
# ──────────────────────────────────────────
def plot_gain_vs_clients(all_results: dict, out_dir: str):
    """
    Grouped bar chart: global Δ and client Δ side by side per N.
    """
    client_counts = sorted(all_results.keys())
    gain_g, gain_c, valid_N = [], [], []

    for N in client_counts:
        run = all_results.get(N, {})
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
        valid_N.append(N)

    if not valid_N:
        print("[Warn] No complete pairs for gain plot"); return

    x     = np.arange(len(valid_N))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "HedonicMFG Gain over FedAvg vs Number of Clients — FashionMNIST",
        fontsize=13, fontweight="bold")

    for ax, gains, title in zip(
            axes,
            [gain_g, gain_c],
            ["Global Accuracy Gain (Δ%)", "Client Accuracy Gain (Δ%)"]):

        bar_colors = ["#2CA02C" if g >= 0 else "#D62728" for g in gains]
        bars = ax.bar(x, gains, width * 2.2,
                      color=bar_colors, alpha=0.82,
                      edgecolor="white", linewidth=0.8)

        for bar, val in zip(bars, gains):
            ypos = bar.get_height() + 0.05 if val >= 0 \
                else bar.get_height() - 0.35
            ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                    f"{val:+.2f}%", ha="center", va="bottom",
                    fontsize=9.5, fontweight="bold")

        ax.axhline(0, color="black", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([f"N={n}" for n in valid_N], fontsize=10)
        ax.set_ylabel("Δ Accuracy (%)", fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.annotate("← FedAvg baseline",
                    xy=(x[-1], 0), xytext=(5, 4),
                    textcoords="offset points", fontsize=8, color="gray")

    plt.tight_layout()
    fname = os.path.join(out_dir, "fmnist_gain_vs_clients.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] {fname}")


# ──────────────────────────────────────────
# PLOT 4 — Scalability: avg round time vs N
# ──────────────────────────────────────────
def plot_scalability(all_results: dict, out_dir: str):
    """
    Line chart: average round time (seconds) vs N for each algorithm.
    Shows computational overhead of HedonicMFG vs FedAvg as N grows.
    """
    client_counts = sorted(all_results.keys())

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Scalability: Avg Round Time vs Num Clients — FashionMNIST",
                 fontsize=13, fontweight="bold")

    for algo in ["FedAvg", "HedonicMFG"]:
        xs, ys = [], []
        for N in client_counts:
            run     = all_results.get(N, {})
            tracker = run.get(algo)
            if tracker is None or not tracker.get("round_times"):
                continue
            xs.append(N)
            ys.append(np.mean(tracker["round_times"]))

        if not xs:
            continue

        ax.plot(xs, ys,
                label=algo,
                color=ALGO_COLORS[algo],
                linestyle=ALGO_LS[algo],
                linewidth=ALGO_LW[algo],
                marker=ALGO_MK[algo],
                markersize=8)

        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.1f}s",
                        xy=(x, y), xytext=(0, 8),
                        textcoords="offset points",
                        ha="center", fontsize=8,
                        color=ALGO_COLORS[algo])

    ax.set_xlabel("Number of Clients (N)", fontsize=11)
    ax.set_ylabel("Avg Round Time (seconds)", fontsize=11)
    ax.set_xticks(client_counts)
    ax.set_xticklabels([str(n) for n in client_counts], fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=10, framealpha=0.9)

    plt.tight_layout()
    fname = os.path.join(out_dir, "fmnist_scalability.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] {fname}")


# ──────────────────────────────────────────
# PLOT 5 — Combined 2×3 master summary grid
# ──────────────────────────────────────────
def plot_combined_summary(all_results: dict, out_dir: str):
    """
    2 rows × 3 cols:
      Row 0: FedAvg convergence per N  | HedonicMFG convergence per N | scalability
      Row 1: accuracy vs N (global)    | accuracy vs N (client)        | gain bars
    """
    client_counts = sorted(all_results.keys())

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle(
        "Client Ablation Study: FedAvg vs HedonicMFG — FashionMNIST\n"
        "Effect of Federation Size (N) on Performance",
        fontsize=14, fontweight="bold", y=1.01)

    # ── Row 0, Col 0 — FedAvg convergence per N ──────────
    ax = axes[0][0]
    for N in client_counts:
        t = all_results.get(N, {}).get("FedAvg")
        if not t: continue
        vals = smooth([v * 100 for v in t["global_accs"]])
        ax.plot(range(1, len(vals)+1), vals,
                label=f"N={N}", color=CLIENT_COLORS.get(N, "#333"),
                linewidth=1.8)
    ax.set_title("FedAvg — Global Acc per N", fontsize=11, fontweight="bold")
    ax.set_xlabel("Round", fontsize=10); ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=8, title="N clients")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # ── Row 0, Col 1 — HedonicMFG convergence per N ──────
    ax = axes[0][1]
    for N in client_counts:
        t = all_results.get(N, {}).get("HedonicMFG")
        if not t: continue
        vals = smooth([v * 100 for v in t["global_accs"]])
        ax.plot(range(1, len(vals)+1), vals,
                label=f"N={N}", color=CLIENT_COLORS.get(N, "#333"),
                linewidth=1.8)
    ax.set_title("HedonicMFG — Global Acc per N", fontsize=11, fontweight="bold")
    ax.set_xlabel("Round", fontsize=10); ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=8, title="N clients")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # ── Row 0, Col 2 — Scalability (round time vs N) ─────
    ax = axes[0][2]
    for algo in ["FedAvg", "HedonicMFG"]:
        xs, ys = [], []
        for N in client_counts:
            t = all_results.get(N, {}).get(algo)
            if not t or not t.get("round_times"): continue
            xs.append(N); ys.append(np.mean(t["round_times"]))
        if xs:
            ax.plot(xs, ys, label=algo,
                    color=ALGO_COLORS[algo], linestyle=ALGO_LS[algo],
                    linewidth=ALGO_LW[algo], marker=ALGO_MK[algo],
                    markersize=7)
    ax.set_title("Avg Round Time vs N", fontsize=11, fontweight="bold")
    ax.set_xlabel("Num Clients (N)", fontsize=10)
    ax.set_ylabel("Time (s)", fontsize=10)
    ax.set_xticks(client_counts)
    ax.set_xticklabels([str(n) for n in client_counts], fontsize=8)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # ── Row 1, Col 0 — Best global acc vs N ──────────────
    ax = axes[1][0]
    for algo in ["FedAvg", "HedonicMFG"]:
        xs, ys = [], []
        for N in client_counts:
            t = all_results.get(N, {}).get(algo)
            if not t or not t["global_accs"]: continue
            xs.append(N); ys.append(max(t["global_accs"]) * 100)
        if xs:
            ax.plot(xs, ys, label=algo,
                    color=ALGO_COLORS[algo], linestyle=ALGO_LS[algo],
                    linewidth=ALGO_LW[algo], marker=ALGO_MK[algo],
                    markersize=8)
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.1f}", xy=(x, y), xytext=(0, 6),
                            textcoords="offset points",
                            ha="center", fontsize=7.5,
                            color=ALGO_COLORS[algo])
    ax.set_title("Best Global Acc vs N", fontsize=11, fontweight="bold")
    ax.set_xlabel("Num Clients (N)", fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.set_xticks(client_counts)
    ax.set_xticklabels([str(n) for n in client_counts], fontsize=8)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # ── Row 1, Col 1 — Best client acc vs N ──────────────
    ax = axes[1][1]
    for algo in ["FedAvg", "HedonicMFG"]:
        xs, ys = [], []
        for N in client_counts:
            t = all_results.get(N, {}).get(algo)
            if not t or not t["avg_client_accs"]: continue
            xs.append(N); ys.append(max(t["avg_client_accs"]) * 100)
        if xs:
            ax.plot(xs, ys, label=algo,
                    color=ALGO_COLORS[algo], linestyle=ALGO_LS[algo],
                    linewidth=ALGO_LW[algo], marker=ALGO_MK[algo],
                    markersize=8)
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.1f}", xy=(x, y), xytext=(0, 6),
                            textcoords="offset points",
                            ha="center", fontsize=7.5,
                            color=ALGO_COLORS[algo])
    ax.set_title("Best Client Acc vs N", fontsize=11, fontweight="bold")
    ax.set_xlabel("Num Clients (N)", fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.set_xticks(client_counts)
    ax.set_xticklabels([str(n) for n in client_counts], fontsize=8)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # ── Row 1, Col 2 — Gain bars ─────────────────────────
    ax = axes[1][2]
    gain_g, gain_c, valid_N = [], [], []
    for N in client_counts:
        run = all_results.get(N, {})
        hm  = run.get("HedonicMFG"); fa = run.get("FedAvg")
        if not hm or not fa: continue
        hg = max(hm["global_accs"])     * 100 if hm["global_accs"]     else 0
        fg = max(fa["global_accs"])     * 100 if fa["global_accs"]     else 0
        hc = max(hm["avg_client_accs"]) * 100 if hm["avg_client_accs"] else 0
        fc = max(fa["avg_client_accs"]) * 100 if fa["avg_client_accs"] else 0
        gain_g.append(hg - fg); gain_c.append(hc - fc); valid_N.append(N)

    if valid_N:
        xp    = np.arange(len(valid_N))
        width = 0.32
        bc_g  = ["#2CA02C" if g >= 0 else "#D62728" for g in gain_g]
        bc_c  = ["#1F77B4" if g >= 0 else "#FF7F0E" for g in gain_c]
        bars_g = ax.bar(xp - width/2, gain_g, width,
                        color=bc_g, alpha=0.85, label="Global Δ",
                        edgecolor="white")
        bars_c = ax.bar(xp + width/2, gain_c, width,
                        color=bc_c, alpha=0.65, label="Client Δ",
                        edgecolor="white", hatch="//")
        for bar, val in list(zip(bars_g, gain_g)) + list(zip(bars_c, gain_c)):
            yp = bar.get_height() + 0.03 if val >= 0 \
                else bar.get_height() - 0.30
            ax.text(bar.get_x() + bar.get_width()/2, yp,
                    f"{val:+.1f}", ha="center", fontsize=7, fontweight="bold")
        ax.axhline(0, color="black", linewidth=0.9)
        ax.set_xticks(xp)
        ax.set_xticklabels([f"N={n}" for n in valid_N], fontsize=8)
        ax.set_ylabel("Δ Accuracy (%)", fontsize=10)
        ax.legend(fontsize=8)

    ax.set_title("HedonicMFG Gain over FedAvg", fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fname = os.path.join(out_dir, "fmnist_combined_summary.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Combined summary → {fname}")


# ──────────────────────────────────────────
# Terminal summary table
# ──────────────────────────────────────────
def print_summary_table(all_results: dict):
    client_counts = sorted(all_results.keys())
    print(f"\n{'='*75}")
    print(f"  Client Ablation Summary — FashionMNIST")
    print(f"{'='*75}")
    print(f"  {'N':<6} {'Algorithm':<14} "
          f"{'Best Global':>12} {'Final Global':>13} "
          f"{'Best Client':>12} {'Final Client':>13}")
    print(f"  {'-'*71}")

    for N in client_counts:
        run = all_results.get(N, {})
        for algo in ["FedAvg", "HedonicMFG"]:
            t = run.get(algo)
            if not t: continue
            bg = max(t["global_accs"])     * 100 if t["global_accs"]     else 0
            fg = t["global_accs"][-1]      * 100 if t["global_accs"]     else 0
            bc = max(t["avg_client_accs"]) * 100 if t["avg_client_accs"] else 0
            fc = t["avg_client_accs"][-1]  * 100 if t["avg_client_accs"] else 0
            print(f"  {N:<6} {algo:<14} "
                  f"{bg:>11.2f}%  {fg:>12.2f}%  "
                  f"{bc:>11.2f}%  {fc:>12.2f}%")

    print(f"\n  HedonicMFG Gain over FedAvg:")
    print(f"  {'N':<6}  {'ΔGlobal':>10}  {'ΔClient':>10}  {'Round Time HM':>14}  {'Round Time FA':>14}")
    for N in client_counts:
        run = all_results.get(N, {})
        hm  = run.get("HedonicMFG"); fa = run.get("FedAvg")
        if not hm or not fa: continue
        dg   = (max(hm["global_accs"])     - max(fa["global_accs"]))     * 100
        dc   = (max(hm["avg_client_accs"]) - max(fa["avg_client_accs"])) * 100
        hm_t = np.mean(hm["round_times"]) if hm.get("round_times") else 0
        fa_t = np.mean(fa["round_times"]) if fa.get("round_times") else 0
        print(f"  {N:<6}  "
              f"{'✓' if dg>0 else '✗'} {dg:>+7.2f}%   "
              f"{'✓' if dc>0 else '✗'} {dc:>+7.2f}%  "
              f"{hm_t:>13.1f}s  {fa_t:>13.1f}s")
    print(f"{'='*75}")


# ──────────────────────────────────────────
# Main
# ──────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Combine client ablation results into plots")
    p.add_argument("--clients", nargs="+", type=int, default=None,
                   help="Client counts to include (default: all in config)")
    p.add_argument("--output",  type=str, default="./results")
    return p.parse_args()


def main():
    args          = parse_args()
    client_counts = args.clients if args.clients else ABLATION_CLIENTS

    print("\n" + "╔" + "═"*54 + "╗")
    print("║  combine_plots.py — Client Ablation Summary       ║")
    print("╠" + "═"*54 + "╣")
    print(f"║  Client counts : {str(client_counts):<36} ║")
    print(f"║  Output dir    : {args.output:<36} ║")
    print("╚" + "═"*54 + "╝\n")

    print(f"[Load] Reading results from {args.output}/")
    all_results = load_all_results(client_counts, args.output)

    if not all_results:
        print("[Error] No results found. Run runner_clients.py first.")
        return

    os.makedirs(args.output, exist_ok=True)

    print(f"\n[Plots] Generating all plots ...")
    plot_convergence_per_N(all_results, args.output)
    plot_accuracy_vs_clients(all_results, args.output)
    plot_gain_vs_clients(all_results, args.output)
    plot_scalability(all_results, args.output)
    plot_combined_summary(all_results, args.output)

    print_summary_table(all_results)

    print(f"\n[Done] All plots saved to {args.output}/")
    print("       fmnist_convergence_per_N_FedAvg.png")
    print("       fmnist_convergence_per_N_HedonicMFG.png")
    print("       fmnist_accuracy_vs_clients.png")
    print("       fmnist_gain_vs_clients.png")
    print("       fmnist_scalability.png")
    print("       fmnist_combined_summary.png")


if __name__ == "__main__":
    main()
