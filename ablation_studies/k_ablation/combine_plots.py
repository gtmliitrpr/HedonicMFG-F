"""
combine_plots.py — Combine per-K results into ablation plots.

Run this AFTER all individual K runs are complete.

Usage:
    python combine_plots.py                      # all K values
    python combine_plots.py --k 2 3 4 5          # subset
    python combine_plots.py --output ./results   # custom dir

Reuse K=3 from alpha ablation (saves one run):
    mkdir -p results/K_003
    cp ../ablation_study/results/mnist/alpha_03/results.json results/K_003/results.json

Folder structure expected:
    results/
        K_002/results.json
        K_003/results.json
        K_004/results.json
        K_005/results.json
        K_006/results.json
        K_008/results.json

Plots generated in results/:
    mnist_convergence_per_K_FedAvg.png       ← convergence, one line per K
    mnist_convergence_per_K_HedonicMFG.png   ← convergence, one line per K
    mnist_accuracy_vs_K.png                  ← best acc vs K (both algos)
    mnist_gain_vs_K.png                      ← HedonicMFG - FedAvg gain vs K
    mnist_optimal_K.png                      ← HedonicMFG-only curve highlighting peak
    mnist_combined_summary.png               ← 2×3 master summary grid
"""

import argparse, os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from config import ABLATION_K

# ──────────────────────────────────────────
# Styles
# ──────────────────────────────────────────
ALGO_COLORS = {"HedonicMFG": "#D62728", "FedAvg": "#1F77B4"}
ALGO_LS     = {"HedonicMFG": "-",       "FedAvg": "--"}
ALGO_LW     = {"HedonicMFG": 2.5,       "FedAvg": 1.8}
ALGO_MK     = {"HedonicMFG": "o",       "FedAvg": "s"}

K_COLORS = {
    2: "#D62728", 3: "#FF7F0E", 4: "#2CA02C",
    5: "#1F77B4", 6: "#9467BD", 8: "#8C564B",
}


def k_tag(k: int) -> str:
    return f"K_{k:03d}"


def smooth(values, window=5):
    if len(values) <= window:
        return np.array(values, dtype=float)
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(values)]


# ──────────────────────────────────────────
# Load all results
# ──────────────────────────────────────────
def load_all_results(k_values: list, base_dir: str) -> dict:
    """Returns: {K: {"FedAvg": dict, "HedonicMFG": dict}}"""
    results = {}
    missing = []
    for k in k_values:
        path = os.path.join(base_dir, k_tag(k), "results.json")
        if not os.path.exists(path):
            missing.append(k); continue
        with open(path) as f:
            data = json.load(f)
        results[k] = data
        print(f"  [Load] K={k}: found {list(data.keys())}")
    if missing:
        print(f"  [Warn] Missing results for K={missing}")
        print(f"         Run: python runner_k.py --k <value>")
    return results


# ──────────────────────────────────────────
# PLOT 1 — Convergence per K, one algo per figure
# ──────────────────────────────────────────
def plot_convergence_per_K(all_results: dict, out_dir: str):
    k_values = sorted(all_results.keys())

    for algo in ["FedAvg", "HedonicMFG"]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f"{algo} — Convergence at Different K\nMNIST  "
            f"(N=20 clients, α=0.3)",
            fontsize=13, fontweight="bold", y=1.02)

        keys   = ["global_accs", "avg_client_accs"]
        titles = ["Global Test Accuracy (%)", "Avg Client Accuracy (%)"]

        for ax, key, title in zip(axes, keys, titles):
            for k in k_values:
                run     = all_results.get(k, {})
                tracker = run.get(algo)
                if tracker is None: continue
                values   = [v * 100 for v in tracker[key]]
                smoothed = smooth(values)
                rounds   = list(range(1, len(values) + 1))
                color    = K_COLORS.get(k, "#333")
                ax.plot(rounds, smoothed,
                        label=f"K={k}", color=color,
                        linewidth=2.0, alpha=0.9)
                bi = int(np.argmax(smoothed))
                ax.scatter(rounds[bi], smoothed[bi],
                           color=color, s=50, zorder=5)

            ax.set_xlabel("Communication Round", fontsize=11)
            ax.set_ylabel("Accuracy (%)", fontsize=11)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(fontsize=9, loc="lower right",
                      framealpha=0.9, title="Num Coalitions K")

        plt.tight_layout()
        fname = os.path.join(
            out_dir,
            f"mnist_convergence_per_K_{algo.replace(' ', '')}.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Plot] {fname}")


# ──────────────────────────────────────────
# PLOT 2 — Best accuracy vs K
# ──────────────────────────────────────────
def plot_accuracy_vs_K(all_results: dict, out_dir: str):
    k_values = sorted(all_results.keys())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Best Accuracy vs Number of Coalitions K — MNIST\n"
        "(N=20 clients, α=0.3, fixed)",
        fontsize=13, fontweight="bold", y=1.02)

    keys   = ["global_accs",          "avg_client_accs"]
    titles = ["Best Global Accuracy (%)", "Best Avg Client Accuracy (%)"]

    for ax, key, title in zip(axes, keys, titles):
        for algo in ["FedAvg", "HedonicMFG"]:
            xs, ys = [], []
            for k in k_values:
                run     = all_results.get(k, {})
                tracker = run.get(algo)
                if tracker is None or not tracker[key]: continue
                xs.append(k)
                ys.append(max(tracker[key]) * 100)
            if not xs: continue

            ax.plot(xs, ys, label=algo,
                    color=ALGO_COLORS[algo],
                    linestyle=ALGO_LS[algo],
                    linewidth=ALGO_LW[algo],
                    marker=ALGO_MK[algo],
                    markersize=9,
                    zorder=10 if algo == "HedonicMFG" else 2)

            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.1f}",
                            xy=(x, y), xytext=(0, 8),
                            textcoords="offset points",
                            ha="center", fontsize=8,
                            color=ALGO_COLORS[algo])

        # Mark optimal K for HedonicMFG
        hm_map = {k: max(all_results[k]["HedonicMFG"][key]) * 100
                  for k in k_values
                  if k in all_results and
                  all_results[k].get("HedonicMFG") and
                  all_results[k]["HedonicMFG"][key]}
        if hm_map:
            best_k   = max(hm_map, key=hm_map.get)
            best_val = hm_map[best_k]
            ax.axvline(best_k, color="#D62728", linewidth=1.2,
                       linestyle=":", alpha=0.7)
            ax.annotate(f"Optimal K={best_k}",
                        xy=(best_k, best_val),
                        xytext=(8, -15), textcoords="offset points",
                        fontsize=9, color="#D62728", fontweight="bold",
                        arrowprops=dict(arrowstyle="->",
                                        color="#D62728", lw=1.2))

        # Shade advantage region
        hm_pts = {k: max(all_results[k]["HedonicMFG"][key]) * 100
                  for k in k_values
                  if k in all_results and
                  all_results[k].get("HedonicMFG") and
                  all_results[k]["HedonicMFG"][key]}
        fa_pts = {k: max(all_results[k]["FedAvg"][key]) * 100
                  for k in k_values
                  if k in all_results and
                  all_results[k].get("FedAvg") and
                  all_results[k]["FedAvg"][key]}
        common = sorted(set(hm_pts) & set(fa_pts))
        if common:
            ax.fill_between(common,
                            [fa_pts[k] for k in common],
                            [hm_pts[k] for k in common],
                            where=[hm_pts[k] > fa_pts[k] for k in common],
                            alpha=0.10, color="#D62728",
                            label="HedonicMFG advantage")

        ax.set_xlabel("Number of Coalitions K", fontsize=11)
        ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(k_values)
        ax.set_xticklabels([str(k) for k in k_values], fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=9, loc="lower left", framealpha=0.9)

    plt.tight_layout()
    fname = os.path.join(out_dir, "mnist_accuracy_vs_K.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] {fname}")


# ──────────────────────────────────────────
# PLOT 3 — HedonicMFG gain over FedAvg vs K
# ──────────────────────────────────────────
def plot_gain_vs_K(all_results: dict, out_dir: str):
    k_values = sorted(all_results.keys())
    gain_g, gain_c, valid_K = [], [], []

    for k in k_values:
        run = all_results.get(k, {})
        hm  = run.get("HedonicMFG"); fa = run.get("FedAvg")
        if not hm or not fa: continue
        hg = max(hm["global_accs"])     * 100 if hm["global_accs"]     else 0
        fg = max(fa["global_accs"])     * 100 if fa["global_accs"]     else 0
        hc = max(hm["avg_client_accs"]) * 100 if hm["avg_client_accs"] else 0
        fc = max(fa["avg_client_accs"]) * 100 if fa["avg_client_accs"] else 0
        gain_g.append(hg - fg); gain_c.append(hc - fc); valid_K.append(k)

    if not valid_K:
        print("[Warn] No complete pairs for gain plot"); return

    x     = np.arange(len(valid_K))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "HedonicMFG Gain over FedAvg vs K — MNIST\n"
        "(N=20 clients, α=0.3, fixed)",
        fontsize=13, fontweight="bold")

    for ax, gains, title in zip(
            axes,
            [gain_g, gain_c],
            ["Global Accuracy Gain (Δ%)", "Client Accuracy Gain (Δ%)"]):

        bar_colors = ["#2CA02C" if g >= 0 else "#D62728" for g in gains]
        bars = ax.bar(x, gains, 0.55,
                      color=bar_colors, alpha=0.82,
                      edgecolor="white", linewidth=0.8)

        for bar, val in zip(bars, gains):
            ypos = bar.get_height() + 0.05 if val >= 0 \
                else bar.get_height() - 0.35
            ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                    f"{val:+.2f}%", ha="center", va="bottom",
                    fontsize=9.5, fontweight="bold")

        # Mark best K
        if gains:
            best_idx = int(np.argmax(gains))
            bars[best_idx].set_edgecolor("#006400")
            bars[best_idx].set_linewidth(2.5)
            ax.annotate(f"Best K={valid_K[best_idx]}",
                        xy=(x[best_idx], gains[best_idx]),
                        xytext=(0, 12), textcoords="offset points",
                        ha="center", fontsize=8.5,
                        color="#006400", fontweight="bold",
                        arrowprops=dict(arrowstyle="->",
                                        color="#006400", lw=1.2))

        ax.axhline(0, color="black", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([f"K={k}" for k in valid_K], fontsize=10)
        ax.set_ylabel("Δ Accuracy (%)", fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fname = os.path.join(out_dir, "mnist_gain_vs_K.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] {fname}")


# ──────────────────────────────────────────
# PLOT 4 — Optimal K analysis (HedonicMFG only)
# Shows the inverted-U curve clearly — peak at optimal K
# ──────────────────────────────────────────
def plot_optimal_K(all_results: dict, out_dir: str):
    """
    HedonicMFG-only plot showing both global and client accuracy
    as a function of K. Highlights the optimal K with annotation.
    This is the key insight plot for the paper.
    """
    k_values = sorted(all_results.keys())

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title(
        "HedonicMFG: Effect of Coalition Count K on Accuracy\n"
        "MNIST  (N=20 clients, α=0.3)",
        fontsize=13, fontweight="bold")

    styles = {
        "global_accs":     ("Best Global Acc",  "#D62728", "o", "-",  2.5),
        "avg_client_accs": ("Best Client Acc",   "#1F77B4", "s", "--", 1.8),
    }

    best_k_global = None
    best_val_global = -1

    for key, (label, color, marker, ls, lw) in styles.items():
        xs, ys = [], []
        for k in k_values:
            run     = all_results.get(k, {})
            tracker = run.get("HedonicMFG")
            if tracker is None or not tracker[key]: continue
            xs.append(k)
            val = max(tracker[key]) * 100
            ys.append(val)
            if key == "global_accs" and val > best_val_global:
                best_val_global = val
                best_k_global   = k

        if not xs: continue
        ax.plot(xs, ys, label=label, color=color,
                marker=marker, linestyle=ls,
                linewidth=lw, markersize=10, zorder=5)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.2f}%",
                        xy=(x, y), xytext=(0, 10),
                        textcoords="offset points",
                        ha="center", fontsize=8.5, color=color)

    # FedAvg reference line (flat — K has no effect on FedAvg)
    fa_vals = []
    for k in k_values:
        run = all_results.get(k, {})
        fa  = run.get("FedAvg")
        if fa and fa["global_accs"]:
            fa_vals.append(max(fa["global_accs"]) * 100)
    if fa_vals:
        fa_mean = np.mean(fa_vals)
        ax.axhline(fa_mean, color="#888888", linewidth=1.5,
                   linestyle=":", label=f"FedAvg baseline ({fa_mean:.1f}%)")

    # Mark optimal K
    if best_k_global is not None:
        ax.axvline(best_k_global, color="#D62728", linewidth=1.5,
                   linestyle=":", alpha=0.6)
        ymin, ymax = ax.get_ylim()
        ax.text(best_k_global + 0.1, ymin + (ymax - ymin) * 0.05,
                f"Optimal K={best_k_global}",
                fontsize=10, color="#D62728", fontweight="bold")

    # Shade underfit / overfit regions
    if len(k_values) >= 3 and best_k_global is not None:
        ax.axvspan(min(k_values), best_k_global,
                   alpha=0.04, color="#FF7F0E", label="Under-clustering")
        ax.axvspan(best_k_global, max(k_values),
                   alpha=0.04, color="#9467BD", label="Over-clustering")

    ax.set_xlabel("Number of Coalitions K", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_xticks(k_values)
    ax.set_xticklabels([str(k) for k in k_values], fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, framealpha=0.9, loc="lower left")

    plt.tight_layout()
    fname = os.path.join(out_dir, "mnist_optimal_K.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] {fname}")


# ──────────────────────────────────────────
# PLOT 5 — Combined 2×3 master summary grid
# ──────────────────────────────────────────
def plot_combined_summary(all_results: dict, out_dir: str):
    """
    2 rows × 3 cols:
      Row 0: FedAvg conv per K  | HedonicMFG conv per K | optimal K curve
      Row 1: global acc vs K    | client acc vs K        | gain bars
    """
    k_values = sorted(all_results.keys())

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle(
        "K Ablation Study: FedAvg vs HedonicMFG — MNIST\n"
        "Effect of Coalition Count K  (N=20 clients, α=0.3 fixed)",
        fontsize=14, fontweight="bold", y=1.01)

    # ── Row 0, Col 0 — FedAvg convergence per K ──────────
    ax = axes[0][0]
    for k in k_values:
        t = all_results.get(k, {}).get("FedAvg")
        if not t: continue
        vals = smooth([v * 100 for v in t["global_accs"]])
        ax.plot(range(1, len(vals)+1), vals,
                label=f"K={k}", color=K_COLORS.get(k, "#333"),
                linewidth=1.8)
    ax.set_title("FedAvg — Global Acc per K",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Round", fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=8, title="K")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Row 0, Col 1 — HedonicMFG convergence per K ──────
    ax = axes[0][1]
    for k in k_values:
        t = all_results.get(k, {}).get("HedonicMFG")
        if not t: continue
        vals = smooth([v * 100 for v in t["global_accs"]])
        ax.plot(range(1, len(vals)+1), vals,
                label=f"K={k}", color=K_COLORS.get(k, "#333"),
                linewidth=1.8)
    ax.set_title("HedonicMFG — Global Acc per K",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Round", fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=8, title="K")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Row 0, Col 2 — Optimal K curve ───────────────────
    ax = axes[0][2]
    for key, (label, color, marker, ls, lw) in {
        "global_accs":     ("Global Acc",  "#D62728", "o", "-",  2.2),
        "avg_client_accs": ("Client Acc",  "#1F77B4", "s", "--", 1.6),
    }.items():
        xs, ys = [], []
        for k in k_values:
            t = all_results.get(k, {}).get("HedonicMFG")
            if not t or not t[key]: continue
            xs.append(k); ys.append(max(t[key]) * 100)
        if xs:
            ax.plot(xs, ys, label=f"HM {label}", color=color,
                    marker=marker, linestyle=ls, linewidth=lw, markersize=8)
    # FedAvg reference
    fa_vals = [max(all_results[k]["FedAvg"]["global_accs"]) * 100
               for k in k_values
               if k in all_results and
               all_results[k].get("FedAvg") and
               all_results[k]["FedAvg"]["global_accs"]]
    if fa_vals:
        ax.axhline(np.mean(fa_vals), color="#888", linewidth=1.4,
                   linestyle=":", label=f"FedAvg ({np.mean(fa_vals):.1f}%)")
    ax.set_title("HedonicMFG: Optimal K Analysis",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("K", fontsize=10); ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.set_xticks(k_values)
    ax.set_xticklabels([str(k) for k in k_values], fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # ── Row 1, Col 0 — Best global acc vs K ──────────────
    ax = axes[1][0]
    for algo in ["FedAvg", "HedonicMFG"]:
        xs, ys = [], []
        for k in k_values:
            t = all_results.get(k, {}).get(algo)
            if not t or not t["global_accs"]: continue
            xs.append(k); ys.append(max(t["global_accs"]) * 100)
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
    ax.set_title("Best Global Acc vs K", fontsize=11, fontweight="bold")
    ax.set_xlabel("K", fontsize=10); ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.set_xticks(k_values)
    ax.set_xticklabels([str(k) for k in k_values], fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # ── Row 1, Col 1 — Best client acc vs K ──────────────
    ax = axes[1][1]
    for algo in ["FedAvg", "HedonicMFG"]:
        xs, ys = [], []
        for k in k_values:
            t = all_results.get(k, {}).get(algo)
            if not t or not t["avg_client_accs"]: continue
            xs.append(k); ys.append(max(t["avg_client_accs"]) * 100)
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
    ax.set_title("Best Client Acc vs K", fontsize=11, fontweight="bold")
    ax.set_xlabel("K", fontsize=10); ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.set_xticks(k_values)
    ax.set_xticklabels([str(k) for k in k_values], fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # ── Row 1, Col 2 — Gain bars ─────────────────────────
    ax = axes[1][2]
    gain_g, gain_c, valid_K = [], [], []
    for k in k_values:
        run = all_results.get(k, {})
        hm  = run.get("HedonicMFG"); fa = run.get("FedAvg")
        if not hm or not fa: continue
        hg = max(hm["global_accs"])     * 100 if hm["global_accs"]     else 0
        fg = max(fa["global_accs"])     * 100 if fa["global_accs"]     else 0
        hc = max(hm["avg_client_accs"]) * 100 if hm["avg_client_accs"] else 0
        fc = max(fa["avg_client_accs"]) * 100 if fa["avg_client_accs"] else 0
        gain_g.append(hg - fg); gain_c.append(hc - fc); valid_K.append(k)

    if valid_K:
        xp    = np.arange(len(valid_K))
        width = 0.32
        bc_g  = ["#2CA02C" if g >= 0 else "#D62728" for g in gain_g]
        bc_c  = ["#1F77B4" if g >= 0 else "#FF7F0E" for g in gain_c]
        bars_g = ax.bar(xp - width/2, gain_g, width,
                        color=bc_g, alpha=0.85, label="Global Δ",
                        edgecolor="white")
        bars_c = ax.bar(xp + width/2, gain_c, width,
                        color=bc_c, alpha=0.65, label="Client Δ",
                        edgecolor="white", hatch="//")
        for bar, val in (list(zip(bars_g, gain_g)) +
                         list(zip(bars_c, gain_c))):
            yp = bar.get_height() + 0.03 if val >= 0 \
                else bar.get_height() - 0.28
            ax.text(bar.get_x() + bar.get_width()/2, yp,
                    f"{val:+.1f}", ha="center",
                    fontsize=7, fontweight="bold")
        ax.axhline(0, color="black", linewidth=0.9)
        ax.set_xticks(xp)
        ax.set_xticklabels([f"K={k}" for k in valid_K], fontsize=8)
        ax.set_ylabel("Δ Accuracy (%)", fontsize=10)
        ax.legend(fontsize=8)

    ax.set_title("HedonicMFG Gain over FedAvg", fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fname = os.path.join(out_dir, "mnist_combined_summary.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Combined summary → {fname}")


# ──────────────────────────────────────────
# Terminal summary table
# ──────────────────────────────────────────
def print_summary_table(all_results: dict):
    k_values = sorted(all_results.keys())
    print(f"\n{'='*75}")
    print(f"  K Ablation Summary — MNIST (N=20, α=0.3)")
    print(f"{'='*75}")
    print(f"  {'K':<5} {'Algorithm':<14} "
          f"{'Best Global':>12} {'Final Global':>13} "
          f"{'Best Client':>12} {'Final Client':>13}")
    print(f"  {'-'*71}")
    for k in k_values:
        run = all_results.get(k, {})
        for algo in ["FedAvg", "HedonicMFG"]:
            t = run.get(algo)
            if not t: continue
            bg = max(t["global_accs"])     * 100 if t["global_accs"]     else 0
            fg = t["global_accs"][-1]      * 100 if t["global_accs"]     else 0
            bc = max(t["avg_client_accs"]) * 100 if t["avg_client_accs"] else 0
            fc = t["avg_client_accs"][-1]  * 100 if t["avg_client_accs"] else 0
            print(f"  {k:<5} {algo:<14} "
                  f"{bg:>11.2f}%  {fg:>12.2f}%  "
                  f"{bc:>11.2f}%  {fc:>12.2f}%")

    print(f"\n  HedonicMFG Gain over FedAvg:")
    print(f"  {'K':<5}  {'ΔGlobal':>10}  {'ΔClient':>10}")
    print(f"  {'-'*30}")
    for k in k_values:
        run = all_results.get(k, {})
        hm  = run.get("HedonicMFG"); fa = run.get("FedAvg")
        if not hm or not fa: continue
        dg = (max(hm["global_accs"])     - max(fa["global_accs"]))     * 100
        dc = (max(hm["avg_client_accs"]) - max(fa["avg_client_accs"])) * 100
        print(f"  {k:<5}  "
              f"{'✓' if dg>0 else '✗'} {dg:>+7.2f}%   "
              f"{'✓' if dc>0 else '✗'} {dc:>+7.2f}%")
    print(f"{'='*75}")


# ──────────────────────────────────────────
# Main
# ──────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Combine K ablation results into plots")
    p.add_argument("--k",      nargs="+", type=int, default=None,
                   help="K values to include (default: all in config)")
    p.add_argument("--output", type=str, default="./results")
    return p.parse_args()


def main():
    args     = parse_args()
    k_values = args.k if args.k else ABLATION_K

    print("\n" + "╔" + "═"*54 + "╗")
    print("║  combine_plots.py — K Ablation Summary            ║")
    print("╠" + "═"*54 + "╣")
    print(f"║  K values  : {str(k_values):<41} ║")
    print(f"║  Dataset   : {'MNIST (N=20, α=0.3)':<41} ║")
    print(f"║  Output    : {args.output:<41} ║")
    print("╚" + "═"*54 + "╝\n")

    print(f"[Load] Reading results from {args.output}/")
    all_results = load_all_results(k_values, args.output)

    if not all_results:
        print("[Error] No results found. Run runner_k.py first.")
        return

    os.makedirs(args.output, exist_ok=True)

    print(f"\n[Plots] Generating all plots ...")
    plot_convergence_per_K(all_results, args.output)
    plot_accuracy_vs_K(all_results, args.output)
    plot_gain_vs_K(all_results, args.output)
    plot_optimal_K(all_results, args.output)
    plot_combined_summary(all_results, args.output)

    print_summary_table(all_results)

    print(f"\n[Done] All plots saved to {args.output}/")
    print("       mnist_convergence_per_K_FedAvg.png")
    print("       mnist_convergence_per_K_HedonicMFG.png")
    print("       mnist_accuracy_vs_K.png")
    print("       mnist_gain_vs_K.png")
    print("       mnist_optimal_K.png          ← key insight plot for paper")
    print("       mnist_combined_summary.png")


if __name__ == "__main__":
    main()
