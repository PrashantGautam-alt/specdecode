"""
Generate the report/README figures from the recorded benchmark numbers.

No GPU and no model needed -- the numbers are the ones measured on the A5000 during the
project and recorded in DAILY_LOG.md / report.tex. Run anywhere matplotlib is installed:

    python scripts/make_figures.py

Writes PNGs to report/figures/.
"""
import os

import matplotlib
matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt

OUT_DIR = os.path.join("report", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# consistent palette by category
C_BASE = "#9aa0a6"   # baseline / naive
C_LOSSLESS = "#4361ee"  # lossless configs
C_LOSSY = "#f08c00"   # typical acceptance (lossy)
C_CONTRIB = "#2f9e44"  # the calibrated tree (the contribution)
C_SLOW = "#c92a2a"    # slower than baseline


def fig_results_ladder():
    """Bar chart: speedup by configuration, colored by category."""
    labels = [
        "Naive\n8B", "SpecDec\nK=4", "Medusa ep1\n(3-pass)",
        "Medusa 4h\nfused greedy", "Medusa 6h\nfused greedy",
        "Medusa 4h\ntypical T=0.8", "Medusa 4h\ncalibrated",
    ]
    speedups = [1.00, 1.17, 0.65, 1.24, 1.14, 1.47, 1.45]
    colors = [C_BASE, C_LOSSLESS, C_SLOW, C_LOSSLESS, C_LOSSLESS, C_LOSSY, C_CONTRIB]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.bar(labels, speedups, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(len(labels) - 0.4, 1.02, "baseline (1.00x)", fontsize=8, alpha=0.7)

    for bar, val in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}x",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Speedup vs naive 8B")
    ax.set_title("Speedup ladder (Llama 3.1 8B, RTX A5000)")
    ax.set_ylim(0, 1.7)

    # legend by category
    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor=C_LOSSLESS, label="lossless"),
        Patch(facecolor=C_CONTRIB, label="calibrated tree (contribution, lossless)"),
        Patch(facecolor=C_LOSSY, label="typical acceptance (lossy)"),
        Patch(facecolor=C_SLOW, label="slower than baseline"),
    ]
    ax.legend(handles=legend, fontsize=8, loc="upper left")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "results_ladder.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def fig_k_sweep():
    """Dual-axis: speedup peaks at K=4 while acceptance keeps rising (optimal-K)."""
    K = [1, 2, 4, 6, 8]
    speedup = [0.73, 1.01, 1.17, 1.03, 0.89]
    accept = [1.00, 1.83, 3.05, 3.65, 3.82]  # avg tokens/round

    fig, ax1 = plt.subplots(figsize=(7, 4.6))
    ax1.plot(K, speedup, "o-", color=C_LOSSLESS, linewidth=2, markersize=7, label="speedup")
    ax1.set_xlabel("K (draft tokens per round)")
    ax1.set_ylabel("Speedup vs naive", color=C_LOSSLESS)
    ax1.tick_params(axis="y", labelcolor=C_LOSSLESS)
    ax1.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax1.axvline(4, color=C_CONTRIB, linestyle=":", linewidth=1.5, alpha=0.7)
    ax1.text(4.1, 0.8, "optimal K=4", color=C_CONTRIB, fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(K, accept, "s--", color=C_LOSSY, linewidth=2, markersize=6, label="acceptance")
    ax2.set_ylabel("Avg tokens accepted / round", color=C_LOSSY)
    ax2.tick_params(axis="y", labelcolor=C_LOSSY)

    ax1.set_title("Optimal K: acceptance keeps rising, but speedup peaks at K=4")
    ax1.set_xticks(K)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "k_sweep.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def fig_calibrated():
    """Grouped bars: Cartesian vs calibrated tree, same node budget."""
    prompts = ["Relativity", "Tea (list)"]
    cartesian = [1.81, 1.92]
    calibrated = [1.91, 2.22]
    gains = [6, 15]  # percent

    x = range(len(prompts))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    b1 = ax.bar([i - w / 2 for i in x], cartesian, w, label="Cartesian (naive tree)",
                color=C_BASE, edgecolor="black", linewidth=0.5)
    b2 = ax.bar([i + w / 2 for i in x], calibrated, w, label="Calibrated tree",
                color=C_CONTRIB, edgecolor="black", linewidth=0.5)

    for bar, val in list(zip(b1, cartesian)) + list(zip(b2, calibrated)):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}",
                ha="center", va="bottom", fontsize=9)
    for i, g in enumerate(gains):
        ax.text(i + w / 2, calibrated[i] + 0.14, f"+{g}%", ha="center",
                color=C_CONTRIB, fontweight="bold", fontsize=10)

    ax.set_xticks(list(x))
    ax.set_xticklabels(prompts)
    ax.set_ylabel("Avg tokens accepted / round")
    ax.set_title("Calibrated tree: more accepted tokens at the same node budget (lossless)")
    ax.set_ylim(0, 2.7)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "calibrated_vs_cartesian.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    fig_results_ladder()
    fig_k_sweep()
    fig_calibrated()
    print("done")
