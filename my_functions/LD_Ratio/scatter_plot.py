import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_ld_ratio_comparison(
    csv_bates="master_thesis/my_functions/LD_Ratio/util/no_fail_dataset/no_fail_dataset_bates.csv",
    csv_star="master_thesis/my_functions/LD_Ratio/util/no_fail_dataset/no_fail_dataset_star.csv",
    csv_endburner="master_thesis/my_functions/LD_Ratio/util/no_fail_dataset/no_fail_dataset_endburner.csv",
    show_plot=True
):
    """
    Create a single plot comparing L/D ratio statistics (min, max, mean, median)
    for Bates, Star, and Endburner after strict range filtering.

    Filtering (strict):
      diameter in (0.0254, 0.0977975)
      length   in (0.132,  0.381)

    Saves 'LD_Ratio_Comparison.png' to the current directory.
    """

    # ---- Ranges (strict) ----
    diameter_min, diameter_max = 0.0254, 0.0977975
    length_min, length_max = 0.132, 0.381

    def load_filter_compute(path, label):
        df = pd.read_csv(path)
        before = len(df)
        df = df[
            (df["diameter"] > diameter_min) & (df["diameter"] < diameter_max) &
            (df["length"] > length_min) & (df["length"] < length_max)
        ].copy()
        after = len(df)
        print(f"[{label}] rows: {before} -> {after} after range filter")

        if after == 0:
            return None

        df["LD_Ratio"] = df["length"] / df["diameter"]
        return {
            "label": label,
            "min": float(df["LD_Ratio"].min()),
            "max": float(df["LD_Ratio"].max()),
            "mean": float(df["LD_Ratio"].mean()),
            "median": float(df["LD_Ratio"].median()),
            "count": int(after),
        }

    # ---- Load all three datasets ----
    candidates = [
        ("Bates", csv_bates),
        ("Star", csv_star),
        ("Endburner", csv_endburner),
    ]
    results = []
    for lbl, path in candidates:
        stats = load_filter_compute(path, lbl)
        if stats is not None:
            results.append(stats)

    if not results:
        print("No data left after filtering — nothing to plot.")
        return

    labels = [r["label"] for r in results]

    # ---- Choose x positions; nudge first/last toward center, add edge padding ----
    n = len(results)
    positions = np.arange(n, dtype=float)

    # Nudge only if we have >= 2 points
    if n >= 2:
        inset = 0.35   # how far to push inward
        positions[0] += inset
        positions[-1] -= inset

    fig, ax = plt.subplots(figsize=(8, 4))

    # Plot vertical min–max lines and markers
    for i, (pos, r) in enumerate(zip(positions, results)):
        ax.vlines(pos, r["min"], r["max"], colors="gray", lw=2)
        ax.scatter(pos, r["min"], color="red", marker="o", label="Min" if i == 0 else "")
        ax.scatter(pos, r["max"], color="orange", marker="o", label="Max" if i == 0 else "")
        ax.scatter(pos, r["mean"], color="blue", marker="x", label="Mean" if i == 0 else "")
        ax.scatter(pos, r["median"], color="green", marker="d", label="Median" if i == 0 else "")

    # Now that data is drawn, place counts at the bottom edge
    y_bottom = ax.get_ylim()[0]
    for pos, r in zip(positions, results):
        ax.text(pos, y_bottom-2, f"n={r['count']}", va="bottom", ha="center", fontsize=9)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("L/D Ratio")
    ax.set_title("L/D Ratio — Bates vs Star vs Endburner")
    ax.grid(True, linestyle="--", alpha=0.6)

    # Legend outside the axes to avoid overlap
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0.)

    # Add left/right padding so points are away from the borders
    if n == 1:
        edge_pad = 0.8
        ax.set_xlim(positions[0] - edge_pad, positions[0] + edge_pad)
    else:
        edge_pad = 0.6
        ax.set_xlim(positions[0] - edge_pad, positions[-1] + edge_pad)

    plt.tight_layout()
    out_name = "master_thesis/my_functions/LD_Ratio/LD_Ratio_Comparison.png"
    fig.savefig(out_name, dpi=300, bbox_inches="tight")
    print(f"Plot saved: ./{out_name}")

    if show_plot:
        plt.show()
    plt.close(fig)

# Example:
# plot_ld_ratio_comparison()

# Example usage:
plot_ld_ratio_comparison()