import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
CSV_STOCH = os.path.join(current_dir, "../../scripts/benchmark/results_64.csv")
CSV_STRUCT = os.path.join(current_dir, "../../scripts/benchmark/results_structural_64.csv")
OUT = os.path.join(current_dir, "../fig_heatmap_seeds_density.pdf")

def get_pivot(csv_path):
    df = pd.read_csv(csv_path)
    # Ensure correct types
    df["Correct"] = pd.to_numeric(df["Correct"], errors="coerce")
    df["Seeds"] = pd.to_numeric(df["Seeds"], errors="coerce")
    df["Density"] = pd.to_numeric(df["Density"], errors="coerce")
    if "structural" in csv_path:
        df["Density"] = df["Density"] + 0.01
    
    # Filter for standard 64k/qwen2/yarn setup

    df = df[(df['Model'] == 'qwen2') & (df['RoPE_Type'] == 'yarn') & (df['Window'] == 65536)]
    
    # Calculate accuracy per run (64 questions)
    summary = (
        df.groupby(["Density", "Seeds"])
        .agg(Accuracy=("Correct", "mean"))
        .reset_index()
    )
    pivot = summary.pivot(index="Density", columns="Seeds", values="Accuracy")
    return pivot

pivot_stoch = get_pivot(CSV_STOCH)
pivot_struct = get_pivot(CSV_STRUCT)

# Common settings
BASELINE = 0.5156
VMIN, VMAX = 0.45, 0.66
CMAP = "RdYlGn"

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

def plot_heatmap(ax, pivot, label):
    im = ax.imshow(
        pivot.values,
        aspect="auto",
        cmap=CMAP,
        vmin=VMIN, vmax=VMAX,
        interpolation="nearest",
    )
    
    row_labels = [f"{d*100:.1f}%" for d in pivot.index]
    col_labels  = [str(int(s)) for s in pivot.columns]
    
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_ylabel("Density", fontsize=10)
    
    ax.set_title(label, loc='left', fontweight='bold', fontsize=12)
    
    # Annotations
    for i, row in enumerate(pivot.index):
        for j, col in enumerate(pivot.columns):
            val = pivot.loc[row, col]
            if pd.isna(val):
                ax.text(j, i, "OOM", ha="center", va="center", fontsize=7, color="grey")
            else:
                pct = f"{val*100:.1f}%"
                colour = "black" if 0.54 <= val <= 0.61 else "white"
                weight = "bold" if val >= 0.62 else "normal"
                ax.text(j, i, pct, ha="center", va="center", fontsize=7, color=colour, fontweight=weight)
                if val >= 0.62:
                    ax.add_patch(mpatches.FancyBboxPatch(
                        (j - 0.48, i - 0.48), 0.96, 0.96,
                        boxstyle="round,pad=0.02", linewidth=1.2, edgecolor="#2166ac", facecolor="none"
                    ))
    return im

im = plot_heatmap(ax1, pivot_stoch, "(a) Stochastic mode")
plot_heatmap(ax2, pivot_struct, "(b) Structural mode")


ax2.set_xticks(range(len(pivot_stoch.columns)))
ax2.set_xticklabels([str(int(s)) for s in pivot_stoch.columns], fontsize=8.5)
ax2.set_xlabel("Number of Seeds", fontsize=10)

# Single colorbar for both
plt.tight_layout()
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label("Accuracy", fontsize=9)
cbar.ax.axhline(y=BASELINE, color="navy", linewidth=1.5, linestyle="--")

plt.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"Dual heatmap saved to {OUT}")
