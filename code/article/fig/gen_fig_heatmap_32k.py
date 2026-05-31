import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(current_dir, "../../scripts/benchmark32/results_baseline_32k.csv")
OUT = os.path.join(current_dir, "../fig_heatmap_32k.pdf")

def get_pivot(csv_path, mode):
    df = pd.read_csv(csv_path)
    # Ensure correct types
    df["Correct"] = pd.to_numeric(df["Correct"], errors="coerce")
    df["Seeds"] = pd.to_numeric(df["Seeds"], errors="coerce")
    df["Density"] = pd.to_numeric(df["Density"], errors="coerce")
    if mode == "structural":
        df.loc[df["RIS_Mode"] == "structural", "Density"] += 0.01

    
    # Filter for standard setup
    df = df[(df['Model'] == 'qwen2') & (df['RoPE_Type'] == 'yarn') & (df['RIS_Mode'] == mode) & (df['Window'] == 32768)]
    
    # Calculate accuracy per run (32 questions)
    summary = (
        df.groupby(["Density", "Seeds"])
        .agg(Accuracy=("Correct", "mean"))
        .reset_index()
    )
    # Pivot
    pivot = summary.pivot(index="Density", columns="Seeds", values="Accuracy")
    
    # Reindex to ensure we have all densities [0.01, 0.02, 0.05] and seeds [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    densities = [0.01, 0.02, 0.05]
    seeds = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    pivot = pivot.reindex(index=densities, columns=seeds)
    return pivot

pivot_stoch = get_pivot(CSV_PATH, "stochastic")
pivot_struct = get_pivot(CSV_PATH, "structural")

# Baseline values for 32k
BASELINE_W0 = 0.59375  # 59.38% (19/32)
DENSE_LIMIT = 0.71875  # 71.88% (23/32)
VMIN, VMAX = 0.50, 0.75
CMAP = "RdYlGn"

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)

def plot_heatmap(ax, pivot, label):
    # Set background color for NaNs to light grey
    ax.set_facecolor('#f0f0f0')
    
    # Use masked array or plot with nan support
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
    ax.set_yticklabels(row_labels, fontsize=9.5)
    ax.set_ylabel("Density", fontsize=11)
    
    ax.set_title(label, loc='left', fontweight='bold', fontsize=12)
    
    # Annotations
    for i, row in enumerate(pivot.index):
        for j, col in enumerate(pivot.columns):
            val = pivot.loc[row, col]
            if pd.isna(val):
                # Draw hatch or just 'N/A'
                ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="#999999")
            else:
                pct = f"{val*100:.1f}%"
                if col in [80, 90, 100] and abs(row - 0.05) < 1e-4:
                    pct += "*"
                # Text coloring for visibility against colormap
                colour = "black" if val < 0.65 else "white"
                weight = "bold" if val >= DENSE_LIMIT else "normal"
                
                # Highlight if it matches or exceeds the Dense Limit
                if val >= DENSE_LIMIT:
                    ax.text(j, i, pct, ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")
                    # Draw a distinct border around the cell
                    ax.add_patch(mpatches.FancyBboxPatch(
                        (j - 0.47, i - 0.47), 0.94, 0.94,
                        boxstyle="round,pad=0.01", linewidth=1.8, edgecolor="#0e4b85", facecolor="none"
                    ))
                else:
                    ax.text(j, i, pct, ha="center", va="center", fontsize=8.5, color=colour, fontweight=weight)
                    # If it beats the w=0 baseline, give a subtle border
                    if val > BASELINE_W0:
                        ax.add_patch(mpatches.FancyBboxPatch(
                            (j - 0.47, i - 0.47), 0.94, 0.94,
                            boxstyle="round,pad=0.01", linewidth=1.0, edgecolor="#555555", facecolor="none", linestyle="--"
                        ))
                        
    return im

im = plot_heatmap(ax1, pivot_stoch, "(a) Stochastic mode")
plot_heatmap(ax2, pivot_struct, "(b) Structural mode")

ax2.set_xticks(range(len(pivot_stoch.columns)))
ax2.set_xticklabels([str(int(s)) for s in pivot_stoch.columns], fontsize=9.5)
ax2.set_xlabel("Number of Seeds ($N$)", fontsize=11)

# Single colorbar for both
plt.tight_layout()
fig.subplots_adjust(right=0.88, hspace=0.3)
cbar_ax = fig.add_axes([0.90, 0.15, 0.022, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label("Accuracy", fontsize=10)

# Add baseline indicators to colorbar
cbar.ax.axhline(y=BASELINE_W0, color="orange", linewidth=1.5, linestyle="--", label="w=0 Floor")
cbar.ax.axhline(y=DENSE_LIMIT, color="blue", linewidth=1.5, linestyle="-", label="Dense Target")

# Save figure
plt.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"32k heatmap saved to {OUT}")
