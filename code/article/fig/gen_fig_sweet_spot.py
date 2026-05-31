import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
CSV_STOCH = os.path.join(current_dir, "../../scripts/benchmark/results_sweet_spot_64.csv")
CSV_STRUCT = os.path.join(current_dir, "../../scripts/benchmark/results_structural_sweet_spot_64.csv")
OUT = os.path.join(current_dir, "../fig_sweet_spot_64k.pdf")

# ── Load & aggregate Stochastic ─────────────────────────────────────────────
df_stoch = pd.read_csv(CSV_STOCH)
df_stoch["Correct"] = pd.to_numeric(df_stoch["Correct"], errors="coerce")
summary_stoch = (
    df_stoch.groupby(["Density", "Seeds"])
    .agg(Accuracy=("Correct", "mean"))
    .reset_index()
)
pivot_stoch = summary_stoch.pivot(index="Density", columns="Seeds", values="Accuracy")

# ── Load & aggregate Structural ─────────────────────────────────────────────
df_struct = pd.read_csv(CSV_STRUCT)
df_struct["Correct"] = pd.to_numeric(df_struct["Correct"], errors="coerce")
df_struct["Density"] = pd.to_numeric(df_struct["Density"], errors="coerce")
if df_struct["Density"].max() < 0.06:  # Only add if it's logging global density (e.g. max is 0.05)
    df_struct["Density"] = df_struct["Density"] + 0.01
summary_struct = (
    df_struct.groupby(["Density", "Seeds"])
    .agg(Accuracy=("Correct", "mean"))
    .reset_index()
)
pivot_struct = summary_struct.pivot(index="Density", columns="Seeds", values="Accuracy")


# Reindex Structural to have the same columns (seeds) as Stochastic
pivot_struct_aligned = pivot_struct.reindex(columns=pivot_stoch.columns)

# ── Pretty labels ───────────────────────────────────────────────────────────
row_labels = [f"{d*100:.1f}%" for d in pivot_stoch.index]
col_labels = [str(int(s)) for s in pivot_stoch.columns]

# ── Colour map: diverging around the zero-context baseline (0.5156) ─────────
BASELINE = 0.5156
VMIN, VMAX = 0.53, 0.63          # tight range to maximise contrast
PEAK = 0.625

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8.5), sharex=True)

def plot_heatmap(ax, pivot, is_structural):
    # Set background color for NaNs/not run configs to light grey
    ax.set_facecolor('#eaeaea')
    
    im = ax.imshow(
        pivot.values,
        aspect="auto",
        cmap="RdYlGn",
        vmin=VMIN, vmax=VMAX,
        interpolation="nearest",
    )
    
    # Axis ticks
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(row_labels, fontsize=9.5)
    ax.set_ylabel("Attention Density ($q$)", fontsize=10.5, labelpad=6)
    
    # Cell annotations
    for i, row in enumerate(pivot.index):
        for j, col in enumerate(pivot.columns):
            val = pivot.loc[row, col]
            if pd.isna(val):
                if is_structural:
                    # For structural, check if this seed was part of the sweep (38-50)
                    if 38 <= col <= 50:
                        # Inside the grid, so it's a pending/not run config
                        ax.text(j, i, "—", ha="center", va="center",
                                fontsize=9, color="grey", fontweight="bold")
                    else:
                        # Outside the grid (not run by design)
                        pass
                else:
                    # For stochastic, NaNs are OOM slots
                    ax.text(j, i, "OOM", ha="center", va="center",
                            fontsize=8, color="grey", style="italic", fontweight="bold")
            else:
                pct = f"{val*100:.2f}%"
                colour = "black" if 0.555 <= val <= 0.615 else "white"
                weight = "bold" if val >= PEAK else "normal"
                ax.text(j, i, pct, ha="center", va="center",
                        fontsize=8, color=colour, fontweight=weight)
                
                # Highlight peak cells with a rectangle
                if val >= PEAK:
                    ax.add_patch(mpatches.FancyBboxPatch(
                        (j - 0.48, i - 0.48), 0.96, 0.96,
                        boxstyle="round,pad=0.05",
                        linewidth=1.6, edgecolor="#2166ac",
                        facecolor="none", zorder=3,
                    ))
    return im

im1 = plot_heatmap(ax1, pivot_stoch, is_structural=False)
ax1.set_title("(a) Stochastic Mode (YaRN, Qwen2-1.5B)", loc='left', fontsize=11, fontweight='bold', pad=8)

im2 = plot_heatmap(ax2, pivot_struct_aligned, is_structural=True)
ax2.set_title("(b) Structural Mode (YaRN, Qwen2-1.5B)", loc='left', fontsize=11, fontweight='bold', pad=8)

ax2.set_xticks(range(len(pivot_stoch.columns)))
ax2.set_xticklabels(col_labels, fontsize=9.5, rotation=45, ha="right")
ax2.set_xlabel("Number of Seeds ($N$)", fontsize=11, labelpad=6)

ax1.tick_params(top=False, bottom=True, labeltop=False, labelbottom=False)
ax2.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

# ── Colour bar ───────────────────────────────────────────────────────────────
plt.tight_layout()
fig.subplots_adjust(right=0.90, hspace=0.25)
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig.colorbar(im1, cax=cbar_ax)
cbar.set_label("Accuracy", fontsize=10)
cbar.ax.axhline(y=BASELINE, color="navy", linewidth=1.5, linestyle="--")
cbar.ax.text(2.6, BASELINE, " baseline\n (51.56%)",
             va="center", fontsize=8, color="navy",
             transform=cbar.ax.get_yaxis_transform())

plt.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"Saved → {OUT}")
