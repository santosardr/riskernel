import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(current_dir, "../../scripts/benchmark32/results_baseline_32k.csv")
OUT = os.path.join(current_dir, "../fig_context_recovery_32k.pdf")

df = pd.read_csv(CSV_PATH)
df["Correct"] = pd.to_numeric(df["Correct"], errors="coerce")
df["Seeds"] = pd.to_numeric(df["Seeds"], errors="coerce")
df["Density"] = pd.to_numeric(df["Density"], errors="coerce")
df.loc[df["RIS_Mode"] == "structural", "Density"] += 0.01


# Filter for standard setup
df = df[(df['Model'] == 'qwen2') & (df['RoPE_Type'] == 'yarn') & (df['Window'] == 32768)]

# Group and calculate accuracy
summary = (
    df.groupby(["RIS_Mode", "Density", "Seeds"])
    .agg(Accuracy=("Correct", "mean"))
    .reset_index()
)
summary["Accuracy_pct"] = summary["Accuracy"] * 100

# Plot setup
plt.figure(figsize=(9, 5.5))

# Baselines for 32k
BASELINE_W0 = 59.375
DENSE_LIMIT = 71.875

# Color map for densities & modes
# Stochastic: Blue shades; Structural: Green shades
styles = {
    ('stochastic', 0.01): {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-', 'label': 'Stochastic 1%'},
    ('stochastic', 0.02): {'color': '#6baed6', 'marker': '^', 'linestyle': '--', 'label': 'Stochastic 2%'},
    ('stochastic', 0.05): {'color': '#08519c', 'marker': 'D', 'linestyle': ':', 'label': 'Stochastic 5%'},
    ('structural', 0.01): {'color': '#2ca02c', 'marker': 's', 'linestyle': '-', 'label': 'Structural 1%'},
    ('structural', 0.02): {'color': '#74c476', 'marker': 'v', 'linestyle': '--', 'label': 'Structural 2%'},
    ('structural', 0.05): {'color': '#006d2c', 'marker': 'X', 'linestyle': ':', 'label': 'Structural 5%'},
}

# Plot each configuration
for (mode, density), style in styles.items():
    sub = summary[(summary['RIS_Mode'] == mode) & (summary['Density'] == density)].sort_values('Seeds')
    if len(sub) > 0:
        plt.plot(sub['Seeds'], sub['Accuracy_pct'], 
                 label=style['label'], 
                 color=style['color'], 
                 marker=style['marker'], 
                 linestyle=style['linestyle'],
                 linewidth=1.8, markersize=7, alpha=0.85)

# Plot reference lines
plt.axhline(y=DENSE_LIMIT, color='#d62728', linestyle='-', linewidth=1.5, label='Full Dense Target (71.88%)')
plt.axhline(y=BASELINE_W0, color='#7f7f7f', linestyle='--', linewidth=1.5, label='w=0 Floor (59.38%)')

plt.xlabel('Ensemble Seeds ($N$)', fontsize=11)
plt.ylabel('Accuracy (%)', fontsize=11)
plt.title('Context Retrieval Recovery Rate (32k Window)', fontsize=13, fontweight='bold', pad=12)

# X-axis ticks
plt.xticks([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
plt.grid(True, which='both', linestyle=':', alpha=0.5)

# Y-axis secondary indicators (Recovery %)
# Gap is 71.875 - 59.375 = 12.5%
# Recovery Rate = (Acc - 59.375) / 12.5 * 100
ax = plt.gca()
ax2 = ax.twinx()
ax2.set_ylabel('Context Recovery Rate (%)', fontsize=11, color='#d62728')
ax2.set_ylim(ax.get_ylim())

# Map the y ticks of ax (Accuracy) to Recovery Rate (%)
y_ticks = ax.get_yticks()
recovery_ticks = [(y - BASELINE_W0) / 12.5 * 100 for y in y_ticks]
ax2.set_yticks(y_ticks)
ax2.set_yticklabels([f"{r:.0f}%" for r in recovery_ticks])
ax2.tick_params(axis='y', labelcolor='#d62728')

# Legends
# Place legend outside or inside nicely
ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5)

plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches='tight')
print(f"Context recovery figure saved to {OUT}")
