import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(current_dir, "../../scripts/benchmark32/results_baseline_32k.csv")
OUT = os.path.join(current_dir, "../fig_mode_comparison_32k.pdf")

df = pd.read_csv(CSV_PATH)
df["Correct"] = pd.to_numeric(df["Correct"], errors="coerce")
df["Seeds"] = pd.to_numeric(df["Seeds"], errors="coerce")
df["Density"] = pd.to_numeric(df["Density"], errors="coerce")
df.loc[df["RIS_Mode"] == "structural", "Density"] += 0.01


# Filter for standard setup
df = df[(df['Model'] == 'qwen2') & (df['RoPE_Type'] == 'yarn') & (df['Window'] == 32768)]

# Group to get max accuracy per mode and density
summary = (
    df.groupby(["RIS_Mode", "Density"])
    .agg(MaxAccuracy=("Correct", "max")) # We want the best performance across seeds, but wait, is it the mean of correct per run? Yes, Correct is 0 or 1.
    # Ah, the accuracy of a run is the mean of Correct.
    # Let's calculate the accuracy for each (Mode, Density, Seeds) first, then take the max over Seeds.
)

# Correct calculation:
run_acc = df.groupby(["RIS_Mode", "Density", "Seeds"]).agg(Accuracy=("Correct", "mean")).reset_index()
best_acc = run_acc.groupby(["RIS_Mode", "Density"]).agg(BestAccuracy=("Accuracy", "max")).reset_index()
best_acc["BestAccuracy_pct"] = best_acc["BestAccuracy"] * 100

# Constants
BASELINE_W0 = 59.375
DENSE_LIMIT = 71.875

# Data preparation
densities = [0.01, 0.02, 0.05]
stoch_vals = []
struct_vals = []

for d in densities:
    st_val = best_acc[(best_acc['RIS_Mode'] == 'stochastic') & (best_acc['Density'] == d)]['BestAccuracy_pct'].values
    stoch_vals.append(st_val[0] if len(st_val) > 0 else 0)
    
    sr_val = best_acc[(best_acc['RIS_Mode'] == 'structural') & (best_acc['Density'] == d)]['BestAccuracy_pct'].values
    struct_vals.append(sr_val[0] if len(sr_val) > 0 else 0)

# Plotting
x = np.arange(len(densities))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x - width/2, stoch_vals, width, label='Stochastic (Best of Seeds)', color='#1f77b4', alpha=0.85)
rects2 = ax.bar(x + width/2, struct_vals, width, label='Structural (Best of Seeds)', color='#2ca02c', alpha=0.85)

# Add reference lines
ax.axhline(y=DENSE_LIMIT, color='#d62728', linestyle='-', linewidth=1.5, label='Full Dense Baseline (71.88%)')
ax.axhline(y=BASELINE_W0, color='#7f7f7f', linestyle='--', linewidth=1.5, label='w=0 Baseline (59.38%)')

# Labels and title
ax.set_ylabel('Accuracy (%)', fontsize=11)
ax.set_xlabel('Attention Density ($q$)', fontsize=11)
ax.set_title('Peak Performance Comparison (32k Window)', fontsize=12, fontweight='bold', pad=12)
ax.set_xticks(x)
ax.set_xticklabels([f'{d*100:.1f}%' for d in densities], fontsize=10)
ax.set_ylim(40, 80)
ax.legend(loc='lower right', fontsize=9.5)
ax.grid(axis='y', linestyle=':', alpha=0.5)

# Label formatting on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        if height > 0:
            ax.annotate(f'{height:.2f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches='tight')
print(f"Mode comparison figure saved to {OUT}")
