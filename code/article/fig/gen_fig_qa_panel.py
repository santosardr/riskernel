import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths relative to this script's directory (riskernel/article/fig)
current_dir = os.path.dirname(os.path.abspath(__file__))
benchmark_dir = os.path.join(current_dir, "../../scripts/benchmark")
results_stoch = os.path.join(benchmark_dir, "results_64.csv")
results_struct = os.path.join(benchmark_dir, "results_structural_64.csv")

def get_data(filepath):
    df = pd.read_csv(filepath)
    df["Density"] = pd.to_numeric(df["Density"], errors="coerce")
    if "structural" in filepath:
        df["Density"] = df["Density"] + 0.01
    # filter for qwen2 + yarn + 65536
    df = df[(df['Model'] == 'qwen2') & (df['RoPE_Type'] == 'yarn') & (df['Window'] == 65536)]

    # group by density and seed to get independent runs (64 questions each)
    g = df.groupby(['Density', 'Seeds']).agg(Correct=('Correct', 'sum'), Total=('Correct', 'count')).reset_index()
    # Filter for complete runs (64 QA)
    g = g[g['Total'] == 64]
    g['Accuracy'] = g['Correct'] / g['Total'] * 100
    # Group by density to get mean accuracy
    summary = g.groupby('Density')['Accuracy'].mean().reset_index()
    return summary

stoch_data = get_data(results_stoch)
struct_data = get_data(results_struct)

# Baseline
dense_baseline = 51.56

# Plot setup
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

# Panel (a) - Stochastic
ax1.plot(stoch_data['Density'], stoch_data['Accuracy'], 'o-', label='Stochastic-YaRN', color='#1f77b4')
ax1.axhline(y=dense_baseline, color='gray', linestyle='--', label='Zero-Context Baseline (w=0)')
ax1.set_xlabel('Density ($q$)')
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('(a) Stochastic Mode', loc='left', fontsize=10)
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

# Panel (b) - Structural
ax2.plot(struct_data['Density'], struct_data['Accuracy'], 's-', label='Structural-YaRN', color='#2ca02c')
ax2.axhline(y=dense_baseline, color='gray', linestyle='--', label='Zero-Context Baseline (w=0)')
ax2.set_xlabel('Density ($q$)')
ax2.set_title('(b) Structural Mode', loc='left', fontsize=10)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

plt.tight_layout()
output_pdf = os.path.join(current_dir, "../fig_qa_panel.pdf")
plt.savefig(output_pdf, bbox_inches='tight')
print(f"Figure saved to {output_pdf} (Cleaned Labels)")
