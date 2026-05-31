import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths relative to this script's directory (riskernel/article/fig)
current_dir = os.path.dirname(os.path.abspath(__file__))
benchmark_dir = os.path.join(current_dir, "../../scripts/benchmark")
results_stoch = os.path.join(benchmark_dir, "results_64.csv")
results_struct = os.path.join(benchmark_dir, "results_structural_64.csv")

def get_scaling_data(filepath):
    df = pd.read_csv(filepath)
    df["Density"] = pd.to_numeric(df["Density"], errors="coerce")
    if "structural" in filepath:
        df["Density"] = df["Density"] + 0.01
        
    # filter for qwen2 + yarn + 65536 + 5% density
    df = df[(df['Model'] == 'qwen2') & 
            (df['RoPE_Type'] == 'yarn') & 
            (df['Window'] == 65536) & 
            (df['Density'] == 0.05)]

    
    # group by Seeds to get mean accuracy
    g = df.groupby(['Seeds']).agg(Correct=('Correct', 'sum'), Total=('Correct', 'count')).reset_index()
    # Filter for complete runs (64 QA)
    g = g[g['Total'] == 64]
    g['Accuracy'] = g['Correct'] / g['Total'] * 100
    return g

stoch_scaling = get_scaling_data(results_stoch)
struct_scaling = get_scaling_data(results_struct)

# Plot setup
plt.figure(figsize=(8, 5))

plt.plot(stoch_scaling['Seeds'], stoch_scaling['Accuracy'], 'o-', label='Stochastic', color='#1f77b4', alpha=0.8)
plt.plot(struct_scaling['Seeds'], struct_scaling['Accuracy'], 's-', label='Structural', color='#2ca02c', alpha=0.8)


# Baseline
plt.axhline(y=51.56, color='gray', linestyle='--', label='Zero-Context Baseline (w=0)', alpha=0.5)

plt.xlabel('Ensemble Seeds ($N$)')
plt.ylabel('Accuracy (%)')
plt.title('Ensemble Scaling Law: Stochastic vs. Structural (Density 5%)')
plt.legend()
plt.grid(alpha=0.3)
plt.ylim(45, 70)

plt.tight_layout()
output_pdf = os.path.join(current_dir, "../fig_ensemble_scaling.pdf")
plt.savefig(output_pdf, bbox_inches='tight')
print(f"Figure saved to {output_pdf} (Comparative)")
