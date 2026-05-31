import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

# Styling
plt.style.use('seaborn-v0_8-whitegrid') if 'seaborn-v0_8-whitegrid' in plt.style.available else plt.style.use('ggplot')

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
CSV_64 = os.path.join(current_dir, "../../scripts/benchmark/results_64.csv")
CSV_32 = os.path.join(current_dir, "../../scripts/benchmark32/results_baseline_32k.csv")
OUT = os.path.join(current_dir, "../fig_accuracy_window.pdf")

# Constants
BASELINE_64 = 51.5625
BASELINE_32 = 59.375
DENSE_32 = 71.875

def load_data():
    df_64 = pd.read_csv(CSV_64)
    df_64 = df_64[df_64['Model'] == 'qwen2'].copy()
    df_64["Correct"] = pd.to_numeric(df_64["Correct"], errors="coerce")
    
    # Merge newly calculated windows in Code Ocean environment
    if os.path.exists("/results/results_64.csv"):
        df_new = pd.read_csv("/results/results_64.csv")
        df_new = df_new[df_new['Model'] == 'qwen2'].copy()
        df_new["Correct"] = pd.to_numeric(df_new["Correct"], errors="coerce")
        
        new_windows = df_new['Window'].unique()
        df_64 = df_64[~df_64['Window'].isin(new_windows)].copy()
        df_64 = pd.concat([df_64, df_new], ignore_index=True)
    
    df_32 = pd.read_csv(CSV_32)
    df_32 = df_32[df_32['Model'] == 'qwen2'].copy()
    df_32["Correct"] = pd.to_numeric(df_32["Correct"], errors="coerce")
    
    return df_64, df_32

def main():
    df_64, df_32 = load_data()
    
    # Calculate run accuracy
    # Group by RIS_Mode as well if it exists, to prevent averaging different modes
    group_cols_64 = ['Window', 'RoPE_Type', 'Density', 'Seeds']
    if 'RIS_Mode' in df_64.columns:
        group_cols_64.append('RIS_Mode')
    run_acc_64 = df_64.groupby(group_cols_64)['Correct'].mean().reset_index()

    group_cols_32 = ['Window', 'RoPE_Type', 'Density', 'Seeds']
    if 'RIS_Mode' in df_32.columns:
        group_cols_32.append('RIS_Mode')
    run_acc_32 = df_32.groupby(group_cols_32)['Correct'].mean().reset_index()
    
    # Get best per window
    # 4k
    acc_4k = run_acc_64[run_acc_64['Window'] == 4096]['Correct'].max() * 100
    # 8k
    acc_8k = run_acc_64[run_acc_64['Window'] == 8192]['Correct'].max() * 100
    # 16k
    acc_16k = run_acc_64[run_acc_64['Window'] == 16384]['Correct'].max() * 100
    # 32k (balanced)
    acc_32k = run_acc_32[run_acc_32['Window'] == 32768]['Correct'].max() * 100
    
    # 64k
    acc_64k_linear = run_acc_64[(run_acc_64['Window'] == 65536) & (run_acc_64['RoPE_Type'] == 'linear')]['Correct'].max() * 100
    acc_64k_yarn = run_acc_64[(run_acc_64['Window'] == 65536) & (run_acc_64['RoPE_Type'] == 'yarn')]['Correct'].max() * 100
    
    print(f"Loaded Accuracies:")
    print(f"4k: {acc_4k:.2f}%")
    print(f"8k: {acc_8k:.2f}%")
    print(f"16k: {acc_16k:.2f}%")
    print(f"32k: {acc_32k:.2f}%")
    print(f"64k Linear: {acc_64k_linear:.2f}%")
    print(f"64k YaRN: {acc_64k_yarn:.2f}%")
    
    windows = [4096, 8192, 16384, 32768, 65536]
    
    # Series
    linear_accs = [acc_4k, acc_8k, acc_16k, acc_32k, acc_64k_linear]
    yarn_accs = [acc_4k, acc_8k, acc_16k, acc_32k, acc_64k_yarn]
    
    plt.figure(figsize=(8.5, 5))
    
    # Plot curves
    plt.plot(windows, linear_accs, marker='o', markersize=6, linewidth=2.0, color='#D62728', label='Linear Interpolation')
    plt.plot(windows, yarn_accs, marker='^', markersize=6, linewidth=2.0, color='#1F77B4', label='YaRN scaling')
    
    # Baselines
    plt.axhline(y=BASELINE_64, color='#7f7f7f', linestyle='--', linewidth=1.2, label='w=0 Floor (64k): 51.56%')
    plt.axhline(y=BASELINE_32, color='#bcbd22', linestyle='--', linewidth=1.2, label='w=0 Floor (32k): 59.38%')
    plt.axhline(y=DENSE_32, color='#2ca02c', linestyle=':', linewidth=1.5, label='Full Dense Target (32k): 71.88%')
    
    # X-axis formatting
    plt.xscale('log', base=2)
    plt.xticks(windows, [f'{w//1024}k' for w in windows])
    plt.minorticks_off()
    
    # Native limit line
    plt.axvline(x=32768, color='gray', linestyle=':', alpha=0.7)
    plt.text(32768 * 1.05, 30, 'Native Limit (32k)', rotation=90, verticalalignment='bottom', fontsize=10, color='gray', fontweight='bold')
    
    # Labels and Title
    plt.xlabel('Context Window Size (Tokens)', fontsize=11)
    plt.ylabel('Accuracy (%)', fontsize=11)
    plt.title('Performance Stability: Accuracy vs Context Window Size\n(Qwen2-1.5B, RIS-Kernel ESP)', fontsize=13, fontweight='bold', pad=15)
    plt.ylim(0, 100)
    plt.xlim(3000, 80000)
    
    # Legend
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5, loc='lower left')
    plt.grid(True, which='both', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(OUT, dpi=300, bbox_inches='tight')
    print(f"Accuracy vs Window figure saved to {OUT}")

if __name__ == "__main__":
    main()
