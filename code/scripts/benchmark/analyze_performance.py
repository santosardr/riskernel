import os
import pandas as pd

def analyze_performance(csv_path=None):
    """
    Calculates the mean accuracy grouped by Density and Seeds.
    """
    if csv_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, "results_sweet_spot_zone.csv")
    try:
        df = pd.read_csv(csv_path)
        
        # Group by Density and Seeds, calculate mean of 'Correct'
        summary = df.groupby(['Density', 'Seeds'])['Correct'].mean().reset_index()
        
        print("Performance Summary (Accuracy by Density and Seeds):")
        print(summary.to_string(index=False))
        
        # Identify the best performing configuration
        best_row = summary.loc[summary['Correct'].idxmax()]
        print(f"\nBest Configuration: Density={best_row['Density']}, Seeds={best_row['Seeds']} with Accuracy={best_row['Correct']:.3f}")
        
    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    analyze_performance()
