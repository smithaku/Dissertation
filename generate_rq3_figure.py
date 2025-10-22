import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def create_performance_barchart():
    """
    Generates the comparative bar chart for system-level performance
    metrics (Figure 4.8) using the final calculated data.
    """
    print("--- Generating Figure 4.8: Comparative System Performance Metrics ---")

    # 1. --- Final Data from the Monitoring Logs ---
    # This data is based on the programmatic analysis of your 14 log files.
    data = {
        'Metric': ['Throughput (TPS)', 'Avg. CPU Usage (cs)', 'Avg. Physical Reads', 'Avg. Physical Writes'],
        'PDB_CONTROL': [1.38, 128691, 84649116, 566219],
        'PDB_PERF_TUNE': [0.00, 150776, 104259929, 384196]
    }
    df = pd.DataFrame(data)
    
    # --- Create a 2x2 Subplot Layout ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Figure 4.8: Comparative System Performance Metrics', fontsize=20, y=1.02)
    
    # Flatten the 2x2 array of axes for easy iteration
    axes = axes.flatten()
    
    colors = ['#FF7F50', '#4CAF50'] # Custom colors for Control and Tuned

    # 2. --- Generate a Bar Chart for Each Metric ---
    for i, metric in enumerate(df['Metric']):
        ax = axes[i]
        pdb_names = ['PDB_CONTROL', 'PDB_PERF_TUNE']
        values = [df[df['Metric'] == metric]['PDB_CONTROL'].values[0], 
                  df[df['Metric'] == metric]['PDB_PERF_TUNE'].values[0]]

        bars = ax.bar(pdb_names, values, color=colors)
        
        # Add titles and labels
        ax.set_title(metric, fontsize=14, pad=15)
        ax.set_ylabel('Value', fontsize=12)
        ax.tick_params(axis='x', labelsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Format y-axis ticks to be more readable (e.g., with commas)
        ax.get_yaxis().set_major_formatter(
            plt.FuncFormatter(lambda x, p: format(int(x), ',')))

        # Add data labels on top of the bars
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval, 
                    f'{yval:,.2f}', 
                    va='bottom',  # Align text above the bar
                    ha='center', 
                    fontsize=12)

    # 3. --- Finalize and Save the Figure ---
    plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to make room for the suptitle
    
    output_path = 'final_figure_4_8_system_performance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"-> Bar chart saved successfully as '{output_path}'")

# --- Run the function ---
if __name__ == "__main__":
    create_performance_barchart()