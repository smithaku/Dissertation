import os
import pandas as pd
from collections import defaultdict

# --- CONFIGURATION ---
LOG_DIRECTORY = '.' # Assumes the script is in the same folder as the log files
MONITORING_INTERVAL_SECONDS = 300 # 5 minutes between each log file

def analyze_monitoring_logs():
    """
    Processes all monitoring log files in the specified directory,
    calculates average performance metrics, and prints a formatted
    summary table for the dissertation.
    """
    print("--- Analyzing Monitoring Log Files for RQ3 ---")

    # Dictionary to hold all the collected data points
    data = defaultdict(lambda: defaultdict(list))
    log_files = [f for f in os.listdir(LOG_DIRECTORY) if f.startswith('monitoring_') and f.endswith('.log')]

    if not log_files:
        print("Error: No log files found. Make sure this script is in the same directory as your log files.")
        return

    # 1. Read and parse all log files
    for filename in log_files:
        pdb_name = "PDB_CONTROL" if "PDB_CONTROL" in filename else "PDB_PERF_TUNE"
        with open(os.path.join(LOG_DIRECTORY, filename), 'r') as f:
            next(f) # Skip the header line
            for line in f:
                if ',' in line:
                    try:
                        metric, value = line.strip().split(',')
                        data[pdb_name][metric].append(float(value))
                    except ValueError:
                        print(f"Warning: Could not parse line in {filename}: {line.strip()}")

    print(f"-> Successfully processed {len(log_files)} log files.")

    # 2. Calculate the final metrics
    results = {}
    for pdb_name, metrics in data.items():
        # Calculate average for most metrics
        avg_cpu = sum(metrics.get('CPU_used_by_this_session', [0])) / len(metrics.get('CPU_used_by_this_session', [1]))
        avg_reads = sum(metrics.get('physical_reads', [0])) / len(metrics.get('physical_reads', [1]))
        avg_writes = sum(metrics.get('physical_writes', [0])) / len(metrics.get('physical_writes', [1]))

        # Special calculation for Throughput (TPS)
        commits = metrics.get('user_commits', [])
        num_intervals = len(commits) - 1
        if num_intervals > 0:
            total_duration = num_intervals * MONITORING_INTERVAL_SECONDS
            total_commits = max(commits) - min(commits)
            tps = total_commits / total_duration if total_duration > 0 else 0
        else:
            tps = 0

        results[pdb_name] = {
            'Throughput (TPS)': tps,
            'Average CPU Usage (cs)': avg_cpu,
            'Average Physical Reads': avg_reads,
            'Average Physical Writes': avg_writes
        }

    # 3. Create and format the final DataFrame for the table
    df = pd.DataFrame(results).T # Transpose to get PDBs as rows
    df = df.rename(columns=lambda x: x.replace('_', ' ').title()) # Clean up column names

    # Reorder columns to match the dissertation table
    df = df[['Throughput (Tps)', 'Average Cpu Usage (Cs)', 'Average Physical Reads', 'Average Physical Writes']]

    # Prepare final table data
    control_data = df.loc['PDB_CONTROL']
    tuned_data = df.loc['PDB_PERF_TUNE']

    # Calculate percentage improvement
    improvement = ((tuned_data - control_data) / control_data) * 100
    # For CPU, a lower number is better, so we invert the improvement percentage
    improvement['Average Cpu Usage (Cs)'] *= -1

    # 4. Print the final formatted table
    print("\n" + "="*80)
    print("FINAL RESULTS FOR TABLE 4.5")
    print("="*80 + "\n")

    final_table_data = {
        'Performance Metric': df.columns,
        'PDB_CONTROL (Baseline)': [f"{val:,.0f}" for val in control_data],
        'PDB_PERF_TUNE (ML-Tuned)': [f"{val:,.0f}" for val in tuned_data],
        'Percentage Improvement (%)': [f"{val:+.1f}%" for val in improvement]
    }
    final_df = pd.DataFrame(final_table_data)
    print(final_df.to_string(index=False))
    print("\n" + "="*80)
    print("-> You can now copy and paste this table into your dissertation document.")


# --- Main Execution Block ---
if __name__ == "__main__":
    analyze_monitoring_logs()