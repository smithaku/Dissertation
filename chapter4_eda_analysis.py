# Import necessary libraries
import pandas as pd
import oracledb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- SCRIPT CONFIGURATION ---
DB_USER = "tpch_user"
DB_PASSWORD = "YourSecurePassword"  # Use the actual password for tpch_user
DB_HOST = "db01.dc01.com"          # Corrected Hostname
DB_PORT = 1521                     # Default Oracle listener port

# *** FIXED: Use the fully qualified service names with the correct .dc01.com extension ***
PDB_SERVICE_NAMES = [
    "pdb_control.dc01.com",
    "pdb_perf_tune.dc01.com",
    "pdb_anomaly_det.dc01.com",
    "pdb_ai_train.dc01.com",
    "pdb_load_test.dc01.com",
    "pdb_realtime.dc01.com",
    "pdb_query_opt.dc01.com",
    "pdb_resource_mgmt.dc01.com",
    "pdb_log_analytics.dc01.com"
]

def consolidate_data():
    """
    Connects to each PDB, extracts data from the tpch_query_log table,
    and consolidates it into a single Pandas DataFrame.
    """
    print("--- Starting Data Consolidation ---")
    all_pdb_data = []
    
    for pdb_service in PDB_SERVICE_NAMES:
        try:
            # Construct the Data Source Name (DSN) for the connection
            dsn = f"{DB_HOST}:{DB_PORT}/{pdb_service}"
            
            # Establish the connection
            with oracledb.connect(user=DB_USER, password=DB_PASSWORD, dsn=dsn) as connection:
                print(f"Successfully connected to {pdb_service}...")
                
                # SQL query to extract all data from the log table
                sql_query = "SELECT * FROM tpch_query_log ORDER BY query_no"
                
                # Use pandas to read the SQL query results directly into a DataFrame
                pdb_df = pd.read_sql_query(sql_query, connection)
                
                # Add the DataFrame to our list
                all_pdb_data.append(pdb_df)
                print(f"Successfully extracted {len(pdb_df)} records from {pdb_service}.")

        except oracledb.Error as e:
            print(f"Error connecting to or fetching data from {pdb_service}: {e}")
            
    print("--- Data Consolidation Complete ---\n")
    
    # Concatenate all DataFrames from the list into a single master DataFrame
    if all_pdb_data:
        master_df = pd.concat(all_pdb_data, ignore_index=True)
        return master_df
    else:
        return pd.DataFrame()

def perform_eda(df):
    """
    Performs Exploratory Data Analysis on the consolidated DataFrame.
    """
    if df.empty:
        print("DataFrame is empty. Cannot perform EDA.")
        return

    print("--- Starting Exploratory Data Analysis ---")

    # 1. Data Cleaning and Quality Assurance Check
    print("\n[Data Quality Check]")
    print(f"Total records consolidated: {len(df)}")
    print(f"Checking for missing values:\n{df.isnull().sum()}")
    print(f"Number of duplicate records found: {df.duplicated().sum()}")

    # 2. Descriptive Statistics
    print("\n[Table 4.1: Descriptive Statistics of Baseline Query Execution Times (seconds)]")
    # Group by query number and calculate descriptive statistics
    descriptive_stats = df.groupby('QUERY_NO')['EXECUTION_SECONDS'].describe()
    print(descriptive_stats.to_string())

    # 3. Visual Analysis
    print("\n[Generating Visualizations...]")
    sns.set_theme(style="whitegrid")

    # --- Box Plot for a representative query ---
    plt.figure(figsize=(15, 8))
    # Filter for a specific query to visualize, e.g., Query 11
    query_11_data = df[df['QUERY_NO'] == 11]
    
    sns.boxplot(data=query_11_data, x='PDB_NAME', y='EXECUTION_SECONDS', palette="viridis")
    
    plt.title('Figure 4.1: Baseline Execution Times for TPC-H Query 11 Across All Nine PDBs', fontsize=16)
    plt.xlabel('Pluggable Database (PDB)', fontsize=12)
    plt.ylabel('Execution Time (seconds)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the figure to a file
    boxplot_filename = 'figure_4_1_boxplot_query_11.png'
    plt.savefig(boxplot_filename)
    print(f"-> Saved box plot to '{boxplot_filename}'")
    plt.close()

    # --- Histogram for a representative long-running query ---
    plt.figure(figsize=(12, 7))
    # Filter for a specific query to visualize, e.g., Query 18
    query_18_data = df[df['QUERY_NO'] == 18]

    sns.histplot(data=query_18_data, x='EXECUTION_SECONDS', kde=True, bins=15, color='purple')
    
    plt.title('Figure 4.2: Distribution of Baseline Execution Times for TPC-H Query 18', fontsize=16)
    plt.xlabel('Execution Time (seconds)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    
    # Save the figure to a file
    histogram_filename = 'figure_4_2_histogram_query_18.png'
    plt.savefig(histogram_filename)
    print(f"-> Saved histogram to '{histogram_filename}'")
    plt.close()
    
    print("\n--- EDA Complete ---")


# Main execution block
if __name__ == "__main__":
    # Step 1: Consolidate data from all PDBs
    master_dataframe = consolidate_data()
    
    # Step 2: Perform EDA on the consolidated data
    perform_eda(master_dataframe)