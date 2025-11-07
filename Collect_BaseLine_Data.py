import oracledb
import pandas as pd
import warnings

# --- Configuration ---
# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# TPC-H user credentials
TPC_USER = "tpch_user"
TPC_PASSWORD = "tpch_user" # Use the correct password you set

# List of all PDB service names
# NOTE: Update these to match your environment's service names/TNS entries
PDB_SERVICE_NAMES = [
    "pdb_control",
    "pdb_perf_tune",
    "pdb_anomaly_det",
    "pdb_ai_train",
    "pdb_load_test",
    "pdb_realtime",
    "pdb_query_opt",
    "pdb_resource_mgmt",
    "pdb_log_analytics"
]

# Output file
OUTPUT_CSV_FILE = "baseline_performance_data.csv"

def consolidate_pdb_logs():
    """
    Connects to each PDB, extracts the tpch_query_log table,
    and consolidates them into a single CSV file.
    """
    print("--- Starting Data Consolidation ---")
    
    all_dataframes = []
    
    for pdb_service in PDB_SERVICE_NAMES:
        try:
            # Construct the DSN (Data Source Name)
            # Assumes the database is running on localhost and default port 1521
            # Adjust 'localhost:1521' if your server is elsewhere
            dsn = f"localhost:1521/{pdb_service}"
            
            # Establish connection
            with oracledb.connect(user=TPC_USER, password=TPC_PASSWORD, dsn=dsn) as connection:
                print(f"Successfully connected to {pdb_service}...")
                
                # SQL query to extract all data from the log table
                sql_query = "SELECT * FROM tpch_query_log"
                
                # Use pandas to read the SQL query results directly into a DataFrame
                df = pd.read_sql(sql_query, con=connection)
                
                print(f"Successfully extracted {len(df)} records from {pdb_service}.")
                
                if not df.empty:
                    all_dataframes.append(df)
                
        except Exception as e:
            print(f"!!! ERROR connecting to or reading from {pdb_service}: {e}")

    if not all_dataframes:
        print("--- No dataframes were collected. Exiting. ---")
        return

    # --- Data Consolidation Complete ---
    print("\n--- Data Consolidation Complete ---")
    
    # Combine all individual DataFrames into one master DataFrame
    master_df = pd.concat(all_dataframes, ignore_index=True)
    
    print(f"\nTotal records consolidated: {master_df.shape[0]}")
    print(f"Total columns: {master_df.shape[1]}")

    # --- Data Quality Check ---
    print("\n[Data Quality Check]")
    print(f"Total records consolidated: {master_df.shape[0]}")
    
    # [cite_start]Check for missing values [cite: 1646]
    print("\nChecking for missing values:")
    missing_values = master_df.isnull().sum()
    print(missing_values)
    
    # [cite_start]Check for duplicate records [cite: 1648]
    duplicate_count = master_df.duplicated().sum()
    print(f"\nNumber of duplicate records found: {duplicate_count}")

    # --- Save to CSV ---
    try:
        master_df.to_csv(OUTPUT_CSV_FILE, index=False)
        print(f"\n--- Successfully saved consolidated data to '{OUTPUT_CSV_FILE}' ---")
    except Exception as e:
        print(f"\n!!! ERROR saving to CSV: {e}")

# --- Main Execution Block ---
if __name__ == "__main__":
    # Initialize Oracle Client
    # This path may be different on your system.
    # Update it to point to your Oracle Instant Client directory.
    try:
         oracledb.init_oracle_client(lib_dir=r"C:\oracle\instantclient_19_22")
    except Exception as e:
         print(f"Warning: Could not initialize Oracle client. {e}")
         print("This may be okay if your environment variables (e.g., PATH) are set correctly.")

    consolidate_pdb_logs()