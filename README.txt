*** DESCRIPTION FOR ALL MAJOR PYTHON SCRIPT THAT CONTAIN THE CORE MACHINE LERNING LOGIC FOR THIS DISSERTATION *** 

## Setting up the environment
TPC-H Utilities (dbgen.exe, qgen.exe) and makefile:
These standard TPC-H tools were used to generate the 50GB dataset (.tbl files) and the 22 benchmark queries (.sql file), that were loaded into the databases initially before the start of the experiment.

The SQL*Loader Control Files (.ctl):
These files provided instructions to Oracle's sqlldr.exe utility on how to load the pipe-delimited data from the .tbl files into the corresponding Oracle tables

SQL DDL Scripts Create_TPC-H_Tablestxt (Table/Constraint/Index Creation):
These SQL scripts were used to create the TPC-H tables, and later, the primary keys, foreign keys, and any necessary indexes within each PDB .

## Baseline data from TPC-H 
Workload Execution Scripts (execute_all_benchmarks.bat and run_and_log_query.sql):
These scripts were used to run the TPC-H benchmark queries systematically across all nine PDBs during the initial baseline data collection phase

## Baseline data collection for this dissertation
 The initial baseline data (baseline_performance_data.csv) used to train the anomaly detection model was collected earlier using a similar process. The TPC-H benchmark was run only on PDB_CONTROL using the automated script framework (execute_all_benchmarks.bat and run_and_log_query.sql), and the query execution times were logged into the tpch_query_log table (CREATE_tpch_query_log_table.sql) within the PDB. This data was then consolidated into the CSV file.


## Final Data Collection for this dissertation
The script used to collect data was the evaluate_rl_agent.py script.
It collected data on Query Execution Time (for RQ1), and System-Level Metrics (for RQ3).
Data collection was performed on both PDB_CONTROL and PDB_PERF_TUNE, and exported into final_performance_results.csv.

## Data Cleaning

The data cleaning process was straightforward due to the controlled nature of the data collection.
The Consolidation Script (chapter4_eda_analysis.py - part of EDA), included the data cleaning, and intergrity check.
The first step involved consolidating the baseline query logs from the tpch_query_log tables in all nine PDBs into a single Pandas DataFrame . This script used the oracledb library to connect and fetch data.

Immediately after consolidation, automated checks were performed within the script to ensure data integrity

I used the pandas isnull().sum() method on the consolidated DataFrame. This check confirmed zero missing values across all columns , validating the robustness of the SQL*Plus logging script (run_and_log_query.sql) used during collection.

I used the pandas duplicated().sum() method. This check confirmed zero duplicate records , verifying that each row represented a unique query execution event.

Because the data collection was fully automated within a controlled virtual environment and the initial checks showed no missing or duplicate data, no further cleaning steps like imputation or aggressive outlier removal were necessary after the data was collected.

The system-level metrics collected in the .log files were also used directly without cleaning, as they were simple key-value pairs captured from reliable system views.

## Machine Learning models
This train_anomaly_model.py primary function is to build and train the Autoencoder neural network responsible for anomaly detection. It learns the patterns of "normal" database operation exclusively from the baseline data collected from the PDB_CONTROL environment above.

This oracle_rl_environment.py defines the custom Reinforcement Learning environment (OracleTuningEnv class) . It acts as the crucial bridge between the RL agent (the "brain") and the live Oracle database (PDB_AI_TRAIN during training). It simulates the "game" the agent plays, defining the rules, actions, and scoring system according to the standard Gymnasium.

This train_rl_agent.py orchestrates the training process for the Deep Q-Network (DQN) agent . It brings together the custom OracleTuningEnv and the DQNAgent class, running them through numerous episodes to allow the agent to learn an effective policy for choosing optimizer hints that reduce query execution time.

## Data visualization

Figure 4.4: Comparative Query Execution Times
Script Name: evaluate_rl_agent.py
Libraries Used: matplotlib, seaborn, pandas
After running the TPC-H benchmark on both PDB_CONTROL and PDB_PERF_TUNE and calculating the final comparative query times (Table 4.2), this same script uses matplotlib and seaborn to generate the side-by-side bar chart (final_figure_4_4_comparative_qet.png).


Figure 4.6: Confusion Matrix
Script Name: generate_dissertation_figures.py
Libraries Used: matplotlib, seaborn, numpy
The evaluate_anomaly_model.py script calculates the raw confusion matrix values (TN, FP, FN, TP) . While it prints these numbers, the final, polished visual plot with labels and percentages was generated by the separate generate_dissertation_figures.py script (or create_confusion_matrix.py that I provided).


Figure 4.7: AUC-ROC Curve
Script Name: generate_auc_roc_curve.py
Libraries Used: matplotlib, scikit-learn, numpy, tensorflow, pandas, pickle
This dedicated script loads the saved anomaly detection model (.h5), the scaler (.pkl), and the test data (.csv). It calculates the model's reconstruction errors for the test set, treating these errors as the prediction scores. It then uses scikit-learn's roc_curve and auc functions to compute the False Positive Rates, True Positive Rates, and the Area Under the Curve


Figure 4.8: Comparative System Performance Metrics
Script Name: generate_rq3_figure.py
Libraries Used: matplotlib, pandas
This script takes the final, averaged system-level metrics (calculated by process_monitoring_logs.py and presented in Table 4.5) as input. It uses matplotlib to create a figure with four distinct subplots, one for each metric (Throughput, CPU Usage, Physical Reads, Physical Writes)


