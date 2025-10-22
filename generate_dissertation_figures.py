import os
import graphviz
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Suppress a font warning from matplotlib on Windows if needed
# os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"

def create_architecture_diagram():
    """
    Generates the High-Level System Architecture diagram (Figure 3.1)
    using Graphviz.
    """
    print("--- Generating Figure 3.1: High-Level System Architecture ---")

    # Create a new directed graph with Left-to-Right layout
    dot = graphviz.Digraph('System_Architecture', comment='Dissertation System Architecture')
    dot.attr(rankdir='LR', splines='ortho', concentrate='true')

    # Define default styles for nodes and edges
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue', fontname='Arial', fontsize='10')
    dot.attr('edge', fontname='Arial', fontsize='9')

    # Subgraph for the Oracle Database Environment (Zone 1)
    with dot.subgraph(name='cluster_oracle') as c:
        c.attr(label='Oracle 19c Database Environment (CDBML)', style='rounded', color='grey', fontname='Arial Bold', fontsize='12')
        # Define the PDB nodes with specific roles and colors
        c.node('pdb_control', 'PDB_CONTROL\n(Baseline / Control)', fillcolor='#FFCCCB') # Light Red
        c.node('pdb_perf_tune', 'PDB_PERF_TUNE\n(ML-Tuned for RQ1/RQ3)', fillcolor='#90EE90') # Light Green
        c.node('pdb_ai_train', 'PDB_AI_TRAIN\n(RL Agent Training)', fillcolor='#FFFFE0') # Light Yellow
        c.node('pdb_anomaly_det', 'PDB_ANOMALY_DET\n(Anomaly Detection Target for RQ2)', fillcolor='#ADD8E6') # Lighter Blue
        # Add other PDBs if needed for completeness, perhaps less prominent
        c.node('other_pdbs', 'Other PDBs\n(Load Test, Realtime, etc.)', fillcolor='lightgrey', shape='ellipse')


    # Subgraph for the Python Machine Learning Pipeline (Zone 2)
    with dot.subgraph(name='cluster_ml') as c:
        c.attr(label='Python Machine Learning Pipeline', style='rounded', color='blue', fontname='Arial Bold', fontsize='12')
        # Define the pipeline stages
        c.node('data_collection', '1. Data Collection\n(Oracle Logs / V$ Views)')
        c.node('feature_engineering', '2. Feature Engineering\n(Temporal, Encoding, Scaling)')
        c.node('model_training', '3. Model Training', shape='Mdiamond', fillcolor='gold') # Highlight training step
        c.node('anomaly_model', 'Anomaly Detection Model\n(Autoencoder)', shape='cylinder', fillcolor='lightcyan')
        c.node('tuning_model', 'Performance Tuning Model\n(DQN Agent)', shape='cylinder', fillcolor='lightcyan')
        c.node('evaluation_engine', '4. Evaluation & Inference\nEngine', shape='Mdiamond', fillcolor='gold') # Highlight evaluation step

    # --- Define the data flows (edges) ---

    # Training Data Flow (Solid arrows, Blue)
    dot.edge('pdb_ai_train', 'data_collection', label=' RL Training Logs', color='blue', fontcolor='blue')
    # Using PDB_CONTROL logs for baseline anomaly model training
    dot.edge('pdb_control', 'data_collection', label=' Anomaly Baseline Logs', color='darkgreen', fontcolor='darkgreen')
    dot.edge('data_collection', 'feature_engineering', label='Cleaned Data')
    dot.edge('feature_engineering', 'model_training', label='Feature Vectors')
    dot.edge('model_training', 'anomaly_model', label=' Trains Anomaly Model')
    dot.edge('model_training', 'tuning_model', label=' Trains Tuning Agent')

    # Evaluation/Inference Flow (Dashed arrows, Purple)
    dot.edge('tuning_model', 'evaluation_engine', label=' Loads Trained Agent', style='dashed', color='purple', fontcolor='purple')
    dot.edge('anomaly_model', 'evaluation_engine', label=' Loads Trained Model', style='dashed', color='purple', fontcolor='purple')
    # Agent applies tuning to PDB_PERF_TUNE
    dot.edge('evaluation_engine', 'pdb_perf_tune', label=' Applies Tuning Policy', style='dashed', color='purple', fontcolor='purple', dir='back')
    # Anomaly model evaluates PDB_ANOMALY_DET
    dot.edge('evaluation_engine', 'pdb_anomaly_det', label=' Monitors for Anomalies', style='dashed', color='purple', fontcolor='purple', dir='back')

    # Final Comparison Flow (Dotted arrows, Red)
    dot.edge('pdb_control', 'evaluation_engine', label=' Performance Comparison (RQ1/RQ3)', style='dotted', constraint='false', color='red', fontcolor='red')
    dot.edge('pdb_perf_tune', 'evaluation_engine', label=' Performance Comparison (RQ1/RQ3)', style='dotted', constraint='false', color='red', fontcolor='red')


    # Render and save the diagram as PNG
    output_path = 'final_figure_3_1_system_architecture'
    try:
        dot.render(output_path, format='png', cleanup=True, view=False)
        print(f" Figure 3.1 saved successfully as '{output_path}.png'")
    except graphviz.backend.execute.CalledProcessError as e:
        print(f" Error rendering Figure 3.1: {e}")
        print("Ensure Graphviz is installed and its 'bin' directory is in your system PATH.")

def create_confusion_matrix_plot():
    """
    Generates the Confusion Matrix plot (Figure 4.6) for the
    anomaly detection model using Seaborn and Matplotlib.
    """
    print("\n--- Generating Figure 4.6: Confusion Matrix ---")

    # --- Final Confusion Matrix Data (From your evaluate_anomaly_model.py output) ---
    cm_data = np.array([[176, 0],   # TN, FP
                        [3, 7]])    # FN, TP

    # Labels for each quadrant
    group_names = ['True Negative (TN)', 'False Positive (FP)', 'False Negative (FN)', 'True Positive (TP)']
    # Raw counts
    group_counts = [f"{value}" for value in cm_data.flatten()]
    # Percentages of the total
    group_percentages = [f"{value:.2%}" for value in cm_data.flatten() / np.sum(cm_data)]

    # Combine labels for annotation
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    # --- Create the plot ---
    plt.figure(figsize=(9, 7)) # Slightly adjusted size
    sns.heatmap(cm_data,
                annot=labels,       # Use combined labels
                fmt='',             # Format is handled by the labels array
                cmap='Blues',       # Color scheme
                cbar=False,         # Hide the color bar legend
                xticklabels=['Predicted Normal', 'Predicted Anomaly'], # X-axis labels
                yticklabels=['Actual Normal', 'Actual Anomaly'],    # Y-axis labels
                annot_kws={"size": 14}) # Font size for annotations

    # Add titles and labels with appropriate font sizes
    plt.ylabel('True Label', fontsize=13)
    plt.xlabel('Predicted Label', fontsize=13)
    plt.title('Figure 4.6: Confusion Matrix for the Anomaly Detection Model', fontsize=15, pad=20)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11, rotation=0) # Keep y-axis labels horizontal

    # --- Save the figure ---
    output_path = 'final_figure_4_6_confusion_matrix.png'
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight') # High resolution, tight layout
        print(f" Figure 4.6 saved successfully as '{output_path}'")
    except Exception as e:
        print(f" Error saving Figure 4.6: {e}")
    plt.close() # Close the plot to free memory


# --- Main Execution Block ---
if __name__ == "__main__":
    create_architecture_diagram()
    create_confusion_matrix_plot()