import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def create_confusion_matrix_plot():
    """
    Generates and saves a professional confusion matrix plot.
    """
    # 1. --- Define Your Data ---
    # The final confusion matrix data from your experiment
    cm_data = np.array([[176, 0],
                        [3, 7]])

    # 2. --- Create Labels for Each Cell ---
    # These labels will include the name, the raw count, and the percentage
    group_names = ['True Negative (TN)', 'False Positive (FP)', 'False Negative (FN)', 'True Positive (TP)']
    group_counts = [f"{value}" for value in cm_data.flatten()]
    group_percentages = [f"{value:.2%}" for value in cm_data.flatten() / np.sum(cm_data)]
    
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    
    # 3. --- Generate the Plot ---
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_data, 
                annot=labels, 
                fmt='', 
                cmap='Blues', 
                cbar=False,
                xticklabels=['Normal', 'Anomaly'], 
                yticklabels=['Normal', 'Anomaly'],
                annot_kws={"size": 16})
    
    # 4. --- Add Titles and Save the Figure ---
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.title('Confusion Matrix for the Anomaly Detection Model', fontsize=16, pad=20)
    
    output_path = 'final_confusion_matrix.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Confusion matrix plot saved successfully as '{output_path}'")

# --- Run the function ---
if __name__ == "__main__":
    create_confusion_matrix_plot()