import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pickle
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
TEST_DATA_PATH = "anomaly_test_data.csv"
MODEL_PATH = "anomaly_detection_model.h5"
SCALER_PATH = "scaler.pkl"
FINAL_FIGURE_PATH = "final_figure_4_7_auc_roc_curve.png"

def generate_and_save_roc_curve():
    """
    Loads the trained model, calculates the ROC curve and AUC score,
    and saves the resulting plot.
    """
    print("--- 1. Loading Saved Model and Data ---")
    try:
        # Load the model with the custom object handler for 'mae'
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"mae": tf.keras.losses.MeanAbsoluteError()})
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        test_df = pd.read_csv(TEST_DATA_PATH)
        print("-> Successfully loaded model, scaler, and test data.")
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        return

    # --- 2. Prepare the Test Data (using the SAME steps as training) ---
    ground_truth = test_df['ground_truth']
    
    # Feature Engineering
    test_df['START_TIME'] = pd.to_datetime(test_df['START_TIME'])
    test_df['hour_of_day'] = test_df['START_TIME'].dt.hour
    test_df['day_of_week'] = test_df['START_TIME'].dt.dayofweek
    test_df['is_weekend'] = (test_df['START_TIME'].dt.dayofweek >= 5).astype(int)
    test_df = pd.get_dummies(test_df, columns=['QUERY_NO'], prefix='query')
    
    test_df_numeric = test_df.drop(columns=['PDB_NAME', 'START_TIME', 'END_TIME', 'EXECUTION_DATE', 'ground_truth'])
    
    # Use the LOADED scaler to transform the test data
    scaled_test_data = scaler.transform(test_df_numeric)
    print("-> Test data prepared and scaled successfully.")

    # --- 3. Get Prediction Scores (Reconstruction Errors) ---
    reconstructions = model.predict(scaled_test_data)
    # The reconstruction error serves as the score for anomaly detection
    reconstruction_errors = tf.keras.losses.mae(reconstructions, scaled_test_data)
    print("-> Calculated reconstruction errors to use as prediction scores.")

    # --- 4. Calculate ROC Curve and AUC ---
    # The roc_curve function returns the false positive rates, true positive rates, and thresholds
    fpr, tpr, thresholds = roc_curve(ground_truth, reconstruction_errors)
    # The auc function calculates the Area Under the Curve
    roc_auc = auc(fpr, tpr)
    print(f"-> Calculated AUC Score: {roc_auc:.4f}")

    # --- 5. Generate and Save the Plot ---
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance (AUC = 0.50)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate (Recall)', fontsize=14)
    plt.title('Figure 4.7: Receiver Operating Characteristic (ROC) Curve', fontsize=16, pad=20)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True)
    
    # Save the figure
    plt.savefig(FINAL_FIGURE_PATH, dpi=300, bbox_inches='tight')
    print(f"-> AUC-ROC curve plot saved to '{FINAL_FIGURE_PATH}'")


# --- Main Execution Block ---
if __name__ == "__main__":
    generate_and_save_roc_curve()