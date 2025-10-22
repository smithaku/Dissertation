# Import necessary libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import pickle
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
BASELINE_DATA_PATH = "baseline_performance_data.csv"
TEST_DATA_PATH = "anomaly_test_data.csv"
MODEL_PATH = "anomaly_detection_model.h5"
SCALER_PATH = "scaler.pkl"
THRESHOLD_PATH = "anomaly_threshold.npy"
NUM_ANOMALIES_TO_INJECT = 10 # We'll inject 10 anomalies into the test set

# --- FUNCTIONS ---

def simulate_test_data():
    """
    Creates a simulated test dataset with both normal and injected anomalous data.
    """
    print("--- 1. Simulating Test Data with Injected Anomalies ---")
    try:
        df = pd.read_csv(BASELINE_DATA_PATH)
    except FileNotFoundError:
        print(f"Error: The baseline data file '{BASELINE_DATA_PATH}' was not found.")
        return

    # Use all data *except* the control PDB as our normal test data
    normal_data = df[df['PDB_NAME'] != 'PDB_CONTROL'].copy()
    normal_data['ground_truth'] = 0 # 0 represents a normal event

    # Create anomalous data by taking a sample and corrupting it
    anomalies_to_create = normal_data.sample(n=NUM_ANOMALIES_TO_INJECT, random_state=42)
    # Simulate a performance anomaly by drastically increasing execution time
    anomalies_to_create['EXECUTION_SECONDS'] = anomalies_to_create['EXECUTION_SECONDS'] * np.random.uniform(50, 100, size=NUM_ANOMALIES_TO_INJECT)
    anomalies_to_create['ground_truth'] = 1 # 1 represents an anomalous event
    
    # Combine normal and anomalous data
    test_df = pd.concat([normal_data, anomalies_to_create])
    
    # Shuffle the dataset to mix normal and anomalous records
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save to a new CSV file
    test_df.to_csv(TEST_DATA_PATH, index=False)
    print(f"-> Successfully created test dataset with {len(test_df)} records ({NUM_ANOMALIES_TO_INJECT} anomalies).")
    print(f"-> Test data saved to '{TEST_DATA_PATH}'")


def evaluate_model():
    """
    Loads the trained model and artifacts, makes predictions on the test data,
    and prints the final performance metrics.
    """
    print("\n--- 2. Evaluating Anomaly Detection Model ---")
    
    # --- a) Load all saved artifacts ---
    try:
        # **MODIFICATION HERE:** Added custom_objects to handle the 'mae' loss function.
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"mae": tf.keras.losses.MeanAbsoluteError()})
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        threshold = np.load(THRESHOLD_PATH)
        test_df = pd.read_csv(TEST_DATA_PATH)
        print("-> Successfully loaded model, scaler, threshold, and test data.")
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        return

    # --- b) Prepare the test data (using the SAME steps as training) ---
    ground_truth = test_df['ground_truth']
    
    # Feature Engineering
    test_df['START_TIME'] = pd.to_datetime(test_df['START_TIME'])
    test_df['hour_of_day'] = test_df['START_TIME'].dt.hour
    test_df['day_of_week'] = test_df['START_TIME'].dt.dayofweek
    test_df['is_weekend'] = (test_df['START_TIME'].dt.dayofweek >= 5).astype(int)
    test_df = pd.get_dummies(test_df, columns=['QUERY_NO'], prefix='query')
    
    # Drop non-numeric and label columns
    test_df_numeric = test_df.drop(columns=['PDB_NAME', 'START_TIME', 'END_TIME', 'EXECUTION_DATE', 'ground_truth'])
    
    # Use the LOADED scaler to transform the test data
    scaled_test_data = scaler.transform(test_df_numeric)
    print("-> Test data prepared and scaled successfully.")

    # --- c) Make Predictions ---
    reconstructions = model.predict(scaled_test_data)
    reconstruction_errors = tf.keras.losses.mae(reconstructions, scaled_test_data)
    
    # --- d) Classify Anomalies ---
    # If the error is greater than the threshold, classify as an anomaly (1)
    predictions = np.where(reconstruction_errors > threshold, 1, 0)
    
    # --- e) Generate and Print Final Results ---
    print("\n--- FINAL PERFORMANCE METRICS FOR RQ2 ---")
    print("\n[Classification Report]")
    print(classification_report(ground_truth, predictions, target_names=['Normal', 'Anomaly']))
    
    print("\n[Confusion Matrix]")
    cm = confusion_matrix(ground_truth, predictions)
    print(cm)
    print("\nFormat: [[True Negative, False Positive], [False Negative, True Positive]]")


# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # First, create the simulated test data file
    simulate_test_data()
    
    # Second, run the evaluation using that file
    evaluate_model()