# Import necessary libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
BASELINE_DATA_PATH = "baseline_performance_data.csv"
MODEL_SAVE_PATH = "anomaly_detection_model.h5"
SCALER_SAVE_PATH = "scaler.pkl"
THRESHOLD_SAVE_PATH = "anomaly_threshold.npy"

# --- FUNCTIONS ---

def prepare_training_data(csv_path):
    """
    Loads the baseline data, filters for the control PDB, and performs
    all necessary feature engineering and preprocessing.
    """
    print("--- 1. Preparing Training Data for Anomaly Model ---")
    try:
        df = pd.read_csv(csv_path)
        print(f"-> Loaded {len(df)} records from '{csv_path}'.")
    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found.")
        return None, None

    # Filter for ONLY the PDB_CONTROL data to define "normal" behavior
    control_df = df[df['PDB_NAME'] == 'PDB_CONTROL'].copy()
    if control_df.empty:
        print("Error: No data found for 'PDB_CONTROL'. Cannot train model.")
        return None, None
        
    print(f"-> Isolated {len(control_df)} records from PDB_CONTROL for training.")

    # Feature Engineering (must be identical to the steps in EDA)
    # a) Temporal Features
    control_df['START_TIME'] = pd.to_datetime(control_df['START_TIME'])
    control_df['hour_of_day'] = control_df['START_TIME'].dt.hour
    control_df['day_of_week'] = control_df['START_TIME'].dt.dayofweek
    control_df['is_weekend'] = (control_df['START_TIME'].dt.dayofweek >= 5).astype(int)

    # b) Categorical Features (only QUERY_NO, as PDB_NAME is constant)
    control_df = pd.get_dummies(control_df, columns=['QUERY_NO'], prefix='query')

    # c) Drop non-numeric columns for training
    control_df_numeric = control_df.drop(columns=['PDB_NAME', 'START_TIME', 'END_TIME', 'EXECUTION_DATE'])
    
    # d) Numerical Scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(control_df_numeric)
    
    print("-> Feature engineering and scaling complete.")
    print(f"-> Final training data shape: {scaled_data.shape}")
    
    return scaled_data, scaler


def build_autoencoder(input_shape):
    """
    Builds the Autoencoder model architecture using Keras.
    """
    print("\n--- 2. Building Autoencoder Model Architecture ---")
    
    # The Autoencoder has two parts: an Encoder and a Decoder
    model = keras.Sequential([
        # Encoder: Compresses the input into a smaller representation
        keras.layers.Input(shape=(input_shape,)),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(8, activation='relu'),  # Bottleneck layer (latent space)
        
        # Decoder: Tries to reconstruct the original input from the compressed representation
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(input_shape, activation='sigmoid') # Output layer has same shape as input
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='mae') # Mean Absolute Error is a good loss for reconstruction
    
    print("-> Model built and compiled successfully.")
    model.summary() # Print the model architecture
    
    return model

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # Step 1: Prepare the data
    training_data, scaler_object = prepare_training_data(BASELINE_DATA_PATH)

    if training_data is not None:
        # Step 2: Build the Autoencoder model
        autoencoder = build_autoencoder(input_shape=training_data.shape[1])

        # Step 3: Train the model
        print("\n--- 3. Training the Autoencoder ---")
        # The model learns by trying to reconstruct the input data.
        # It's trained on itself (input=training_data, output=training_data).
        history = autoencoder.fit(
            training_data,
            training_data,
            epochs=50,
            batch_size=16,
            shuffle=True,
            validation_split=0.1, # Use 10% of data for validation
            verbose=1
        )
        print("-> Model training complete.")

        # Step 4: Determine the anomaly detection threshold
        print("\n--- 4. Calculating Anomaly Threshold ---")
        # Use the trained model to reconstruct the training data
        reconstructions = autoencoder.predict(training_data)
        # Calculate the Mean Absolute Error for each record
        reconstruction_errors = tf.keras.losses.mae(reconstructions, training_data)
        
        # Set the threshold as the mean error + 3 standard deviations
        threshold = np.mean(reconstruction_errors) + 3 * np.std(reconstruction_errors)
        print(f"-> Calculated anomaly threshold: {threshold:.4f}")

        # Step 5: Save the trained model and artifacts for later use
        print("\n--- 5. Saving Model and Artifacts ---")
        # Save the trained model
        autoencoder.save(MODEL_SAVE_PATH)
        print(f"-> Model saved to '{MODEL_SAVE_PATH}'")

        # Save the scaler object (important for preprocessing new data)
        import pickle
        with open(SCALER_SAVE_PATH, 'wb') as f:
            pickle.dump(scaler_object, f)
        print(f"-> Scaler object saved to '{SCALER_SAVE_PATH}'")

        # Save the calculated threshold
        np.save(THRESHOLD_SAVE_PATH, threshold)
        print(f"-> Anomaly threshold saved to '{THRESHOLD_SAVE_PATH}'")
        
        print("\n--- Anomaly Detection Model Training Complete ---")