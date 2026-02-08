import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
INPUT_FILE = os.path.join("results", "merged_bearing_data.csv")
RESULTS_DIR = "results"

# We will use the first 50% of the data for training (Healthy state)
TRAIN_SPLIT_PCT = 0.5 

def preprocess_data():
    print(f"--- STEP 4: PREPROCESSING & SPLITTING ---")
    
    # 1. Load the merged data
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: {INPUT_FILE} not found. Run Step 2 first!")
        return
        
    df = pd.read_csv(INPUT_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # We only care about Bearing 1 for the model (since we want to detect ITS failure)
    # But often we train on all to learn correlations. Let's stick to Bearing 1 for simplicity first, 
    # OR use all 4 features. Let's use ALL 4 so the model is robust.
    data = df.values
    
    # 2. Split into Train and Test
    train_size = int(len(data) * TRAIN_SPLIT_PCT)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    print(f"   -> Total Data Points: {len(data)}")
    print(f"   -> Training Set (Healthy): {len(train_data)} points (First {TRAIN_SPLIT_PCT*100}%)")
    print(f"   -> Test Set (Unknown): {len(test_data)} points")

    # 3. Normalize the data (Scale between 0 and 1)
    # This is CRITICAL for Neural Networks to work efficiently
    scaler = MinMaxScaler()
    
    # We fit the scaler ONLY on training data (to avoid 'data leakage')
    train_data_scaled = scaler.fit_transform(train_data)
    
    # We apply the same scaler to the test data
    test_data_scaled = scaler.transform(test_data)
    
    # 4. Save the processed data for Step 5 (Training)
    # We save as binary .npy files (faster for Python to read)
    import numpy as np
    np.save(os.path.join(RESULTS_DIR, "train_data.npy"), train_data_scaled)
    np.save(os.path.join(RESULTS_DIR, "test_data.npy"), test_data_scaled)
    
    print(f"   -> ✅ Processed data saved to .npy files in '{RESULTS_DIR}'")

    # 5. Visualize the Split
    plt.figure(figsize=(12, 6))
    plt.plot(df.index[:train_size], df['Bearing 1'][:train_size], color='green', label='Training Data (Learns this)')
    plt.plot(df.index[train_size:], df['Bearing 1'][train_size:], color='red', label='Test Data (Detects this)')
    plt.axvline(df.index[train_size], color='black', linestyle='--', label='Split Point')
    plt.title("Data Split: Training (Healthy) vs Testing (Failure)", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save split graph
    plt.savefig(os.path.join(RESULTS_DIR, "Data_Split_Visualization.png"))
    print(f"   -> Visualization saved to {RESULTS_DIR}")
    plt.show()

if __name__ == "__main__":
    preprocess_data()