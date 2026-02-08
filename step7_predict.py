import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from step5_model import LSTMAutoencoder 

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
RESULTS_DIR = "results"
TRAIN_DATA_PATH = os.path.join(RESULTS_DIR, "train_data.npy")
TEST_DATA_PATH = os.path.join(RESULTS_DIR, "test_data.npy")
MODEL_PATH = os.path.join(RESULTS_DIR, "lstm_autoencoder.pth")

# Hyperparameters (Must match training!)
SEQUENCE_LENGTH = 10
INPUT_DIM = 4
HIDDEN_DIM = 16

def create_sequences(data, seq_length):
    xs = []
    for i in range(len(data) - seq_length):
        xs.append(data[i:(i + seq_length)])
    return np.array(xs)

def predict():
    print("--- STEP 7: PREDICTION & ANOMALY DETECTION ---")

    # 1. Load Data
    print("   -> Loading test data...")
    if not os.path.exists(TEST_DATA_PATH):
        print(f"ERROR: {TEST_DATA_PATH} not found.")
        return
        
    train_data = np.load(TRAIN_DATA_PATH)
    test_data = np.load(TEST_DATA_PATH)
    
    # 2. Prepare Sequences (These are NumPy Arrays!)
    X_train = create_sequences(train_data, SEQUENCE_LENGTH)
    X_test = create_sequences(test_data, SEQUENCE_LENGTH)
    
    # Create Tensors for the Model
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    # 3. Load Model
    print(f"   -> Loading trained model...")
    model = LSTMAutoencoder(INPUT_DIM, HIDDEN_DIM, SEQUENCE_LENGTH)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # 4. Calculate Threshold (Baseline)
    print("   -> Calculating baseline threshold...")
    with torch.no_grad():
        # Get Model Prediction (This is a Tensor)
        train_pred = model(X_train_tensor)
        
        # Calculate Error
        # FIX: train_pred is Tensor -> needs .numpy()
        # FIX: X_train is ALREADY NumPy -> NO .numpy() needed!
        train_loss = np.mean(np.abs(train_pred.numpy() - X_train), axis=2)
        train_score = np.mean(train_loss, axis=1)
        
        threshold = np.max(train_score)
        print(f"      Threshold set to: {threshold:.4f}")

    # 5. Detect Anomalies (Test Data)
    print("   -> Detecting anomalies in Test Data...")
    with torch.no_grad():
        test_pred = model(X_test_tensor)
        
        # FIX: Same here. Remove .numpy() from X_test
        test_loss = np.mean(np.abs(test_pred.numpy() - X_test), axis=2)
        test_score = np.mean(test_loss, axis=1)

    # 6. Combine Results
    anomaly_score = np.concatenate([train_score, test_score])

    # 7. Visualization
    print("   -> Generating Final Graph...")
    plt.figure(figsize=(14, 6))
    
    plt.plot(anomaly_score, label='Anomaly Score (Reconstruction Error)', color='blue', linewidth=1.5)
    plt.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.4f})')
    
    # Mark Areas
    plt.axvline(x=len(train_score), color='black', linestyle='-')
    plt.text(len(train_score)/2, plt.ylim()[1]*0.9, 'TRAINING (Healthy)', color='green', fontweight='bold', ha='center')
    plt.text(len(train_score) + len(test_score)/2, plt.ylim()[1]*0.9, 'TESTING (Failure)', color='orange', fontweight='bold', ha='center')

    plt.title("AI Predictive Maintenance: Run-to-Failure Detection", fontsize=16)
    plt.ylabel("Reconstruction Error (MSE)")
    plt.xlabel("Time Steps")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(RESULTS_DIR, "Final_Result_Graph.png")
    plt.savefig(save_path)
    print(f"   -> ✅ SUCCESS! Graph saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    predict()