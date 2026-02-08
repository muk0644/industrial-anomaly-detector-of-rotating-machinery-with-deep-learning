import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

# --- 1. SETUP & CONFIGURATION ---
st.set_page_config(page_title="AI Predictive Maintenance", layout="wide")

RESULTS_DIR = "results"
TRAIN_DATA_PATH = os.path.join(RESULTS_DIR, "train_data.npy")
TEST_DATA_PATH = os.path.join(RESULTS_DIR, "test_data.npy")
MODEL_PATH = os.path.join(RESULTS_DIR, "lstm_autoencoder.pth")

# Model Parameters (Must match training!)
SEQUENCE_LENGTH = 10
INPUT_DIM = 4
HIDDEN_DIM = 16

# --- 2. MODEL DEFINITION (Copy from step5_model.py) ---
# We define it here again to make this file standalone for the dashboard
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len, num_layers=1):
        super(LSTMAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.encoder = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.decoder = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        _, (hidden_n, _) = self.encoder(x)
        repeated_hidden = hidden_n[-1, :, :].unsqueeze(1).repeat(1, self.seq_len, 1)
        decoder_output, _ = self.decoder(repeated_hidden)
        reconstructed_x = self.output_layer(decoder_output)
        return reconstructed_x

# --- 3. HELPER FUNCTIONS ---
@st.cache_data
def load_data():
    if not os.path.exists(TRAIN_DATA_PATH) or not os.path.exists(TEST_DATA_PATH):
        st.error("Data files not found. Please run Step 4 first.")
        return None, None
    train_data = np.load(TRAIN_DATA_PATH)
    test_data = np.load(TEST_DATA_PATH)
    return train_data, test_data

def create_sequences(data, seq_length):
    xs = []
    for i in range(len(data) - seq_length):
        xs.append(data[i:(i + seq_length)])
    return np.array(xs)

# --- 4. MAIN APP LOGIC ---
st.title("🏭 AI-Driven Predictive Maintenance Dashboard")
st.markdown("### Real-Time Condition Monitoring of Rotating Machinery")
st.markdown("This dashboard visualizes the **Health Score** of industrial bearings using an **LSTM Autoencoder**.")

# Load Data
train_data, test_data = load_data()

if train_data is not None:
    # Sidebar Controls
    st.sidebar.header("Configuration")
    threshold_factor = st.sidebar.slider("Sensitivity (Threshold Factor)", 0.8, 1.5, 1.0)
    
    # Prepare Data
    X_train = create_sequences(train_data, SEQUENCE_LENGTH)
    X_test = create_sequences(test_data, SEQUENCE_LENGTH)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    # Load Model
    if os.path.exists(MODEL_PATH):
        model = LSTMAutoencoder(INPUT_DIM, HIDDEN_DIM, SEQUENCE_LENGTH)
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        
        # Calculate Anomaly Scores
        with torch.no_grad():
            # Training Data (Healthy)
            train_pred = model(X_train_tensor)
            train_loss = np.mean(np.abs(train_pred.numpy() - X_train), axis=2)
            train_score = np.mean(train_loss, axis=1)
            
            # Test Data (Unknown/Failure)
            test_pred = model(X_test_tensor)
            test_loss = np.mean(np.abs(test_pred.numpy() - X_test), axis=2)
            test_score = np.mean(test_loss, axis=1)

        # Dynamic Threshold Calculation
        base_threshold = np.max(train_score)
        final_threshold = base_threshold * threshold_factor

        # --- DASHBOARD VISUALS ---
        
        # 1. KPI Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Model Status", "Active", "LSTM Autoencoder")
        col2.metric("Safety Threshold", f"{final_threshold:.4f}", "MSE Loss")
        
        # Check current status (last data point)
        current_score = test_score[-1]
        if current_score > final_threshold:
            col3.error(f"⚠️ CRITICAL ALERT: {current_score:.4f}")
        else:
            col3.success(f"✅ System Healthy: {current_score:.4f}")

        # 2. Main Plot
        st.subheader("Live Anomaly Detection Feed")
        
        # Combine data for plotting
        full_score = np.concatenate([train_score, test_score])
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(full_score, label='Reconstruction Error (Health Score)', color='#1f77b4', linewidth=1.5)
        ax.axhline(y=final_threshold, color='red', linestyle='--', linewidth=2, label=f'Alarm Threshold ({final_threshold:.4f})')
        
        # Color areas
        ax.axvspan(0, len(train_score), color='green', alpha=0.1, label='Training Phase (Healthy)')
        ax.axvspan(len(train_score), len(full_score), color='orange', alpha=0.1, label='Testing Phase (Monitoring)')
        
        ax.set_title("Machine Health Monitoring (Run-to-Failure)")
        ax.set_ylabel("MSE Loss")
        ax.set_xlabel("Time Steps")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)

        # 3. Data Inspection
        with st.expander("🔍 View Raw Sensor Data"):
            st.write("First 100 rows of Test Data (Vibration Signals):")
            st.dataframe(pd.DataFrame(test_data[:100], columns=["Bearing 1", "Bearing 2", "Bearing 3", "Bearing 4"]))

    else:
        st.error(f"Model file not found at {MODEL_PATH}. Please train the model first.")