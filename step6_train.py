import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
# Ensure your model file is actually named 'step5_model.py'!
from step5_model import LSTMAutoencoder  

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
RESULTS_DIR = "results"
TRAIN_DATA_PATH = os.path.join(RESULTS_DIR, "train_data.npy")
MODEL_SAVE_PATH = os.path.join(RESULTS_DIR, "lstm_autoencoder.pth")

# Hyperparameters
SEQUENCE_LENGTH = 10   # Look-back window size
INPUT_DIM = 4          # 4 Bearings (Features)
HIDDEN_DIM = 16        # Compressed representation size
NUM_EPOCHS = 2000       # Number of training loops
BATCH_SIZE = 32
LEARNING_RATE = 0.001

def create_sequences(data, seq_length):
    """
    Creates sliding window sequences from the data.
    """
    xs = []
    for i in range(len(data) - seq_length):
        xs.append(data[i:(i + seq_length)])
    return np.array(xs)

def train():
    print("--- STEP 6: TRAINING START ---")
    
    # 1. Load Data
    if not os.path.exists(TRAIN_DATA_PATH):
        print(f"ERROR: {TRAIN_DATA_PATH} not found. Run Step 4 first!")
        return

    train_data = np.load(TRAIN_DATA_PATH)
    print(f"   -> Loaded training data. Shape: {train_data.shape}")

    # 2. Prepare Sequences
    X_train = create_sequences(train_data, SEQUENCE_LENGTH)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    print(f"   -> Created sequences. Shape: {X_train_tensor.shape}")

    # 3. Initialize Model
    # IMPORTANT: The arguments must match the order in step5_model.py!
    # Order: (input_dim, hidden_dim, seq_len) -> (4, 16, 10)
    model = LSTMAutoencoder(INPUT_DIM, HIDDEN_DIM, SEQUENCE_LENGTH)
    
    criterion = nn.MSELoss()  # Mean Squared Error Loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Training Loop
    loss_history = []
    model.train()
    
    print(f"   -> Training for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        optimizer.zero_grad()
        
        # Forward Pass (Model makes a guess)
        output = model(X_train_tensor)
        
        # Loss Calculation (How bad was the guess?)
        loss = criterion(output, X_train_tensor)
        
        # Backward Pass (Update weights)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        # Print progress every 10 epochs
        if (epoch+1) % 10 == 0:
            print(f"      Epoch {epoch+1}: Loss = {loss.item():.6f}")

    # 5. Save Model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"   -> ✅ Model saved to {MODEL_SAVE_PATH}")

    # 6. Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label="Training Loss")
    plt.title("Model Training Progress")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.grid(True)
    
    save_plot_path = os.path.join(RESULTS_DIR, "training_loss.png")
    plt.savefig(save_plot_path)
    print(f"   -> Training graph saved to: {save_plot_path}")
    plt.show()

if __name__ == "__main__":
    train()