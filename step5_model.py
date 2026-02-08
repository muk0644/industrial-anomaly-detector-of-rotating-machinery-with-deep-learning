import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    # CORRECT ORDER: input_dim (4), hidden_dim (16), seq_len (10)
    # This matches exactly how step6_train.py calls it.
    def __init__(self, input_dim, hidden_dim, seq_len, num_layers=1):
        super(LSTMAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.num_layers = num_layers

        # ---------------------------------------------------------
        # ENCODER (Compress the data)
        # ---------------------------------------------------------
        # Input Shape: [Batch, Sequence Length, Features]
        # Output: Compressed representation (Hidden State)
        self.encoder = nn.LSTM(
            input_size=input_dim,   # Correctly set to 4
            hidden_size=hidden_dim, # Correctly set to 16
            num_layers=num_layers,
            batch_first=True
        )

        # ---------------------------------------------------------
        # DECODER (Reconstruct the data)
        # ---------------------------------------------------------
        # It tries to recreate the original input from the compressed hidden state
        self.decoder = nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=hidden_dim, # We keep it hidden size internally
            num_layers=num_layers,
            batch_first=True
        )
        
        # Output layer to map back to original dimensions (16 -> 4)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # 1. Encode
        # We only care about the final hidden state (hidden_n)
        _, (hidden_n, _) = self.encoder(x)
        
        # 2. Prepare for Decode
        # The hidden state is [Layers, Batch, Hidden]. 
        # We need to repeat it for the sequence length to reconstruct the time series.
        # Shape becomes: [Batch, Sequence Length, Hidden]
        
        # Get the hidden state from the last layer
        hidden_n = hidden_n[-1, :, :] 
        
        # Repeat it for every time step in the sequence
        repeated_hidden = hidden_n.unsqueeze(1).repeat(1, self.seq_len, 1)
        
        # 3. Decode
        decoder_output, _ = self.decoder(repeated_hidden)
        
        # 4. Final Reconstruction
        reconstructed_x = self.output_layer(decoder_output)
        
        return reconstructed_x