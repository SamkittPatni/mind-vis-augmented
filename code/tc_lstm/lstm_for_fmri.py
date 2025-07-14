import torch
import torch.nn as nn
import tc_lstm.utils as ut
import numpy as np

class LSTMforFMRI(nn.Module):
    """
    LSTM model for fMRI data.
    Input shape: (B, T, D) where B is batch size, T is time steps, D is feature dimension.
    Output shape: (B, T, embed_dim) where embed_dim is the dimension of the output embeddings.
    """
    def __init__(self, input_dim=224, hidden_size=1024, num_layers=1, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, num_layers=num_layers, 
                            batch_first=True, bidirectional=bidirectional)
        self.norm = nn.LayerNorm(hidden_size)

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.embed_dim = hidden_size

    def forward(self, x, lengths=None):
        """
        Forward pass through the LSTM model.
        """
        # x: (B, T, D) [can be padded]
        # lengths: (B,) [optional, for packed sequences, contains the actual lengths of sequences]
        if lengths is not None:
            # If input is padded, pack the sequences and then pass into the model
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            out_packed, _ = self.lstm(packed)
            # Unpack the sequences to get the output
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        else:
            lstm_out, _ = self.lstm(x)
        z = self.norm(lstm_out)  # Apply layer normalization
        loss = ut.symmetric_info_nce_loss(z)  # Compute the loss

        return loss, z  # Return the loss and the output embeddings