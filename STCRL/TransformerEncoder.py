import torch
import torch.nn as nn
from .PositionalEncoder import PEncoding

class STCRLTransformer(nn.Module):
    def __init__(self, seq_len=512, input_dim=3, hidden_dim=64, nhead=8,
                 num_layers=2, dropout=0.1, metadata_dim=1):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim  # 3 for x, y, t
        self.hidden_dim = hidden_dim

        # Input embedding for trajectory
        self.embedding = nn.Linear(input_dim, hidden_dim)

        # Positional encoding
        self.pos_encoder = PEncoding(hidden_dim)

        # Metadata embedding (e.g., for task_type)
        self.metadata_embedding = nn.Sequential(
            nn.Linear(metadata_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Projection heads for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Decoder for trajectory reconstruction
        self.decoder = nn.Linear(hidden_dim + hidden_dim // 2, seq_len * input_dim)

    def forward(self, x, metadata=None, return_projection=True):
        # Handle input shape to ensure it's [batch_size, seq_len, input_dim]
        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        # Embed trajectory features
        x = self.embedding(x)

        # Apply positional encoding and transformer
        x = self.pos_encoder(x)
        x = self.transformer(x)

        # Get encoded representation
        encoded_traj = x.mean(dim=1)  # (batch_size, hidden_dim)

        # Process metadata if provided
        if metadata is not None:
            encoded_meta = self.metadata_embedding(metadata)
            encoded = torch.cat([encoded_traj, encoded_meta], dim=1)
        else:
            # If no metadata, duplicate the last part
            encoded = torch.cat([encoded_traj, torch.zeros_like(encoded_traj[:, :self.hidden_dim // 2])], dim=1)

        # Project for contrastive learning
        projected = self.projection(encoded)

        # Decode the sequence
        decoded = self.decoder(encoded)  # (batch_size, seq_len * input_dim)
        decoded = decoded.reshape(-1, self.seq_len, self.input_dim)  # (batch_size, seq_len, input_dim)

        if return_projection:
            return encoded_traj, projected, decoded
        return encoded_traj, decoded