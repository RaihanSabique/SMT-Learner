"""
Baseline models for comparison with STCRL in the rebuttal.
Implements STTraj2Vec, VAE, Seq2Seq with Attention, and Transformer AutoEncoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class STTraj2Vec(nn.Module):
    """
    Spatiotemporal Trajectory to Vector model
    Based on LSTM encoder-decoder architecture for trajectory embedding
    """
    def __init__(self, input_dim=3, hidden_dim=128, embedding_dim=64, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        # Encoder: LSTM that processes the trajectory sequence
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Embedding layer: Maps final hidden state to fixed-size embedding
        self.embedding_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, embedding_dim)
        )
        
        # Decoder: LSTM that reconstructs the trajectory
        self.decoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Output layer: Maps decoder hidden states back to trajectory coordinates
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Encode trajectory
        encoded_sequence, (h_n, c_n) = self.encoder(x)
        
        # Create embedding from final hidden state
        embedding = self.embedding_layer(h_n[-1])  # Use final layer's hidden state
        
        # Prepare decoder input by repeating embedding across sequence length
        decoder_input = embedding.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Decode to reconstruct trajectory
        decoded_sequence, _ = self.decoder(decoder_input)
        output = self.output_layer(decoded_sequence)
        
        return output, embedding


class TrajectoryVAE(nn.Module):
    """
    Variational AutoEncoder for trajectory embedding
    Uses probabilistic latent space with reparameterization trick
    """
    def __init__(self, input_dim=3, hidden_dim=128, latent_dim=64, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        
        # Encoder: LSTM + variational layers
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Variational layers for mean and log variance
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder: Latent to trajectory reconstruction
        self.decoder_lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Encode
        encoded_sequence, (h_n, c_n) = self.encoder_lstm(x)
        
        # Get variational parameters
        mu = self.mu_layer(h_n[-1])
        logvar = self.logvar_layer(h_n[-1])
        
        # Sample from latent distribution
        z = self.reparameterize(mu, logvar)
        
        # Decode
        decoder_input = z.unsqueeze(1).repeat(1, seq_len, 1)
        decoded_sequence, _ = self.decoder_lstm(decoder_input)
        output = self.output_layer(decoded_sequence)
        
        # Store mu and logvar as attributes for potential KL loss computation if needed
        self.mu = mu
        self.logvar = logvar
        
        # Always return 4 values - training code expects this, evaluation code will handle it
        return output, z, mu, logvar


class Seq2SeqAttention(nn.Module):
    """
    Sequence-to-Sequence model with attention mechanism
    Uses bidirectional encoder and attention-based decoder
    """
    def __init__(self, input_dim=3, hidden_dim=128, embedding_dim=64, num_layers=2, num_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        # Bidirectional encoder
        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # Bidirectional hidden size
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        
        # Embedding layer from concatenated final states
        self.embedding_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2 * num_layers, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # Attention output projection
        self.attention_projection = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        
        # Final output layer (from bidirectional encoder output)
        self.output_layer = nn.Linear(hidden_dim * 2, input_dim)
        
    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Encode with bidirectional GRU
        encoded_sequence, hidden_states = self.encoder(x)
        
        # Create embedding from all hidden states
        # hidden_states shape: (num_layers * 2, batch_size, hidden_dim)
        hidden_concat = hidden_states.transpose(0, 1).contiguous().view(batch_size, -1)
        embedding = self.embedding_layer(hidden_concat)
        
        # Simplified approach: use self-attention on encoder output for reconstruction
        attended_output, attention_weights = self.attention(
            encoded_sequence, encoded_sequence, encoded_sequence
        )
        
        # Project attention output and generate final reconstruction
        projected_output = self.attention_projection(attended_output)
        reconstructed = self.output_layer(projected_output)
        
        return reconstructed, embedding


class TransformerAutoEncoder(nn.Module):
    """
    Pure Transformer AutoEncoder without contrastive learning
    Serves as ablation baseline to isolate the contribution of contrastive components
    """
    def __init__(self, input_dim=3, hidden_dim=128, embedding_dim=64, num_layers=4, num_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(512, hidden_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Embedding projection layer
        self.embedding_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, embedding_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, input_dim)
        )
        
    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Input embedding
        x_embedded = self.input_embedding(x)
        
        # Add positional encoding
        x_embedded = x_embedded + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Transform
        encoded_sequence = self.transformer_encoder(x_embedded)
        
        # Create global embedding (mean pooling)
        global_embedding = encoded_sequence.mean(dim=1)
        embedding = self.embedding_layer(global_embedding)
        
        # Decode each timestep
        reconstructed = self.decoder(encoded_sequence)
        
        return reconstructed, embedding


class DummySTCRL(nn.Module):
    """
    Simplified version of STCRL for comparison
    Removes contrastive learning components to serve as ablation
    """
    def __init__(self, seq_len=512, input_dim=3, hidden_dim=64, nhead=8, num_layers=3):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        # Same architecture as STCRL but without contrastive components
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # Simplified positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(seq_len, hidden_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Decoder
        self.decoder = nn.Linear(hidden_dim, seq_len * input_dim)
        
    def forward(self, x, metadata=None, return_projection=False):
        # Embed trajectory features
        x_embedded = self.embedding(x)
        
        # Add positional encoding
        x_embedded = x_embedded + self.pos_encoding[:x.size(1)].unsqueeze(0)
        
        # Transform
        encoded = self.transformer(x_embedded)
        
        # Global representation
        global_repr = encoded.mean(dim=1)
        
        # Decode
        decoded = self.decoder(global_repr)
        decoded = decoded.reshape(-1, self.seq_len, x.size(-1))
        
        if return_projection:
            # Return in same format as STCRL: (embeddings, projection, reconstruction)
            return global_repr, global_repr, decoded  # Use same embedding as projection
        else:
            return decoded, global_repr


# Factory function to create all baseline models
def create_baseline_models(input_dim=3, hidden_dim=128, embedding_dim=64):
    """
    Factory function to create all baseline models with consistent parameters
    """
    models = {
        'STTraj2Vec': STTraj2Vec(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_layers=2
        ),
        'VAE': TrajectoryVAE(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=embedding_dim,
            num_layers=2
        ),
        'Seq2Seq_Attention': Seq2SeqAttention(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_layers=2,
            num_heads=8
        ),
        'Transformer_AE': TransformerAutoEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_layers=3,
            num_heads=8
        ),
        'STCRL_NoContrastive': DummySTCRL(
            seq_len=512,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            nhead=8,
            num_layers=3
        )
    }
    
    return models
