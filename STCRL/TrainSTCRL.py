import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import math
from torch.optim import lr_scheduler
import copy
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from .STCRLDataset import STCRLModelFittingDataset
from .TransformerEncoder import STCRLTransformer


def train_stcrl_model(df, loss_fn, hidden_dim=64, nhead=8, num_layers=3, epochs=50,
                         batch_size=32, val_split=0.1, early_stop_patience=10, within_subject_weight=0.7):
    """
    Train the enhanced trajectory transformer model with the specified loss function.

    Args:
        df: Pandas DataFrame containing trajectory data
        loss_fn: Loss function to use for contrastive learning
        hidden_dim: Hidden dimension size for the model
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        epochs: Number of training epochs
        batch_size: Batch size for training
        val_split: Validation split ratio
        early_stop_patience: Number of epochs to wait before early stopping
        within_subject_weight: Weight for within-subject vs between-subject learning

    Returns:
        model: Trained model
        history: Training history dictionary
    """
    # Split into train and validation
    # Split into train and validation
    train_size = int((1 - val_split) * len(df))
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]

    # Create datasets
    train_dataset = STCRLModelFittingDataset(train_df)
    val_dataset = STCRLModelFittingDataset(val_df)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = STCRLTransformer(
        seq_len=512,
        input_dim=3,  # x, y, t coordinates
        hidden_dim=hidden_dim,
        nhead=nhead,
        num_layers=num_layers,
        metadata_dim=1  # For task_type
    ).to(device)

    # Loss functions
    reconstruction_loss = nn.MSELoss()
    contrastive_loss = loss_fn

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    # For early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model = None
    best_optimizer = None  # Save best optimizer state too

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'recon_loss': [],
        'contrastive_loss': []
    }

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        total_recon_loss = 0
        total_contrastive_loss = 0

        for batch_traj, temporal_batch in train_dataloader:
            batch_traj = batch_traj.to(device)
            task_type = temporal_batch['task_type'].float().unsqueeze(1).to(device)
            temporal_batch = {k: v.to(device) for k, v in temporal_batch.items()}

            encoded, projected, decoded = model(batch_traj, task_type)

            # Calculate losses
            recon_loss = reconstruction_loss(decoded, batch_traj)
            contr_loss = contrastive_loss(projected, temporal_batch)

            loss = recon_loss + contr_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate losses
            total_train_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_contrastive_loss += contr_loss.item()

        # Calculate average losses
        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_recon_loss = total_recon_loss / len(train_dataloader)
        avg_contrastive_loss = total_contrastive_loss / len(train_dataloader)

        # Validation phase
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch_traj, temporal_batch in val_dataloader:
                batch_traj = batch_traj.to(device)
                task_type = temporal_batch['task_type'].float().unsqueeze(1).to(device)
                temporal_batch = {k: v.to(device) for k, v in temporal_batch.items()}

                encoded, projected, decoded = model(batch_traj, task_type)

                recon_loss = reconstruction_loss(decoded, batch_traj)
                contr_loss = contrastive_loss(projected, temporal_batch)

                val_loss = recon_loss + contr_loss
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        scheduler.step(avg_val_loss)

        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['recon_loss'].append(avg_recon_loss)
        history['contrastive_loss'].append(avg_contrastive_loss)

        # Print progress
        print(f'Epoch [{epoch + 1}/{epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        print(f'Recon Loss: {avg_recon_loss:.4f}, Contrastive Loss: {avg_contrastive_loss:.4f}')

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = copy.deepcopy(model.state_dict())
            best_optimizer = copy.deepcopy(optimizer.state_dict())  # Also save optimizer state
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f'\nEarly stopping triggered after {epoch + 1} epochs')
                break

    # Load best model if early stopping was triggered
    if best_model is not None:
        model.load_state_dict(best_model)
        optimizer.load_state_dict(best_optimizer)  # Also load best optimizer state

    return model, optimizer, history

