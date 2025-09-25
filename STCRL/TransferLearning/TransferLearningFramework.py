import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import copy
import os
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
from .AdaptiveLoss import AdaptiveLossWeighter

class TransferLearningFramework:
    """
    Comprehensive transfer learning framework for STCRL with cross-task and cross-subject capabilities.
    """

    def __init__(self, source_model, loss_functions_dict):
        self.source_model = source_model
        self.loss_functions = loss_functions_dict
        self.device = next(source_model.parameters()).device

    def calculate_domain_similarity(self, source_data, target_data):
        """
        Calculate similarity between source and target domains.
        Returns a value between 0 (completely different) and 1 (identical).
        """
        # Simple implementation - can be made more sophisticated
        try:
            # Compare task type distributions
            source_tasks = source_data['task_type'].value_counts(normalize=True).sort_index()
            target_tasks = target_data['task_type'].value_counts(normalize=True).sort_index()

            # Compare completion time distributions
            source_time_mean = source_data['completion_time'].mean()
            target_time_mean = target_data['completion_time'].mean()
            time_similarity = 1.0 - min(1.0, abs(source_time_mean - target_time_mean) / max(source_time_mean,
                                                                                            target_time_mean))

            # Compare RMSD distributions
            source_rmsd_mean = source_data['rmsd'].mean()
            target_rmsd_mean = target_data['rmsd'].mean()
            rmsd_similarity = 1.0 - min(1.0, abs(source_rmsd_mean - target_rmsd_mean) / max(source_rmsd_mean,
                                                                                            target_rmsd_mean))

            # Combine similarities
            overall_similarity = (time_similarity + rmsd_similarity) / 2.0

            return max(0.1, min(0.9, overall_similarity))  # Clamp between 0.1 and 0.9

        except Exception as e:
            print(f"Error calculating domain similarity: {e}")
            return 0.5  # Default to medium similarity

    def zero_shot_transfer(self, target_dataset, batch_size=32):
        """
        Perform zero-shot transfer by directly applying pre-trained model to new data.
        """
        print("Performing zero-shot transfer...")

        target_dataloader = DataLoader(target_dataset, batch_size=batch_size, shuffle=False)

        self.source_model.eval()
        results = []

        with torch.no_grad():
            for batch_traj, temporal_batch in target_dataloader:
                batch_traj = batch_traj.to(self.device)
                task_type = temporal_batch['task_type'].float().unsqueeze(1).to(self.device)

                # Forward pass with frozen model
                encoded, projected, decoded = self.source_model(batch_traj, task_type)

                # Store results for analysis
                results.append({
                    'encoded': encoded.cpu(),
                    'projected': projected.cpu(),
                    'decoded': decoded.cpu(),
                    'metadata': {k: v.cpu() for k, v in temporal_batch.items()}
                })

        print("Zero-shot transfer completed.")
        return results

    def cross_subject_transfer(self, target_df, epochs=10, batch_size=16, learning_rate=0.0005):
        """
        Perform cross-subject transfer learning.
        Fine-tunes only projection and decoder layers for new subjects.
        """
        print("Starting cross-subject transfer learning...")

        # Create target dataset
        from STCRL.STCRLDataset import STCRLModelFittingDataset  # Adjust import as needed
        target_dataset = STCRLModelFittingDataset(target_df)

        # Split for validation
        val_size = max(1, int(0.1 * len(target_dataset)))
        train_size = len(target_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(target_dataset, [train_size, val_size])

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Create target model as copy of source model
        target_model = copy.deepcopy(self.source_model)

        # Freeze transformer layers - only fine-tune projection and decoder
        for name, param in target_model.named_parameters():
            if 'transformer' in name or 'embedding' in name or 'pos_encoder' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        # Setup adaptive loss weighter for cross-subject transfer
        loss_weighter = AdaptiveLossWeighter(
            num_losses=5,
            transfer_type='cross_subject',
            adapt_speed=0.1
        )
        loss_weighter.set_transfer_phase('fine_tune')

        # Calculate domain similarity
        # Note: You'll need to pass source_df as parameter or store it in the class
        domain_similarity = 0.7  # Default for cross-subject (usually high similarity)

        # Setup optimizer for trainable parameters only
        trainable_params = [p for p in target_model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'individual_losses': [],
            'weights': []
        }

        best_val_loss = float('inf')
        patience_counter = 0
        patience = 5

        for epoch in range(epochs):
            # Training phase
            target_model.train()
            epoch_train_loss = 0
            epoch_individual_losses = []
            epoch_weights = []

            for batch_traj, temporal_batch in train_dataloader:
                batch_traj = batch_traj.to(self.device)
                task_type = temporal_batch['task_type'].float().unsqueeze(1).to(self.device)
                temporal_batch = {k: v.to(self.device) for k, v in temporal_batch.items()}

                # Forward pass
                encoded, projected, decoded = target_model(batch_traj, task_type)

                # Calculate individual losses
                individual_losses = self._calculate_individual_losses(
                    projected, decoded, batch_traj, temporal_batch
                )

                # Update loss weighter
                loss_weighter.update_history([l.item() for l in individual_losses])

                # Get adaptive weights
                current_weights = loss_weighter.get_updated_weights(domain_similarity)

                # Calculate weighted total loss
                total_loss = sum(w * l for w, l in zip(current_weights, individual_losses))

                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()

                # Update metrics
                epoch_train_loss += total_loss.item()
                epoch_individual_losses.append([l.item() for l in individual_losses])
                epoch_weights.append(current_weights.tolist())

            # Validation phase
            target_model.eval()
            epoch_val_loss = 0
            val_individual_losses = []

            with torch.no_grad():
                for batch_traj, temporal_batch in val_dataloader:
                    batch_traj = batch_traj.to(self.device)
                    task_type = temporal_batch['task_type'].float().unsqueeze(1).to(self.device)
                    temporal_batch = {k: v.to(self.device) for k, v in temporal_batch.items()}

                    encoded, projected, decoded = target_model(batch_traj, task_type)

                    individual_losses = self._calculate_individual_losses(
                        projected, decoded, batch_traj, temporal_batch
                    )

                    total_loss = sum(w * l for w, l in zip(current_weights, individual_losses))

                    epoch_val_loss += total_loss.item()
                    val_individual_losses.append([l.item() for l in individual_losses])

            # Calculate averages
            avg_train_loss = epoch_train_loss / len(train_dataloader)
            avg_val_loss = epoch_val_loss / len(val_dataloader)
            avg_individual_losses = np.mean(epoch_individual_losses, axis=0).tolist()
            avg_weights = np.mean(epoch_weights, axis=0).tolist()

            # Update history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['individual_losses'].append(avg_individual_losses)
            history['weights'].append(avg_weights)

            # Learning rate scheduling
            scheduler.step(avg_val_loss)

            # Print progress
            print(f"Epoch {epoch + 1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = copy.deepcopy(target_model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    target_model.load_state_dict(best_model_state)
                    break

        print("Cross-subject transfer learning completed.")
        return target_model, history

    def cross_task_transfer(self, target_df, epochs=15, batch_size=16, learning_rate=0.0003):
        """
        Perform cross-task transfer learning.
        Fine-tunes projection and decoder layers for new tasks with stronger adaptation.
        """
        print("Starting cross-task transfer learning...")

        # Create target dataset
        from STCRL.STCRLDataset import STCRLModelFittingDataset
        target_dataset = STCRLModelFittingDataset(target_df)

        # Split for validation
        val_size = max(1, int(0.15 * len(target_dataset)))  # Larger validation set for cross-task
        train_size = len(target_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(target_dataset, [train_size, val_size])

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Create target model
        target_model = copy.deepcopy(self.source_model)

        # For cross-task, we might need to fine-tune more layers
        # Phase 1: Freeze transformer, fine-tune projection/decoder
        for name, param in target_model.named_parameters():
            if 'transformer' in name or 'embedding' in name or 'pos_encoder' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        # Setup adaptive loss weighter for cross-task transfer
        loss_weighter = AdaptiveLossWeighter(
            num_losses=5,
            transfer_type='cross_task',
            adapt_speed=0.15  # Faster adaptation for cross-task
        )
        loss_weighter.set_transfer_phase('fine_tune')

        # Domain similarity is typically lower for cross-task
        domain_similarity = 0.4  # Default for cross-task

        # Two-phase training
        # Phase 1: Fine-tune projection and decoder
        trainable_params = [p for p in target_model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.7)

        history = {'train_loss': [], 'val_loss': [], 'individual_losses': [], 'weights': [], 'phase': []}

        # Phase 1: Projection/Decoder fine-tuning
        phase1_epochs = epochs // 2
        print(f"Phase 1: Fine-tuning projection/decoder layers ({phase1_epochs} epochs)")

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(phase1_epochs):
            # Training
            target_model.train()
            epoch_train_loss = 0
            epoch_individual_losses = []
            epoch_weights = []

            for batch_traj, temporal_batch in train_dataloader:
                batch_traj = batch_traj.to(self.device)
                task_type = temporal_batch['task_type'].float().unsqueeze(1).to(self.device)
                temporal_batch = {k: v.to(self.device) for k, v in temporal_batch.items()}

                encoded, projected, decoded = target_model(batch_traj, task_type)

                individual_losses = self._calculate_individual_losses(
                    projected, decoded, batch_traj, temporal_batch
                )

                loss_weighter.update_history([l.item() for l in individual_losses])
                current_weights = loss_weighter.get_updated_weights(domain_similarity)

                total_loss = sum(w * l for w, l in zip(current_weights, individual_losses))

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()

                epoch_train_loss += total_loss.item()
                epoch_individual_losses.append([l.item() for l in individual_losses])
                epoch_weights.append(current_weights.tolist())

            # Validation
            target_model.eval()
            epoch_val_loss = 0

            with torch.no_grad():
                for batch_traj, temporal_batch in val_dataloader:
                    batch_traj = batch_traj.to(self.device)
                    task_type = temporal_batch['task_type'].float().unsqueeze(1).to(self.device)
                    temporal_batch = {k: v.to(self.device) for k, v in temporal_batch.items()}

                    encoded, projected, decoded = target_model(batch_traj, task_type)
                    individual_losses = self._calculate_individual_losses(
                        projected, decoded, batch_traj, temporal_batch
                    )
                    total_loss = sum(w * l for w, l in zip(current_weights, individual_losses))
                    epoch_val_loss += total_loss.item()

            avg_train_loss = epoch_train_loss / len(train_dataloader)
            avg_val_loss = epoch_val_loss / len(val_dataloader)

            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['individual_losses'].append(np.mean(epoch_individual_losses, axis=0).tolist())
            history['weights'].append(np.mean(epoch_weights, axis=0).tolist())
            history['phase'].append(1)

            scheduler.step(avg_val_loss)

            print(f"Phase 1 - Epoch {epoch + 1}/{phase1_epochs}: Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = copy.deepcopy(target_model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 3:
                    print("Early stopping in Phase 1")
                    target_model.load_state_dict(best_model_state)
                    break

        # Phase 2: Unfreeze top transformer layers for deeper adaptation
        print(f"Phase 2: Fine-tuning with top transformer layers ({epochs - phase1_epochs} epochs)")

        # Unfreeze the last transformer layer
        for name, param in target_model.named_parameters():
            if 'transformer.layers.1' in name or 'transformer.layers.2' in name:  # Unfreeze last layers
                param.requires_grad = True

        # Reduce learning rate for the transformer layers
        trainable_params = [p for p in target_model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate * 0.3, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.8)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(phase1_epochs, epochs):
            # Training
            target_model.train()
            epoch_train_loss = 0
            epoch_individual_losses = []
            epoch_weights = []

            for batch_traj, temporal_batch in train_dataloader:
                batch_traj = batch_traj.to(self.device)
                task_type = temporal_batch['task_type'].float().unsqueeze(1).to(self.device)
                temporal_batch = {k: v.to(self.device) for k, v in temporal_batch.items()}

                encoded, projected, decoded = target_model(batch_traj, task_type)

                individual_losses = self._calculate_individual_losses(
                    projected, decoded, batch_traj, temporal_batch
                )

                loss_weighter.update_history([l.item() for l in individual_losses])
                current_weights = loss_weighter.get_updated_weights(domain_similarity)

                total_loss = sum(w * l for w, l in zip(current_weights, individual_losses))

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 0.5)  # Smaller clipping for transformer layers
                optimizer.step()

                epoch_train_loss += total_loss.item()
                epoch_individual_losses.append([l.item() for l in individual_losses])
                epoch_weights.append(current_weights.tolist())

            # Validation
            target_model.eval()
            epoch_val_loss = 0

            with torch.no_grad():
                for batch_traj, temporal_batch in val_dataloader:
                    batch_traj = batch_traj.to(self.device)
                    task_type = temporal_batch['task_type'].float().unsqueeze(1).to(self.device)
                    temporal_batch = {k: v.to(self.device) for k, v in temporal_batch.items()}

                    encoded, projected, decoded = target_model(batch_traj, task_type)
                    individual_losses = self._calculate_individual_losses(
                        projected, decoded, batch_traj, temporal_batch
                    )
                    total_loss = sum(w * l for w, l in zip(current_weights, individual_losses))
                    epoch_val_loss += total_loss.item()

            avg_train_loss = epoch_train_loss / len(train_dataloader)
            avg_val_loss = epoch_val_loss / len(val_dataloader)

            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['individual_losses'].append(np.mean(epoch_individual_losses, axis=0).tolist())
            history['weights'].append(np.mean(epoch_weights, axis=0).tolist())
            history['phase'].append(2)

            scheduler.step(avg_val_loss)

            print(f"Phase 2 - Epoch {epoch + 1}/{epochs}: Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = copy.deepcopy(target_model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 3:
                    print("Early stopping in Phase 2")
                    target_model.load_state_dict(best_model_state)
                    break

        print("Cross-task transfer learning completed.")
        return target_model, history

    def _calculate_individual_losses(self, projected, decoded, batch_traj, temporal_batch):
        """Helper method to calculate individual loss components"""
        # Reconstruction loss
        reconstruction_loss = nn.MSELoss()(decoded, batch_traj)

        # Contrastive losses
        completion_loss = self.loss_functions['completion_time'](projected, temporal_batch)
        task_loss = self.loss_functions['task_type'](projected, temporal_batch)
        rmsd_loss = self.loss_functions['rmsd'](projected, temporal_batch)
        success_loss = self.loss_functions['success'](projected, temporal_batch)

        return [reconstruction_loss, completion_loss, task_loss, rmsd_loss, success_loss]

    def evaluate_transfer_performance(self, model, test_dataset, batch_size=32):
        """Evaluate the performance of a transferred model"""
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model.eval()
        total_loss = 0
        individual_losses = []

        with torch.no_grad():
            for batch_traj, temporal_batch in test_dataloader:
                batch_traj = batch_traj.to(self.device)
                task_type = temporal_batch['task_type'].float().unsqueeze(1).to(self.device)
                temporal_batch = {k: v.to(self.device) for k, v in temporal_batch.items()}

                encoded, projected, decoded = model(batch_traj, task_type)

                losses = self._calculate_individual_losses(projected, decoded, batch_traj, temporal_batch)
                individual_losses.append([l.item() for l in losses])

                total_loss += sum(losses).item()

        avg_total_loss = total_loss / len(test_dataloader)
        avg_individual_losses = np.mean(individual_losses, axis=0)

        results = {
            'total_loss': avg_total_loss,
            'reconstruction_loss': avg_individual_losses[0],
            'completion_time_loss': avg_individual_losses[1],
            'task_type_loss': avg_individual_losses[2],
            'rmsd_loss': avg_individual_losses[3],
            'success_loss': avg_individual_losses[4]
        }

        return results

    def save_transferred_model(self, model, history, transfer_type, filepath, hyperparams=None):
        """Save transferred model with transfer learning metadata"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Create hyperparams if not provided
        if hyperparams is None:
            hyperparams = {
                'transfer_type': transfer_type,
                'hidden_dim': model.hidden_dim,
                'seq_len': model.seq_len,
                'input_dim': model.input_dim
            }
        else:
            hyperparams['transfer_type'] = transfer_type

        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'hyperparams': hyperparams,
            'transfer_type': transfer_type
        }, filepath + '.pt')

        # Save history
        with open(filepath + '_history.json', 'w') as f:
            json.dump(history, f)

        # Save architecture info
        model_info = {
            'type': model.__class__.__name__,
            'hidden_dim': model.hidden_dim,
            'seq_len': model.seq_len,
            'input_dim': model.input_dim,
            'transfer_type': transfer_type
        }

        with open(filepath + '_architecture.json', 'w') as f:
            json.dump(model_info, f)

        print(f"Transferred model saved to {filepath}")