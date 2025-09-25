import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr
from sklearn.manifold import TSNE
import copy
import os
import time


class STCRLModelEvaluator:
    def __init__(self, model, test_dataloader, device):
        self.model = model
        self.test_dataloader = test_dataloader
        self.device = device
        self.model.eval()

    def evaluate_reconstruction(self):
        """Evaluate trajectory reconstruction quality with spatial metrics"""
        total_mse = 0
        total_samples = 0

        # Spatial metrics
        endpoint_errors = []
        path_lengths = []
        curvature_errors = []

        with torch.no_grad():
            for batch_traj, temporal_batch in self.test_dataloader:
                batch_traj = batch_traj.to(self.device)
                task_type = temporal_batch['task_type'].float().unsqueeze(1).to(self.device)

                # Use the enhanced model with task_type metadata
                _, _, decoded = self.model(batch_traj, task_type)

                # Overall MSE
                mse = torch.nn.functional.mse_loss(decoded, batch_traj)
                total_mse += mse.item() * batch_traj.size(0)
                total_samples += batch_traj.size(0)

                # Endpoint error - account for 3D trajectories
                endpoint_error = torch.norm(decoded[:, -1, :] - batch_traj[:, -1, :], dim=1)
                endpoint_errors.extend(endpoint_error.cpu().numpy())

                # Path length difference - account for 3D trajectories
                orig_lengths = torch.norm(batch_traj[:, 1:] - batch_traj[:, :-1], dim=2).sum(dim=1)
                recon_lengths = torch.norm(decoded[:, 1:] - decoded[:, :-1], dim=2).sum(dim=1)
                path_length_diff = torch.abs(orig_lengths - recon_lengths)
                path_lengths.extend(path_length_diff.cpu().numpy())

                # Curvature preservation - only use x,y for angles (first two dimensions)
                orig_vectors = batch_traj[:, 1:, :2] - batch_traj[:, :-1, :2]
                recon_vectors = decoded[:, 1:, :2] - decoded[:, :-1, :2]

                # Handle zero vectors by adding small epsilon
                orig_vectors = orig_vectors + 1e-6
                recon_vectors = recon_vectors + 1e-6

                orig_angles = torch.atan2(orig_vectors[..., 1], orig_vectors[..., 0])
                recon_angles = torch.atan2(recon_vectors[..., 1], recon_vectors[..., 0])
                angle_diff = torch.abs(orig_angles - recon_angles).mean(dim=1)
                curvature_errors.extend(angle_diff.cpu().numpy())

        return {
            'reconstruction_mse': total_mse / total_samples,
            'endpoint_error_mean': np.mean(endpoint_errors),
            'endpoint_error_std': np.std(endpoint_errors),
            'path_length_error_mean': np.mean(path_lengths),
            'path_length_error_std': np.std(path_lengths),
            'curvature_error_mean': np.mean(curvature_errors),
            'curvature_error_std': np.std(curvature_errors)
        }

    def evaluate_temporal_correlation(self):
        """Evaluate correlation between embeddings and temporal features"""
        all_embeddings = []
        temporal_features = {
            'completion_time': [],
            'task_type': [],
            'rmsd': [],
            'is_success': [],
            'participant_id': []
        }

        with torch.no_grad():
            for batch_traj, temporal_batch in self.test_dataloader:
                batch_traj = batch_traj.to(self.device)
                task_type = temporal_batch['task_type'].float().unsqueeze(1).to(self.device)

                # Get encoded representation (not the projection)
                encoded, _, _ = self.model(batch_traj, task_type)
                encoded = encoded.cpu().numpy()

                all_embeddings.append(encoded)
                for key in temporal_features:
                    if key in temporal_batch:
                        temporal_features[key].extend(temporal_batch[key].cpu().numpy())

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        embedding_norms = np.linalg.norm(all_embeddings, axis=1)

        correlations = {}
        # Calculate correlations for continuous variables
        for key in ['completion_time', 'rmsd']:
            if temporal_features[key]:
                corr, _ = pearsonr(embedding_norms, temporal_features[key])
                correlations[f'{key}_correlation'] = corr

        # Calculate task_type prediction accuracy
        if temporal_features['task_type']:
            task_pred = (embedding_norms > np.median(embedding_norms)).astype(int)
            task_acc = accuracy_score(temporal_features['task_type'], task_pred)
            correlations['task_type_accuracy'] = task_acc

        # Calculate success prediction accuracy
        if temporal_features['is_success']:
            success_pred = (embedding_norms > np.median(embedding_norms)).astype(int)
            success_acc = accuracy_score(temporal_features['is_success'], success_pred)
            correlations['success_accuracy'] = success_acc

        # Evaluate participant ID classification
        if temporal_features['participant_id']:
            # Use t-SNE to reduce dimensions and then calculate silhouette score
            tsne = TSNE(n_components=2, random_state=42)
            embeddings_2d = tsne.fit_transform(all_embeddings)
            silhouette = silhouette_score(embeddings_2d, temporal_features['participant_id'])
            correlations['participant_silhouette'] = silhouette

        return correlations

    def evaluate_neighborhood_consistency(self, k=5):
        """Evaluate consistency of temporal features in local neighborhoods"""
        all_embeddings = []
        all_trajectories = []
        temporal_features = {
            'completion_time': [],
            'task_type': [],
            'rmsd': [],
            'is_success': [],
            'participant_id': []
        }

        with torch.no_grad():
            for batch_traj, temporal_batch in self.test_dataloader:
                batch_traj = batch_traj.to(self.device)
                task_type = temporal_batch['task_type'].float().unsqueeze(1).to(self.device)

                encoded, _, _ = self.model(batch_traj, task_type)
                encoded = encoded.cpu().numpy()

                all_embeddings.append(encoded)
                all_trajectories.extend(batch_traj.cpu().numpy())
                for key in temporal_features:
                    if key in temporal_batch:
                        temporal_features[key].extend(temporal_batch[key].cpu().numpy())

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_trajectories = np.array(all_trajectories)

        # Find k-nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(all_embeddings)
        distances, indices = nbrs.kneighbors(all_embeddings)

        # Calculate consistencies
        consistency_scores = {}

        # For temporal features
        for feature_name, feature_values in temporal_features.items():
            if not feature_values:  # Skip empty features
                continue

            feature_values = np.array(feature_values)
            if feature_name in ['is_success', 'task_type', 'participant_id']:
                # For categorical features
                neighbor_agreements = []
                for idx_list in indices:
                    neighborhood = feature_values[idx_list[1:]]
                    agreement = np.mean(neighborhood == feature_values[idx_list[0]])
                    neighbor_agreements.append(agreement)
                consistency_scores[f'{feature_name}_consistency'] = np.mean(neighbor_agreements)
            else:
                # For continuous features
                neighbor_stds = []
                for idx_list in indices:
                    neighborhood = feature_values[idx_list[1:]]
                    # Handle empty neighborhoods
                    if len(neighborhood) > 0:
                        neighbor_stds.append(np.std(neighborhood))
                if neighbor_stds:
                    consistency_scores[f'{feature_name}_consistency'] = 1 / (1 + np.mean(neighbor_stds))

        # For trajectory shapes
        shape_similarities = []
        for idx_list in indices:
            center_traj = all_trajectories[idx_list[0]]
            neighbor_trajs = all_trajectories[idx_list[1:]]

            # Calculate Euclidean distances between trajectories
            similarities = []
            for neighbor_traj in neighbor_trajs:
                similarity = np.mean(np.sqrt(np.sum((center_traj - neighbor_traj) ** 2, axis=1)))
                similarities.append(similarity)
            if similarities:
                shape_similarities.append(np.mean(similarities))

        if shape_similarities:
            consistency_scores['trajectory_shape_consistency'] = 1 / (1 + np.mean(shape_similarities))

        return consistency_scores

    def visualize_embeddings(self, save_path=None):
        """Visualize embeddings with t-SNE colored by features"""
        all_embeddings = []
        all_trajectories = []
        temporal_features = {
            'completion_time': [],
            'task_type': [],
            'rmsd': [],
            'is_success': [],
            'participant_id': []
        }

        with torch.no_grad():
            for batch_traj, temporal_batch in self.test_dataloader:
                batch_traj = batch_traj.to(self.device)
                task_type = temporal_batch['task_type'].float().unsqueeze(1).to(self.device)

                encoded, _, _ = self.model(batch_traj, task_type)
                encoded = encoded.cpu().numpy()

                all_embeddings.append(encoded)
                all_trajectories.extend(batch_traj.cpu().numpy())
                for key in temporal_features:
                    if key in temporal_batch:
                        temporal_features[key].extend(temporal_batch[key].cpu().numpy())

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_trajectories = np.array(all_trajectories)

        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(all_embeddings)

        # Create visualization
        fig = plt.figure(figsize=(20, 15))

        # Plot temporal features
        subplot_idx = 1
        for feature_name, feature_values in temporal_features.items():
            if not feature_values:  # Skip empty features
                continue

            plt.subplot(3, 2, subplot_idx)
            if feature_name in ['is_success', 'task_type', 'participant_id']:
                scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                                      c=feature_values, cmap='tab10', alpha=0.6)
            else:
                scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                                      c=feature_values, cmap='viridis', alpha=0.6)
            plt.title(f'Embeddings colored by {feature_name}')
            plt.colorbar(scatter)
            subplot_idx += 1

        # Plot sample trajectories (just x,y)
        plt.subplot(3, 2, 5)
        sample_indices = np.random.choice(len(all_trajectories), 5)
        for idx in sample_indices:
            traj = all_trajectories[idx]
            plt.plot(traj[:, 0], traj[:, 1], alpha=0.6)
        plt.title('Sample Trajectories (x,y)')

        # Plot embedding space with trajectory previews
        plt.subplot(3, 2, 6)
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.1, c='gray')
        for idx in sample_indices:
            x, y = embeddings_2d[idx]
            traj = all_trajectories[idx]
            plt.plot(traj[:, 0] * 0.1 + x, traj[:, 1] * 0.1 + y, alpha=0.8)
        plt.title('Embedding Space with Trajectory Previews')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def evaluate_all(self, save_viz_path=None):
        """Run all evaluations and return combined results"""
        results = {}
        results.update(self.evaluate_reconstruction())
        results.update(self.evaluate_temporal_correlation())
        results.update(self.evaluate_neighborhood_consistency())

        if save_viz_path:
            self.visualize_embeddings(save_viz_path)

        return results


