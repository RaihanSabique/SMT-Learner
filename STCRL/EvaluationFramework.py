"""
Comprehensive evaluation framework for baseline model comparison
Implements all metrics and statistical tests for the NeurIPS rebuttal
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class EvaluationMetrics:
    """
    Comprehensive evaluation metrics for trajectory embedding models
    """
    
    def __init__(self):
        self.metrics = {}
    
    def trajectory_reconstruction_quality(self, original: torch.Tensor, reconstructed: torch.Tensor) -> float:
        """
        Calculate Mean Squared Error between original and reconstructed trajectories
        
        Args:
            original: Original trajectories [batch_size, seq_len, input_dim]
            reconstructed: Reconstructed trajectories [batch_size, seq_len, input_dim]
        
        Returns:
            MSE reconstruction error
        """
        mse = torch.mean((original - reconstructed) ** 2)
        return mse.item()
    
    def endpoint_error(self, original: torch.Tensor, reconstructed: torch.Tensor) -> float:
        """
        Calculate distance between final positions
        
        Args:
            original: Original trajectories [batch_size, seq_len, input_dim]
            reconstructed: Reconstructed trajectories [batch_size, seq_len, input_dim]
        
        Returns:
            Average endpoint error
        """
        # Take x, y coordinates of final non-zero point for each trajectory
        orig_endpoints = []
        recon_endpoints = []
        
        for i in range(original.size(0)):
            # Find last non-zero point
            orig_traj = original[i]
            recon_traj = reconstructed[i]
            
            # Find last non-zero timestep
            non_zero_mask = torch.sum(torch.abs(orig_traj), dim=1) > 1e-6
            if non_zero_mask.any():
                last_idx = torch.where(non_zero_mask)[0][-1]
                orig_endpoints.append(orig_traj[last_idx, :2])  # x, y coordinates
                recon_endpoints.append(recon_traj[last_idx, :2])
            else:
                # If all zeros, use last point
                orig_endpoints.append(orig_traj[-1, :2])
                recon_endpoints.append(recon_traj[-1, :2])
        
        orig_endpoints = torch.stack(orig_endpoints)
        recon_endpoints = torch.stack(recon_endpoints)
        
        error = torch.mean(torch.norm(orig_endpoints - recon_endpoints, dim=1))
        return error.item()
    
    def calculate_curvature(self, trajectories: torch.Tensor) -> torch.Tensor:
        """
        Calculate curvature for trajectories
        
        Args:
            trajectories: Trajectories [batch_size, seq_len, input_dim]
        
        Returns:
            Average curvature per trajectory
        """
        curvatures = []
        
        for i in range(trajectories.size(0)):
            traj = trajectories[i, :, :2]  # x, y coordinates only
            
            # Calculate first and second derivatives
            dx = torch.diff(traj[:, 0])
            dy = torch.diff(traj[:, 1])
            
            if len(dx) < 2:
                curvatures.append(torch.tensor(0.0))
                continue
                
            ddx = torch.diff(dx)
            ddy = torch.diff(dy)
            
            # Curvature formula: |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
            numerator = torch.abs(dx[:-1] * ddy - dy[:-1] * ddx)
            denominator = (dx[:-1]**2 + dy[:-1]**2)**(3/2)
            
            # Avoid division by zero
            denominator = torch.clamp(denominator, min=1e-8)
            curvature = numerator / denominator
            
            # Handle NaN values
            curvature = torch.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)
            avg_curvature = torch.mean(curvature)
            curvatures.append(avg_curvature)
        
        return torch.stack(curvatures)
    
    def curvature_error(self, original: torch.Tensor, reconstructed: torch.Tensor) -> float:
        """
        Calculate difference in path curvature
        
        Args:
            original: Original trajectories
            reconstructed: Reconstructed trajectories
        
        Returns:
            Average curvature error
        """
        orig_curvature = self.calculate_curvature(original)
        recon_curvature = self.calculate_curvature(reconstructed)
        error = torch.mean(torch.abs(orig_curvature - recon_curvature))
        return error.item()
    
    def temporal_correlation(self, embeddings: np.ndarray, metadata: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Calculate correlation between embeddings and temporal features
        
        Args:
            embeddings: Embedding vectors [n_samples, embedding_dim]
            metadata: Dictionary with temporal features
        
        Returns:
            Dictionary of correlation values
        """
        # Reduce embeddings to scalar via mean
        embedding_scalars = np.mean(embeddings, axis=1)
        
        correlations = {}
        
        # Completion time correlation
        if 'completion_time' in metadata:
            completion_times = np.array(metadata['completion_time'])
            if len(completion_times) > 1 and np.std(completion_times) > 1e-8:
                corr = np.corrcoef(embedding_scalars, completion_times)[0, 1]
                correlations['completion_time'] = corr if not np.isnan(corr) else 0.0
            else:
                correlations['completion_time'] = 0.0
        
        # RMSD correlation
        if 'rmsd' in metadata:
            rmsd_values = np.array(metadata['rmsd'])
            if len(rmsd_values) > 1 and np.std(rmsd_values) > 1e-8:
                corr = np.corrcoef(embedding_scalars, rmsd_values)[0, 1]
                correlations['rmsd'] = corr if not np.isnan(corr) else 0.0
            else:
                correlations['rmsd'] = 0.0
        
        # Success rate correlation
        if 'is_success' in metadata:
            success_values = np.array(metadata['is_success'])
            if len(success_values) > 1 and np.std(success_values) > 1e-8:
                corr = np.corrcoef(embedding_scalars, success_values)[0, 1]
                correlations['success'] = corr if not np.isnan(corr) else 0.0
            else:
                correlations['success'] = 0.0
        
        return correlations
    
    def clustering_consistency(self, embeddings: np.ndarray, metadata: Dict[str, np.ndarray], 
                             metric_name: str = 'completion_time', k: int = 5) -> float:
        """
        Calculate k-nearest neighbor consistency for clustering
        
        Args:
            embeddings: Embedding vectors [n_samples, embedding_dim]
            metadata: Dictionary with performance metrics
            metric_name: Name of the metric to evaluate consistency for
            k: Number of nearest neighbors to consider
        
        Returns:
            Average consistency score
        """
        if metric_name not in metadata:
            return 0.0
        
        metric_values = np.array(metadata[metric_name])
        
        if len(metric_values) < k + 1:
            return 0.0
        
        # Fit k-NN model
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(embeddings)  # +1 to exclude self
        distances, indices = nbrs.kneighbors(embeddings)
        
        consistency_scores = []
        
        for i in range(len(embeddings)):
            target_value = metric_values[i]
            neighbor_indices = indices[i][1:]  # Exclude self (first neighbor)
            neighbor_values = metric_values[neighbor_indices]
            
            # Calculate similarity based on metric type
            if metric_name == 'is_success':
                # For binary success, use exact match
                similarities = (neighbor_values == target_value).astype(float)
            else:
                # For continuous metrics, use inverse distance similarity
                differences = np.abs(neighbor_values - target_value)
                max_diff = np.std(metric_values) if np.std(metric_values) > 1e-8 else 1.0
                similarities = np.exp(-differences / max_diff)
            
            consistency_scores.append(np.mean(similarities))
        
        return np.mean(consistency_scores)
    
    def silhouette_analysis(self, embeddings: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculate silhouette score for cluster quality
        
        Args:
            embeddings: Embedding vectors [n_samples, embedding_dim]
            labels: Cluster labels
        
        Returns:
            Silhouette score
        """
        if len(np.unique(labels)) < 2:
            return 0.0
        
        try:
            score = silhouette_score(embeddings, labels)
            return score
        except:
            return 0.0


class BaselineTrainer:
    """
    Unified trainer for all baseline models
    """
    
    def __init__(self, device='cpu'):
        self.device = device
    
    def train_model(self, model, train_loader, val_loader, config):
        """
        Train a baseline model with specified configuration
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration dictionary
        
        Returns:
            Trained model and training history
        """
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(config['epochs']):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                trajectories, metadata = batch
                trajectories = trajectories.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass - handle different model types
                if hasattr(model, '__class__') and 'VAE' in model.__class__.__name__:
                    reconstructed, embeddings, mu, logvar = model(trajectories)
                    # VAE loss with KL divergence
                    recon_loss = nn.MSELoss()(reconstructed, trajectories)
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = recon_loss + config.get('beta', 0.001) * kl_loss
                else:
                    reconstructed, embeddings = model(trajectories)
                    loss = nn.MSELoss()(reconstructed, trajectories)
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    trajectories, metadata = batch
                    trajectories = trajectories.to(self.device)
                    
                    if hasattr(model, '__class__') and 'VAE' in model.__class__.__name__:
                        reconstructed, embeddings, mu, logvar = model(trajectories)
                        recon_loss = nn.MSELoss()(reconstructed, trajectories)
                        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                        loss = recon_loss + config.get('beta', 0.001) * kl_loss
                    else:
                        reconstructed, embeddings = model(trajectories)
                        loss = nn.MSELoss()(reconstructed, trajectories)
                    
                    val_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), f'best_{model.__class__.__name__}.pth')
            else:
                patience_counter += 1
                if patience_counter >= config['early_stopping_patience']:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}')
        
        # Load best model
        model.load_state_dict(torch.load(f'best_{model.__class__.__name__}.pth'))
        return model, history


class ComprehensiveComparison:
    """
    Main class for comprehensive baseline comparison
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        self.evaluator = EvaluationMetrics()
        self.trainer = BaselineTrainer(device)
    
    def evaluate_all_models(self, models: Dict, test_loader, metadata: Dict) -> Dict:
        """
        Evaluate all models on standardized metrics
        
        Args:
            models: Dictionary of models to evaluate
            test_loader: Test data loader
            metadata: Test metadata
        
        Returns:
            Dictionary of results for each model
        """
        results = {}
        
        for model_name, model in models.items():
            print(f"Evaluating {model_name}...")
            model.eval()
            model = model.to(self.device)
            
            model_results = {
                'reconstruction_mse': [],
                'endpoint_error': [],
                'curvature_error': [],
                'embeddings': [],
                'reconstructions': []
            }
            
            with torch.no_grad():
                for batch in test_loader:
                    trajectories, batch_metadata = batch
                    trajectories = trajectories.to(self.device)
                    
                    # Forward pass - handle different model output formats
                    if hasattr(model, '__class__') and 'VAE' in model.__class__.__name__:
                        reconstructed, embeddings, _, _ = model(trajectories)
                    elif hasattr(model, '__class__') and 'STCRLTransformer' in model.__class__.__name__:
                        # Real STCRL models return (embeddings, projection, reconstruction)
                        embeddings, projection, reconstructed = model(trajectories, return_projection=True)
                    elif hasattr(model, '__class__') and ('STCRL' in model.__class__.__name__ or 'DummySTCRL' in model.__class__.__name__):
                        # Dummy STCRL and other STCRL variants
                        try:
                            embeddings, projection, reconstructed = model(trajectories, return_projection=True)
                        except TypeError:
                            # Fallback for models that don't support return_projection
                            reconstructed, embeddings = model(trajectories)
                    else:
                        reconstructed, embeddings = model(trajectories)
                    
                    # Calculate metrics
                    recon_mse = self.evaluator.trajectory_reconstruction_quality(trajectories, reconstructed)
                    endpoint_err = self.evaluator.endpoint_error(trajectories, reconstructed)
                    curvature_err = self.evaluator.curvature_error(trajectories, reconstructed)
                    
                    model_results['reconstruction_mse'].append(recon_mse)
                    model_results['endpoint_error'].append(endpoint_err)
                    model_results['curvature_error'].append(curvature_err)
                    model_results['embeddings'].append(embeddings.cpu().numpy())
                    model_results['reconstructions'].append(reconstructed.cpu().numpy())
            
            # Aggregate embeddings and calculate correlations
            all_embeddings = np.vstack(model_results['embeddings'])
            temporal_corrs = self.evaluator.temporal_correlation(all_embeddings, metadata)
            
            # Calculate clustering consistency for different metrics
            consistency_metrics = {}
            for metric_name in ['completion_time', 'rmsd', 'is_success']:
                if metric_name in metadata:
                    consistency = self.evaluator.clustering_consistency(
                        all_embeddings, metadata, metric_name
                    )
                    consistency_metrics[f'{metric_name}_consistency'] = consistency
            
            # Store aggregated results
            results[model_name] = {
                'reconstruction_mse': np.mean(model_results['reconstruction_mse']),
                'reconstruction_mse_std': np.std(model_results['reconstruction_mse']),
                'endpoint_error': np.mean(model_results['endpoint_error']),
                'endpoint_error_std': np.std(model_results['endpoint_error']),
                'curvature_error': np.mean(model_results['curvature_error']),
                'curvature_error_std': np.std(model_results['curvature_error']),
                **temporal_corrs,
                **consistency_metrics
            }
        
        return results
    
    def statistical_comparison(self, results: Dict, baseline_results: Dict, metric_name: str) -> Dict:
        """
        Perform statistical tests comparing STCRL with baselines
        
        Args:
            results: Results from comprehensive evaluation
            baseline_results: Detailed results for statistical testing
            metric_name: Metric to compare
        
        Returns:
            Statistical comparison results
        """
        stcrl_values = baseline_results.get('STCRL', {}).get(f'{metric_name}_values', [])
        
        comparisons = {}
        
        for model_name, model_results in baseline_results.items():
            if model_name == 'STCRL':
                continue
            
            model_values = model_results.get(f'{metric_name}_values', [])
            
            if len(stcrl_values) > 1 and len(model_values) > 1:
                statistic, p_value = stats.ttest_ind(stcrl_values, model_values)
                
                comparisons[f'STCRL_vs_{model_name}'] = {
                    'statistic': statistic,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'stcrl_better': statistic < 0 if 'error' in metric_name or 'mse' in metric_name else statistic > 0
                }
        
        return comparisons
    
    def create_comparison_table(self, results: Dict) -> pd.DataFrame:
        """
        Create a formatted comparison table
        
        Args:
            results: Results from comprehensive evaluation
        
        Returns:
            Pandas DataFrame with comparison results
        """
        metrics = [
            'reconstruction_mse', 'endpoint_error', 'curvature_error',
            'completion_time', 'rmsd', 'success',
            'completion_time_consistency', 'rmsd_consistency', 'is_success_consistency'
        ]
        
        # Filter metrics that exist in results
        available_metrics = []
        for metric in metrics:
            if any(metric in model_results for model_results in results.values()):
                available_metrics.append(metric)
        
        df = pd.DataFrame.from_dict(results, orient='index')
        df = df.reindex(columns=available_metrics, fill_value=0.0)
        
        # Round values for display
        df = df.round(4)
        
        # Add ranking for each metric
        for metric in available_metrics:
            if 'error' in metric or 'mse' in metric:
                # Lower is better
                df[f'{metric}_rank'] = df[metric].rank(method='min')
            else:
                # Higher is better
                df[f'{metric}_rank'] = df[metric].rank(method='min', ascending=False)
        
        return df
    
    def visualize_results(self, results: Dict, save_path: str = 'baseline_comparison.png'):
        """
        Create comprehensive visualization of comparison results
        
        Args:
            results: Results dictionary
            save_path: Path to save the visualization
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        models = list(results.keys())
        
        # 1. Reconstruction Quality Metrics
        recon_metrics = ['reconstruction_mse', 'endpoint_error', 'curvature_error']
        available_recon = [m for m in recon_metrics if m in results[models[0]]]
        
        if available_recon:
            ax = axes[0, 0]
            x = np.arange(len(available_recon))
            width = 0.15
            
            for i, model in enumerate(models):
                values = [results[model].get(metric, 0) for metric in available_recon]
                ax.bar(x + i * width, values, width, label=model)
            
            ax.set_xlabel('Reconstruction Metrics')
            ax.set_ylabel('Error Value')
            ax.set_title('Reconstruction Quality Comparison')
            ax.set_xticks(x + width * len(models) / 2)
            ax.set_xticklabels([m.replace('_', ' ').title() for m in available_recon], rotation=45)
            ax.legend()
            ax.set_yscale('log')  # Log scale for better visualization
        
        # 2. Temporal Correlations
        corr_metrics = ['completion_time', 'rmsd', 'success']
        available_corr = [m for m in corr_metrics if m in results[models[0]]]
        
        if available_corr:
            ax = axes[0, 1]
            x = np.arange(len(available_corr))
            
            for i, model in enumerate(models):
                values = [results[model].get(metric, 0) for metric in available_corr]
                ax.bar(x + i * width, values, width, label=model)
            
            ax.set_xlabel('Correlation Metrics')
            ax.set_ylabel('Correlation Value')
            ax.set_title('Temporal Feature Correlations')
            ax.set_xticks(x + width * len(models) / 2)
            ax.set_xticklabels([f'{m} Correlation' for m in available_corr], rotation=45)
            ax.legend()
        
        # 3. Clustering Consistency
        cons_metrics = ['completion_time_consistency', 'rmsd_consistency', 'is_success_consistency']
        available_cons = [m for m in cons_metrics if m in results[models[0]]]
        
        if available_cons:
            ax = axes[1, 0]
            x = np.arange(len(available_cons))
            
            for i, model in enumerate(models):
                values = [results[model].get(metric, 0) for metric in available_cons]
                ax.bar(x + i * width, values, width, label=model)
            
            ax.set_xlabel('Consistency Metrics')
            ax.set_ylabel('Consistency Score')
            ax.set_title('Clustering Consistency')
            ax.set_xticks(x + width * len(models) / 2)
            ax.set_xticklabels([m.replace('_consistency', '').replace('_', ' ').title() for m in available_cons], rotation=45)
            ax.legend()
        
        # 4. Overall Performance Heatmap
        ax = axes[1, 1]
        
        # Create normalized scores for heatmap
        all_metrics = []
        for metric in ['reconstruction_mse', 'endpoint_error', 'curvature_error', 'completion_time', 'rmsd', 'completion_time_consistency', 'rmsd_consistency']:
            if metric in results[models[0]]:
                all_metrics.append(metric)
        
        if all_metrics:
            heatmap_data = []
            for model in models:
                model_scores = []
                for metric in all_metrics:
                    score = results[model].get(metric, 0)
                    # Normalize scores (invert for error metrics)
                    if 'error' in metric or 'mse' in metric:
                        score = -score  # Invert so higher is better
                    model_scores.append(score)
                heatmap_data.append(model_scores)
            
            heatmap_data = np.array(heatmap_data)
            # Normalize each column to [0, 1]
            for j in range(heatmap_data.shape[1]):
                col = heatmap_data[:, j]
                if np.std(col) > 1e-8:
                    heatmap_data[:, j] = (col - np.min(col)) / (np.max(col) - np.min(col))
            
            sns.heatmap(heatmap_data, 
                       xticklabels=[m.replace('_', ' ').title() for m in all_metrics],
                       yticklabels=models,
                       annot=True,
                       cmap='viridis',
                       ax=ax)
            ax.set_title('Normalized Performance Heatmap')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
