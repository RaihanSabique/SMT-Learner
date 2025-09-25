import torch
import numpy as np
import pandas as pd
import os
import json
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from .STCRLDataset import STCRLModelFittingDataset
from .Evaluator import STCRLModelEvaluator
from .ContrastiveLossFunctions import CompletionTimeLoss, RMSDLoss, \
    TaskTypeLoss, SuccessLoss, WithinBetweenSubjectLoss, MultiTemporalLoss
from .TrainSTCRL import train_stcrl_model

def evaluate_stcrl_model(model, test_df, batch_size=32, save_viz_path=None):
    """Wrapper function for model evaluation"""
    # Create test dataset and dataloader
    test_dataset = STCRLModelFittingDataset(test_df)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create evaluator
    device = next(model.parameters()).device
    evaluator = STCRLModelEvaluator(model, test_dataloader, device)

    # Run evaluation
    results = evaluator.evaluate_all(save_viz_path)

    # Print results
    print("\nModel Evaluation Results:")
    print("-------------------------")
    print(f"Reconstruction MSE: {results.get('reconstruction_mse', 'N/A')}")

    print("\nTemporal Correlations:")
    for key in ['completion_time_correlation', 'task_type_accuracy', 'rmsd_correlation',
                'success_accuracy', 'participant_silhouette']:
        if key in results:
            print(f"{key}: {results[key]:.4f}")
        else:
            print(f"{key}: N/A")

    print("\nNeighborhood Consistency:")
    for key in ['completion_time_consistency', 'task_type_consistency', 'rmsd_consistency',
                'is_success_consistency', 'participant_id_consistency', 'trajectory_shape_consistency']:
        if key in results:
            print(f"{key}: {results[key]:.4f}")
        else:
            print(f"{key}: N/A")

    return results

def save_model_for_transfer(model, optimizer, history, hyperparams, filepath):
        """
        Save the trained model along with training history and hyperparameters.

        Args:
            model: The trained PyTorch model
            optimizer: The optimizer used for training
            history: Dictionary containing training metrics
            hyperparams: Dictionary of model hyperparameters
            filepath: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save model state dictionary
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'hyperparams': hyperparams,
        }, filepath + '.pt')

        # Save training history separately (as JSON)
        with open(filepath + '_history.json', 'w') as f:
            json.dump(history, f)

        # Save model architecture information for future reference
        model_info = {
            'type': model.__class__.__name__,
            'hidden_dim': model.hidden_dim,
            'num_layers': len(model.fusion_transformer.layers),
            'nhead': model.fusion_transformer.layers[0].self_attn.num_heads,
            'seq_len': model.seq_len,
            'input_dim': model.input_dim,
        }

        with open(filepath + '_architecture.json', 'w') as f:
            json.dump(model_info, f)

        print(f"Model saved successfully to {filepath}")


def train_and_evaluate_models(train_df, test_df, output_dir='model_results', epochs=10, batch_size=16):
    """Train and evaluate multiple models with different loss functions"""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define loss functions to test
    loss_functions = {
        # 'completion_time': CompletionTimeLoss(),
        # 'task_type': TaskTypeLoss(),
        # 'rmsd': RMSDLoss(),
        # 'success': SuccessLoss(),
        # 'within_between_subject': WithinBetweenSubjectLoss(within_weight=0.7),
        'multi_loss': MultiTemporalLoss(
            weights=(0.3, 0.2, 0.2, 0.2, 0.1),
            within_subject_weight=0.7
        )
    }

    results = {}
    models = {}
    optimizers = {}  # Store optimizers
    histories = {}
    all_hyperparams = {}  # Store hyperparameters

    start_time = time.time()

    # Train and evaluate each model
    for loss_name, loss_fn in loss_functions.items():
        print(f"\n{'=' * 50}")
        print(f"Training model with {loss_name} loss...")
        print(f"{'=' * 50}")

        # Define hyperparameters for this model
        hyperparams = {
            'hidden_dim': 512,
            'nhead': 8,
            'num_layers': 3,
            'epochs': epochs,
            'batch_size': batch_size,
            'loss_type': loss_name
        }
        # If it's a multi-loss, add the weights to hyperparams
        if loss_name == 'multi_loss':
            hyperparams['loss_weights'] = loss_fn.weights
            hyperparams['within_subject_weight'] = loss_fn.within_subject_weight

        all_hyperparams[loss_name] = hyperparams

        # Train the model - modify to return optimizer as well
        model, optimizer, history = train_stcrl_model(
            train_df,
            loss_fn=loss_fn,
            hidden_dim=hyperparams['hidden_dim'],
            nhead=hyperparams['nhead'],
            num_layers=hyperparams['num_layers'],
            epochs=hyperparams['epochs'],
            batch_size=hyperparams['batch_size']
        )

        models[loss_name] = model
        optimizers[loss_name] = optimizer
        histories[loss_name] = history

        # Save the model using the save_model_for_transfer function
        model_path = os.path.join(output_dir, f"{loss_name}_model")
        save_model_for_transfer(
            model=model,
            optimizer=optimizer,
            history=history,
            hyperparams=hyperparams,
            filepath=model_path
        )
        # Evaluate the model
        viz_path = os.path.join(output_dir, f"{loss_name}_embeddings.png")
        results[loss_name] = evaluate_stcrl_model(model, test_df, save_viz_path=viz_path)

        # Save the results
        results_path = os.path.join(output_dir, f"{loss_name}_results.csv")
        pd.DataFrame(results[loss_name], index=[0]).to_csv(results_path)

        # Save the training history
        history_path = os.path.join(output_dir, f"{loss_name}_history.csv")
        pd.DataFrame(history).to_csv(history_path)

    # Calculate total training time
    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time / 60:.2f} minutes")

    # Create comparison dataframe
    metrics = [
        'reconstruction_mse', 'endpoint_error_mean', 'curvature_error_mean',
        'completion_time_correlation', 'task_type_accuracy', 'rmsd_correlation',
        'success_accuracy', 'participant_silhouette',
        'completion_time_consistency', 'task_type_consistency', 'rmsd_consistency',
        'is_success_consistency', 'participant_id_consistency', 'trajectory_shape_consistency'
    ]

    comparison_df = pd.DataFrame(index=metrics)
    for loss_name in loss_functions.keys():
        comparison_df[loss_name] = [results[loss_name].get(metric, np.nan) for metric in metrics]

    # Save comparison results
    comparison_path = os.path.join(output_dir, "model_comparison.csv")
    comparison_df.to_csv(comparison_path)

    print("\nMetric Comparison:")
    print(comparison_df)

    # Visualize comparison
    plt.figure(figsize=(15, 12))
    ax = comparison_df.plot(kind='bar', figsize=(15, 12))
    plt.title('Model Performance Comparison')
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_bar.png"))
    plt.close()

    # Plot training histories
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 1, 1)
    for loss_name, history in histories.items():
        plt.plot(history['train_loss'], label=f'{loss_name} (train)')
        plt.plot(history['val_loss'], '--', label=f'{loss_name} (val)')
    plt.title('Training and Validation Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.subplot(2, 1, 2)
    for loss_name, history in histories.items():
        plt.plot(history['recon_loss'], label=f'{loss_name} (recon)')
        plt.plot(history['contrastive_loss'], '--', label=f'{loss_name} (contrastive)')
    plt.title('Loss Components Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_history.png"))
    plt.close()

    # Create radar plot for model comparison
    # First, handle metrics where lower is better (like MSE)
    radar_df = comparison_df.copy()
    for col in radar_df.columns:
        # Normalize MSE and error values (lower is better)
        for metric in ['reconstruction_mse', 'endpoint_error_mean', 'curvature_error_mean']:
            if not pd.isna(radar_df.loc[metric, col]):
                max_val = radar_df.loc[metric].max()
                if max_val > 0:
                    radar_df.loc[metric, col] = 1 - (radar_df.loc[metric, col] / max_val)

    # Then normalize all values between 0 and 1
    for idx in radar_df.index:
        row_max = radar_df.loc[idx].max()
        if row_max > 0:
            radar_df.loc[idx] = radar_df.loc[idx] / row_max

    # Plot radar chart
    plt.figure(figsize=(12, 12))

    # Prepare the radar plot
    categories = metrics
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # close the loop

    ax = plt.subplot(111, polar=True)

    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], categories, color='grey', size=8)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=7)
    plt.ylim(0, 1)

    # Plot each model
    for i, (loss_name, values) in enumerate(radar_df.items()):
        values = values.values.flatten().tolist()
        values += values[:1]  # close the loop

        # Handle NaN values
        values = [0 if pd.isna(v) else v for v in values]

        ax.plot(angles, values, linewidth=1, linestyle='solid', label=loss_name)
        ax.fill(angles, values, alpha=0.1)

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Model Performance Comparison Radar Chart')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "radar_chart.png"))
    plt.close()

    return results, comparison_df, models, optimizers, histories, all_hyperparams

