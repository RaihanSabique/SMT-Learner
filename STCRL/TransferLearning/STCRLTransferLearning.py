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
from .TransferLearningFramework import TransferLearningFramework
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
from STCRL.ContrastiveLossFunctions import CompletionTimeLoss, RMSDLoss, \
    TaskTypeLoss, SuccessLoss, WithinBetweenSubjectLoss, MultiTemporalLoss

class STCRLTransferLearningRunner:
    """
    Main runner class for STCRL transfer learning experiments.
    Handles cross-subject, cross-task, and zero-shot transfer scenarios.
    """

    def __init__(self, source_model_path, save_dir="transfer_learning_results"):
        self.source_model_path = source_model_path
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Load source model
        self.source_model = self._load_source_model()

        # Initialize loss functions
        self.loss_functions = {
            'completion_time': CompletionTimeLoss(),
            'task_type': TaskTypeLoss(),
            'rmsd': RMSDLoss(),
            'success': SuccessLoss(),
            'within_subject': WithinBetweenSubjectLoss(within_weight=0.7)
        }

        # Initialize transfer framework
        self.transfer_framework = TransferLearningFramework(
            self.source_model, self.loss_functions
        )

    def _load_source_model(self):
        """Load the pre-trained source model"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load architecture info
        with open(self.source_model_path + '_architecture.json', 'r') as f:
            model_info = json.load(f)

        # Create model
        from STCRL.TransformerEncoder import STCRLTransformer
        model = STCRLTransformer(
            seq_len=model_info['seq_len'],
            input_dim=model_info['input_dim'],
            hidden_dim=model_info['hidden_dim'],
            nhead=model_info['nhead'],
            num_layers=model_info['num_layers'],
            metadata_dim=1
        ).to(device)

        # Load weights
        checkpoint = torch.load(self.source_model_path + '.pt', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        print(f"Source model loaded from {self.source_model_path}")
        return model

    def run_cross_subject_transfer(self, target_df, subject_ids=None, epochs=10):
        """
        Run cross-subject transfer learning experiment.

        Args:
            target_df: DataFrame with target subject data
            subject_ids: List of subject IDs to transfer to (if None, use all)
            epochs: Number of training epochs
        """
        print("=== Cross-Subject Transfer Learning ===")

        if subject_ids is not None:
            target_df = target_df[target_df['participant_id'].isin(subject_ids)]

        print(f"Target subjects: {target_df['participant_id'].unique()}")
        print(f"Target dataset size: {len(target_df)}")

        # Perform transfer learning
        transferred_model, history = self.transfer_framework.cross_subject_transfer(
            target_df, epochs=epochs, batch_size=16
        )

        # Save results
        model_path = os.path.join(self.save_dir, "cross_subject_model")
        self.transfer_framework.save_transferred_model(
            transferred_model, history, "cross_subject", model_path
        )

        # Evaluate
        from STCRL.STCRLDataset import STCRLModelFittingDataset
        test_dataset = STCRLModelFittingDataset(target_df)
        results = self.transfer_framework.evaluate_transfer_performance(
            transferred_model, test_dataset
        )

        print("Cross-subject transfer results:")
        for key, value in results.items():
            print(f"  {key}: {value:.4f}")

        return transferred_model, history, results

    def run_cross_task_transfer(self, target_df, task_types=None, epochs=15):
        """
        Run cross-task transfer learning experiment.

        Args:
            target_df: DataFrame with target task data
            task_types: List of task types to transfer to (if None, use all)
            epochs: Number of training epochs
        """
        print("=== Cross-Task Transfer Learning ===")

        if task_types is not None:
            target_df = target_df[target_df['task_type'].isin(task_types)]

        print(f"Target tasks: {target_df['task_type'].unique()}")
        print(f"Target dataset size: {len(target_df)}")

        # Perform transfer learning
        transferred_model, history = self.transfer_framework.cross_task_transfer(
            target_df, epochs=epochs, batch_size=16
        )

        # Save results
        model_path = os.path.join(self.save_dir, "cross_task_model")
        self.transfer_framework.save_transferred_model(
            transferred_model, history, "cross_task", model_path
        )

        # Evaluate
        from STCRL.STCRLDataset import STCRLModelFittingDataset
        test_dataset = STCRLModelFittingDataset(target_df)
        results = self.transfer_framework.evaluate_transfer_performance(
            transferred_model, test_dataset
        )

        print("Cross-task transfer results:")
        for key, value in results.items():
            print(f"  {key}: {value:.4f}")

        return transferred_model, history, results

    def run_zero_shot_transfer(self, target_df):
        """
        Run zero-shot transfer (direct application of source model).

        Args:
            target_df: DataFrame with target data
        """
        print("=== Zero-Shot Transfer ===")

        print(f"Target dataset size: {len(target_df)}")

        # Create dataset
        from STCRL.STCRLDataset import STCRLModelFittingDataset
        target_dataset = STCRLModelFittingDataset(target_df)

        # Perform zero-shot transfer
        zero_shot_results = self.transfer_framework.zero_shot_transfer(
            target_dataset, batch_size=32
        )

        # Evaluate zero-shot performance
        results = self.transfer_framework.evaluate_transfer_performance(
            self.source_model, target_dataset
        )

        print("Zero-shot transfer results:")
        for key, value in results.items():
            print(f"  {key}: {value:.4f}")

        return results

    def run_comprehensive_transfer_study(self, target_df, test_split=0.2):
        """
        Run a comprehensive transfer learning study comparing all methods.

        Args:
            target_df: DataFrame with target data
            test_split: Fraction of data to use for testing
        """
        print("=== Comprehensive Transfer Learning Study ===")

        # Split target data for training and testing
        train_df, test_df = train_test_split(
            target_df, test_size=test_split, random_state=42,
            stratify=target_df['task_type'] if 'task_type' in target_df.columns else None
        )

        all_results = {}
        all_histories = {}

        # 1. Zero-shot transfer
        print("\n1. Running zero-shot transfer...")
        zero_shot_results = self.run_zero_shot_transfer(test_df)
        all_results['zero_shot'] = zero_shot_results
        all_histories['zero_shot'] = {'train_loss': [], 'val_loss': []}  # Empty history for zero-shot

        # 2. Cross-subject transfer
        print("\n2. Running cross-subject transfer...")
        cs_model, cs_history, cs_results = self.run_cross_subject_transfer(train_df, epochs=10)

        # Test on held-out data
        from STCRL.STCRLDataset import STCRLModelFittingDataset
        test_dataset = STCRLModelFittingDataset(test_df)
        cs_test_results = self.transfer_framework.evaluate_transfer_performance(cs_model, test_dataset)

        all_results['cross_subject'] = cs_test_results
        all_histories['cross_subject'] = cs_history

        # 3. Cross-task transfer
        print("\n3. Running cross-task transfer...")
        ct_model, ct_history, ct_results = self.run_cross_task_transfer(train_df, epochs=15)

        # Test on held-out data
        ct_test_results = self.transfer_framework.evaluate_transfer_performance(ct_model, test_dataset)

        all_results['cross_task'] = ct_test_results
        all_histories['cross_task'] = ct_history

        # Create comprehensive comparison
        self._create_transfer_comparison_report(all_results, all_histories)

        # Visualize results
        # visualize_transfer_learning_results(
        #     list(all_histories.values()),
        #     list(all_histories.keys()),
        #     os.path.join(self.save_dir, "transfer_comparison.png")
        # )

        return all_results, all_histories

    def _create_transfer_comparison_report(self, results, histories):
        """Create a comprehensive comparison report"""
        # Create results DataFrame
        comparison_df = pd.DataFrame(results).T

        # Save results
        comparison_df.to_csv(os.path.join(self.save_dir, "transfer_comparison.csv"))

        # Create detailed report
        report_path = os.path.join(self.save_dir, "transfer_report.txt")
        with open(report_path, 'w') as f:
            f.write("STCRL Transfer Learning Comprehensive Report\n")
            f.write("=" * 50 + "\n\n")

            f.write("Results Summary:\n")
            f.write("-" * 20 + "\n")
            f.write(str(comparison_df))
            f.write("\n\n")

            # Analysis
            f.write("Analysis:\n")
            f.write("-" * 20 + "\n")

            best_method = comparison_df['total_loss'].idxmin()
            f.write(f"Best performing method: {best_method}\n")
            f.write(f"Best total loss: {comparison_df.loc[best_method, 'total_loss']:.4f}\n\n")

            # Convergence analysis
            f.write("Convergence Analysis:\n")
            for method, history in histories.items():
                if history['train_loss']:
                    final_loss = history['train_loss'][-1]
                    epochs = len(history['train_loss'])
                    f.write(f"{method}: {epochs} epochs, final loss: {final_loss:.4f}\n")

            f.write("\nRecommendations:\n")
            f.write("-" * 15 + "\n")
            if best_method == 'zero_shot':
                f.write("Zero-shot transfer is sufficient. The source domain generalizes well.\n")
            elif best_method == 'cross_subject':
                f.write("Cross-subject adaptation is most effective. Focus on subject-specific fine-tuning.\n")
            else:
                f.write("Cross-task adaptation is most effective. Task-specific features need adaptation.\n")

        print(f"Comprehensive report saved to {report_path}")