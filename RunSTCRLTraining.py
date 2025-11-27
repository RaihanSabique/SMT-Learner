import numpy as np
import pandas as pd
import torch
import json
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
from DataProcessing.Normalization import normalize_trajectory_sequence_3d
from STCRL.TrainAndEvaluate import train_and_evaluate_models
from STCRL.TransformerEncoder import STCRLTransformer

class RunSTCRLTraining:
    def __init__(self):
        self.data_path = "Dataset/SMT_Dataset/preprocessed_human_smt_dataset.csv"
        self.save_dir = save_dir = "saved_models/STCRL/"

    def loadAndProcessDataset(self):
        print("Loading dataset...")
        df = pd.read_csv(self.data_path)
        df['participant_id'], unique_participants = pd.factorize(df['participant_id'])
        df["normalized_trajectory"] = df.apply(
            lambda x: normalize_trajectory_sequence_3d(x['path'], x['time_diff_ms']), axis=1)
        # df = df[:2400]
        print("Data loaded successfully")
        print(df.head(5))
        return df

    def train_and_evaluate(self, df):
        train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
        print(f"Split data into {len(train_df)} training and {len(test_df)} testing samples")
        # Train and evaluate models
        results, comparison_df, models, optimizers, histories, all_hyperparams = train_and_evaluate_models(
            train_df=train_df,
            test_df=test_df,
            output_dir=self.save_dir,
            epochs=5,
            batch_size=24
        )

        # Save outputs
        comparison_df.to_csv(os.path.join(self.save_dir, "final_results.csv"))

        print(f"Training and evaluation complete. Results saved to {self.save_dir}")
        print("\nFinal Comparison Results:")
        print(comparison_df)
        return results, comparison_df, models, optimizers, histories, all_hyperparams

    def load_model_for_transfer(filepath, device=None):
        """
        Load a saved model for transfer learning.

        Args:
            filepath: Path to the saved model
            device: Device to load the model to (cuda/cpu)

        Returns:
            model: The loaded model
            hyperparams: The hyperparameters used to train the model
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model architecture information
        with open(filepath + '_architecture.json', 'r') as f:
            model_info = json.load(f)

        # Create model with the same architecture
        model = STCRLTransformer(
            seq_len=model_info['seq_len'],
            input_dim=model_info['input_dim'],
            hidden_dim=model_info['hidden_dim'],
            nhead=model_info['nhead'],
            num_layers=model_info['num_layers'],
            metadata_dim=1  # Assuming metadata_dim is 1
        ).to(device)

        # Load model weights
        checkpoint = torch.load(filepath + '.pt', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load training history for reference
        try:
            with open(filepath + '_history.json', 'r') as f:
                history = json.load(f)
            print("Training history loaded successfully")
        except FileNotFoundError:
            history = None
            print("No training history found")

        return model, checkpoint['hyperparams'], history

if __name__ == '__main__':
    trainer = RunSTCRLTraining()
    df = trainer.loadAndProcessDataset()
    results, comparison_df, models, optimizers, histories, all_hyperparams = trainer.train_and_evaluate(df)
    # trainer.save_model_for_transfer()

