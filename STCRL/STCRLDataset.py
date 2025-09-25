import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import math
import copy
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class STCRLModelFittingDataset(Dataset):
    """Dataset class for temporal contrastive learning with 3D trajectories"""
    def __init__(self, df):
        self.trajectories = []
        self.temporal_data = {
            'completion_time': [],
            'task_type': [],
            'rmsd': [],
            'is_success': [],
            'participant_id': []
        }

        print("Starting dataset processing...")
        print(f"Total rows in dataframe: {len(df)}")

        for idx, row in df.iterrows():
            try:
                traj = row['normalized_trajectory']

                # Handle different data types
                if isinstance(traj, str):
                    # If it's a string representation, try to parse it
                    try:
                        import ast
                        traj = np.array(ast.literal_eval(traj))
                    except:
                        # If parsing fails, skip this row
                        continue

                # Ensure traj is numpy array with shape (512, 3)
                if isinstance(traj, np.ndarray) and len(traj.shape) == 2 and traj.shape[1] >= 3:
                    # Take x, y, t coordinates (first three columns)
                    spatial_temp_traj = traj[:, :3]

                    # Ensure we have 512 time steps (pad or truncate)
                    if len(spatial_temp_traj) > 512:
                        spatial_temp_traj = spatial_temp_traj[:512]
                    elif len(spatial_temp_traj) < 512:
                        padding = np.zeros((512 - len(spatial_temp_traj), 3))
                        spatial_temp_traj = np.vstack([spatial_temp_traj, padding])

                    self.trajectories.append(torch.FloatTensor(spatial_temp_traj))

                    # Extract temporal data
                    self.temporal_data['completion_time'].append(float(row.get('completion_time', 0.0)))
                    self.temporal_data['task_type'].append(int(row.get('task_type', 0)))
                    self.temporal_data['rmsd'].append(float(row.get('rmsd', 0.0)))
                    self.temporal_data['is_success'].append(int(row.get('is_success', 0)))
                    self.temporal_data['participant_id'].append(int(row.get('participant_id', 0)))

            except Exception as e:
                print(f"Error processing row {idx}: {str(e)}")
                continue

        if len(self.trajectories) == 0:
            raise ValueError("No valid trajectories were loaded!")

        print(f"Successfully loaded {len(self.trajectories)} valid trajectories")

        # Convert to tensors
        for key in self.temporal_data:
            self.temporal_data[key] = torch.tensor(self.temporal_data[key])

        # Verify tensor shapes
        print(f"Trajectory shape: {self.trajectories[0].shape}")
        print(f"Number of temporal features: {len(self.temporal_data)}")

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        temporal_batch = {k: v[idx] for k, v in self.temporal_data.items()}
        return self.trajectories[idx], temporal_batch