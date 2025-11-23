import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import math
import copy
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from DataProcessing.Normalization import normalize_trajectory_sequence_3d

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

        skipped = 0
        skip_reasons = {
            'missing_path_time': 0,
            'norm_exception': 0,
            'parse_str_fail': 0,
            'reshape_fail': 0,
            'shape_invalid': 0,
            'not_ndarray': 0,
            'nan_or_inf': 0
        }
        sample_skips = []
        for idx, row in df.iterrows():
            try:
                # Prefer existing normalized_trajectory; if missing or invalid, compute canonical normalization
                traj = row.get('normalized_trajectory', None)
                needs_norm = False
                if traj is None:
                    needs_norm = True
                elif isinstance(traj, (list, tuple)) and len(traj) == 0:
                    needs_norm = True
                elif isinstance(traj, str) and (traj.strip() == '' or traj.strip().lower() in ['nan', '[]']):
                    needs_norm = True

                if needs_norm:
                    # Require path and time_diff_ms to exist
                    path = row.get('path', None)
                    tdiff = row.get('time_diff_ms', None)
                    # If missing time diffs, derive from 'Time' (MATLAB-style) if present
                    if (tdiff is None or (isinstance(tdiff, str) and tdiff.strip() == '')) and ('Time' in row):
                        try:
                            s = str(row['Time']).strip().strip('[]')
                            parts = [p for p in s.split(';') if p != ''] if ';' in s else [p for p in s.replace('\n', ',').split(',') if p != '']
                            tdiff = np.array([float(val) * 1000.0 for val in parts], dtype=float)
                        except Exception:
                            tdiff = None
                    if path is None or tdiff is None:
                        skipped += 1
                        skip_reasons['missing_path_time'] += 1
                        if len(sample_skips) < 10:
                            sample_skips.append((idx, 'missing_path_time'))
                        continue
                    try:
                        traj = normalize_trajectory_sequence_3d(path, tdiff)
                    except Exception as e:
                        print(f"Row {idx} normalization failed: {e}")
                        skipped += 1
                        skip_reasons['norm_exception'] += 1
                        if len(sample_skips) < 10:
                            sample_skips.append((idx, 'norm_exception'))
                        continue

                # Handle different data types
                if isinstance(traj, str):
                    # If it's a string representation, try to parse it
                    try:
                        import ast
                        traj = np.array(ast.literal_eval(traj))
                    except Exception:
                        skipped += 1
                        skip_reasons['parse_str_fail'] += 1
                        if len(sample_skips) < 10:
                            sample_skips.append((idx, 'parse_str_fail'))
                        continue
                elif isinstance(traj, (list, tuple)):
                    traj = np.array(traj)
                elif isinstance(traj, pd.Series):
                    traj = traj.to_numpy()

                # Ensure traj is numpy array with shape (512, C) (C >= 3)
                if isinstance(traj, np.ndarray):
                    if traj.ndim == 1:
                        if traj.size % 3 == 0 and traj.size > 0:
                            try:
                                traj = traj.reshape(-1, 3)
                            except Exception:
                                skipped += 1
                                skip_reasons['reshape_fail'] += 1
                                if len(sample_skips) < 10:
                                    sample_skips.append((idx, 'reshape_fail'))
                                continue
                        else:
                            skipped += 1
                            skip_reasons['shape_invalid'] += 1
                            if len(sample_skips) < 10:
                                sample_skips.append((idx, 'shape_invalid_dim1'))
                            continue

                    # Drop rows with NaN/Inf
                    if not np.isfinite(traj).all():
                        skipped += 1
                        skip_reasons['nan_or_inf'] += 1
                        if len(sample_skips) < 10:
                            sample_skips.append((idx, 'nan_or_inf'))
                        continue

                    if traj.ndim == 2 and traj.shape[1] >= 3 and traj.shape[0] > 0:
                        spatial_temp_traj = traj  # keep all available channels (x,y,t,(theta),(rotation))
                        channels = spatial_temp_traj.shape[1]

                        # Pad or truncate to 512 timesteps
                        if len(spatial_temp_traj) > 512:
                            spatial_temp_traj = spatial_temp_traj[:512, :]
                        elif len(spatial_temp_traj) < 512:
                            padding = np.zeros((512 - len(spatial_temp_traj), channels))
                            spatial_temp_traj = np.vstack([spatial_temp_traj, padding])

                        self.trajectories.append(torch.FloatTensor(spatial_temp_traj))

                        # Extract temporal data
                        self.temporal_data['completion_time'].append(float(row.get('completion_time', 0.0)))
                        self.temporal_data['task_type'].append(int(row.get('task_type', 0)))
                        self.temporal_data['rmsd'].append(float(row.get('rmsd', 0.0)))
                        self.temporal_data['is_success'].append(int(row.get('is_success', 0)))
                        self.temporal_data['participant_id'].append(int(row.get('participant_id', 0)))
                    else:
                        skipped += 1
                        skip_reasons['shape_invalid'] += 1
                        if len(sample_skips) < 10:
                            sample_skips.append((idx, f"shape_invalid_{getattr(traj,'shape',None)}"))
                        continue
                else:
                    skipped += 1
                    skip_reasons['not_ndarray'] += 1
                    if len(sample_skips) < 10:
                        sample_skips.append((idx, 'not_ndarray'))
                    continue

            except Exception as e:
                print(f"Error processing row {idx}: {str(e)}")
                skipped += 1
                continue

        if len(self.trajectories) == 0:
            raise ValueError("No valid trajectories were loaded!")

        print(f"Successfully loaded {len(self.trajectories)} valid trajectories")
        if skipped:
            print(f"Skipped {skipped} rows due to invalid/parse errors")
            print("Skip reasons summary:")
            for k, v in skip_reasons.items():
                if v:
                    print(f"  - {k}: {v}")
            if sample_skips:
                print("Sample skipped rows (index, reason):", sample_skips)

        # Convert to tensors
        for key in self.temporal_data:
            self.temporal_data[key] = torch.tensor(self.temporal_data[key])

        # Verify tensor shapes
        print(f"Trajectory shape: {self.trajectories[0].shape}")
        print(f"Number of temporal features: {len(self.temporal_data)}")

        # Store input dimension for downstream model config use
        self.input_dim = self.trajectories[0].shape[1]

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        temporal_batch = {k: v[idx] for k, v in self.temporal_data.items()}
        return self.trajectories[idx], temporal_batch

    def get_input_dim(self):
        return self.input_dim