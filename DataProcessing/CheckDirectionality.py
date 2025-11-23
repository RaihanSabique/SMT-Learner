import pandas as pd, numpy as np, ast
from numpy import degrees, arctan2

from DataProcessing.Normalization import normalize_trajectory_sequence_3d, \
    normalize_trajectory_sequence_3d_directionality


def angles_from_start_deg(traj_xy):
    """Per-point angle (degrees) from the first point to each point in (N,2)."""
    xy = np.asarray(traj_xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] < 2:
        return np.array([])
    v = xy - xy[0]  # vectors from start
    ang = np.degrees(np.arctan2(v[:, 1], v[:, 0]))
    # define angle at step 0 as 0 for convenience
    if ang.size > 0:
        ang[0] = 0.0
    return ang


def parse_path(x):
    """Parse raw path column to (N,2)."""
    arr = np.array(ast.literal_eval(x)) if isinstance(x, str) else np.array(x)
    return arr[:, :2]


def parse_normalized_trajectory(row):
    """Use the project parsing logic for normalized_trajectory and return (N,3)."""
    traj = row['normalized_trajectory']

    # Handle different data types
    if isinstance(traj, str):
        try:
            traj = np.array(ast.literal_eval(traj))
        except Exception:
            return None
    else:
        # Accept list-of-lists or other array-likes
        traj = np.array(traj)

    # Ensure traj is numpy array with shape (512, 3) or at least 3 cols
    if isinstance(traj, np.ndarray) and len(traj.shape) == 2 and traj.shape[1] >= 3:
        spatial_temp_traj = traj[:, :3]
        # Ensure 512 time steps (pad or truncate) to match dataset behavior
        if len(spatial_temp_traj) > 512:
            spatial_temp_traj = spatial_temp_traj[:512]
        elif len(spatial_temp_traj) < 512:
            padding = np.zeros((512 - len(spatial_temp_traj), 3))
            spatial_temp_traj = np.vstack([spatial_temp_traj, padding])
        return spatial_temp_traj
    return None


def last_valid_index_xy(arr3):
    """Return last index with non-zero x or y in (N,3) array; -1 if none."""
    xy = arr3[:, :2]
    nz = np.where(np.any(np.abs(xy) > 0, axis=1))[0]
    return int(nz[-1]) if nz.size > 0 else -1


if __name__ == '__main__':
    print("Compare per-point angles: raw vs normalized")
    # Load full preprocessed dataset; do not recompute normalization here
    df = pd.read_csv('../Dataset/SMT_Dataset/preprocessed_human_smt_dataset.csv')
    df = df[:24]
    # Compute normalization with directionality metadata and attach to df
    dir_meta_series = df.apply(
        lambda x: normalize_trajectory_sequence_3d_directionality(x['path'], x['time_diff_ms']), axis=1)
    dir_meta_df = pd.DataFrame(dir_meta_series.tolist())
    # Store normalized trajectory as list-of-lists for safe CSV round-trip
    df['normalized_trajectory'] = dir_meta_df['normalized_trajectory'].apply(
        lambda a: a.tolist() if isinstance(a, np.ndarray) else a)
    df['original_target_angle'] = dir_meta_df['original_target_angle']
    df['rotation_angle'] = dir_meta_df['rotation_angle']
    df['original_end_vector'] = dir_meta_df['original_end_vector'].apply(
        lambda v: v.tolist() if isinstance(v, np.ndarray) else v)
    df.to_csv('../DataProcessing/check_directionality_full_dataset.csv', index=False)
    out_rows = []
    max_rows = 10  # limit output size
    for idx, r in df.head(max_rows).iterrows():
        try:
            raw_xy = parse_path(r['path'])
            norm_3d = parse_normalized_trajectory(r)
            if norm_3d is None:
                continue
            end_idx = last_valid_index_xy(norm_3d)
            if end_idx <= 0:
                continue
            norm_xy = norm_3d[:end_idx + 1, :2]

            raw_angles = angles_from_start_deg(raw_xy)
            norm_angles = angles_from_start_deg(norm_xy)

            # Align by minimum length to compare per-step
            L = int(min(len(raw_angles), len(norm_angles)))
            raw_angles = raw_angles[:L]
            norm_angles = norm_angles[:L]
            diff = (raw_angles - norm_angles)

            for t in range(L):
                out_rows.append({
                    'row_idx': int(idx),
                    'step': t,
                    'raw_angle_deg': float(round(raw_angles[t], 4)),
                    'normalized_angle_deg': float(round(norm_angles[t], 4)),
                    'angle_diff_deg': float(round(diff[t], 4))
                })
        except Exception as e:
            print(f"Row {idx} error: {e}")

    out_df = pd.DataFrame(out_rows)
    save_path = '../DataProcessing/check_directionality_per_point.csv'
    out_df.to_csv(save_path, index=False)
    # Print a small preview
    print(out_df.head(12).to_string(index=False))
    print(f"\nSaved per-point angle comparison to: {save_path}")