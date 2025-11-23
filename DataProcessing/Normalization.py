import numpy as np
import ast


    import ast
    # Parse input trajectory
    try:
        if isinstance(path, str):
            trajectory = np.array(ast.literal_eval(path))
        else:
            trajectory = np.array(path)
    except Exception:
        # Fallback: try eval as last resort
        try:
            trajectory = np.array(eval(path) if isinstance(path, str) else path)
        except Exception:
            trajectory = np.array([])
    """
    # Parse time sequence (differences in ms)
    try:
        if isinstance(time_diff_ms, str):
            time_sequence = np.array(ast.literal_eval(time_diff_ms))
        else:
            time_sequence = np.array(time_diff_ms)
    except Exception:
        try:
            time_sequence = np.array(eval(time_diff_ms) if isinstance(time_diff_ms, str) else time_diff_ms)
        except Exception:
            time_sequence = None

    Args:
        trajectory: numpy array of shape (num_points, 2) containing x,y coordinates
        time_sequence: numpy array of timestamps in milliseconds for each point
        # Coerce trajectory to 2D (N,2) if flattened
        if trajectory.ndim == 1 and trajectory.size % 2 == 0:
            trajectory = trajectory.reshape(-1, 2)
        # If trajectory has more than 2 columns, keep first two as spatial
        if trajectory.ndim == 2 and trajectory.shape[1] > 2:
            trajectory = trajectory[:, :2]

        # Build absolute time from diffs if available, else fallback to linspace
        if time_sequence is None or not isinstance(time_sequence, np.ndarray) or time_sequence.size == 0:
            time_abs = np.linspace(0, 1000.0, trajectory.shape[0])
        else:
            # If provided diffs length mismatches, pad/trim
            diffs = time_sequence.astype(float).flatten()
            if diffs.size == trajectory.shape[0]:
                time_abs = diffs
            else:
                # If diffs are per-step (N-1), expand to N by prepending 0 and cumsum
                if diffs.size == trajectory.shape[0] - 1:
                    time_abs = np.cumsum(np.insert(diffs, 0, 0.0))
                else:
                    # Interpolate to match number of points
                    t_src = np.linspace(0, 1, max(diffs.size, 2))
                    t_dst = np.linspace(0, 1, trajectory.shape[0])
                    diffs_interp = np.interp(t_dst, t_src, diffs)
                    time_abs = np.cumsum(diffs_interp)

        # Normalize to 3D with time in seconds
        norm_traj_3d = normalize_trajectory_3d(trajectory, time_abs, target_position)
        Normalized 3D trajectory (x, y, time_in_seconds)
    """
    # Convert time from milliseconds to seconds
    time_sequence_seconds = time_sequence / 1000.0

    # Spatial normalization
    start_pos = trajectory[0]
    centered = trajectory - start_pos

    # Get current target position after centering
    current_target = centered[-1]

    # Calculate rotation angle to align with desired target
    current_angle = np.arctan2(current_target[1], current_target[0])
    desired_angle = np.arctan2(target_position[1], target_position[0])
    rotation_angle = desired_angle - current_angle

    # Create rotation matrix
    cos_theta = np.cos(rotation_angle)
    sin_theta = np.sin(rotation_angle)
    rotation_matrix = np.array([[cos_theta, -sin_theta],
                                [sin_theta, cos_theta]])

    # Rotate trajectory
    rotated = np.dot(centered, rotation_matrix.T)

    # Scale to match target length
    current_length = np.linalg.norm(rotated[-1])
    target_length = np.linalg.norm(target_position)
    scale_factor = target_length / current_length if current_length > 0 else 1

    normalized_spatial = rotated * scale_factor

    # Create 3D trajectory with time in seconds
    trajectory_3d = np.column_stack((normalized_spatial, time_sequence_seconds))

    return trajectory_3d


def normalize_trajectory_sequence_3d(path, time_diff_ms, target_position=np.array([1.0, 0.0]), target_length=512):
    """
    Normalize a trajectory into 3D (x, y, time_in_seconds).

    Args:
        path: trajectory coordinates as string or numpy array
        time_diff_ms: time differences in milliseconds
        target_position: desired end position (default: [1,0])
        target_length: desired number of points after resampling (default: 100)

    Returns:
        Normalized 3D trajectory with time in seconds
    """
    # Parse input trajectory
    trajectory = np.array(eval(path) if isinstance(path, str) else path)

    # Parse time sequence
    time_sequence = np.array(eval(time_diff_ms) if isinstance(time_diff_ms, str) else time_diff_ms)

    norm_traj_3d = np.array([])

    if isinstance(trajectory, np.ndarray) and trajectory.size > 0 and not np.all(trajectory == 0):
        # Normalize to 3D with time in seconds
        norm_traj_3d = normalize_trajectory_3d(trajectory, time_sequence, target_position)

        # Optional resampling (only for spatial coordinates)
        if target_length is not None and target_length > 2:
            t = np.linspace(0, 1, target_length)
            t_original = np.linspace(0, 1, len(norm_traj_3d))

            # Resample spatial coordinates
            resampled_spatial = np.vstack([
                np.interp(t, t_original, norm_traj_3d[:, 0]),
                np.interp(t, t_original, norm_traj_3d[:, 1])
            ]).T

            # Resample time to maintain correspondence
            resampled_time = np.interp(t, t_original, norm_traj_3d[:, 2])

            norm_traj_3d = np.column_stack((resampled_spatial, resampled_time))

    return norm_traj_3d


def normalize_trajectory_sequence_3d_directionality(path, time_diff_ms, target_position=np.array([1.0, 0.0]), target_length=512, return_dict=True):
    """Normalize trajectory while preserving and returning original directional information.

    Args:
        path: trajectory coordinates as string or numpy array of shape (N,2) or (N,>=2)
        time_diff_ms: time differences in milliseconds (string or array)
        target_position: canonical end position after rotation/scale (default [1,0])
        target_length: optional resampled length (default 512)
        return_dict: if True returns a dict with metadata; else returns tuple

    Returns:
        dict or tuple containing:
            normalized_trajectory: (L,3) array after centering, rotation, scaling, time (seconds)
            original_target_angle: angle (radians) start->original end before normalization
            rotation_angle: applied rotation (desired_angle - original_target_angle)
            original_end_vector: raw end-start vector before normalization (length and direction)
    """
    trajectory = np.array(eval(path) if isinstance(path, str) else path)
    time_sequence = np.array(eval(time_diff_ms) if isinstance(time_diff_ms, str) else time_diff_ms)

    if not (isinstance(trajectory, np.ndarray) and trajectory.size > 0 and not np.all(trajectory == 0)):
        empty = np.array([])
        if return_dict:
            return {
                'normalized_trajectory': empty,
                'original_target_angle': 0.0,
                'rotation_angle': 0.0,
                'original_end_vector': np.array([0.0, 0.0])
            }
        return empty, 0.0, 0.0, np.array([0.0, 0.0])

    start_pos = trajectory[0]
    end_pos = trajectory[-1]
    original_vec = end_pos - start_pos
    original_target_angle = float(np.arctan2(original_vec[1], original_vec[0]))
    desired_angle = float(np.arctan2(target_position[1], target_position[0]))
    rotation_angle = desired_angle - original_target_angle

    time_sequence_seconds = time_sequence / 1000.0
    centered = trajectory - start_pos
    cos_theta = np.cos(rotation_angle)
    sin_theta = np.sin(rotation_angle)
    rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    rotated = np.dot(centered, rotation_matrix.T)
    current_length = np.linalg.norm(rotated[-1])
    target_length_unit = np.linalg.norm(target_position)
    scale_factor = target_length_unit / current_length if current_length > 0 else 1.0
    normalized_spatial = rotated * scale_factor
    traj_3d = np.column_stack((normalized_spatial, time_sequence_seconds))

    if target_length is not None and target_length > 2:
        t = np.linspace(0, 1, target_length)
        t_original = np.linspace(0, 1, len(traj_3d))
        resampled_spatial = np.vstack([
            np.interp(t, t_original, traj_3d[:, 0]),
            np.interp(t, t_original, traj_3d[:, 1])
        ]).T
        resampled_time = np.interp(t, t_original, traj_3d[:, 2])
        traj_3d = np.column_stack((resampled_spatial, resampled_time))

    if return_dict:
        return {
            'normalized_trajectory': traj_3d,
            'original_target_angle': original_target_angle,
            'rotation_angle': rotation_angle,
            'original_end_vector': original_vec
        }
    return traj_3d, original_target_angle, rotation_angle, original_vec