import numpy as np
import ast


def normalize_trajectory_3d(trajectory, time_sequence, target_position=np.array([1.0, 0.0])):
    """
    Normalize trajectory spatially while preserving original time.

    Args:
        trajectory: numpy array of shape (num_points, 2) containing x,y coordinates
        time_sequence: numpy array of timestamps in milliseconds for each point
        target_position: desired end position for all trajectories (default: [1,0])

    Returns:
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