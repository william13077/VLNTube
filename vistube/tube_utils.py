"""
Shared utility functions for the VLN pipeline.

Contains deduplicated functions used across multiple pipeline stages.
"""

import numpy as np
import math

# Default camera forward direction (used with rot3_from_O_to_AB)
DEFAULT_CAMERA_FORWARD = np.array([1, 0, 0])


def remove_initial_turns(actions, path_xy):
    """Remove leading turn actions from an action sequence.

    Scans the path for the first position change, then trims the action
    and path arrays so the sequence starts at the last stationary point.

    Args:
        actions: List of discrete action IDs.
        path_xy: Array of (x, y) waypoints corresponding to each action/state.

    Returns:
        Tuple of (trimmed_actions, trimmed_path, initial_angle_rad).
        initial_angle_rad is the heading (in radians) implied by the first
        actual movement, suitable for setting the agent's initial orientation.
    """
    first_p = path_xy[0]
    diff_idx = -1

    # Find the first waypoint whose position differs from the start
    for i, cur_p in enumerate(path_xy):
        if not np.array_equal(first_p, cur_p):
            diff_idx = i
            break

    # If every waypoint is identical (turns only, no forward), return as-is
    if diff_idx == -1:
        return actions, path_xy, 0.0

    # Compute the heading angle from the last stationary point to the first moved point
    dx = path_xy[diff_idx][0] - path_xy[diff_idx - 1][0]
    dy = path_xy[diff_idx][1] - path_xy[diff_idx - 1][1]
    initial_angle = math.atan2(dy, dx)

    # Keep from diff_idx-1 onward (preserves the starting position)
    return actions[diff_idx - 1:], path_xy[diff_idx - 1:], initial_angle


def extract_object_type_outer(full_id):
    """Extract the object type name from a full instance ID.

    Splits on underscores and keeps only the leading non-numeric parts.
    For example, ``"table_lamp_0003"`` -> ``"table_lamp"``.

    Args:
        full_id: Full object instance ID string (e.g. ``"chair_0012/mesh"``).

    Returns:
        The object type prefix with numeric suffixes removed.
    """
    parts = full_id.split('_')
    non_numeric_parts = []
    for part in parts:
        if part.isdigit():
            break  # Stop at first numeric part
        non_numeric_parts.append(part)
    return '_'.join(non_numeric_parts) if non_numeric_parts else full_id


def rot3_from_O_to_AB(O, A, B):
    """Compute a 3x3 rotation matrix that rotates direction O to point from A toward B.

    Uses the Rodrigues rotation formula. Returns the identity matrix when
    A and B coincide or when O already points in the target direction.

    Args:
        O: Reference forward direction (3-vector, will be normalized).
        A: Origin point (3-vector).
        B: Target point (3-vector).

    Returns:
        A 3x3 numpy rotation matrix.
    """
    v = O / np.linalg.norm(O)
    vec_ab = B - A
    if np.linalg.norm(vec_ab) < 1e-6:
        return np.eye(3)
    w = vec_ab / np.linalg.norm(vec_ab)
    d = np.dot(v, w)

    if np.isclose(d, 1.0):
        return np.eye(3)
    if np.isclose(d, -1.0):
        # 180-degree rotation: pick an arbitrary perpendicular axis
        axis = np.cross(v, [1, 0, 0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(v, [0, 1, 0])
        axis /= np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        return np.eye(3) + 2 * (K @ K)

    # Standard Rodrigues rotation
    k = np.cross(v, w)
    k /= np.linalg.norm(k)
    theta = np.arccos(np.clip(d, -1.0, 1.0))
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
