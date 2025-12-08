#!/usr/bin/env python3
"""
SERL Dataset Preprocessing for EE Control (Memory-Optimized)
=============================================================

Applies preprocessing transformations to SERL pickle datasets.
Takes raw SERL pickle (from hdf5_to_serl_pkl_ee.py) and applies normalizations.

OPTIMIZED: Modifies arrays in-place to avoid memory explosion with large datasets.

=== EE CONTROL FORMATS (Variable Dimensionality) ===

Different rotation representations result in different action/state dimensions:

  angle_axis (14D): [L_pos(3), L_rot(3), L_grip(1), R_pos(3), R_rot(3), R_grip(1)]
  euler      (14D): [L_pos(3), L_rot(3), L_grip(1), R_pos(3), R_rot(3), R_grip(1)]
  quat       (16D): [L_pos(3), L_quat(4), L_grip(1), R_pos(3), R_quat(4), R_grip(1)]
  ortho6d    (20D): [L_pos(3), L_o6d(6), L_grip(1), R_pos(3), R_o6d(6), R_grip(1)]

The script auto-detects the format from the action dimension and uses correct indices.

=== AVAILABLE PREPROCESSING ===

1. Gripper Normalization (--normalize_gripper):
   - Raw gripper values: 0.0 (closed) to ~0.044 (open)
   - Normalized to: -1.0 (closed) to +1.0 (open)
   - Applied to gripper indices (auto-detected based on rotation mode)

2. Action Shifting (--shift_actions N):
   - Shifts actions forward by N timesteps
   - At time t, action becomes action[t+N]
   - Compensates for tracking delay in teleoperation

3. Filter Static Transitions (--filter_static THRESHOLD):
   - Removes transitions where robot barely moved
   - Based on EE position/rotation change magnitude

4. Statistics Analysis (--stats):
   - Shows per-dimension min/max/mean/std
   - Helps identify normalization issues

=== USAGE ===

    # Normalize gripper values (auto-detects rotation mode)
    python -m trossen_arm_mujoco.dataset_utils.serl_ds_preprocessing \\
        ./serl_demos_raw.pkl ./serl_demos_normalized.pkl --normalize_gripper

    # Shift actions by 3 steps (compensate for tracking lag)
    python -m trossen_arm_mujoco.dataset_utils.serl_ds_preprocessing \\
        ./serl_demos.pkl ./serl_demos_shifted.pkl --shift_actions 3

    # Filter out static steps (threshold on state change)
    python -m trossen_arm_mujoco.dataset_utils.serl_ds_preprocessing \\
        ./serl_demos.pkl ./serl_demos_cleaned.pkl --filter_static 0.01

    # Full pipeline: normalize + shift + filter
    python -m trossen_arm_mujoco.dataset_utils.serl_ds_preprocessing \\
        ./serl_demos.pkl ./serl_demos_processed.pkl \\
        --normalize_gripper --shift_actions 3 --filter_static 0.01

    # Show stats only (dry run)
    python -m trossen_arm_mujoco.dataset_utils.serl_ds_preprocessing \\
        ./serl_demos.pkl ./output.pkl --stats
"""

import pickle
import numpy as np
import argparse
from pathlib import Path
import logging
import gc

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# === GRIPPER CONSTANTS ===
# Physical gripper limits (in meters)
GRIPPER_MIN = 0.0    # Fully closed
GRIPPER_MAX = 0.044   # Fully open (approximate)


# === ROTATION MODE CONFIGURATIONS ===
# Maps rotation mode -> (rotation_dim, total_action_dim)
# Format per arm: [pos(3), rot(N), grip(1)] x 2 arms
ROT_MODE_CONFIG = {
    'angle_axis': {'rot_dim': 3, 'total_dim': 14},
    'euler':      {'rot_dim': 3, 'total_dim': 14},
    'quat':       {'rot_dim': 4, 'total_dim': 16},
    'ortho6d':    {'rot_dim': 6, 'total_dim': 20},
}


def detect_rotation_mode(action_dim: int) -> str:
    """
    Auto-detect rotation mode from action dimension.
    
    Args:
        action_dim: Dimension of action vector
        
    Returns:
        Rotation mode string: 'angle_axis', 'euler', 'quat', or 'ortho6d'
    """
    for mode, config in ROT_MODE_CONFIG.items():
        if config['total_dim'] == action_dim:
            return mode
    
    raise ValueError(f"Unknown action dimension {action_dim}. "
                     f"Expected one of {[c['total_dim'] for c in ROT_MODE_CONFIG.values()]}")


def get_indices_for_mode(rot_mode: str) -> dict:
    """
    Get index mappings for a specific rotation mode.
    
    Returns dict with:
        - left_gripper_idx: int
        - right_gripper_idx: int
        - left_pos_idx: list
        - left_rot_idx: list
        - right_pos_idx: list
        - right_rot_idx: list
        - dim_labels: list of dimension names
    """
    config = ROT_MODE_CONFIG[rot_mode]
    rot_dim = config['rot_dim']
    
    # Per-arm layout: [pos(3), rot(N), grip(1)]
    arm_dim = 3 + rot_dim + 1  # pos + rot + grip
    
    # Left arm indices
    left_pos_idx = [0, 1, 2]
    left_rot_idx = list(range(3, 3 + rot_dim))
    left_gripper_idx = 3 + rot_dim
    
    # Right arm indices (starts at arm_dim)
    right_pos_idx = [arm_dim + i for i in range(3)]
    right_rot_idx = [arm_dim + 3 + i for i in range(rot_dim)]
    right_gripper_idx = arm_dim + 3 + rot_dim
    
    # Generate dimension labels
    dim_labels = []
    # Left arm
    dim_labels.extend(['L_pos_x', 'L_pos_y', 'L_pos_z'])
    for i in range(rot_dim):
        dim_labels.append(f'L_rot_{i}')
    dim_labels.append('L_grip')
    # Right arm
    dim_labels.extend(['R_pos_x', 'R_pos_y', 'R_pos_z'])
    for i in range(rot_dim):
        dim_labels.append(f'R_rot_{i}')
    dim_labels.append('R_grip')
    
    return {
        'left_gripper_idx': left_gripper_idx,
        'right_gripper_idx': right_gripper_idx,
        'left_pos_idx': left_pos_idx,
        'left_rot_idx': left_rot_idx,
        'right_pos_idx': right_pos_idx,
        'right_rot_idx': right_rot_idx,
        'dim_labels': dim_labels,
        'rot_dim': rot_dim,
        'arm_dim': arm_dim,
    }


# === LEGACY COMPATIBILITY (for 14D angle_axis/euler) ===
# These are kept for backward compatibility but should use get_indices_for_mode() instead
LEFT_GRIPPER_IDX = 6
RIGHT_GRIPPER_IDX = 13

# Dimension labels for analysis (14D default - use get_indices_for_mode for dynamic)
DIM_LABELS = [
    'L_pos_x', 'L_pos_y', 'L_pos_z',  # 0-2
    'L_rot_x', 'L_rot_y', 'L_rot_z',  # 3-5
    'L_grip',                          # 6
    'R_pos_x', 'R_pos_y', 'R_pos_z',  # 7-9
    'R_rot_x', 'R_rot_y', 'R_rot_z',  # 10-12
    'R_grip',                          # 13
]

# Index groups for filtering (14D default)
LEFT_POS_IDX = [0, 1, 2]
LEFT_ROT_IDX = [3, 4, 5]
RIGHT_POS_IDX = [7, 8, 9]
RIGHT_ROT_IDX = [10, 11, 12]


def normalize_gripper(value: np.ndarray) -> np.ndarray:
    """
    Normalize gripper value from [0, 0.044] to [-1, +1].
    
    Handles noisy sensor readings by:
    1. FIRST clipping to physical limits [0, 0.044] to remove noise
    2. THEN normalizing to [-1, +1]
    
    This ensures normalized values are always exactly in [-1, +1].
    
    Args:
        value: Gripper value(s), may be slightly outside [0, 0.044] due to noise
        
    Returns:
        Normalized value(s) in [-1, +1]
    """
    # Step 1: Clip to physical limits first (removes sensor noise)
    clipped = np.clip(value, GRIPPER_MIN, GRIPPER_MAX)
    
    # Step 2: Normalize [0, 0.044] -> [-1, +1]
    # normalized = 2 * (value - min) / (max - min) - 1
    normalized = 2.0 * (clipped - GRIPPER_MIN) / (GRIPPER_MAX - GRIPPER_MIN) - 1.0
    
    return normalized


def denormalize_gripper(value: np.ndarray) -> np.ndarray:
    """
    Denormalize gripper value from [-1, +1] back to [0, 0.044].
    
    Handles policy outputs that may be slightly outside [-1, +1] by:
    1. FIRST clipping to [-1, +1]
    2. THEN denormalizing to [0, 0.044]
    
    Args:
        value: Normalized gripper value(s), may be slightly outside [-1, +1]
        
    Returns:
        Raw value(s) in [0, 0.044]
    """
    # Step 1: Clip normalized values first
    clipped = np.clip(value, -1.0, 1.0)
    
    # Step 2: Denormalize [-1, +1] -> [0, 0.044]
    # raw = (normalized + 1) / 2 * (max - min) + min
    raw = (clipped + 1.0) / 2.0 * (GRIPPER_MAX - GRIPPER_MIN) + GRIPPER_MIN
    
    return raw


def normalize_gripper_in_vector(vec: np.ndarray, indices: dict = None) -> np.ndarray:
    """
    Normalize gripper values in an action/state vector (IN-PLACE).
    
    Args:
        vec: Shape (D,) or (1, D) - action or state vector (D depends on rotation mode)
        indices: Dict from get_indices_for_mode() with gripper indices.
                 If None, auto-detects from vector dimension.
        
    Returns:
        Same vector with gripper values normalized (modified in-place)
    """
    # Auto-detect indices if not provided
    if indices is None:
        vec_dim = vec.shape[-1]  # Handle both (D,) and (1, D)
        rot_mode = detect_rotation_mode(vec_dim)
        indices = get_indices_for_mode(rot_mode)
    
    left_idx = indices['left_gripper_idx']
    right_idx = indices['right_gripper_idx']
    
    if vec.ndim == 1:
        vec[left_idx] = normalize_gripper(vec[left_idx])
        vec[right_idx] = normalize_gripper(vec[right_idx])
    elif vec.ndim == 2:
        vec[:, left_idx] = normalize_gripper(vec[:, left_idx])
        vec[:, right_idx] = normalize_gripper(vec[:, right_idx])
    return vec


def preprocess_transitions_inplace(transitions: list, normalize_gripper_flag: bool = True,
                                    indices: dict = None) -> None:
    """
    Apply preprocessing to SERL transitions IN-PLACE (memory efficient).
    
    Modifies the transitions list directly without creating copies.
    This is critical for large datasets (30k+ transitions with images).
    
    Args:
        transitions: List of SERL transition dicts (modified in-place)
        normalize_gripper_flag: Whether to normalize gripper values
        indices: Dict from get_indices_for_mode() (auto-detected if None)
    """
    n = len(transitions)
    
    # Auto-detect indices from first transition if not provided
    if indices is None and n > 0:
        action_dim = transitions[0]['actions'].shape[-1]
        rot_mode = detect_rotation_mode(action_dim)
        indices = get_indices_for_mode(rot_mode)
        logger.info(f"  Auto-detected rotation mode: {rot_mode} (action_dim={action_dim})")
    
    for i, t in enumerate(transitions):
        if i % 5000 == 0:
            logger.info(f"  Processing {i}/{n} transitions...")
            gc.collect()  # Help memory management
        
        if normalize_gripper_flag:
            # Normalize action gripper values - in-place
            normalize_gripper_in_vector(t['actions'], indices)
            
            # Normalize state gripper values in observations
            if 'state' in t['observations']:
                normalize_gripper_in_vector(t['observations']['state'], indices)
            
            # Normalize state in next_observations
            if 'state' in t['next_observations']:
                normalize_gripper_in_vector(t['next_observations']['state'], indices)
    
    logger.info(f"  Processed {n}/{n} transitions.")


def analyze_dataset_fast(transitions: list, sample_size: int = 1000) -> dict:
    """
    Analyze gripper value statistics in dataset (memory-efficient sampling).
    
    For large datasets, samples transitions to avoid memory issues.
    Auto-detects rotation mode from action dimension.
    """
    n = len(transitions)
    
    if n == 0:
        return {'num_transitions': 0, 'sampled': 0}
    
    # Auto-detect rotation mode
    action_dim = transitions[0]['actions'].shape[-1]
    rot_mode = detect_rotation_mode(action_dim)
    idx_config = get_indices_for_mode(rot_mode)
    left_grip_idx = idx_config['left_gripper_idx']
    right_grip_idx = idx_config['right_gripper_idx']
    
    # Sample indices if dataset is large
    if n > sample_size:
        sample_indices = np.linspace(0, n-1, sample_size, dtype=int)
    else:
        sample_indices = range(n)
    
    # Collect just the gripper values (not full arrays)
    action_left = []
    action_right = []
    state_left = []
    state_right = []
    
    for i in sample_indices:
        t = transitions[i]
        action_left.append(t['actions'][left_grip_idx])
        action_right.append(t['actions'][right_grip_idx])
        
        state = t['observations']['state']
        if state.ndim == 2:
            state_left.append(state[0, left_grip_idx])
            state_right.append(state[0, right_grip_idx])
        else:
            state_left.append(state[left_grip_idx])
            state_right.append(state[right_grip_idx])
    
    return {
        'num_transitions': n,
        'sampled': len(sample_indices),
        'rot_mode': rot_mode,
        'action_dim': action_dim,
        'action_left_gripper': {
            'min': float(np.min(action_left)),
            'max': float(np.max(action_left)),
            'mean': float(np.mean(action_left)),
        },
        'action_right_gripper': {
            'min': float(np.min(action_right)),
            'max': float(np.max(action_right)),
            'mean': float(np.mean(action_right)),
        },
        'state_left_gripper': {
            'min': float(np.min(state_left)),
            'max': float(np.max(state_left)),
            'mean': float(np.mean(state_left)),
        },
        'state_right_gripper': {
            'min': float(np.min(state_right)),
            'max': float(np.max(state_right)),
            'mean': float(np.mean(state_right)),
        },
    }


def get_episode_boundaries(transitions: list) -> list:
    """
    Find episode boundaries based on 'dones' flag.
    
    Returns list of (start_idx, end_idx) tuples for each episode.
    """
    boundaries = []
    start = 0
    
    for i, t in enumerate(transitions):
        if t['dones']:
            boundaries.append((start, i + 1))  # end is exclusive
            start = i + 1
    
    # Handle case where last transition isn't marked done
    if start < len(transitions):
        boundaries.append((start, len(transitions)))
    
    return boundaries


def process_episode_streaming(input_pkl: Path, output_pkl: Path, 
                               normalize_gripper_flag: bool = True) -> dict:
    """
    Process dataset episode-by-episode to minimize memory usage.
    
    Loads full dataset once, processes in chunks by episode, 
    and saves incrementally.
    
    Returns stats dict.
    """
    # Load dataset
    logger.info(f"Loading: {input_pkl}")
    with open(input_pkl, 'rb') as f:
        transitions = pickle.load(f)
    
    n_total = len(transitions)
    logger.info(f"Loaded {n_total} transitions")
    
    # Find episode boundaries
    boundaries = get_episode_boundaries(transitions)
    n_episodes = len(boundaries)
    logger.info(f"Found {n_episodes} episodes")
    
    # Collect stats before (sample-based for memory)
    logger.info("\n=== BEFORE PREPROCESSING (sampled) ===")
    before_stats = analyze_dataset_fast(transitions)
    _print_stats(before_stats)
    
    # Process episode by episode (in-place)
    logger.info("\n=== PROCESSING EPISODES ===")
    
    for ep_idx, (start, end) in enumerate(boundaries):
        if ep_idx % 10 == 0:
            logger.info(f"  Episode {ep_idx + 1}/{n_episodes} (transitions {start}-{end})")
        
        # Process this episode's transitions in-place
        for i in range(start, end):
            t = transitions[i]
            
            if normalize_gripper_flag:
                # Normalize action gripper values (in-place)
                normalize_gripper_in_vector(t['actions'])
                
                # Normalize state in observations
                if 'state' in t['observations']:
                    normalize_gripper_in_vector(t['observations']['state'])
                
                # Normalize state in next_observations
                if 'state' in t['next_observations']:
                    normalize_gripper_in_vector(t['next_observations']['state'])
        
        # Periodic garbage collection
        if ep_idx % 20 == 0:
            gc.collect()
    
    logger.info(f"  Processed all {n_episodes} episodes")
    
    # Collect stats after
    logger.info("\n=== AFTER PREPROCESSING (sampled) ===")
    after_stats = analyze_dataset_fast(transitions)
    _print_stats(after_stats)
    
    # Save
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"\nSaving to: {output_pkl}")
    with open(output_pkl, 'wb') as f:
        pickle.dump(transitions, f)
    
    logger.info(f"✓ Saved {n_total} transitions")
    
    return {'before': before_stats, 'after': after_stats, 'n_episodes': n_episodes}


def _print_stats(stats: dict):
    """Print gripper stats."""
    logger.info(f"Action Left Gripper:  min={stats['action_left_gripper']['min']:.4f}, "
                f"max={stats['action_left_gripper']['max']:.4f}, "
                f"mean={stats['action_left_gripper']['mean']:.4f}")
    logger.info(f"Action Right Gripper: min={stats['action_right_gripper']['min']:.4f}, "
                f"max={stats['action_right_gripper']['max']:.4f}, "
                f"mean={stats['action_right_gripper']['mean']:.4f}")
    logger.info(f"State Left Gripper:   min={stats['state_left_gripper']['min']:.4f}, "
                f"max={stats['state_left_gripper']['max']:.4f}, "
                f"mean={stats['state_left_gripper']['mean']:.4f}")
    logger.info(f"State Right Gripper:  min={stats['state_right_gripper']['min']:.4f}, "
                f"max={stats['state_right_gripper']['max']:.4f}, "
                f"mean={stats['state_right_gripper']['mean']:.4f}")


# ============================================================================
# COMPREHENSIVE STATISTICS
# ============================================================================

def compute_full_stats(transitions: list, sample_size: int = 5000) -> dict:
    """
    Compute comprehensive per-dimension statistics for actions and states.
    
    Auto-detects rotation mode and returns stats with appropriate dimension labels.
    """
    n = len(transitions)
    
    if n == 0:
        return {'n_transitions': 0, 'n_sampled': 0, 'action': {}, 'state': {}}
    
    # Auto-detect rotation mode
    action_dim = transitions[0]['actions'].shape[-1]
    rot_mode = detect_rotation_mode(action_dim)
    idx_config = get_indices_for_mode(rot_mode)
    dim_labels = idx_config['dim_labels']
    
    # Sample if large
    if n > sample_size:
        sample_indices = np.random.choice(n, sample_size, replace=False)
    else:
        sample_indices = range(n)
    
    actions = np.array([transitions[i]['actions'] for i in sample_indices])
    
    states = []
    for i in sample_indices:
        s = transitions[i]['observations']['state']
        if s.ndim == 2:
            s = s[0]
        states.append(s)
    states = np.array(states)
    
    stats = {
        'n_transitions': n,
        'n_sampled': len(sample_indices),
        'rot_mode': rot_mode,
        'action_dim': action_dim,
        'dim_labels': dim_labels,
        'action': {},
        'state': {},
    }
    
    for dim_idx, label in enumerate(dim_labels):
        stats['action'][label] = {
            'min': float(actions[:, dim_idx].min()),
            'max': float(actions[:, dim_idx].max()),
            'mean': float(actions[:, dim_idx].mean()),
            'std': float(actions[:, dim_idx].std()),
        }
        stats['state'][label] = {
            'min': float(states[:, dim_idx].min()),
            'max': float(states[:, dim_idx].max()),
            'mean': float(states[:, dim_idx].mean()),
            'std': float(states[:, dim_idx].std()),
        }
    
    return stats


def print_full_stats(stats: dict):
    """Pretty print comprehensive stats."""
    rot_mode = stats.get('rot_mode', 'unknown')
    action_dim = stats.get('action_dim', '?')
    dim_labels = stats.get('dim_labels', DIM_LABELS)  # Fallback to default for old stats dicts
    
    print("\n" + "=" * 80)
    print(f"DATASET STATISTICS ({stats['n_sampled']}/{stats['n_transitions']} transitions sampled)")
    print(f"Rotation Mode: {rot_mode} | Action Dim: {action_dim}")
    print("=" * 80)
    
    print("\n" + "-" * 80)
    print("ACTION STATISTICS")
    print("-" * 80)
    print(f"{'Dimension':<12} {'Min':>12} {'Max':>12} {'Mean':>12} {'Std':>12}")
    print("-" * 80)
    
    for label in dim_labels:
        s = stats['action'][label]
        print(f"{label:<12} {s['min']:>12.5f} {s['max']:>12.5f} {s['mean']:>12.5f} {s['std']:>12.5f}")
    
    print("\n" + "-" * 80)
    print("STATE STATISTICS")
    print("-" * 80)
    print(f"{'Dimension':<12} {'Min':>12} {'Max':>12} {'Mean':>12} {'Std':>12}")
    print("-" * 80)
    
    for label in dim_labels:
        s = stats['state'][label]
        print(f"{label:<12} {s['min']:>12.5f} {s['max']:>12.5f} {s['mean']:>12.5f} {s['std']:>12.5f}")
    
    # Highlight potential issues
    print("\n" + "-" * 80)
    print("POTENTIAL ISSUES")
    print("-" * 80)
    
    issues_found = False
    
    # Check for static dimensions
    for label in dim_labels:
        if stats['action'][label]['std'] < 0.001:
            print(f"⚠️  Action {label} has very low variance (std={stats['action'][label]['std']:.6f}) - likely static")
            issues_found = True
    
    # Check for out-of-range values
    for label in dim_labels:
        a = stats['action'][label]
        if a['max'] > 1.0 or a['min'] < -1.0:
            print(f"⚠️  Action {label} outside [-1, 1]: [{a['min']:.4f}, {a['max']:.4f}]")
            issues_found = True
    
    # Check gripper normalization
    for grip_label in ['L_grip', 'R_grip']:
        if grip_label not in stats['action']:
            continue
        a = stats['action'][grip_label]
        s = stats['state'][grip_label]
        
        # If raw gripper values (0 to 0.044)
        if a['max'] < 0.1 and a['min'] >= -0.01:
            print(f"⚠️  Action {grip_label} appears unnormalized (range [{a['min']:.4f}, {a['max']:.4f}])")
            issues_found = True
        if s['max'] < 0.1 and s['min'] >= -0.01:
            print(f"⚠️  State {grip_label} appears unnormalized (range [{s['min']:.4f}, {s['max']:.4f}])")
            issues_found = True
    
    if not issues_found:
        print("✅ No obvious issues detected")


# ============================================================================
# ACTION SHIFTING (Compensate for tracking delay)
# ============================================================================

def shift_actions_inplace(transitions: list, k_shift: int) -> int:
    """
    Shift actions forward by k steps within each episode.
    
    At time t, the action becomes action[t + k].
    This compensates for teleop tracking delay.
    
    Args:
        transitions: List of SERL transitions (modified in-place)
        k_shift: Number of steps to shift forward
        
    Returns:
        Number of episodes processed
    """
    logger.info(f"⏩ Shifting actions by {k_shift} steps...")
    
    boundaries = get_episode_boundaries(transitions)
    n_episodes = len(boundaries)
    
    for ep_idx, (start, end) in enumerate(boundaries):
        ep_len = end - start
        
        # Collect actions for this episode
        ep_actions = [transitions[i]['actions'].copy() for i in range(start, end)]
        
        # Shift: action[t] = action[t + k]
        for t in range(ep_len):
            future_t = min(t + k_shift, ep_len - 1)
            transitions[start + t]['actions'] = ep_actions[future_t]
    
    logger.info(f"   Shifted actions in {n_episodes} episodes")
    return n_episodes


# ============================================================================
# FILTER STATIC TRANSITIONS
# ============================================================================

# def filter_static_transitions(transitions: list, threshold: float = 0.01) -> list:
#     """
#     Remove transitions where the robot barely moved.
    
#     Computes magnitude of state change between consecutive steps.
#     Keeps only steps where change exceeds threshold.
    
#     NOTE: This creates a NEW list (not in-place) because it changes the length.
    
#     Args:
#         transitions: List of SERL transitions
#         threshold: Minimum state change magnitude to keep
        
#     Returns:
#         New list with static transitions removed
#     """
#     logger.info(f"🧹 Filtering static transitions (threshold={threshold})...")
    
#     boundaries = get_episode_boundaries(transitions)
#     new_transitions = []
#     total_kept = 0
#     total_dropped = 0
    
#     for ep_idx, (start, end) in enumerate(boundaries):
#         ep_len = end - start
#         if ep_len < 2:
#             continue
        
#         # Always keep first step
#         keep_indices = [start]
        
#         # Get previous state
#         prev_state = transitions[start]['observations']['state']
#         if prev_state.ndim == 2:
#             prev_state = prev_state[0]
        
#         for i in range(start + 1, end):
#             curr_state = transitions[i]['observations']['state']
#             if curr_state.ndim == 2:
#                 curr_state = curr_state[0]
            
#             # Compute change magnitude (exclude grippers for this check)
#             # Use position + rotation dims only
#             pos_rot_dims = LEFT_POS_IDX + LEFT_ROT_IDX + RIGHT_POS_IDX + RIGHT_ROT_IDX
#             diff = np.linalg.norm(curr_state[pos_rot_dims] - prev_state[pos_rot_dims])
            
#             if diff > threshold:
#                 keep_indices.append(i)
#                 prev_state = curr_state
        
#         # Always keep last step (for done signal)
#         if end - 1 not in keep_indices:
#             keep_indices.append(end - 1)
        
#         # Copy kept transitions
#         for idx in keep_indices:
#             new_transitions.append(transitions[idx])
        
#         # Update done flag for new last step
#         if new_transitions:
#             new_transitions[-1]['dones'] = True
        
#         total_kept += len(keep_indices)
#         total_dropped += (ep_len - len(keep_indices))
    
#     logger.info(f"   Kept {total_kept} transitions, dropped {total_dropped} static ones")
#     return new_transitions
def filter_static_transitions(transitions: list, threshold: float = 0.002) -> list:
    """
    Remove transitions where the robot barely moved, BUT preserve gripper actions
    and prevent long gaps in data.
    
    Args:
        transitions: List of SERL transitions
        threshold: Minimum arm movement magnitude to keep (default 2mm)
        
    Returns:
        New list with static transitions removed
    """
    logger.info(f"🧹 Filtering static transitions (arm_thresh={threshold})...")
    
    if len(transitions) == 0:
        return []
    
    # Auto-detect rotation mode from first transition
    action_dim = transitions[0]['actions'].shape[-1]
    rot_mode = detect_rotation_mode(action_dim)
    idx_config = get_indices_for_mode(rot_mode)
    logger.info(f"   Detected rotation mode: {rot_mode} (action_dim={action_dim})")
    
    # Get indices for this rotation mode
    left_pos_idx = idx_config['left_pos_idx']
    left_rot_idx = idx_config['left_rot_idx']
    right_pos_idx = idx_config['right_pos_idx']
    right_rot_idx = idx_config['right_rot_idx']
    left_grip_idx = idx_config['left_gripper_idx']
    right_grip_idx = idx_config['right_gripper_idx']
    pos_rot_dims = left_pos_idx + left_rot_idx + right_pos_idx + right_rot_idx
    
    boundaries = get_episode_boundaries(transitions)
    new_transitions = []
    total_kept = 0
    total_dropped = 0
    
    # 1. LOWER THRESHOLD: Sim grippers move very slowly (tiny values). 
    # 0.0005 ensures we catch the start of a squeeze.
    GRIPPER_THRESHOLD = 0.0005 
    
    # 2. MAX SKIP: Force a frame save every N steps even if static.
    # At 50Hz, 15 steps = 0.3 seconds. This preserves "holding" actions.
    MAX_SKIP = 15 
    
    for ep_idx, (start, end) in enumerate(boundaries):
        ep_len = end - start
        if ep_len < 2:
            continue
        
        # Always keep first step
        keep_indices = [start]
        
        # Get previous state (Obs of the last KEPT frame)
        prev_state = transitions[start]['observations']['state']
        if prev_state.ndim == 2:
            prev_state = prev_state[0]
            
        steps_since_last_keep = 0 # Counter for Max Skip
        
        for i in range(start + 1, end):
            curr_state = transitions[i]['observations']['state']
            if curr_state.ndim == 2:
                curr_state = curr_state[0]
            
            # --- Check Arm Movement (Position + Rotation) ---
            diff_arm = np.linalg.norm(curr_state[pos_rot_dims] - prev_state[pos_rot_dims])
            
            # --- Check Gripper Movement ---
            diff_l_grip = abs(curr_state[left_grip_idx] - prev_state[left_grip_idx])
            diff_r_grip = abs(curr_state[right_grip_idx] - prev_state[right_grip_idx])
            
            # --- DECISION LOGIC ---
            # Keep if:
            # 1. Arm moved significantly
            # 2. OR Gripper moved (even slightly)
            # 3. OR We haven't saved a frame in a while (MAX_SKIP reached)
            
            is_moving = (diff_arm > threshold) or \
                        (diff_l_grip > GRIPPER_THRESHOLD) or \
                        (diff_r_grip > GRIPPER_THRESHOLD)
            
            if is_moving or (steps_since_last_keep >= MAX_SKIP):
                keep_indices.append(i)
                prev_state = curr_state  # Update prev_state ONLY when we keep a frame
                steps_since_last_keep = 0
            else:
                # Drop this frame
                # Do NOT update prev_state (this allows movement to accumulate)
                steps_since_last_keep += 1
        
        # Always keep last step (for done signal)
        if end - 1 not in keep_indices:
            keep_indices.append(end - 1)
        
        # Copy kept transitions
        for idx in keep_indices:
            new_transitions.append(transitions[idx])
        
        # Update done flag for new last step
        if new_transitions:
            new_transitions[-1]['dones'] = True
        
        total_kept += len(keep_indices)
        total_dropped += (ep_len - len(keep_indices))
    
    logger.info(f"   Kept {total_kept} transitions, dropped {total_dropped} static ones")
    return new_transitions

# ============================================================================
# FILTER BAD EPISODES (Gripper never changes state)
# ============================================================================

def analyze_episode_gripper_activity(transitions: list) -> list:
    """
    Analyze each episode's gripper activity.
    
    Returns list of dicts with episode info:
    - ep_idx: episode index
    - start, end: transition indices
    - r_grip_min, r_grip_max, r_grip_range: right gripper stats
    - l_grip_min, l_grip_max, l_grip_range: left gripper stats
    - has_grip_change: bool - whether gripper state changed significantly
    """
    if len(transitions) == 0:
        return []
    
    # Auto-detect rotation mode
    action_dim = transitions[0]['actions'].shape[-1]
    rot_mode = detect_rotation_mode(action_dim)
    idx_config = get_indices_for_mode(rot_mode)
    left_grip_idx = idx_config['left_gripper_idx']
    right_grip_idx = idx_config['right_gripper_idx']
    
    boundaries = get_episode_boundaries(transitions)
    results = []
    
    for ep_idx, (start, end) in enumerate(boundaries):
        # Collect gripper values for this episode
        r_grips = []
        l_grips = []
        
        for i in range(start, end):
            action = transitions[i]['actions']
            l_grips.append(action[left_grip_idx])
            r_grips.append(action[right_grip_idx])
        
        r_grips = np.array(r_grips)
        l_grips = np.array(l_grips)
        
        r_range = r_grips.max() - r_grips.min()
        l_range = l_grips.max() - l_grips.min()
        
        # For cube stacking, right arm does the work
        # Gripper should open and close (range > threshold)
        # If normalized: range should be > 0.5 (from -1 to +1)
        # If raw: range should be > 0.02 (from 0 to 0.044)
        
        results.append({
            'ep_idx': ep_idx,
            'start': start,
            'end': end,
            'length': end - start,
            'r_grip_min': float(r_grips.min()),
            'r_grip_max': float(r_grips.max()),
            'r_grip_range': float(r_range),
            'l_grip_min': float(l_grips.min()),
            'l_grip_max': float(l_grips.max()),
            'l_grip_range': float(l_range),
        })
    
    return results


def filter_bad_episodes(transitions: list, 
                        min_grip_range: float = 0.5,
                        check_arm: str = 'right') -> tuple:
    """
    Filter out episodes where gripper never changes state.
    
    For pick-and-place tasks, the gripper should open and close.
    Episodes where gripper range is too small are likely failed demos.
    
    Args:
        transitions: List of SERL transitions
        min_grip_range: Minimum gripper range to keep episode
                        (0.5 for normalized [-1,1], 0.02 for raw [0,0.044])
        check_arm: 'right', 'left', or 'both'
        
    Returns:
        (new_transitions, removed_episodes) - filtered list and list of removed ep indices
    """
    logger.info(f"🔍 Filtering episodes with insufficient gripper activity...")
    logger.info(f"   Minimum grip range: {min_grip_range}")
    logger.info(f"   Checking: {check_arm} arm")
    
    episode_stats = analyze_episode_gripper_activity(transitions)
    
    # Determine which episodes to keep
    keep_episodes = []
    remove_episodes = []
    
    for ep in episode_stats:
        keep = True
        
        if check_arm in ['right', 'both']:
            if ep['r_grip_range'] < min_grip_range:
                keep = False
        
        if check_arm in ['left', 'both']:
            if ep['l_grip_range'] < min_grip_range:
                keep = False
        
        if keep:
            keep_episodes.append(ep)
        else:
            remove_episodes.append(ep['ep_idx'])
            logger.info(f"   ❌ Episode {ep['ep_idx']}: R_grip range={ep['r_grip_range']:.3f}, "
                       f"L_grip range={ep['l_grip_range']:.3f} (too small)")
    
    if not remove_episodes:
        logger.info("   ✅ All episodes have sufficient gripper activity")
        return transitions, []
    
    # Build new transitions list from kept episodes
    new_transitions = []
    for ep in keep_episodes:
        for i in range(ep['start'], ep['end']):
            new_transitions.append(transitions[i])
    
    # Fix done flags for new episode boundaries
    boundaries = get_episode_boundaries(new_transitions)
    for start, end in boundaries:
        # Reset all dones to False
        for i in range(start, end - 1):
            new_transitions[i]['dones'] = False
        # Set last transition done to True
        new_transitions[end - 1]['dones'] = True
    
    logger.info(f"   Removed {len(remove_episodes)} episodes: {remove_episodes}")
    logger.info(f"   Kept {len(keep_episodes)} episodes, {len(new_transitions)} transitions")
    
    return new_transitions, remove_episodes


def remove_episodes_by_index(transitions: list, episode_indices: list) -> list:
    """
    Remove specific episodes by their index.
    
    Args:
        transitions: List of SERL transitions
        episode_indices: List of episode indices to remove (0-indexed)
        
    Returns:
        New transitions list with specified episodes removed
    """
    if not episode_indices:
        return transitions
    
    logger.info(f"🗑️ Removing episodes by index: {episode_indices}")
    
    boundaries = get_episode_boundaries(transitions)
    indices_to_remove = set(episode_indices)
    
    new_transitions = []
    kept_count = 0
    
    for ep_idx, (start, end) in enumerate(boundaries):
        if ep_idx not in indices_to_remove:
            for i in range(start, end):
                new_transitions.append(transitions[i])
            kept_count += 1
    
    # Fix done flags
    if new_transitions:
        boundaries = get_episode_boundaries(new_transitions)
        for start, end in boundaries:
            for i in range(start, end - 1):
                new_transitions[i]['dones'] = False
            new_transitions[end - 1]['dones'] = True
    
    logger.info(f"   Removed {len(indices_to_remove)} episodes, kept {kept_count}")
    logger.info(f"   New total: {len(new_transitions)} transitions")
    
    return new_transitions


def print_episode_gripper_summary(transitions: list):
    """
    Print a summary table of gripper activity per episode.
    Useful for identifying bad episodes manually.
    """
    stats = analyze_episode_gripper_activity(transitions)
    
    print("\n" + "=" * 90)
    print("EPISODE GRIPPER ACTIVITY SUMMARY")
    print("=" * 90)
    print(f"{'Ep':<4} {'Len':<6} {'R_min':<8} {'R_max':<8} {'R_range':<8} "
          f"{'L_min':<8} {'L_max':<8} {'L_range':<8} {'Status':<10}")
    print("-" * 90)
    
    for ep in stats:
        # Determine status based on gripper ranges
        # Assume normalized values: good range > 0.5
        r_ok = ep['r_grip_range'] > 0.5
        l_ok = ep['l_grip_range'] > 0.5 or ep['l_grip_range'] < 0.01  # Left often static
        
        status = "✅ OK" if r_ok else "❌ BAD"
        
        print(f"{ep['ep_idx']:<4} {ep['length']:<6} "
              f"{ep['r_grip_min']:<8.3f} {ep['r_grip_max']:<8.3f} {ep['r_grip_range']:<8.3f} "
              f"{ep['l_grip_min']:<8.3f} {ep['l_grip_max']:<8.3f} {ep['l_grip_range']:<8.3f} "
              f"{status:<10}")
    
    print("-" * 90)
    
    # Summary
    good_eps = sum(1 for ep in stats if ep['r_grip_range'] > 0.5)
    bad_eps = len(stats) - good_eps
    print(f"Total: {len(stats)} episodes | Good: {good_eps} | Bad (R_grip range < 0.5): {bad_eps}")
    
    if bad_eps > 0:
        bad_indices = [ep['ep_idx'] for ep in stats if ep['r_grip_range'] <= 0.5]
        print(f"Bad episode indices: {bad_indices}")


# ============================================================================
# Z-SCORE NORMALIZATION
# ============================================================================

def compute_zscore_stats(transitions: list) -> tuple:
    """
    Compute mean and std for z-score normalization.
    
    Returns:
        (mean, std) - numpy arrays of shape (14,)
    """
    actions = np.array([t['actions'] for t in transitions])
    mean = np.mean(actions, axis=0)
    std = np.std(actions, axis=0)
    std[std < 1e-6] = 1.0  # Avoid divide by zero
    return mean, std


def zscore_normalize_actions(transitions: list, mean: np.ndarray = None, 
                              std: np.ndarray = None, clip: float = 5.0) -> tuple:
    """
    Apply z-score normalization to actions (in-place).
    
    Formula: z = (action - mean) / std
    
    Args:
        transitions: List of SERL transitions (modified in-place)
        mean: Pre-computed mean (if None, computed from data)
        std: Pre-computed std (if None, computed from data)
        clip: Clip z-scores to [-clip, clip] to remove outliers
        
    Returns:
        (mean, std) used for normalization
    """
    if mean is None or std is None:
        mean, std = compute_zscore_stats(transitions)
    
    logger.info(f"📊 Z-score normalizing actions (clip={clip})...")
    
    for t in transitions:
        z = (t['actions'] - mean) / std
        if clip:
            z = np.clip(z, -clip, clip)
        t['actions'] = z
    
    return mean, std


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess SERL pickle dataset for EE control (memory-optimized)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('input_pkl', type=Path, help='Input SERL pickle file')
    parser.add_argument('output_pkl', type=Path, nargs='?', default=None,
                        help='Output preprocessed pickle file (optional for --stats)')
    
    # Preprocessing options
    parser.add_argument('--normalize_gripper', action='store_true',
                        help='Normalize gripper values from [0, 0.044] to [-1, +1]')
    parser.add_argument('--shift_actions', type=int, default=0, metavar='N',
                        help='Shift actions forward by N steps (compensate tracking lag)')
    parser.add_argument('--filter_static', type=float, default=0, metavar='THRESH',
                        help='Remove transitions with state change < THRESH')
    parser.add_argument('--zscore', action='store_true',
                        help='Apply z-score normalization to actions')
    parser.add_argument('--zscore_clip', type=float, default=5.0,
                        help='Clip z-scores to +/- this value (default: 5.0)')
    
    # Episode filtering options
    parser.add_argument('--filter_bad_episodes', action='store_true',
                        help='Remove episodes where gripper never changes state (pick-place failed)')
    parser.add_argument('--min_grip_range', type=float, default=0.5,
                        help='Minimum gripper range to keep episode (default: 0.5 for normalized)')
    parser.add_argument('--remove_episodes', type=str, default='',
                        help='Comma-separated list of episode indices to remove (e.g., "4,7,12")')
    parser.add_argument('--show_episodes', action='store_true',
                        help='Show episode gripper activity summary table')
    
    # Analysis options
    parser.add_argument('--stats', action='store_true',
                        help='Show comprehensive statistics (no output file needed)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Show what would change without saving')
    
    args = parser.parse_args()
    
    if not args.input_pkl.exists():
        logger.error(f"Input file not found: {args.input_pkl}")
        return 1
    
    # Load dataset
    logger.info(f"Loading: {args.input_pkl}")
    with open(args.input_pkl, 'rb') as f:
        transitions = pickle.load(f)
    
    n_original = len(transitions)
    logger.info(f"Loaded {n_original} transitions")
    
    boundaries = get_episode_boundaries(transitions)
    logger.info(f"Found {len(boundaries)} episodes")
    
    # Stats-only mode
    if args.stats:
        stats = compute_full_stats(transitions)
        print_full_stats(stats)
        return 0
    
    # Show episodes mode
    if args.show_episodes:
        print_episode_gripper_summary(transitions)
        return 0
    
    # Parse remove_episodes argument
    remove_ep_indices = []
    if args.remove_episodes:
        try:
            remove_ep_indices = [int(x.strip()) for x in args.remove_episodes.split(',') if x.strip()]
        except ValueError:
            logger.error(f"Invalid --remove_episodes format: {args.remove_episodes}")
            logger.error("Use comma-separated integers, e.g., '4,7,12'")
            return 1
    
    # Check if any preprocessing requested
    has_preprocessing = (args.normalize_gripper or args.shift_actions > 0 or 
                         args.filter_static > 0 or args.zscore or 
                         args.filter_bad_episodes or remove_ep_indices)
    
    if not has_preprocessing:
        logger.warning("No preprocessing options specified. Use --help to see options.")
        logger.info("Showing stats instead:")
        stats = compute_full_stats(transitions)
        print_full_stats(stats)
        return 0
    
    # Show before stats
    logger.info("\n=== BEFORE PREPROCESSING ===")
    before_stats = compute_full_stats(transitions)
    print_full_stats(before_stats)
    
    # Apply preprocessing in order
    
    # 0. Remove specific episodes first
    if remove_ep_indices:
        transitions = remove_episodes_by_index(transitions, remove_ep_indices)
    
    # 1. Filter bad episodes (gripper never changes)
    removed_bad_eps = []
    if args.filter_bad_episodes:
        transitions, removed_bad_eps = filter_bad_episodes(
            transitions, 
            min_grip_range=args.min_grip_range,
            check_arm='right'
        )
    
    # 2. Filter static transitions
    if args.filter_static > 0:
        transitions = filter_static_transitions(transitions, args.filter_static)
    
    # 3. Shift actions
    if args.shift_actions > 0:
        shift_actions_inplace(transitions, args.shift_actions)
    
    # 4. Normalize grippers
    if args.normalize_gripper:
        logger.info("🔧 Normalizing gripper values...")
        preprocess_transitions_inplace(transitions, normalize_gripper_flag=True)
    
    # 5. Z-score normalize (should be last)
    if args.zscore:
        mean, std = zscore_normalize_actions(transitions, clip=args.zscore_clip)
        logger.info(f"   Action mean: {mean}")
        logger.info(f"   Action std:  {std}")
    
    # Show after stats
    logger.info("\n=== AFTER PREPROCESSING ===")
    after_stats = compute_full_stats(transitions)
    print_full_stats(after_stats)
    
    # Get final episode count
    final_boundaries = get_episode_boundaries(transitions)
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("PREPROCESSING SUMMARY")
    logger.info("=" * 80)
    logger.info(f"  Original transitions: {n_original}")
    logger.info(f"  Original episodes:    {len(boundaries)}")
    logger.info(f"  Final transitions:    {len(transitions)}")
    logger.info(f"  Final episodes:       {len(final_boundaries)}")
    if remove_ep_indices:
        logger.info(f"  Removed episodes:     {remove_ep_indices}")
    if removed_bad_eps:
        logger.info(f"  Bad episodes removed: {removed_bad_eps}")
    if args.filter_static > 0:
        logger.info(f"  Static filter:        threshold={args.filter_static}")
    if args.shift_actions > 0:
        logger.info(f"  Action shift:         {args.shift_actions} steps")
    if args.normalize_gripper:
        logger.info(f"  Gripper normalized:   Yes")
    if args.zscore:
        logger.info(f"  Z-score normalized:   Yes (clip={args.zscore_clip})")
    
    if args.dry_run:
        logger.info("\n[DRY RUN] No file saved.")
        return 0
    
    # Save
    if args.output_pkl is None:
        logger.error("Output file required for saving. Use --dry_run to skip saving.")
        return 1
    
    args.output_pkl.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"\nSaving to: {args.output_pkl}")
    with open(args.output_pkl, 'wb') as f:
        pickle.dump(transitions, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    logger.info(f"✓ Saved {len(transitions)} transitions")
    
    return 0


if __name__ == '__main__':
    exit(main())


"""
# Show comprehensive stats
python -m trossen_arm_mujoco.dataset_utils.serl_ds_preprocessing ./input.pkl --stats

# Normalize grippers only
python -m trossen_arm_mujoco.dataset_utils.serl_ds_preprocessing \
    ./input.pkl ./output.pkl --normalize_gripper

# Shift actions (compensate tracking lag)
python -m trossen_arm_mujoco.dataset_utils.serl_ds_preprocessing \
    ./input.pkl ./output.pkl --shift_actions 3

# Filter static transitions
python -m trossen_arm_mujoco.dataset_utils.serl_ds_preprocessing \
    ./input.pkl ./output.pkl --filter_static 0.01

# Full pipeline
python -m trossen_arm_mujoco.dataset_utils.serl_ds_preprocessing \
    /home/qte9489/personal_abhi/temp/serl/trossen_arm_mujoco/sim_recordings_ee_serl/sim_dataset_ee_ortho6d.pkl /home/qte9489/personal_abhi/temp/serl/trossen_arm_mujoco/sim_recordings_ee_serl/sim_dataset_ee_ortho6d_grip_norm_shifted3_filtered0002.pkl --normalize_gripper --shift_actions 3 --filter_static 0.002

# Dry run (see changes without saving)
python -m trossen_arm_mujoco.dataset_utils.serl_ds_preprocessing \
    ./input.pkl ./output.pkl --normalize_gripper --dry_run

# Z-score normalize actions
python -m trossen_arm_mujoco.dataset_utils.serl_ds_preprocessing \
    ./input.pkl ./output.pkl --zscore --zscore_clip 5.0

# Filter bad episodes (gripper never changes)
python -m trossen_arm_mujoco.dataset_utils.serl_ds_preprocessing /home/qte9489/personal_abhi/temp/serl/trossen_arm_mujoco/sim_recordings_ee_serl/sim_dataset_ee_angleaxis_v1_grip_norm_shifted5_filtered0002.pkl --show_episodes

# Remove bad episodes by index
python -m trossen_arm_mujoco.dataset_utils.serl_ds_preprocessing \
    /home/qte9489/personal_abhi/temp/serl/trossen_arm_mujoco/sim_recordings_ee_serl/sim_dataset_ee_ortho6d_grip_norm_shifted3_filtered0002_v1.pkl \
    /home/qte9489/personal_abhi/temp/serl/trossen_arm_mujoco/sim_recordings_ee_serl/sim_dataset_ee_ortho6d_grip_norm_shifted3_filtered0002_v1_cclean.pkl \
    --remove_episodes 4

or pass many indices like "4,7,12"

# Or auto-filter all bad episodes
python -m trossen_arm_mujoco.dataset_utils.serl_ds_preprocessing \
    input.pkl output.pkl --filter_bad_episodes --min_grip_range 0.5
"""