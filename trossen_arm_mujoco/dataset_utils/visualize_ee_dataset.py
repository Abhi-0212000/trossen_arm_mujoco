#!/usr/bin/env python3
"""
EE Control Dataset Visualization Tools
=======================================

Visualization utilities for SERL pickle datasets using End-Effector (EE) control mode.

Supports multiple rotation representations:
  - angle_axis (14D): [L_pos(3), L_rot(3), L_grip(1), R_pos(3), R_rot(3), R_grip(1)]
  - euler      (14D): [L_pos(3), L_rot(3), L_grip(1), R_pos(3), R_rot(3), R_grip(1)]
  - quat       (16D): [L_pos(3), L_quat(4), L_grip(1), R_pos(3), R_quat(4), R_grip(1)]
  - ortho6d    (20D): [L_pos(3), L_o6d(6), L_grip(1), R_pos(3), R_o6d(6), R_grip(1)]

The script auto-detects the rotation mode from the action dimension.

=== USAGE ===

# Plot all dimensions for an episode
python -m trossen_arm_mujoco.dataset_utils.visualize_ee_dataset \\
    ./serl_demos.pkl --episode 0

# Plot only right arm
python -m trossen_arm_mujoco.dataset_utils.visualize_ee_dataset \\
    ./serl_demos.pkl --robot right

# Plot only position dimensions
python -m trossen_arm_mujoco.dataset_utils.visualize_ee_dataset \\
    ./serl_demos.pkl --dim pos

# Plot specific dimension (0-6 for each arm)
python -m trossen_arm_mujoco.dataset_utils.visualize_ee_dataset \\
    ./serl_demos.pkl --dim 3  # Rotation X

# Show dataset statistics
python -m trossen_arm_mujoco.dataset_utils.visualize_ee_dataset \\
    ./serl_demos.pkl --stats

# Compare action vs state (tracking analysis)
python -m trossen_arm_mujoco.dataset_utils.visualize_ee_dataset \\
    ./serl_demos.pkl --tracking

# Visualize action distribution (histograms)
python -m trossen_arm_mujoco.dataset_utils.visualize_ee_dataset \\
    ./serl_demos.pkl --histogram

# Video playback of camera images
python -m trossen_arm_mujoco.dataset_utils.visualize_ee_dataset \\
    ./serl_demos.pkl --video
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# ROTATION MODE DETECTION (shared with serl_ds_preprocessing.py)
# ============================================================================

# Maps rotation mode -> (rotation_dim, total_action_dim)
ROT_MODE_CONFIG = {
    'angle_axis': {'rot_dim': 3, 'total_dim': 14},
    'euler':      {'rot_dim': 3, 'total_dim': 14},
    'quat':       {'rot_dim': 4, 'total_dim': 16},
    'ortho6d':    {'rot_dim': 6, 'total_dim': 20},
}


def detect_rotation_mode(action_dim: int) -> str:
    """Auto-detect rotation mode from action dimension."""
    for mode, config in ROT_MODE_CONFIG.items():
        if config['total_dim'] == action_dim:
            return mode
    raise ValueError(f"Unknown action dimension {action_dim}. "
                     f"Expected one of {[c['total_dim'] for c in ROT_MODE_CONFIG.values()]}")


def get_indices_for_mode(rot_mode: str) -> dict:
    """
    Get index mappings for a specific rotation mode.
    
    Returns dict with:
        - left_gripper_idx, right_gripper_idx
        - left_pos_idx, left_rot_idx, right_pos_idx, right_rot_idx
        - left_idx, right_idx (all indices per arm)
        - dim_labels
    """
    config = ROT_MODE_CONFIG[rot_mode]
    rot_dim = config['rot_dim']
    
    # Per-arm layout: [pos(3), rot(N), grip(1)]
    arm_dim = 3 + rot_dim + 1
    
    # Left arm indices
    left_pos_idx = [0, 1, 2]
    left_rot_idx = list(range(3, 3 + rot_dim))
    left_gripper_idx = 3 + rot_dim
    left_idx = list(range(arm_dim))
    
    # Right arm indices
    right_pos_idx = [arm_dim + i for i in range(3)]
    right_rot_idx = [arm_dim + 3 + i for i in range(rot_dim)]
    right_gripper_idx = arm_dim + 3 + rot_dim
    right_idx = list(range(arm_dim, 2 * arm_dim))
    
    # Generate dimension labels
    dim_labels = []
    dim_labels.extend(['L_pos_x', 'L_pos_y', 'L_pos_z'])
    for i in range(rot_dim):
        dim_labels.append(f'L_rot_{i}')
    dim_labels.append('L_grip')
    dim_labels.extend(['R_pos_x', 'R_pos_y', 'R_pos_z'])
    for i in range(rot_dim):
        dim_labels.append(f'R_rot_{i}')
    dim_labels.append('R_grip')
    
    return {
        'rot_mode': rot_mode,
        'rot_dim': rot_dim,
        'arm_dim': arm_dim,
        'left_gripper_idx': left_gripper_idx,
        'right_gripper_idx': right_gripper_idx,
        'left_pos_idx': left_pos_idx,
        'left_rot_idx': left_rot_idx,
        'right_pos_idx': right_pos_idx,
        'right_rot_idx': right_rot_idx,
        'left_idx': left_idx,
        'right_idx': right_idx,
        'dim_labels': dim_labels,
    }


# ============================================================================
# LEGACY DEFAULTS (14D angle_axis) - for backward compatibility
# ============================================================================

DIM_LABELS = [
    'L_pos_x', 'L_pos_y', 'L_pos_z',  # 0-2
    'L_rot_x', 'L_rot_y', 'L_rot_z',  # 3-5
    'L_grip',                          # 6
    'R_pos_x', 'R_pos_y', 'R_pos_z',  # 7-9
    'R_rot_x', 'R_rot_y', 'R_rot_z',  # 10-12
    'R_grip',                          # 13
]

LEFT_POS_IDX = [0, 1, 2]
LEFT_ROT_IDX = [3, 4, 5]
LEFT_GRIP_IDX = 6
RIGHT_POS_IDX = [7, 8, 9]
RIGHT_ROT_IDX = [10, 11, 12]
RIGHT_GRIP_IDX = 13
LEFT_IDX = [0, 1, 2, 3, 4, 5, 6]
RIGHT_IDX = [7, 8, 9, 10, 11, 12, 13]


# ============================================================================
# DATASET LOADING
# ============================================================================

def load_dataset(pkl_path: str) -> tuple:
    """
    Load SERL pickle dataset and detect rotation mode.
    
    Returns:
        (transitions, idx_config) - list of transitions and index configuration dict
    """
    logger.info(f"Loading: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    # Handle different formats
    if isinstance(data, list):
        transitions = data
    elif isinstance(data, dict) and 'episodes' in data:
        transitions = data['episodes']
    else:
        raise ValueError(f"Unknown dataset structure: {type(data)}")
    
    logger.info(f"Loaded {len(transitions)} transitions")
    
    # Auto-detect rotation mode from first transition
    if len(transitions) > 0:
        action_dim = transitions[0]['actions'].shape[-1]
        rot_mode = detect_rotation_mode(action_dim)
        idx_config = get_indices_for_mode(rot_mode)
        logger.info(f"Detected rotation mode: {rot_mode} (action_dim={action_dim})")
    else:
        # Fallback to default 14D
        idx_config = get_indices_for_mode('angle_axis')
    
    return transitions, idx_config


def get_episode_boundaries(transitions: list) -> list:
    """
    Find episode boundaries based on 'dones' flag.
    Returns list of (start_idx, end_idx) tuples.
    """
    boundaries = []
    start = 0
    
    for i, t in enumerate(transitions):
        if t.get('dones', False):
            boundaries.append((start, i + 1))
            start = i + 1
    
    # Handle case where last transition isn't marked done
    if start < len(transitions):
        boundaries.append((start, len(transitions)))
    
    return boundaries


def extract_episode_data(transitions: list, episode_idx: int) -> tuple:
    """
    Extract actions and states for a specific episode.
    
    Returns:
        (actions, states, time_steps) - numpy arrays
    """
    boundaries = get_episode_boundaries(transitions)
    
    if episode_idx >= len(boundaries):
        raise ValueError(f"Episode {episode_idx} not found. Only {len(boundaries)} episodes.")
    
    start, end = boundaries[episode_idx]
    logger.info(f"Episode {episode_idx}: steps {start} to {end} ({end - start} transitions)")
    
    actions = []
    states = []
    
    for i in range(start, end):
        t = transitions[i]
        actions.append(t['actions'])
        
        state = t['observations']['state']
        if state.ndim == 2:
            state = state[0]  # Remove batch dim
        states.append(state)
    
    return np.array(actions), np.array(states), np.arange(len(actions))


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_episode_tracking(pkl_path: str, episode_idx: int = 0, 
                          robot: str = 'both', dim_filter: str = None):
    """
    Plot action commands vs observed states for an episode.
    
    Args:
        pkl_path: Path to SERL pickle
        episode_idx: Episode index to visualize
        robot: 'left', 'right', or 'both'
        dim_filter: 'pos', 'rot', 'grip', or None for all
    """
    transitions, idx_config = load_dataset(pkl_path)
    actions, states, time_steps = extract_episode_data(transitions, episode_idx)
    
    # Get indices from config
    dim_labels = idx_config['dim_labels']
    left_idx = idx_config['left_idx']
    right_idx = idx_config['right_idx']
    left_pos_idx = idx_config['left_pos_idx']
    left_rot_idx = idx_config['left_rot_idx']
    left_grip_idx = idx_config['left_gripper_idx']
    right_pos_idx = idx_config['right_pos_idx']
    right_rot_idx = idx_config['right_rot_idx']
    right_grip_idx = idx_config['right_gripper_idx']
    arm_dim = idx_config['arm_dim']
    
    # Determine which dimensions to plot
    if robot == 'left':
        arm_indices = left_idx
        arm_labels = ['left']
    elif robot == 'right':
        arm_indices = right_idx
        arm_labels = ['right']
    else:
        arm_indices = left_idx + right_idx
        arm_labels = ['left', 'right']
    
    # Filter by dimension type if specified
    if dim_filter == 'pos':
        indices = [i for i in arm_indices if i in left_pos_idx + right_pos_idx]
    elif dim_filter == 'rot':
        indices = [i for i in arm_indices if i in left_rot_idx + right_rot_idx]
    elif dim_filter == 'grip':
        indices = [i for i in arm_indices if i in [left_grip_idx, right_grip_idx]]
    elif dim_filter is not None and dim_filter.isdigit():
        # Specific dimension per arm (0 to arm_dim-1)
        d = int(dim_filter)
        indices = []
        if 'left' in arm_labels:
            indices.append(d)
        if 'right' in arm_labels:
            indices.append(d + arm_dim)
    else:
        indices = arm_indices
    
    # Setup plot
    n_dims = len(indices)
    fig, axes = plt.subplots(n_dims, 1, figsize=(12, 2.5 * n_dims), sharex=True)
    if n_dims == 1:
        axes = [axes]
    
    rot_mode = idx_config['rot_mode']
    fig.suptitle(f'Episode {episode_idx} - Action vs State Tracking\n'
                 f'({robot.upper()} arm(s), dims={dim_filter or "all"}, mode={rot_mode})', fontsize=12)
    
    for ax_idx, dim_idx in enumerate(indices):
        ax = axes[ax_idx]
        label = dim_labels[dim_idx]
        
        ax.plot(time_steps, actions[:, dim_idx], 'b--', label='Action (cmd)', linewidth=1.5)
        ax.plot(time_steps, states[:, dim_idx], 'g-', label='State (obs)', alpha=0.7)
        
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        
        if ax_idx == 0:
            ax.legend(loc='upper right')
    
    axes[-1].set_xlabel('Time Step')
    plt.tight_layout()
    plt.show()


def plot_dataset_stats(pkl_path: str, sample_size: int = 5000):
    """
    Show comprehensive dataset statistics.
    """
    transitions, idx_config = load_dataset(pkl_path)
    boundaries = get_episode_boundaries(transitions)
    
    dim_labels = idx_config['dim_labels']
    left_grip_idx = idx_config['left_gripper_idx']
    right_grip_idx = idx_config['right_gripper_idx']
    rot_mode = idx_config['rot_mode']
    action_dim = idx_config['arm_dim'] * 2
    
    n_trans = len(transitions)
    n_eps = len(boundaries)
    
    # Sample transitions for stats
    if n_trans > sample_size:
        sample_indices = np.random.choice(n_trans, sample_size, replace=False)
    else:
        sample_indices = range(n_trans)
    
    actions = np.array([transitions[i]['actions'] for i in sample_indices])
    states = np.array([transitions[i]['observations']['state'] for i in sample_indices])
    if states.ndim == 3:
        states = states[:, 0, :]  # Remove batch dim
    
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    print(f"Total transitions: {n_trans}")
    print(f"Total episodes:    {n_eps}")
    print(f"Rotation mode:     {rot_mode}")
    print(f"Action dimension:  {action_dim}")
    print(f"Avg episode len:   {n_trans / n_eps:.1f} steps")
    print(f"Sampled for stats: {len(sample_indices)} transitions")
    
    # Episode lengths
    ep_lengths = [end - start for start, end in boundaries]
    print(f"\nEpisode lengths: min={min(ep_lengths)}, max={max(ep_lengths)}, "
          f"mean={np.mean(ep_lengths):.1f}, std={np.std(ep_lengths):.1f}")
    
    print("\n" + "-" * 70)
    print(f"ACTION STATISTICS ({action_dim}D EE Control - {rot_mode})")
    print("-" * 70)
    print(f"{'Dimension':<12} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}")
    print("-" * 70)
    
    for i, label in enumerate(dim_labels):
        print(f"{label:<12} {actions[:, i].min():>10.4f} {actions[:, i].max():>10.4f} "
              f"{actions[:, i].mean():>10.4f} {actions[:, i].std():>10.4f}")
    
    print("\n" + "-" * 70)
    print(f"STATE STATISTICS ({action_dim}D EE Control - {rot_mode})")
    print("-" * 70)
    print(f"{'Dimension':<12} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}")
    print("-" * 70)
    
    for i, label in enumerate(dim_labels):
        print(f"{label:<12} {states[:, i].min():>10.4f} {states[:, i].max():>10.4f} "
              f"{states[:, i].mean():>10.4f} {states[:, i].std():>10.4f}")
    
    # Highlight potential issues
    print("\n" + "-" * 70)
    print("POTENTIAL ISSUES")
    print("-" * 70)
    
    # Check for static dimensions
    for i, label in enumerate(dim_labels):
        if actions[:, i].std() < 0.001:
            print(f"⚠️  Action {label} has very low variance (std={actions[:, i].std():.6f}) - might be static")
    
    # Check for out-of-range values
    for i, label in enumerate(dim_labels):
        if actions[:, i].max() > 1.0 or actions[:, i].min() < -1.0:
            print(f"⚠️  Action {label} outside [-1, 1]: [{actions[:, i].min():.4f}, {actions[:, i].max():.4f}]")
    
    # Check gripper values
    for grip_idx in [left_grip_idx, right_grip_idx]:
        label = dim_labels[grip_idx]
        if actions[:, grip_idx].max() < 0.1 and actions[:, grip_idx].min() > -0.1:
            print(f"⚠️  {label} range is very small - might not be normalized")
        if states[:, grip_idx].max() < 0.1 and states[:, grip_idx].min() > -0.1:
            print(f"⚠️  State {label} range is very small - might not be normalized")


def plot_histograms(pkl_path: str, sample_size: int = 5000):
    """
    Plot action value distributions as histograms.
    """
    transitions, idx_config = load_dataset(pkl_path)
    dim_labels = idx_config['dim_labels']
    arm_dim = idx_config['arm_dim']
    rot_mode = idx_config['rot_mode']
    
    n_trans = len(transitions)
    if n_trans > sample_size:
        sample_indices = np.random.choice(n_trans, sample_size, replace=False)
    else:
        sample_indices = range(n_trans)
    
    actions = np.array([transitions[i]['actions'] for i in sample_indices])
    
    # Create subplots (2 rows x arm_dim cols)
    fig, axes = plt.subplots(2, arm_dim, figsize=(2.5 * arm_dim, 6))
    fig.suptitle(f'Action Distribution Histograms (n={len(sample_indices)} samples, mode={rot_mode})', fontsize=12)
    
    for i, label in enumerate(dim_labels):
        row = 0 if i < arm_dim else 1
        col = i if i < arm_dim else i - arm_dim
        ax = axes[row, col]
        
        ax.hist(actions[:, i], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_title(label, fontsize=9)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        
        # Show range
        ax.set_xlabel(f'[{actions[:, i].min():.2f}, {actions[:, i].max():.2f}]', fontsize=7)
    
    plt.tight_layout()
    plt.show()


def plot_episode_trajectories(pkl_path: str, n_episodes: int = 5):
    """
    Overlay multiple episodes to see trajectory variance.
    """
    transitions, idx_config = load_dataset(pkl_path)
    boundaries = get_episode_boundaries(transitions)
    
    dim_labels = idx_config['dim_labels']
    right_pos_idx = idx_config['right_pos_idx']
    right_grip_idx = idx_config['right_gripper_idx']
    rot_mode = idx_config['rot_mode']
    
    n_to_plot = min(n_episodes, len(boundaries))
    
    # Focus on right arm position + gripper (the one that moves)
    dims_to_plot = right_pos_idx + [right_grip_idx]  # R_pos_x, y, z, grip
    
    fig, axes = plt.subplots(len(dims_to_plot), 1, figsize=(12, 8), sharex=False)
    fig.suptitle(f'Right Arm Trajectories ({n_to_plot} episodes overlaid, mode={rot_mode})', fontsize=12)
    
    colors = plt.cm.viridis(np.linspace(0, 1, n_to_plot))
    
    for ep_idx in range(n_to_plot):
        start, end = boundaries[ep_idx]
        ep_actions = np.array([transitions[i]['actions'] for i in range(start, end)])
        time_steps = np.arange(len(ep_actions))
        
        for ax_idx, dim_idx in enumerate(dims_to_plot):
            axes[ax_idx].plot(time_steps, ep_actions[:, dim_idx], 
                             color=colors[ep_idx], alpha=0.6, linewidth=1)
    
    for ax_idx, dim_idx in enumerate(dims_to_plot):
        axes[ax_idx].set_ylabel(dim_labels[dim_idx])
        axes[ax_idx].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time Step')
    plt.tight_layout()
    plt.show()


def video_playback(pkl_path: str, episode_idx: int = 0, fps: int = 15):
    """
    Video playback of camera images with action overlay.
    """
    try:
        import cv2
    except ImportError:
        logger.error("OpenCV not installed. Run: pip install opencv-python")
        return
    
    transitions, idx_config = load_dataset(pkl_path)
    boundaries = get_episode_boundaries(transitions)
    
    # Get indices for this rotation mode
    right_pos_idx = idx_config['right_pos_idx']
    right_rot_idx = idx_config['right_rot_idx']
    right_grip_idx = idx_config['right_gripper_idx']
    rot_mode = idx_config['rot_mode']
    
    if episode_idx >= len(boundaries):
        logger.error(f"Episode {episode_idx} not found. Max: {len(boundaries) - 1}")
        return
    
    start, end = boundaries[episode_idx]
    logger.info(f"Playing episode {episode_idx} ({end - start} frames), mode={rot_mode}")
    
    # Find camera keys
    first_obs = transitions[start]['observations']
    cam_keys = sorted([k for k, v in first_obs.items() 
                       if isinstance(v, np.ndarray) and v.dtype == np.uint8 and v.ndim >= 3])
    
    if not cam_keys:
        logger.error("No camera images found in observations")
        return
    
    logger.info(f"Cameras: {cam_keys}")
    logger.info("Controls: [SPACE] Pause | [ESC] Quit | [→] Next | [←] Prev")
    
    cv2.namedWindow("EE Dataset Viewer", cv2.WINDOW_NORMAL)
    
    idx = start
    paused = False
    
    while start <= idx < end:
        t = transitions[idx]
        obs = t['observations']
        action = t['actions']
        
        # Collect images
        images = []
        for cam in cam_keys:
            img = np.array(obs[cam])
            if img.ndim == 4:
                img = img[0]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.putText(img, cam, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            images.append(img)
        
        # Grid layout
        if len(images) >= 4:
            top = np.hstack(images[:2])
            bot = np.hstack(images[2:4])
            grid = np.vstack((top, bot))
        elif len(images) == 2:
            grid = np.hstack(images)
        else:
            grid = images[0]
        
        # Info bar
        h, w = grid.shape[:2]
        info_bar = np.zeros((80, w, 3), dtype=np.uint8)
        
        status = "PAUSED" if paused else "PLAYING"
        cv2.putText(info_bar, f"Step: {idx - start}/{end - start} | {status} | {rot_mode}", 
                   (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show right arm action using correct indices
        r_pos = action[right_pos_idx]
        r_rot = action[right_rot_idx]
        r_grip = action[right_grip_idx]
        
        cv2.putText(info_bar, f"R_pos: [{r_pos[0]:.3f}, {r_pos[1]:.3f}, {r_pos[2]:.3f}]",
                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(info_bar, f"R_rot: [{r_rot[0]:.3f}, ...] grip: {r_grip:.3f}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        if t.get('dones', False):
            cv2.putText(info_bar, "DONE", (w - 60, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        final = np.vstack((grid, info_bar))
        cv2.imshow("EE Dataset Viewer", final)
        
        # Controls
        wait_ms = 0 if paused else int(1000 / fps)
        key = cv2.waitKey(wait_ms) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == ord(' '):
            paused = not paused
        elif key == 83 or key == ord('d'):  # Right arrow
            idx = min(idx + 1, end - 1)
        elif key == 81 or key == ord('a'):  # Left arrow
            idx = max(idx - 1, start)
        elif not paused:
            idx += 1
    
    cv2.destroyAllWindows()


def plot_action_vs_state_diff(pkl_path: str, episode_idx: int = 0):
    """
    Plot the difference between commanded action and observed state.
    Helps identify tracking lag or offset.
    """
    transitions, idx_config = load_dataset(pkl_path)
    actions, states, time_steps = extract_episode_data(transitions, episode_idx)
    
    # Get indices for this rotation mode
    dim_labels = idx_config['dim_labels']
    right_pos_idx = idx_config['right_pos_idx']
    right_rot_idx = idx_config['right_rot_idx']
    right_grip_idx = idx_config['right_gripper_idx']
    rot_mode = idx_config['rot_mode']
    
    # Focus on right arm (the one that moves)
    dims = right_pos_idx + right_rot_idx + [right_grip_idx]
    
    fig, axes = plt.subplots(len(dims), 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f'Episode {episode_idx}: Action - State Difference (Right Arm, mode={rot_mode})', fontsize=12)
    
    for ax_idx, dim_idx in enumerate(dims):
        ax = axes[ax_idx]
        label = dim_labels[dim_idx]
        
        diff = actions[:, dim_idx] - states[:, dim_idx]
        
        ax.plot(time_steps, diff, 'r-', linewidth=1)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_ylabel(f'Δ {label}')
        ax.grid(True, alpha=0.3)
        
        # Show stats
        ax.text(0.02, 0.95, f'mean={diff.mean():.4f}, std={diff.std():.4f}',
               transform=ax.transAxes, fontsize=8, verticalalignment='top')
    
    axes[-1].set_xlabel('Time Step')
    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Visualize SERL EE Control Dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('pkl_path', type=str, help='Path to SERL pickle file')
    parser.add_argument('--episode', type=int, default=0, help='Episode index (default: 0)')
    parser.add_argument('--robot', choices=['left', 'right', 'both'], default='both',
                        help='Which arm to plot')
    parser.add_argument('--dim', type=str, default=None,
                        help='Dimension filter: pos, rot, grip, or index 0-6')
    
    # Mode flags
    parser.add_argument('--stats', action='store_true', help='Show dataset statistics')
    parser.add_argument('--histogram', action='store_true', help='Plot action histograms')
    parser.add_argument('--tracking', action='store_true', help='Plot action vs state')
    parser.add_argument('--trajectories', action='store_true', help='Overlay episode trajectories')
    parser.add_argument('--diff', action='store_true', help='Plot action-state difference')
    parser.add_argument('--video', action='store_true', help='Video playback of images')
    parser.add_argument('--fps', type=int, default=15, help='Video playback FPS')
    
    args = parser.parse_args()
    
    if not Path(args.pkl_path).exists():
        logger.error(f"File not found: {args.pkl_path}")
        return 1
    
    # Execute requested mode
    if args.stats:
        plot_dataset_stats(args.pkl_path)
    elif args.histogram:
        plot_histograms(args.pkl_path)
    elif args.trajectories:
        plot_episode_trajectories(args.pkl_path, n_episodes=5)
    elif args.diff:
        plot_action_vs_state_diff(args.pkl_path, args.episode)
    elif args.video:
        video_playback(args.pkl_path, args.episode, args.fps)
    else:
        # Default: tracking plot
        plot_episode_tracking(args.pkl_path, args.episode, args.robot, args.dim)
    
    return 0


if __name__ == '__main__':
    exit(main())


"""
# Show stats
python -m trossen_arm_mujoco.dataset_utils.visualize_ee_dataset /home/qte9489/personal_abhi/temp/serl/trossen_arm_mujoco/sim_recordings_ee_serl/sim_dataset_ee_angleaxis_v1_grip_norm_shifted5_filtered005.pkl --stats

# Plot action histograms
python -m trossen_arm_mujoco.dataset_utils.visualize_ee_dataset /home/qte9489/personal_abhi/temp/serl/trossen_arm_mujoco/sim_recordings_ee_serl/sim_dataset_ee_angleaxis_v1_grip_norm_shifted5_filtered005.pkl --histogram

# Overlay episode trajectories
python -m trossen_arm_mujoco.dataset_utils.visualize_ee_dataset /home/qte9489/personal_abhi/temp/serl/trossen_arm_mujoco/sim_recordings_ee_serl/sim_dataset_ee_angleaxis_v1_grip_norm_shifted5_filtered005.pkl --trajectories

# Plot action vs state tracking for an episode
python -m trossen_arm_mujoco.dataset_utils.visualize_ee_dataset /home/qte9489/personal_abhi/temp/serl/trossen_arm_mujoco/sim_recordings_ee_serl/sim_dataset_ee_angleaxis_v1_grip_norm_shifted5_filtered005.pkl --tracking --episode 0

# Video playback
python -m trossen_arm_mujoco.dataset_utils.visualize_ee_dataset /home/qte9489/personal_abhi/temp/serl/trossen_arm_mujoco/sim_recordings_ee_serl/sim_dataset_ee_angleaxis_v1_grip_norm_shifted5_filtered005.pkl --video

"""