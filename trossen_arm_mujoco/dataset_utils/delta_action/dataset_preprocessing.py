import pickle
import numpy as np
import matplotlib.pyplot as plt
from trossen_arm_mujoco.dataset_utils.delta_action_ds.cleanup_utils import load_transitions, get_episode_boundaries
from scipy.spatial.transform import Rotation as R
from trossen_arm_mujoco.ee_transforms import state_16d_to_14d

# --- CONFIGURATION ---
# Adjust these based on your robot's physical limits and data format

# For 14D angle-axis actions: [L_Arm(6), L_Grip(1), R_Arm(6), R_Grip(1)]
GRIPPER_INDICES = {
    'left': 6,    # Index 6 in 14D action space
    'right': 13   # Index 13 in 14D action space
}

# For 14D state: [L_Pos(3), L_Rot(3), L_Grip(1), R_Pos(3), R_Rot(3), R_Grip(1)]
# Indices for the RIGHT arm (which seems to be your focus)
RIGHT_ARM_POS = slice(7, 10)   # x, y, z (indices 7, 8, 9)
RIGHT_ARM_ROT = slice(10, 13)  # rx, ry, rz (angle-axis) (indices 10, 11, 12)

# Physical limits for Trossen Grippers (Example values, change to yours!)
# If your raw data is 0 to 1, set min=0, max=1. 
# If it's PWM (200-800), set min=200, max=800.
GRIPPER_LIMITS = {
    'min': 0.0, 
    'max': 0.044  
}
# ---------------------


# Indices for the RIGHT arm position in the 14D state (x, y, z are at 7, 8, 9)
# State Structure: [L_Pos(3), L_Rot(3), L_Grip(1), R_Pos(3), R_Rot(3), R_Grip(1)]
RIGHT_ARM_POS = slice(7, 10)

def clean_static_starts(input_path: str, output_path: str, threshold: float = 0.05):
    """
    Loads a dataset, removes static periods at the start of each episode, 
    and saves the cleaned dataset.

    and we only care about observation's state and especially the right arm position (x, y, z) only.
    
    Logic:
    - Keeps the very first frame (reset state).
    - Drops all subsequent frames until movement > threshold is detected.
    - Resumes recording from the first movement frame.
    """
    # 1. Load Data
    print(f"Loading from: {input_path}")
    transitions = load_transitions(input_path) # Assumes this helper exists
    boundaries = get_episode_boundaries(transitions) # Assumes this helper exists
    print(f"Loaded {len(transitions)} transitions across {len(boundaries)} episodes.")

    # Helper to extract 3D position vector regardless of shape (1, 14) or (14,)
    def get_pos(state):
        if state.ndim == 2: return state[0][RIGHT_ARM_POS]
        return state[RIGHT_ARM_POS]

    indices_to_drop = set()
    total_dropped = 0

    # 2. Identify Static Frames
    for ep_idx, (start, end) in enumerate(boundaries):
        # Anchor: The Reset State (Keep this)
        anchor_state = get_pos(transitions[start]['observations']['state'])
        
        # We start checking at start + 1
        movement_found_at = -1
        
        for i in range(start + 1, end):
            curr_state = get_pos(transitions[i]['observations']['state'])
            displacement = np.linalg.norm(curr_state - anchor_state)
            
            if displacement > threshold:
                movement_found_at = i
                print(f"  Ep {ep_idx}: Motion detected at step {i-start} (disp: {displacement:.4f})")
                break
        
        # If movement was found, drop everything between start and that frame
        # Range to drop: [start + 1, movement_found_at)
        if movement_found_at != -1:
            drop_range = range(start + 1, movement_found_at)
            for idx in drop_range:
                indices_to_drop.add(idx)
            total_dropped += len(drop_range)
        else:
            # Optional: Warning if an episode NEVER moves
            print(f"  ⚠️ Ep {ep_idx}: No movement detected > {threshold}. Keeping all frames.")

    # 3. Filter Data
    cleaned_transitions = [t for i, t in enumerate(transitions) if i not in indices_to_drop]

    print(f"\n✂️  Dropped {total_dropped} static frames.")
    print(f"   Original Size: {len(transitions)}")
    print(f"   New Size:      {len(cleaned_transitions)}")

    # 4. Save Data
    print(f"Saving to: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(cleaned_transitions, f)
    print("✅ Done.")



def plot_episode_displacements(displacements: list, episodes_to_plot: list = None, threshold_line: float = 0.002):
    """
    Plots the displacement over time for specific episodes.
    
    Args:
        displacements: List of numpy arrays from calculate_displacements
        episodes_to_plot: List of episode indices (e.g., [0, 1, 2]). If None, plots all.
        threshold_line: Where to draw the red reference line.
    """
    if episodes_to_plot is None:
        episodes_to_plot = range(len(displacements))

    plt.figure(figsize=(12, 6))
    
    for ep_idx in episodes_to_plot:
        if ep_idx >= len(displacements):
            print(f"Skipping episode {ep_idx} (out of range)")
            continue
            
        data = displacements[ep_idx]
        steps = np.arange(len(data))
        plt.plot(steps, data, label=f'Episode {ep_idx}')

    # Draw the threshold reference line
    plt.axhline(y=threshold_line, color='r', linestyle='--', label=f'Threshold ({threshold_line})')
    
    plt.title("Robot Arm Displacement from Episode Start")
    plt.xlabel("Time Step (within episode)")
    plt.ylabel("Displacement (L2 Norm)")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_raw_state_actions(transitions: list, boundaries: list, episodes_to_plot: list):
    """
    Plot raw state and action values for right robot (position, rotation, gripper).
    This helps verify gripper normalization: actions should be [-1, 1], states should be [0, 0.044].
    
    Args:
        transitions: List of transitions
        boundaries: Episode boundaries
        episodes_to_plot: List of episode indices to plot
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    
    def get_vec(state):
        if state.ndim == 2:
            return state[0]
        return state
    
    for ep_idx in episodes_to_plot:
        if ep_idx >= len(boundaries):
            print(f"Skipping episode {ep_idx} (out of range)")
            continue
        
        start, end = boundaries[ep_idx]
        ep_transitions = transitions[start:end]
        
        # Extract right robot data
        r_pos_state = []
        r_rot_state = []
        r_grip_state = []
        r_grip_action = []
        
        for t in ep_transitions:
            state = get_vec(t['observations']['state'])
            action = t['actions']
            
            # Right arm: state[7:10] = pos, state[10:13] = rot, state[13] = grip
            r_pos_state.append(np.linalg.norm(state[RIGHT_ARM_POS]))  # Magnitude for simplicity
            r_rot_state.append(np.linalg.norm(state[RIGHT_ARM_ROT]))
            r_grip_state.append(state[GRIPPER_INDICES['right']])
            
            # Right gripper action: action[13]
            r_grip_action.append(action[GRIPPER_INDICES['right']])
        
        steps = np.arange(len(ep_transitions))
        
        # Plot position magnitude
        axes[0].plot(steps, r_pos_state, 'o-', alpha=0.7, markersize=3, label=f'Ep {ep_idx}')
        
        # Plot rotation magnitude
        axes[1].plot(steps, r_rot_state, 'o-', alpha=0.7, markersize=3, label=f'Ep {ep_idx}')
        
        # Plot gripper STATE (should be [0, 0.044])
        axes[2].plot(steps, r_grip_state, 'o-', alpha=0.7, markersize=3, label=f'Ep {ep_idx} (state)')
        
        # Plot gripper ACTION (should be [-1, 1] after normalization)
        axes[3].plot(steps, r_grip_action, 's-', alpha=0.7, markersize=3, label=f'Ep {ep_idx} (action)')
    
    # Add reference lines for gripper
    axes[2].axhline(y=0.0, color='red', linestyle='--', linewidth=1, label='Closed (0.0)')
    axes[2].axhline(y=0.044, color='green', linestyle='--', linewidth=1, label='Open (0.044)')
    
    axes[3].axhline(y=-1.0, color='red', linestyle='--', linewidth=1, label='Normalized -1')
    axes[3].axhline(y=1.0, color='green', linestyle='--', linewidth=1, label='Normalized +1')
    axes[3].axhline(y=0.0, color='gray', linestyle=':', linewidth=1, label='Zero')
    
    # Labels
    axes[0].set_ylabel('Position Magnitude (m)')
    axes[0].set_title('Right Arm Position Magnitude')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_ylabel('Rotation Magnitude (rad)')
    axes[1].set_title('Right Arm Rotation Magnitude')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_ylabel('Gripper State (m)')
    axes[2].set_title('Right Gripper STATE (should be [0.0, 0.044])')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    axes[3].set_ylabel('Gripper Action')
    axes[3].set_title('Right Gripper ACTION (should be [-1, +1] after normalization)')
    axes[3].set_xlabel('Time Step (within episode)')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_initial_states(transitions: list, boundaries: list, episodes_to_plot='all'):
    """
    Plot initial states (Step 0) across episodes to visualize variance.
    
    Args:
        transitions: List of transition dicts
        boundaries: List of (start, end) tuples for each episode
        episodes_to_plot: 'all' or list of episode indices (e.g., [0, 1, 2, 5])
    """
    print(f"\n📊 Plotting Initial States Across Episodes...")
    
    # Helper to get state vector
    def get_vec(state):
        if state.ndim == 2:
            return state[0]
        return state
    
    # Determine which episodes to analyze
    if episodes_to_plot == 'all':
        ep_indices = range(len(boundaries))
    else:
        ep_indices = episodes_to_plot
    
    # Collect initial states
    initial_states = []
    valid_ep_indices = []
    
    for ep_idx in ep_indices:
        if ep_idx >= len(boundaries):
            print(f"  Skipping episode {ep_idx} (out of range)")
            continue
        start, end = boundaries[ep_idx]
        init_state = get_vec(transitions[start]['observations']['state'])
        initial_states.append(init_state)
        valid_ep_indices.append(ep_idx)
    
    initial_states = np.array(initial_states)  # Shape: (N_episodes, 14)
    
    # Calculate statistics
    mean_state = np.mean(initial_states, axis=0)
    std_state = np.std(initial_states, axis=0)
    min_state = np.min(initial_states, axis=0)
    max_state = np.max(initial_states, axis=0)
    range_state = max_state - min_state
    
    print(f"\n  Analyzed {len(initial_states)} episodes")
    print(f"  State dimensions: {initial_states.shape[1]}")
    print(f"\n  Statistics across initial states:")
    print(f"    Mean:  {mean_state}")
    print(f"    Std:   {std_state}")
    print(f"    Min:   {min_state}")
    print(f"    Max:   {max_state}")
    print(f"    Range: {range_state}")
    print(f"\n  Max variance in dimension: {np.argmax(std_state)} (std={std_state[np.argmax(std_state)]:.6f})")
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Initial State Analysis Across Episodes', fontsize=16)
    
    # 1. Box plot for each dimension
    ax = axes[0, 0]
    ax.boxplot(initial_states, vert=True, patch_artist=True)
    ax.set_xlabel('State Dimension')
    ax.set_ylabel('Value')
    ax.set_title('Distribution per Dimension (Box Plot)')
    ax.grid(True, alpha=0.3)
    
    # 2. Heatmap of all initial states
    ax = axes[0, 1]
    im = ax.imshow(initial_states.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_xlabel('Episode Index')
    ax.set_ylabel('State Dimension')
    ax.set_title('Initial States Heatmap')
    ax.set_xticks(range(0, len(valid_ep_indices), max(1, len(valid_ep_indices)//10)))
    ax.set_xticklabels(valid_ep_indices[::max(1, len(valid_ep_indices)//10)])
    plt.colorbar(im, ax=ax)
    
    # 3. Left arm position (dims 0-2)
    ax = axes[1, 0]
    for dim, label in zip([0, 1, 2], ['X', 'Y', 'Z']):
        ax.plot(valid_ep_indices, initial_states[:, dim], 'o-', label=f'Left Arm {label}', alpha=0.7)
    ax.axhline(y=mean_state[0], color='r', linestyle='--', alpha=0.3, label='Mean X')
    ax.set_xlabel('Episode Index')
    ax.set_ylabel('Position (m)')
    ax.set_title('Left Arm Initial Position')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Right arm position (dims 7-9)
    ax = axes[1, 1]
    for dim, label in zip([7, 8, 9], ['X', 'Y', 'Z']):
        ax.plot(valid_ep_indices, initial_states[:, dim], 'o-', label=f'Right Arm {label}', alpha=0.7)
    ax.axhline(y=mean_state[7], color='r', linestyle='--', alpha=0.3, label='Mean X')
    ax.set_xlabel('Episode Index')
    ax.set_ylabel('Position (m)')
    ax.set_title('Right Arm Initial Position')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Gripper states (dims 6, 13)
    ax = axes[2, 0]
    ax.plot(valid_ep_indices, initial_states[:, 6], 'o-', label='Left Gripper', alpha=0.7)
    ax.plot(valid_ep_indices, initial_states[:, 13], 's-', label='Right Gripper', alpha=0.7)
    ax.axhline(y=mean_state[6], color='b', linestyle='--', alpha=0.3, label='Left Mean')
    ax.axhline(y=mean_state[13], color='orange', linestyle='--', alpha=0.3, label='Right Mean')
    ax.set_xlabel('Episode Index')
    ax.set_ylabel('Gripper State (m)')
    ax.set_title('Gripper Initial States')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Standard deviation bar chart
    ax = axes[2, 1]
    dims = np.arange(14)
    colors = ['blue']*7 + ['orange']*7  # Left arm = blue, right arm = orange
    ax.bar(dims, std_state, color=colors, alpha=0.7)
    ax.set_xlabel('State Dimension')
    ax.set_ylabel('Standard Deviation')
    ax.set_title('Variance per Dimension (Higher = More Inconsistent)')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add dimension labels
    dim_labels = ['L_X', 'L_Y', 'L_Z', 'L_Rx', 'L_Ry', 'L_Rz', 'L_Grip',
                  'R_X', 'R_Y', 'R_Z', 'R_Rx', 'R_Ry', 'R_Rz', 'R_Grip']
    ax.set_xticks(dims)
    ax.set_xticklabels(dim_labels, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()
    
    return mean_state, std_state, initial_states


def standardize_initial_states(transitions: list, boundaries: list, target_state='average', epsilon=1e-6):
    """
    Make all episodes start from the exact same initial state.
    
    Args:
        transitions: List of transition dicts
        boundaries: List of (start, end) tuples for each episode
        target_state: 'average' (use mean across episodes) or numpy array (14,) for specific state
        epsilon: Small noise to add for numerical stability (optional)
    
    Returns:
        modified_transitions: Transitions with standardized initial states
    """
    print(f"\n🔧 Standardizing Initial States...")
    
    # Helper to get state vector
    def get_vec(state):
        if state.ndim == 2:
            return state[0]
        return state
    
    # Collect all initial states
    initial_states = []
    for start, end in boundaries:
        init_state = get_vec(transitions[start]['observations']['state'])
        initial_states.append(init_state)
    
    initial_states = np.array(initial_states)
    
    # Determine target state
    if isinstance(target_state, str) and target_state == 'average':
        target = np.mean(initial_states, axis=0)
        print(f"  Using AVERAGE initial state across {len(initial_states)} episodes")
    else:
        target = np.array(target_state)
        print(f"  Using CUSTOM initial state: {target}")
    
    print(f"\n  Target Initial State (14D):")
    print(f"    {target}")
    
    # Calculate differences before standardization
    diffs_before = []
    for init_state in initial_states:
        diff = np.linalg.norm(init_state - target)
        diffs_before.append(diff)
    
    print(f"\n  Before standardization:")
    print(f"    Mean distance from target: {np.mean(diffs_before):.6f}")
    print(f"    Max distance from target:  {np.max(diffs_before):.6f}")
    print(f"    Std distance from target:  {np.std(diffs_before):.6f}")
    
    # Modify transitions
    modified_transitions = []
    for i, trans in enumerate(transitions):
        new_trans = trans.copy()
        
        # Check if this is the start of an episode
        for ep_idx, (start, end) in enumerate(boundaries):
            if i == start:
                # This is the first frame of an episode - replace state
                old_state = get_vec(trans['observations']['state'])
                
                # Add tiny noise for diversity if desired
                if epsilon > 0:
                    noise = np.random.uniform(-epsilon, epsilon, size=target.shape)
                    new_state = target + noise
                else:
                    new_state = target.copy()
                
                # Update the state (handle both (14,) and (1, 14) shapes)
                new_trans['observations'] = new_trans['observations'].copy()
                if trans['observations']['state'].ndim == 2:
                    new_trans['observations']['state'] = new_state.reshape(1, -1)
                else:
                    new_trans['observations']['state'] = new_state
                
                print(f"  Episode {ep_idx}: Changed initial state")
                print(f"    Before: {old_state[:7]}...")
                print(f"    After:  {new_state[:7]}...")
                print(f"    L2 Diff: {np.linalg.norm(old_state - new_state):.6f}")
                break
        
        modified_transitions.append(new_trans)

    print(f"First spisode inittial statee: {list(modified_transitions[0]['observations']['state'])}")
    
    print(f"\n✅ Standardization complete! All episodes now start from the same state.")
    return modified_transitions


def shift_actions_and_save(input_path, output_path):
    print(f"🔄 Shifting actions (S_t, A_t+1 -> A_t)... Processing: {input_path}")
    
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    # 1. Detect Episode Boundaries
    boundaries = []
    start = 0
    for i, t in enumerate(data):
        if t.get('dones', False):
            boundaries.append((start, i + 1))
            start = i + 1
    if start < len(data): boundaries.append((start, len(data)))

    # 2. Shift Actions per Episode
    for start, end in boundaries:
        # Stop one step before the end, because the last frame has no 'next' to pull from
        for i in range(start, end - 1):
            # Pull Action and Next Obs from the future frame (i+1)
            data[i]['actions'] = data[i+1]['actions']
            
            # Standard RL: S_next should be the observation of the next step
            if 'observations' in data[i+1]:
                data[i]['next_observations'] = data[i+1]['observations']

        # Last frame (end-1) keeps its original action (usually 'stop' or 'success')
        # No change needed for data[end-1]

    print(f"💾 Saving {len(data)} aligned transitions to: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)


def plot_episodes(file_path, episode_indices=[0], 
                  stage_name='Dataset', robots='both', components=['pos', 'rot', 'grip']):
    """
    Modular plotter.
    Args:
        robots: 'left', 'right', or 'both'
        components: list of 'pos', 'rot', 'grip'
    """
    transitions = load_transitions(file_path)
    print(f"Total transitions loaded: {len(transitions)}")
    boundaries = get_episode_boundaries(transitions)
    print(f"Total episodes found: {len(boundaries)}")
    print(f"\n📊 Plotting Episodes: {episode_indices} | Robots: {robots} | components: {components}")

    # --- 1. CONFIGURATION ---
    # Map (robot, component) -> (indices, labels)
    CFG = {
        'left': {
            'pos':  {'idxs': [0, 1, 2],    'names': ['L_X', 'L_Y', 'L_Z']},
            'rot':  {'idxs': [3, 4, 5],    'names': ['L_Rx', 'L_Ry', 'L_Rz']},
            'grip': {'idxs': [6],          'names': ['L_Grip']}
        },
        'right': {
            'pos':  {'idxs': [7, 8, 9],    'names': ['R_X', 'R_Y', 'R_Z']},
            'rot':  {'idxs': [10, 11, 12], 'names': ['R_Rx', 'R_Ry', 'R_Rz']},
            'grip': {'idxs': [13],         'names': ['R_Grip']}
        }
    }

    target_robots = ['left', 'right'] if robots == 'both' else [robots]

    # --- 2. PREPARE DATA ---
    episodes_data = []
    for ep_idx in episode_indices:
        if ep_idx >= len(boundaries): continue
        start, end = boundaries[ep_idx]
        
        # Actions are already 14D
        actions = np.array([t['actions'] for t in transitions[start:end]])
        
        # States are 16D -> Convert to 14D
        raw_states = np.array([t['observations']['state'].flatten() for t in transitions[start:end]])
        states = np.array([state_16d_to_14d(s) for s in raw_states])
        
        episodes_data.append({'idx': ep_idx, 'act': actions, 'state': states})

    if not episodes_data:
        print("No data found for indices.")
        return

    # --- 3. PLOTTING LOOP ---
    for robot in target_robots:
        for comp in components:
            if comp not in CFG[robot]: continue
            
            info = CFG[robot][comp]
            indices = info['idxs']
            labels = info['names']
            
            # Create a separate figure for each component block (Position, Rotation, etc)
            fig, axes = plt.subplots(len(indices), 1, figsize=(10, 3 * len(indices)))
            if len(indices) == 1: axes = [axes] # Handle single subplot case
            
            fig.suptitle(f"{robot.upper()} Robot - {comp.upper()} ({stage_name})", fontsize=14)
            
            for i, ax in enumerate(axes):
                dim_idx = indices[i]
                for ep in episodes_data:
                    ax.plot(ep['act'][:, dim_idx], label=f"Ep{ep['idx']} Act", linestyle='-', alpha=0.8)
                    ax.plot(ep['state'][:, dim_idx], label=f"Ep{ep['idx']} State", linestyle='--', alpha=0.8)
                
                ax.set_ylabel(labels[i], fontweight='bold')
                ax.grid(True, alpha=0.3)
                if i == 0: ax.legend(fontsize='small', loc='upper right')
            
            plt.xlabel("Timestep")
            plt.tight_layout()
            plt.show()

def get_right_arm_indices(state_dim):
    """
    Automatically detects indices for Right Arm Position & Gripper
    based on state dimension (14D vs 16D).
    """
    if state_dim == 16: # [L_Pos(3), L_Quat(4), L_Grip(1), R_Pos(3), R_Quat(4), R_Grip(1)]
        return slice(8, 11), slice(11, 15), 15, "16D (Quat)"
    elif state_dim == 14: # [L_Pos(3), L_Rot(3), L_Grip(1), R_Pos(3), R_Rot(3), R_Grip(1)]
        return slice(7, 10), slice(10, 13), 13, "14D (AxisAngle)"
    else:
        raise ValueError(f"Unknown state dimension: {state_dim}")

def get_state_vec(t):
    """Safely extracts the flat state vector."""
    s = t['observations']['state']
    return s[0] if s.ndim == 2 else s


def analyze_data_for_thresholds(file_path):
    """
    Scans the dataset to determine optimal subsampling thresholds based on signal-to-noise heuristics.
    
    Logic:
        1. Calculates frame-to-frame deltas for Position, Rotation, and Gripper.
        2. Distinguishes 'Noise' (idle jitter) from 'Signal' (intentional movement) using percentiles.
        3. Recommendation Formula: threshold = max(2.0 * median, 0.1 * p95)
           - 2.0 * median: Rejects the noise floor (hand tremors/sensor jitter).
           - 0.1 * p95: Ensures threshold is at least 10% of peak movement speed (sensitivity safety net).
    
    Args:
        file_path (str): Path to the .pkl dataset file.
    """
    print(f"\n📊 Analyzing: {file_path}...")
    with open(file_path, 'rb') as f:
        transitions = pickle.load(f)

    if not transitions:
        print("Empty dataset.")
        return

    # Detect dimensions
    sample_state = get_state_vec(transitions[0])
    idx_pos, idx_rot, idx_grip, fmt = get_right_arm_indices(len(sample_state))
    print(f"   Detected format: {fmt}")

    deltas_pos, deltas_rot, deltas_grip = [], [], []

    for i in range(len(transitions) - 1):
        # Skip episode boundaries
        if transitions[i].get('dones', False): continue

        s1 = get_state_vec(transitions[i])
        s2 = get_state_vec(transitions[i+1])

        # Position Delta
        d_pos = np.linalg.norm(s2[idx_pos] - s1[idx_pos])
        
        # Rotation Delta
        if len(sample_state) == 16: # Quaternion
            dot = np.abs(np.dot(s1[idx_rot], s2[idx_rot]))
            d_rot = 2 * np.arccos(np.clip(dot, 0, 1.0))
        else: # Axis Angle
            d_rot = np.linalg.norm(s2[idx_rot] - s1[idx_rot])

        # Gripper Delta
        d_grip = np.abs(s2[idx_grip] - s1[idx_grip])

        deltas_pos.append(d_pos)
        deltas_rot.append(d_rot)
        deltas_grip.append(d_grip)

    print("\n" + "="*60)
    print("   🔍 SUGGESTED THRESHOLDS (Copy these values)")
    print("="*60)
    
    metrics = [("Position (m)", deltas_pos), ("Rotation (rad)", deltas_rot), ("Gripper", deltas_grip)]
    
    suggestions = {}
    
    for name, data in metrics:
        if len(data) == 0: continue
        noise = np.percentile(data, 50)  # Median
        signal = np.percentile(data, 95) # Active Motion
        
        # Heuristic: 2x noise floor, clamped to be at least 10% of active signal
        rec = max(noise * 2.0, signal * 0.1)
        
        print(f"{name}:")
        print(f"   Noise Floor: {noise:.5f} | Max Speed: {signal:.5f}")
        print(f"   -> Suggestion: {rec:.5f}")
        suggestions[name] = rec
        print("-" * 30)
        
    return suggestions

def clean_and_subsample_dataset(input_path, output_path, 
                                static_pos_thresh=0.002,  # From analysis (Position)
                                subsample_pos=0.002,      # Same as static
                                subsample_rot=0.02,       # From analysis (Rotation)
                                grip_hold_thresh=0.02,    # Gripper value < this means "Holding Object"
                                max_skip=5):
    """
    Combined Pipeline:
    1. Static Drop: Trims start of episode until arm moves > static_pos_thresh.
    2. Grasp-Aware Subsampling: Keeps frames based on movement, but is MORE sensitive
       when the gripper is closed (holding object) to preserve smooth transport.
    """
    print(f"\n🧹 Processing: {input_path}")
    
    with open(input_path, 'rb') as f:
        transitions = pickle.load(f)
    
    # 1. Get Boundaries
    boundaries = []
    start = 0
    for i, t in enumerate(transitions):
        if t.get('dones', False):
            boundaries.append((start, i + 1))
            start = i + 1
    if start < len(transitions): boundaries.append((start, len(transitions)))
    
    # Setup Indices
    sample_state = get_state_vec(transitions[0])
    idx_pos, idx_rot, idx_grip, _ = get_right_arm_indices(len(sample_state))

    cleaned_transitions = []
    
    # --- PROCESSING LOOP ---
    for ep_idx, (start, end) in enumerate(boundaries):
        
        # --- PHASE 1: STATIC DROP (Start of Episode) ---
        # We always keep the very first frame (reset state)
        anchor_state = get_state_vec(transitions[start])
        
        movement_start_idx = -1
        
        # Scan forward to find first movement
        for i in range(start + 1, end):
            curr_state = get_state_vec(transitions[i])
            disp = np.linalg.norm(curr_state[idx_pos] - anchor_state[idx_pos])
            if disp > static_pos_thresh:
                movement_start_idx = i
                break
        
        if movement_start_idx == -1: 
            # No movement in whole episode? Keep all (safe fallback)
            movement_start_idx = start + 1
        
        # Add the first frame (Reset State)
        cleaned_transitions.append(transitions[start])
        
        # --- PHASE 2: GRASP-AWARE SUBSAMPLING (Rest of Episode) ---
        # Start from where movement began
        
        last_kept_state = get_state_vec(transitions[movement_start_idx])
        
        # Add the first moving frame
        if movement_start_idx < end:
            cleaned_transitions.append(transitions[movement_start_idx])
        
        skipped_count = 0
        
        for i in range(movement_start_idx + 1, end):
            curr_trans = transitions[i]
            curr_state = get_state_vec(curr_trans)
            
            # Deltas from LAST KEPT frame
            d_pos = np.linalg.norm(curr_state[idx_pos] - last_kept_state[idx_pos])
            
            # Rotation Delta
            if len(sample_state) == 16:
                dot = np.abs(np.dot(curr_state[idx_rot], last_kept_state[idx_rot]))
                d_rot = 2 * np.arccos(np.clip(dot, 0, 1.0))
            else:
                d_rot = np.linalg.norm(curr_state[idx_rot] - last_kept_state[idx_rot])

            # Grasp Logic
            curr_grip = curr_state[idx_grip]
            prev_grip = last_kept_state[idx_grip]
            
            # Are we holding something? (Gripper < threshold)
            # If holding, we halve the thresholds (2x sensitivity) to be smoother
            is_holding = (curr_grip < grip_hold_thresh)
            eff_pos_thresh = subsample_pos * (0.5 if is_holding else 1.0)
            eff_rot_thresh = subsample_rot * (0.5 if is_holding else 1.0)
            
            # Did gripper state change? (Open <-> Close) -> ALWAYS KEEP
            grip_changed = (curr_grip < grip_hold_thresh) != (prev_grip < grip_hold_thresh)
            
            keep = False
            if grip_changed: keep = True
            elif d_pos > eff_pos_thresh: keep = True
            elif d_rot > eff_rot_thresh: keep = True
            elif skipped_count >= max_skip: keep = True
            elif curr_trans.get('dones', False): keep = True # Always keep end
            
            if keep:
                cleaned_transitions.append(curr_trans)
                last_kept_state = curr_state
                skipped_count = 0
            else:
                skipped_count += 1
                
        # Fix the 'done' flag for the last frame of this new episode sequence
        cleaned_transitions[-1]['dones'] = True

    # --- SAVE ---
    print(f"\n💾 Saving {len(cleaned_transitions)} frames to: {output_path}")
    print(f"   (Reduction: {100 * (1 - len(cleaned_transitions)/len(transitions)):.1f}%)")
    
    with open(output_path, 'wb') as f:
        pickle.dump(cleaned_transitions, f)


if __name__ == "__main__":
    # 1. getting Static drop threshold. Visualize the Right robot position and zoom in when tehre is movement in robot and get teh value.
    # file_path = "/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/Tasks/Right_moving_to_cube/merged_pkl.pkl"
    # plot_episodes(file_path, episode_indices=[0, 1, 2], robots='right', components=['pos', 'rot'])

    # 2. Static Drop Filtering. good deefault threshold is given. justtune it afterdrop and visualizing again.
    # file_path = "/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/Tasks/Right_moving_to_cube/merged_pkl.pkl"
    # output_path = "/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/Tasks/Right_moving_to_cube/merged_pkl_static_filtered.pkl"
    # l2norm_right_position_threshold = 0.0015  # Set based on noise floor analysis
    # clean_static_starts(
    #     input_path=file_path,
    #     output_path=output_path,
    #     threshold=l2norm_right_position_threshold
    # )
    # plot_episodes(output_path, episode_indices=[0, 1, 49, 55, 77], robots='right', components=['pos'])


    # 3. Load Data and Prepare for Subsampling by analyzing the thresholds
    # file_path = "/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/Tasks/Right_moving_to_cube/merged_pkl_static_filtered.pkl"
    # sug = analyze_data_for_thresholds(file_path)  # suggests you thresholds for subsampling of state data and to drop almost static transitions.
    # print(f"Suggested Thresholds: {sug}")
    # out_path="/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/Tasks/Right_moving_to_cube/merged_pkl_static_filtered_subsampled.pkl"
    # if sug:            # 4. Run cleaning with suggested numbers
    #         clean_and_subsample_dataset(
    #             file_path, 
    #             out_path,
    #             static_pos_thresh=sug.get("Position (m)", 0.006),
    #             subsample_pos=sug.get("Position (m)", 0.017),
    #             subsample_rot=sug.get("Rotation (rad)", 0.0002),
    #             grip_hold_thresh=0.02, # Usually fixed for your gripper ,
    #             max_skip=5
    #         )
    # plot_episodes(out_path, episode_indices=[0, 1, 49, 55, 77], robots='right', components=['pos', 'rot', 'grip'])

    # 4. shift actions by one to fix temporal misalignment
    # file_path="/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/Tasks/Right_moving_to_cube/merged_pkl_static_filtered_subsampled.pkl"
    # output_path="/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/Tasks/Right_moving_to_cube/merged_pkl_static_filtered_subsampled_action_shifted.pkl"
    # shift_actions_and_save(file_path, output_path)
    # plot_episodes(output_path, episode_indices=[0, 1, 49, 55, 77], robots='right', components=['pos', 'rot', 'grip'])

