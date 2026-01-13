import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from scipy.spatial.transform import Rotation as R
from typing import Union, List
from trossen_arm_mujoco.dataset_utils.delta_action.cleanup_utils import load_transitions, get_episode_boundaries, pretty_print_obs, quaternion_to_angle_axis, wrap_angle
from trossen_arm_mujoco.ee_transforms import action_14d_robot_to_world_aa
from typing import Union, List

def convert_actions_to_delta_act_including_gripper(file_path: str = None, output_file_path: str = None):
    # file_path = "/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/random_cube_pose/merged_recordings_static_dropped.pkl"
    # output_file_path = "/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/random_cube_pose/merged_recordings_static_dropped_delta_act.pkl"
    
    transitions = load_transitions(file_path)
    episode_boundaries = get_episode_boundaries(transitions)

    print("Converting actions to delta actions...")
    
    for ep_idx, (start, end) in enumerate(tqdm(episode_boundaries)):
        # Slice the episode
        original_episode = transitions[start:end]

        for i, original_trans in enumerate(original_episode):
            # Get current observation (world frame, 16D with quaternions)
            current_obs = original_trans["observations"]["state"].reshape(-1)
            
            target_action_world = original_trans["actions"].reshape(-1)
            
            # Transform action from robot frame to world frame (both 14D angle-axis)
            # target_action_world = action_14d_robot_to_world_aa(target_action_robot)
            
            # --- Extract Current State in angle-axis format ---
            # Left Arm
            l_pos_curr = current_obs[0:3]
            l_quat_curr = current_obs[3:7]  # [w,x,y,z]
            l_grip_curr = current_obs[7:8]
            
            # Right Arm
            r_pos_curr = current_obs[8:11]
            r_quat_curr = current_obs[11:15] # [w,x,y,z]
            r_grip_curr = current_obs[15:16]
            
            # Convert quaternions to angle-axis
            l_rot_curr = quaternion_to_angle_axis(l_quat_curr)
            r_rot_curr = quaternion_to_angle_axis(r_quat_curr)

            # Construct Current State in 14D angle-axis format
            current_state_aa = np.concatenate([
                l_pos_curr, l_rot_curr, l_grip_curr,
                r_pos_curr, r_rot_curr, r_grip_curr
            ])
            
            # --- Calculate Deltas ---
            # Now both are in world frame and angle-axis format!
            delta_action = target_action_world - current_state_aa
            
            # Wrap rotation deltas to [-pi, pi]
            delta_action[3:6] = wrap_angle(delta_action[3:6])     # Left rotation
            delta_action[10:13] = wrap_angle(delta_action[10:13]) # Right rotation

            # Update in place
            original_trans["actions"] = delta_action

    print(f"Saving processed data to {output_file_path}...")
    with open(output_file_path, 'wb') as f:
        pickle.dump(transitions, f)
    print("Save complete.")

from tqdm import tqdm
def update_action_scale(file_path, output_file_path, ACTION_SCALE):
    # Position scale: 0.02 meters (2cm) -> Means output of 1.0 equals 2cm move
    # Rotation scale: 0.05 radians -> Means output of 1.0 equals 0.05 rad move
    # ACTION_SCALE = np.array([0.02, 0.05])
    
    # file_path = "/home/qte9489/personal_abhi/temp/serl/sim_recordings_ee/v1/same_cube_pose/sim_dataset_replayed_and_verified_delta_act.pkl"
    # output_file_path = "/home/qte9489/personal_abhi/temp/serl/sim_recordings_ee/v1/same_cube_pose/sim_dataset_replayed_and_verified_delta_act_scaled.pkl"
    
    transitions = load_transitions(file_path)
    print("Scaling actions (Normalizing)...")
    
    for trans in tqdm(transitions):
        # 1. Copy to avoid accidental reference issues
        # Ensure it's float64 or float32 so division works cleanly
        delta_action = trans['actions'].astype(np.float32).copy()
        
        # 2. Divide by Scale (Normalization)
        # Note: We use [:, indices] to handle the (1, 14) shape correctly
        
        # Left Arm Position (0-3) & Right Arm Position (7-10)
        delta_action[0:3]  /= ACTION_SCALE[0]
        delta_action[7:10] /= ACTION_SCALE[0]
        
        # Left Arm Rotation (3-6) & Right Arm Rotation (10-13)
        delta_action[3:6]   /= ACTION_SCALE[1]
        delta_action[10:13] /= ACTION_SCALE[1] 
        
        # Grippers (6) & (13) scaled too using ACTION_SCALE[2] and clipped to [-1, 1]
        delta_action[6]   /= ACTION_SCALE[2]
        delta_action[13]  /= ACTION_SCALE[2]
        # Clip grippers to ensure within bounds
        delta_action[6] = np.clip(delta_action[6], -1.0, 1.0)
        delta_action[13] = np.clip(delta_action[13], -1.0, 1.0)

        # 3. Clip (Optional but Recommended)
        # Sometimes a tiny outlier exists. Clipping to [-1, 1] keeps training stable.
        # Clip everything except grippers since they are already in [-1, 1]
        delta_action[0:6] = np.clip(delta_action[0:6], -1.0, 1.0)
        delta_action[7:13] = np.clip(delta_action[7:13], -1.0, 1.0)
        
        trans['actions'] = delta_action

    print(f"Saving scaled data to {output_file_path}...")
    with open(output_file_path, 'wb') as f:
        pickle.dump(transitions, f)
    print("Save complete.")


import numpy as np
import pickle
from tqdm import tqdm

def get_per_episode_stats_with_steps(file_path: str):
    # Update with your actual file path
    # file_path = "/home/qte9489/personal_abhi/temp/serl/sim_recordings_ee/v1/same_cube_pose/sim_dataset_replayed_and_verified_delta_act_CLEANED_no_jump_regenerate_FINAL_list_corrected_masks.pkl"
    transitions = load_transitions(file_path)
    episode_boundaries = get_episode_boundaries(transitions)
    
    print(f"Found {len(episode_boundaries)} episodes.")
    print("=" * 180)
    # Header
    # P = Position, R = Rotation, G = Gripper
    print(f"{'Ep':<4} | {'Side':<5} | {'Pos Min (Step)':<16} | {'Pos Max (Step)':<16} || {'Rot Min (Step)':<16} | {'Rot Max (Step)':<16} || {'Grip Min (Step)':<16} | {'Grip Max (Step)':<16}")
    print("=" * 180)

    for ep_idx, (start, end) in enumerate(episode_boundaries):
        # 1. Extract signed actions (Shape: [Steps, 14])
        # We use signed values now to distinguish between -1 and 1
        ep_actions = np.vstack([t['actions'] for t in transitions[start:end]])
        
        # --- Helper to process a slice of the action vector ---
        def get_extremes(indices):
            slice_data = ep_actions[:, indices]
            
            # --- Minimum (Most Negative) ---
            # We want the "true" min (e.g. -1.0)
            min_val = np.min(slice_data)
            if slice_data.ndim == 1:
                min_step = np.argmin(slice_data)
            else:
                # If 2D (e.g. pos x,y,z), flatten to find the single lowest number
                min_step = np.unravel_index(np.argmin(slice_data), slice_data.shape)[0]

            # --- Maximum (Most Positive) ---
            max_val = np.max(slice_data)
            if slice_data.ndim == 1:
                max_step = np.argmax(slice_data)
            else:
                max_step = np.unravel_index(np.argmax(slice_data), slice_data.shape)[0]
                
            return min_val, min_step, max_val, max_step

        # --- LEFT ROBOT (0-6) ---
        l_pos_min, l_pos_min_s, l_pos_max, l_pos_max_s = get_extremes(slice(0, 3))
        l_rot_min, l_rot_min_s, l_rot_max, l_rot_max_s = get_extremes(slice(3, 6))
        l_grp_min, l_grp_min_s, l_grp_max, l_grp_max_s = get_extremes(slice(6, 7))

        # --- RIGHT ROBOT (7-13) ---
        r_pos_min, r_pos_min_s, r_pos_max, r_pos_max_s = get_extremes(slice(7, 10))
        r_rot_min, r_rot_min_s, r_rot_max, r_rot_max_s = get_extremes(slice(10, 13))
        r_grp_min, r_grp_min_s, r_grp_max, r_grp_max_s = get_extremes(slice(13, 14))

        # --- Formatting Helper ---
        def fmt(val, step):
            # Flag values close to edges (-1 or 1) with "!"
            # We check if abs(val) > 0.99
            mark = "!" if abs(val) >= 0.999 else ""
            return f"{val: .4f} (s{step}){mark}"

        # Prepare strings for Left
        lp_min_str = fmt(l_pos_min, l_pos_min_s)
        lp_max_str = fmt(l_pos_max, l_pos_max_s)
        lr_min_str = fmt(l_rot_min, l_rot_min_s)
        lr_max_str = fmt(l_rot_max, l_rot_max_s)
        lg_min_str = fmt(l_grp_min, l_grp_min_s)
        lg_max_str = fmt(l_grp_max, l_grp_max_s)

        # Prepare strings for Right
        rp_min_str = fmt(r_pos_min, r_pos_min_s)
        rp_max_str = fmt(r_pos_max, r_pos_max_s)
        rr_min_str = fmt(r_rot_min, r_rot_min_s)
        rr_max_str = fmt(r_rot_max, r_rot_max_s)
        rg_min_str = fmt(r_grp_min, r_grp_min_s)
        rg_max_str = fmt(r_grp_max, r_grp_max_s)

        # Print Left Row
        print(f"{ep_idx:<4} | {'LEFT':<5} | {lp_min_str:<16} | {lp_max_str:<16} || {lr_min_str:<16} | {lr_max_str:<16} || {lg_min_str:<16} | {lg_max_str:<16}")
        # Print Right Row
        print(f"{'':<4} | {'RIGHT':<5} | {rp_min_str:<16} | {rp_max_str:<16} || {rr_min_str:<16} | {rr_max_str:<16} || {rg_min_str:<16} | {rg_max_str:<16}")
        print("-" * 180)
    



# --- Main Inspection Function ---
def inspect_start_states(file_path: str = None):
    # Use the UNCLEANED file (the one with the jumps) to see the true starts
    # file_path = "/home/qte9489/personal_abhi/temp/serl/sim_recordings_ee/v1/same_cube_pose/sim_dataset_replayed_and_verified_delta_act.pkl"
    
    transitions = load_transitions(file_path)
    episode_boundaries = get_episode_boundaries(transitions)
    
    print(f"Found {len(episode_boundaries)} episodes.")
    print("-" * 140)
    # Header for readability
    print(f"{'Ep':<4} | {'Left Start Pos (x,y,z)':<32} | {'Right Start Pos (x,y,z)':<32}")
    print("-" * 140)

    # Lists to calculate averages later
    left_starts = []
    right_starts = []
    
    for ep_idx, (start, end) in enumerate(episode_boundaries):
        # Get the VERY FIRST transition of the episode
        first_trans = transitions[start]
        
        # Extract state (Flatten just in case)
        state = first_trans['observations']['state'].reshape(-1)
        
        # Extract positions (Modify indices if your 16D structure is different!)
        # 0:3 = Left Pos, 3:7 = Left Quat, 7 = Left Grip
        # 8:11 = Right Pos, 11:15 = Right Quat, 15 = Right Grip
        l_pos = state[0:3]
        r_pos = state[8:11]
        
        left_starts.append(l_pos)
        right_starts.append(r_pos)
        
        # Format string for cleaner printing
        l_str = f"[{l_pos[0]:.4f}, {l_pos[1]:.4f}, {l_pos[2]:.4f}]"
        r_str = f"[{r_pos[0]:.4f}, {r_pos[1]:.4f}, {r_pos[2]:.4f}]"
        
        print(f"{ep_idx:<4} | {l_str:<32} | {r_str:<32}")

    # --- STATISTICS ---
    left_starts = np.array(left_starts)
    right_starts = np.array(right_starts)
    
    print("-" * 140)
    print("STATISTICS (To copy into your Env Constants):")
    
    l_mean = np.mean(left_starts, axis=0)
    l_std  = np.std(left_starts, axis=0)
    
    r_mean = np.mean(right_starts, axis=0)
    r_std  = np.std(right_starts, axis=0)
    
    print(f"LEFT ARM MEAN START:  np.array([{l_mean[0]:.5f}, {l_mean[1]:.5f}, {l_mean[2]:.5f}])")
    print(f"   (Variation/Std):   [{l_std[0]:.5f}, {l_std[1]:.5f}, {l_std[2]:.5f}]")
    print("")
    print(f"RIGHT ARM MEAN START: np.array([{r_mean[0]:.5f}, {r_mean[1]:.5f}, {r_mean[2]:.5f}])")
    print(f"   (Variation/Std):   [{r_std[0]:.5f}, {r_std[1]:.5f}, {r_std[2]:.5f}]")

    # Also print the Quaternions for the "Mean" episode (usually Ep 0 or close to mean)
    # Just grabbing the first one as a reference for orientation
    ref_state = transitions[episode_boundaries[0][0]]['observations']['state'].reshape(-1)
    l_quat = ref_state[3:7]
    r_quat = ref_state[11:15]
    print("-" * 50)
    print("REFERENCE QUATERNIONS (From Ep 0):")
    print(f"LEFT QUAT:  {l_quat}")
    print(f"RIGHT QUAT: {r_quat}")



def check_frame_1_consistency(file_path: str = None):
    # Use your CURRENT uncleaned file
    
    transitions = load_transitions(file_path)
    episode_boundaries = get_episode_boundaries(transitions)
    
    print(f"Checking consistency of FRAME 1 (The New Start) across {len(episode_boundaries)} episodes...")
    print("-" * 100)
    print(f"{'Ep':<4} | {'Left Frame 1 Pos':<32} | {'Right Frame 1 Pos':<32}")
    print("-" * 100)

    l_starts = []
    r_starts = []
    
    valid_episodes = 0

    for ep_idx, (start, end) in enumerate(episode_boundaries):
        # We need at least 2 frames to check Frame 1
        if (end - start) < 2:
            print(f"{ep_idx:<4} | SKIPPED (Too short)")
            continue

        # Get Frame 1 (The transition AFTER the jump)
        frame_1 = transitions[start + 1]
        state = frame_1['observations']['state'].reshape(-1)
        
        # Extract positions
        l_pos = state[0:3]
        r_pos = state[8:11]
        
        l_starts.append(l_pos)
        r_starts.append(r_pos)
        valid_episodes += 1
        
        l_str = f"[{l_pos[0]:.4f}, {l_pos[1]:.4f}, {l_pos[2]:.4f}]"
        r_str = f"[{r_pos[0]:.4f}, {r_pos[1]:.4f}, {r_pos[2]:.4f}]"
        
        # Print first 10 to inspect visually, then just summaries
        if ep_idx < 10:
            print(f"{ep_idx:<4} | {l_str:<32} | {r_str:<32}")

    l_starts = np.array(l_starts)
    r_starts = np.array(r_starts)
    
    print("-" * 100)
    print(f"Analyzed {valid_episodes} episodes.")
    
    # Calculate Standard Deviation (Variance)
    l_std = np.std(l_starts, axis=0)
    r_std = np.std(r_starts, axis=0)
    
    l_mean = np.mean(l_starts, axis=0)
    r_mean = np.mean(r_starts, axis=0)
    
    print(f"LEFT New Start Mean:  {l_mean}")
    print(f"LEFT Variation (Std): {l_std}")
    print("")
    print(f"RIGHT New Start Mean:  {r_mean}")
    print(f"RIGHT Variation (Std): {r_std}")
    
    # THE VERDICT
    max_var = np.max([np.max(l_std), np.max(r_std)])
    print("-" * 100)
    if max_var < 0.02: # Less than 2cm variance
        print("✅ GOOD NEWS: The start positions are consistent (Variance < 2cm).")
        print("   You CAN use the constant Mean values in your Env Reset.")
    else:
        print("⚠️ WARNING: High variance detected!")
        print("   You CANNOT use a single constant. You must use the Teleport Loop or random resets.")


def prune_first_frame_and_save():
    """
    To drop teh idx 0 of each episode (The Jump Frame) and save a new cleaned dataset.
    """
    # --- CONFIGURATION ---
    # INPUT: Your current "Delta" dataset (The one with the jump)
    input_file_path = "/home/qte9489/personal_abhi/temp/serl/sim_recordings_ee/v1/same_cube_pose/sim_dataset_replayed_and_verified_delta_act_scaled.pkl"
    
    # OUTPUT: The final Cleaned dataset ready for training
    output_file_path = "/home/qte9489/personal_abhi/temp/serl/sim_recordings_ee/v1/same_cube_pose/sim_dataset_replayed_and_verified_delta_act_CLEANED_no_jump.pkl"
    # ---------------------

    transitions = load_transitions(input_file_path)
    episode_boundaries = get_episode_boundaries(transitions)
    
    print(f"Found {len(transitions)} total transitions across {len(episode_boundaries)} episodes.")
    print("Pruning Frame 0 (The Jump) from every episode...")
    
    cleaned_transitions = []
    skipped_eps = 0
    
    for ep_idx, (start, end) in enumerate(tqdm(episode_boundaries)):
        episode_data = transitions[start:end]
        
        # Safety check: Episode must be at least 2 frames long to prune one
        if len(episode_data) < 2:
            print(f"Warning: Episode {ep_idx} is too short ({len(episode_data)} frames). Skipping.")
            skipped_eps += 1
            continue
            
        # --- THE PRUNE ---
        # Take from Index 1 to End (Dropping Index 0)
        new_episode = episode_data[1:]
        
        cleaned_transitions.extend(new_episode)

    # --- SAVE ---
    print("-" * 60)
    print(f"Original Count: {len(transitions)}")
    print(f"New Count:      {len(cleaned_transitions)}")
    print(f"Removed:        {len(transitions) - len(cleaned_transitions)} frames (1 per episode).")
    
    # Save as a dictionary if your downstream loader expects it, or list if it expects list.
    # Usually standard SERL/RL loaders handle list-of-dicts or dict-of-lists. 
    # Your previous code loaded 'transitions' key, so let's keep that structure.
    final_data = {'transitions': cleaned_transitions}
    
    print(f"Saving to {output_file_path}...")
    with open(output_file_path, 'wb') as f:
        pickle.dump(final_data, f)
    print("Done! Dataset is clean.")



def delete_specific_episodes_and_save():
    # --- CONFIGURATION ---
    input_file_path = "/home/qte9489/personal_abhi/temp/serl/sim_recordings_ee/v1/same_cube_pose/sim_dataset_replayed_and_verified_delta_act_CLEANED_no_jump_regenerate.pkl"
    output_file_path = "/home/qte9489/personal_abhi/temp/serl/sim_recordings_ee/v1/same_cube_pose/sim_dataset_replayed_and_verified_delta_act_CLEANED_no_jump_regenerate_FINAL.pkl"
    
    # LIST OF EPISODES TO DELETE (0-based indices)
    # Example: [0, 41, 5]
    episodes_to_delete = [50, 42] 
    # ---------------------

    transitions = load_transitions(input_file_path)
    episode_boundaries = get_episode_boundaries(transitions)
    
    total_episodes = len(episode_boundaries)
    print(f"Found {len(transitions)} transitions across {total_episodes} episodes.")
    print(f"Deleting episodes: {episodes_to_delete}")

    kept_transitions = []
    kept_ep_count = 0
    deleted_ep_count = 0

    # Iterate through all episodes
    for ep_idx, (start, end) in enumerate(tqdm(episode_boundaries)):
        
        # If this index is in our delete list, skip it
        if ep_idx in episodes_to_delete:
            print(f" -> Deleting Episode {ep_idx} (Indices {start}-{end})")
            deleted_ep_count += 1
            continue
            
        # Otherwise, keep these transitions
        episode_data = transitions[start:end]
        kept_transitions.extend(episode_data)
        kept_ep_count += 1

    # --- SAVE ---
    print("-" * 60)
    print(f"Original Episodes: {total_episodes}")
    print(f"Deleted Episodes:  {deleted_ep_count}")
    print(f"Remaining Episodes:{kept_ep_count}")
    print(f"Transitions:       {len(transitions)} -> {len(kept_transitions)}")
    
    # Save in standard dictionary format
    final_data = {'transitions': kept_transitions}
    
    print(f"Saving to {output_file_path}...")
    with open(output_file_path, 'wb') as f:
        pickle.dump(final_data, f)
    print("Done.")


def make_delta_gripper_actions(input_file_path: str = None, output_file_path: str = None):
    """Convert absolute normalized gripper actions to delta gripper actions."""
    # 1. Load dataset and get episode boundaries
    transitions = load_transitions(input_file_path)
    episode_boundaries = get_episode_boundaries(transitions)
    
    print(f"Found {len(transitions)} transitions across {len(episode_boundaries)} episodes.")
    print("Converting gripper actions from absolute to delta...")
    
    # 2-3. Process each transition to compute delta gripper actions
    for trans in tqdm(transitions):
        # Get current observation gripper state (RAW, not normalized)
        current_obs = trans['observations']['state'].reshape(-1)
        current_left_grip_raw = current_obs[7]  # State index 7 - raw physical value
        current_right_grip_raw = current_obs[15]  # State index 15 - raw physical value
        
        # Get target gripper actions (normalized in [-1, 1])
        actions = trans['actions'].copy()
        # grippers are already denormalized
        target_left_grip_phys = actions[6]
        target_right_grip_phys = actions[13]
        
        # Compute delta: action_gripper - obs_gripper
        delta_left_grip = target_left_grip_phys - current_left_grip_raw
        delta_right_grip = target_right_grip_phys - current_right_grip_raw
        
        # Replace action's gripper indices with deltas
        actions[6] = delta_left_grip
        actions[13] = delta_right_grip
        
        trans['actions'] = actions
    
    # 4. Save the new dataset
    print(f"Saving delta gripper dataset to {output_file_path}...")
    with open(output_file_path, 'wb') as f:
        pickle.dump(transitions, f)
    print("Save complete.")

def rescale_gripper_actions_and_save(action_scale, input_file_path=None, output_file_path=None):
    """Rescale gripper actions by dividing by suggested gripper scale."""

    if len(action_scale) < 3:
        raise ValueError("action_scale must have 3 elements: [position_scale, rotation_scale, gripper_scale]")
    
    gripper_scale = action_scale[2]
    print(f"Using gripper scale: {gripper_scale:.6f}")
    
    # 1. Load dataset
    transitions = load_transitions(input_file_path)
    episode_boundaries = get_episode_boundaries(transitions)
    
    print(f"Found {len(transitions)} transitions across {len(episode_boundaries)} episodes.")
    print("Rescaling gripper actions...")
    
    # 2. Rescale gripper actions
    for trans in tqdm(transitions):
        actions = trans['actions'].astype(np.float32).copy()
        
        # Divide gripper deltas by scale to normalize to [-1, 1]
        actions[6] = actions[6] / gripper_scale  # Left gripper
        actions[13] = actions[13] / gripper_scale  # Right gripper
        
        # Clip to ensure within bounds
        actions[6] = np.clip(actions[6], -1.0, 1.0)
        actions[13] = np.clip(actions[13], -1.0, 1.0)
        
        trans['actions'] = actions
    
    # 3. Save the new dataset
    print(f"Saving rescaled dataset to {output_file_path}...")
    with open(output_file_path, 'wb') as f:
        pickle.dump(transitions, f)
    print("Save complete.")
    
    # 4. Verify with statistics
    print("\n" + "="*80)
    print("VERIFICATION: Checking gripper action ranges after rescaling...")
    print("="*80)
    
    left_grips = [trans['actions'][6] for trans in transitions]
    right_grips = [trans['actions'][13] for trans in transitions]
    
    print(f"Left Gripper:  Min={np.min(left_grips):.4f}, Max={np.max(left_grips):.4f}")
    print(f"Right Gripper: Min={np.min(right_grips):.4f}, Max={np.max(right_grips):.4f}")
    
    if np.max(np.abs(left_grips)) <= 1.0 and np.max(np.abs(right_grips)) <= 1.0:
        print("✅ All gripper actions are within [-1, 1]")
    else:
        print("⚠️  WARNING: Some gripper actions exceed [-1, 1]!")
    print("="*80 + "\n")

def add_epsilon_to_near_edge_actions(input_file_path: str = None, output_file_path: str = None):
    """Add small epsilon to actions at -1 or 1 to avoid saturation."""
    # input_file_path = "/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/same_cube_pose/sim_dataset_replayed_and_verified_delta_act_CLEANED_no_jump_regenerate_FINAL_list_corrected_masks_delta_gripper_scaled_005_regenerated_states.pkl"
    # output_file_path = "/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/same_cube_pose/sim_dataset_replayed_and_verified_delta_act_CLEANED_no_jump_regenerate_FINAL_list_corrected_masks_delta_gripper_scaled_005_regenerated_states_epsilon.pkl"
    
    epsilon = 1e-4
    
    # 1. Load dataset
    transitions = load_transitions(input_file_path)
    episode_boundaries = get_episode_boundaries(transitions)
    
    print(f"Found {len(transitions)} transitions across {len(episode_boundaries)} episodes.")
    print(f"Adding epsilon={epsilon} to actions at edges [-1, 1]...")
    
    edge_count = 0
    

    # 2. Process each transition
    for trans in tqdm(transitions):
        # This single line handles both -1 and +1 for ALL indices safely
        trans['actions'] = np.clip(trans['actions'], -1.0 + epsilon, 1.0 - epsilon)
    
    print(f"Modified {edge_count} action components that were at edges.")
    
    # 3. Save the new dataset
    print(f"Saving edge-corrected dataset to {output_file_path}...")
    with open(output_file_path, 'wb') as f:
        pickle.dump(transitions, f)
    print("Save complete.")
    
    # 4. Verify no exact edges remain
    print("\n" + "="*80)
    print("VERIFICATION: Checking for exact edge values...")
    print("="*80)
    
    exact_edges = 0
    for trans in transitions:
        actions = trans['actions']
        exact_edges += np.sum((actions == 1.0) | (actions == -1.0))
    
    if exact_edges == 0:
        print("✅ No actions are exactly at -1 or 1")
    else:
        print(f"⚠️  WARNING: Found {exact_edges} actions still at edges!")
    print("="*80 + "\n")

def visualize_data(
    file_path: str,
    component: str,
    episode_indices: Union[int, List[int], None] = None,
    data_source: str = 'action'
):
    """
    Visualize action or observation data for specified episodes.
    
    Args:
        file_path: Path to the pickle file containing transitions
        component: What to visualize - 'gripper', 'position', 'rotation', or 'all'
        episode_indices: Single episode index, list of indices, or None for all episodes
        data_source: 'action' or 'observation' to specify data source
    
    Example usage:
        # Visualize gripper actions for episode 5
        visualize_data('dataset.pkl', 'gripper', episode_indices=5)
        
        # Visualize position observations for episodes 0, 5, 10
        visualize_data('dataset.pkl', 'position', episode_indices=[0, 5, 10], data_source='observation')
        
        # Visualize rotation actions for all episodes
        visualize_data('dataset.pkl', 'rotation', episode_indices=None)
    """
    # Load data
    transitions = load_transitions(file_path)
    episode_boundaries = get_episode_boundaries(transitions)
    
    # Handle episode_indices input
    if episode_indices is None:
        episode_list = list(range(len(episode_boundaries)))
        print(f"Visualizing all {len(episode_list)} episodes...")
    elif isinstance(episode_indices, int):
        episode_list = [episode_indices]
    else:
        episode_list = list(episode_indices)
    
    # Validate episode indices
    for ep_idx in episode_list:
        if ep_idx < 0 or ep_idx >= len(episode_boundaries):
            raise ValueError(f"Episode index {ep_idx} out of range [0, {len(episode_boundaries)-1}]")
    
    # Define component mappings for actions
    # Action space: [L_Pos(3), L_Rot(3), L_Grip(1), R_Pos(3), R_Rot(3), R_Grip(1)]
    component_map_action = {
        'gripper': {'left': [6], 'right': [13], 'labels': ['Gripper']},
        'position': {'left': [0, 1, 2], 'right': [7, 8, 9], 'labels': ['X', 'Y', 'Z']},
        'rotation': {'left': [3, 4, 5], 'right': [10, 11, 12], 'labels': ['RX', 'RY', 'RZ']},
    }
    
    # Define component mappings for observations
    # Observation state: [L_Pos(3), L_Quat(4), L_Grip(1), R_Pos(3), R_Quat(4), R_Grip(1)]
    component_map_obs = {
        'gripper': {'left': [7], 'right': [15], 'labels': ['Gripper']},
        'position': {'left': [0, 1, 2], 'right': [8, 9, 10], 'labels': ['X', 'Y', 'Z']},
        'rotation': {'left': [3, 4, 5, 6], 'right': [11, 12, 13, 14], 'labels': ['QW', 'QX', 'QY', 'QZ']},
    }
    
    # Select the appropriate mapping
    if data_source == 'action':
        component_map = component_map_action
        data_key = 'actions'
    elif data_source == 'observation':
        component_map = component_map_obs
        data_key = 'observations'
    else:
        raise ValueError(f"data_source must be 'action' or 'observation', got {data_source}")
    
    # Validate component
    if component not in component_map:
        raise ValueError(f"component must be one of {list(component_map.keys())}, got {component}")
    
    comp_info = component_map[component]
    left_indices = comp_info['left']
    right_indices = comp_info['right']
    labels = comp_info['labels']
    
    # Number of subplots needed
    n_dims = len(left_indices)
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_dims, 2, figsize=(14, 4 * n_dims))
    if n_dims == 1:
        axes = axes.reshape(1, -1)
    
    # Colors for different episodes
    colors = plt.cm.tab10(np.linspace(0, 1, min(len(episode_list), 10)))
    if len(episode_list) > 10:
        colors = plt.cm.viridis(np.linspace(0, 1, len(episode_list)))
    
    # Plot each episode
    for color_idx, ep_idx in enumerate(episode_list):
        start, end = episode_boundaries[ep_idx]
        episode_data = transitions[start:end]
        
        # Extract data
        if data_source == 'action':
            data_array = np.vstack([t[data_key] for t in episode_data])
        else:  # observation
            data_array = np.vstack([t[data_key]['state'].reshape(-1) for t in episode_data])
        
        steps = np.arange(len(episode_data))
        color = colors[color_idx % len(colors)]
        
        # Plot each dimension
        for dim_idx in range(n_dims):
            # Left robot
            left_ax = axes[dim_idx, 0]
            left_data = data_array[:, left_indices[dim_idx]]
            print(f"Plotting Left Robot - {component.capitalize()} - {labels[dim_idx]} for Episode {ep_idx} and left_data is {left_data}")
            left_ax.plot(steps, left_data, color=color, label=f'Ep {ep_idx}', linewidth=1.5, alpha=0.7)
            left_ax.set_xlabel('Step', fontsize=11)
            left_ax.set_ylabel(f'{labels[dim_idx]}', fontsize=11)
            left_ax.set_title(f'Left Robot - {component.capitalize()} - {labels[dim_idx]}', fontsize=12, fontweight='bold')
            left_ax.grid(True, alpha=0.3)
            if len(episode_list) <= 10:  # Only show legend if not too crowded
                left_ax.legend(fontsize=9)
            
            # Right robot
            right_ax = axes[dim_idx, 1]
            right_data = data_array[:, right_indices[dim_idx]]
            print(f"Plotting Right Robot - {component.capitalize()} - {labels[dim_idx]} for Episode {ep_idx} and right_data is {right_data}")
            right_ax.plot(steps, right_data, color=color, label=f'Ep {ep_idx}', linewidth=1.5, alpha=0.7)
            right_ax.set_xlabel('Step', fontsize=11)
            right_ax.set_ylabel(f'{labels[dim_idx]}', fontsize=11)
            right_ax.set_title(f'Right Robot - {component.capitalize()} - {labels[dim_idx]}', fontsize=12, fontweight='bold')
            right_ax.grid(True, alpha=0.3)
            if len(episode_list) <= 10:
                right_ax.legend(fontsize=9)
    
    fig.suptitle(f'{component.capitalize()} Data from {data_source.capitalize()} - Episodes: {episode_list if len(episode_list) <= 5 else f"{len(episode_list)} episodes"}', 
                 fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout()
    plt.show()
    
    print(f"Visualized {component} data from {data_source} for {len(episode_list)} episode(s)")
    



def binarize_gripper_deltas(trigger_threshhold, input_file_path: str = None, output_file_path: str = None):
    # Update to your scaled/epsilon dataset path
    # input_file_path = "/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/same_cube_pose/sim_dataset_replayed_and_verified_delta_act_CLEANED_no_jump_regenerate_FINAL_list_corrected_masks_delta_gripper_scaled_005_regenerated_states_epsilon.pkl"
    # output_file_path = "/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/same_cube_pose/sim_dataset_binarized_gripper.pkl"
 
    # === PARAMETERS ===
    # Reduced threshold to catch the "weak" closes we saw in your plot (Ep 1 only went to -0.15)
    TRIGGER_THRESHOLD = trigger_threshhold
    
    # Force strong commands
    STATE_OPEN = 1.0
    STATE_CLOSED = -1.0
    
    with open(input_file_path, 'rb') as f:
        transitions = pickle.load(f)

    print(f"Processing {len(transitions)} transitions...")
    print(f"Aggressively fixing 'Zero Delta' trap. Threshold: +/- {TRIGGER_THRESHOLD}")
    
    # Assume robot starts OPEN
    current_state_left = STATE_OPEN
    current_state_right = STATE_OPEN
    
    overwritten_count = 0
    state_changes = 0

    for i, trans in enumerate(tqdm(transitions)):
        actions = trans['actions'].copy()
        
        # --- RIGHT GRIPPER (Index 13) ---
        val_right = actions[13]
        
        # LOGIC:
        # 1. If we see a distinct movement (Trigger), update our State Memory.
        # 2. ALWAYS overwrite the action with the State Memory.
        #    This turns spikes into solid blocks.
        
        if val_right < -TRIGGER_THRESHOLD:
            current_state_right = STATE_CLOSED
            state_changes += 1
        elif val_right > TRIGGER_THRESHOLD:
            current_state_right = STATE_OPEN
            state_changes += 1
            
        # FORCE the action to match the state. 
        # We replace the noisy delta (e.g. -0.4 or 0.0) with the pure intent (-1.0)
        actions[13] = current_state_right
        
        # --- LEFT GRIPPER (Index 6) ---
        val_left = actions[6]
        if val_left < -TRIGGER_THRESHOLD:
            current_state_left = STATE_CLOSED
        elif val_left > TRIGGER_THRESHOLD:
            current_state_left = STATE_OPEN
        actions[6] = current_state_left

        trans['actions'] = actions

    print(f"Saving to {output_file_path}...")
    with open(output_file_path, 'wb') as f:
        pickle.dump(transitions, f)
        
    print(f"Finished. Detected {state_changes} open/close events.")

    # --- VISUALIZATION VERIFICATION ---
    print("\nVerifying the fix on Episode 0...")
    ep0_right_gripper = []
    # Extract just the first episode (roughly 150 steps based on your plots)
    for i in range(150):
        ep0_right_gripper.append(transitions[i]['actions'][13])
    
    plt.figure(figsize=(10, 4))
    plt.plot(ep0_right_gripper, label="Corrected Signal")
    plt.title("Episode 0 Gripper Action (Should look like a square wave)")
    plt.xlabel("Step")
    plt.ylabel("Action (-1.0 = Close, 1.0 = Open)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    print("Check the plot. You should see a solid block of -1.0 starting around step 60.")


def compute_action_ranges(file_path: str):
    """
    Load dataset and compute min/max ranges for each action dimension.
    Useful for setting better action space bounds for random exploration.
    
    Args:
        file_path: Path to pickle file containing transitions
        
    Prints:
        Two lists: action_mins and action_maxs (each 14D)
    """
    print(f"\n{'='*70}")
    print(f"Computing action ranges from: {file_path}")
    print(f"{'='*70}\n")
    
    # Load transitions
    transitions = load_transitions(file_path)
    print(f"Loaded {len(transitions)} transitions")
    print(f"action dimensions: {transitions[0]['actions'].shape}")
    # Collect all actions (flatten across episodes)
    all_actions = []
    for trans in transitions:
        action = trans['actions']
        # Handle different shapes
        if action.ndim > 1:
            action = action.reshape(-1)
        all_actions.append(action)
    
    # Stack into (N, 14) array
    all_actions = np.array(all_actions)
    print(f"Action array shape: {all_actions.shape}")
    
    # Compute min/max per dimension
    action_mins = np.min(all_actions, axis=0)
    action_maxs = np.max(all_actions, axis=0)
    action_means = np.mean(all_actions, axis=0)
    action_stds = np.std(all_actions, axis=0)
    
    # Print results
    print(f"\n{'='*70}")
    print("ACTION RANGE STATISTICS (14D)")
    print(f"{'='*70}")
    print(f"Dimension labels: [L_pos(3), L_rot(3), L_grip(1), R_pos(3), R_rot(3), R_grip(1)]")
    print(f"\nMIN values:")
    print(f"  {action_mins.tolist()}")
    print(f"\nMAX values:")
    print(f"  {action_maxs.tolist()}")
    print(f"\nMEAN values:")
    print(f"  {action_means.tolist()}")
    print(f"\nSTD values:")
    print(f"  {action_stds.tolist()}")
    
    # Pretty print per dimension
    print(f"\n{'='*70}")
    print("PER-DIMENSION BREAKDOWN:")
    print(f"{'='*70}")
    dim_labels = [
        "L_pos_x", "L_pos_y", "L_pos_z",
        "L_rot_x", "L_rot_y", "L_rot_z",
        "L_grip",
        "R_pos_x", "R_pos_y", "R_pos_z",
        "R_rot_x", "R_rot_y", "R_rot_z",
        "R_grip"
    ]
    
    for i, label in enumerate(dim_labels):
        print(f"{i:2d}. {label:10s}: min={action_mins[i]:7.4f}, max={action_maxs[i]:7.4f}, "
              f"mean={action_means[i]:7.4f}, std={action_stds[i]:7.4f}")
    
    # Suggest action space bounds for Gym
    print(f"\n{'='*70}")
    print("SUGGESTED ACTION SPACE FOR RANDOM SAMPLING:")
    print(f"{'='*70}")
    print(f"action_space = spaces.Box(")
    print(f"    low=np.array({action_mins.tolist()}),")
    print(f"    high=np.array({action_maxs.tolist()}),")
    print(f"    shape=(14,),")
    print(f"    dtype=np.float32")
    print(f")")
    print(f"\n{'='*70}\n")
    
    return action_mins, action_maxs


def clip_and_clean_dataset_within_bounds(
    input_file_path: str,
    output_file_path: str,
    left_bounds: tuple,
    right_bounds: tuple
):
    """
    Clips position actions in the dataset to strictly adhere to Cartesian bounds.
    
    Args:
        input_file_path: Path to source .pkl dataset.
        output_file_path: Path to save clipped dataset.
        left_bounds: Tuple (min_array, max_array) for Left Robot [x, y, z].
        right_bounds: Tuple (min_array, max_array) for Right Robot [x, y, z].
    """
    print(f"Loading dataset from {input_file_path}...")
    transitions = load_transitions(input_file_path)
    # episode_boundaries = get_episode_boundaries(transitions) # Not strictly needed for per-frame clipping, but good for validation
    
    print(f"Clipping {len(transitions)} transitions to bounds:")
    print(f"  Left:  {left_bounds[0]} to {left_bounds[1]}")
    print(f"  Right: {right_bounds[0]} to {right_bounds[1]}")

    print(f"shape of actions before clipping: {transitions[0]['actions'].shape}")

    # time.sleep(10)

    left_min, left_max = left_bounds
    right_min, right_max = right_bounds
    
    clipped_count = 0

    for trans in tqdm(transitions):
        # Ensure float32 for consistency
        actions = trans['actions'].astype(np.float32)
        
        # --- LEFT ROBOT (Indices 0, 1, 2) ---
        original_left = actions[0:3].copy()
        actions[0:3] = np.clip(actions[0:3], left_min, left_max)
        
        # --- RIGHT ROBOT (Indices 7, 8, 9) ---
        original_right = actions[7:10].copy()
        actions[7:10] = np.clip(actions[7:10], right_min, right_max)
        
        # Check if anything changed (for reporting)
        if not np.allclose(original_left, actions[0:3]) or not np.allclose(original_right, actions[7:10]):
            clipped_count += 1

        # Apply changes in place
        trans['actions'] = actions

    print(f"\n✂️  Clipped {clipped_count} frames that were out of bounds.")
    
    print(f"Saving cleaned dataset to {output_file_path}...")
    with open(output_file_path, 'wb') as f:
        pickle.dump(transitions, f)
    print("✅ Save complete.")




def visualize_absolute_gripper_trajectory(
    file_path: str,
    episode_indices: Union[int, List[int], None] = None,
    action_scale_gripper: float = 0.02
):
    """
    Reconstructs and visualizes the ABSOLUTE gripper trajectory by integrating
    delta actions starting from the initial observation.
    
    Logic mirrors the environment step function:
    1. Start with initial observation (Left: idx 7, Right: idx 15)
    2. Apply scaled delta (Action * 0.02)
    3. Clip between 0.0 and 0.044
    """
    # Load data (assuming these helper functions exist based on your previous code)
    transitions = load_transitions(file_path)
    episode_boundaries = get_episode_boundaries(transitions)
    
    # Handle episode_indices input
    if episode_indices is None:
        episode_list = list(range(len(episode_boundaries)))
    elif isinstance(episode_indices, int):
        episode_list = [episode_indices]
    else:
        episode_list = list(episode_indices)

    # INDICES based on your provided info:
    # Obs: [L_Pos(3), L_Quat(4), L_Grip(1)...] -> L_Grip is idx 7
    # Obs: [..., R_Pos(3), R_Quat(4), R_Grip(1)] -> R_Grip is idx 15
    OBS_IDX_L = 7
    OBS_IDX_R = 15
    
    # Acts: [L_Pos(3), L_Rot(3), L_Grip(1)...] -> L_Grip is idx 6
    # Acts: [..., R_Pos(3), R_Rot(3), R_Grip(1)] -> R_Grip is idx 13
    ACT_IDX_L = 6
    ACT_IDX_R = 13
    
    # Physics Limits from your step function
    GRIP_MIN = 0.0
    GRIP_MAX = 0.044

    # Setup Plot
    num_episodes = len(episode_list)
    fig, axes = plt.subplots(num_episodes, 1, figsize=(12, 4 * num_episodes), sharex=False)
    if num_episodes == 1:
        axes = [axes]

    for plot_idx, ep_idx in enumerate(episode_list):
        start_t, end_t = episode_boundaries[ep_idx]
        episode_data = transitions[start_t:end_t]
        
        # 1. Get Initial State from the very first observation of the episode
        # We need this anchor point to start adding deltas to.
        initial_obs = episode_data[0]['observations']

        raw_obs = episode_data[0]['observations']
        
        # Check if observation is a dict (Pixel Agent) or Array (State Agent)
        if isinstance(raw_obs, dict):
            # Try to find the state vector key
            if 'state' in raw_obs:
                state_vec = raw_obs['state']
            elif 'proprio' in raw_obs:
                state_vec = raw_obs['proprio']
            else:
                # Fallback: Print keys to debug
                raise KeyError(f"Could not find state vector in observation dict. Keys found: {list(raw_obs.keys())}")
        else:
            state_vec = raw_obs # It's already an array

        # 2. Squeeze the time dimension: (1, 16) -> (16,)
        state_vec = np.array(state_vec) # Ensure numpy
        if state_vec.ndim == 2 and state_vec.shape[0] == 1:
            state_vec = state_vec[0]
            
        # Initialize with the retrieved state
        curr_l = state_vec[OBS_IDX_L]
        curr_r = state_vec[OBS_IDX_R]
        
        # History arrays for plotting
        l_hist = [curr_l]
        r_hist = [curr_r]
        
        # 2. Replay the actions to reconstruction absolute position
        for t in episode_data:
            action = t['actions']
            
            # Extract Raw Deltas
            raw_delta_l = action[ACT_IDX_L]
            raw_delta_r = action[ACT_IDX_R]
            
            # Apply Scale (Your logic: multiply by 0.02)
            delta_l = raw_delta_l * action_scale_gripper
            delta_r = raw_delta_r * action_scale_gripper
            
            # Apply Update (Add to current)
            curr_l += delta_l
            curr_r += delta_r
            
            # Apply Clipping (Your logic: 0.0 to 0.044)
            curr_l = np.clip(curr_l, GRIP_MIN, GRIP_MAX)
            curr_r = np.clip(curr_r, GRIP_MIN, GRIP_MAX)
            
            l_hist.append(curr_l)
            r_hist.append(curr_r)
            
        # 3. Plotting
        ax = axes[plot_idx]
        steps = range(len(l_hist))
        
        # Left Gripper Line
        ax.plot(steps, l_hist, label='Left Gripper (Reconstructed)', color='blue', linewidth=2)
        # Right Gripper Line
        ax.plot(steps, r_hist, label='Right Gripper (Reconstructed)', color='orange', linewidth=2)
        
        # Add "Open" and "Closed" reference lines
        ax.axhline(y=GRIP_MAX, color='green', linestyle='--', alpha=0.5, label='Max Open (0.044)')
        ax.axhline(y=GRIP_MIN, color='red', linestyle='--', alpha=0.5, label='Closed (0.0)')
        
        ax.set_title(f'Episode {ep_idx}: Absolute Gripper Trajectory (Integrated Deltas)')
        ax.set_ylabel('Position (m)')
        ax.set_xlabel('Time Steps')
        ax.set_ylim(-0.01, 0.055) # Give a little margin around limits
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # 1. convert ac to delta act except gippers
    # file_path = "/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/random_cube_pose/cleaned/merged_pkl_data_action_in_world_frame_static_filtered_subsampled_action_shifted_regenerated_deleted_some_clipped_inBbox.pkl"
    # output_file_path = "/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/random_cube_pose/cleaned/merged_pkl_data_action_in_world_frame_static_filtered_subsampled_action_shifted_regenerated_deleted_some_clipped_inBbox_fulldelta_act.pkl"
    # convert_actions_to_delta_act_including_gripper(file_path=file_path, output_file_path=output_file_path)
    # 2. check frame 0 consistency
    # file_path = "/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/random_cube_pose/cleaned/merged_pkl_data_action_in_world_frame_static_filtered_subsampled_action_shifted_regenerated_deleted_some_clipped_inBbox_fulldelta_act.pkl"
    # check_frame_1_consistency(file_path=file_path)
    # inspect_start_states(file_path=file_path)
    # 3. prune first frame and save if needed
    # prune_first_frame_and_save()
    # 4. delete specific episodes and save
    # delete_specific_episodes_and_save()
    # 5. get per episode stats with steps
    
    # get_per_episode_stats_with_steps(file_path=file_path)
    # ACTION_SCALE = np.array([0.025, 0.1, 0.02]) # based on above output pos, rot, grip
    # file_path = "/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/random_cube_pose/cleaned/merged_pkl_data_action_in_world_frame_static_filtered_subsampled_action_shifted_regenerated_deleted_some_clipped_inBbox_fulldelta_act.pkl"
    # output_file_path = "/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/random_cube_pose/cleaned/merged_pkl_data_action_in_world_frame_static_filtered_subsampled_action_shifted_regenerated_deleted_some_clipped_inBbox_fulldelta_act_scaled_act.pkl"
    # update_action_scale(file_path, output_file_path, ACTION_SCALE)
    # visualize_data(output_file_path, 'gripper', episode_indices=[0,1,24], data_source='action')


    # ===================================================================
    # Gripper delta action conversion steps
    # input_file_path = "/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/random_cube_pose/cleaned/merged_pkl_data_action_in_world_frame_static_filtered_subsampled_action_shifted_regenerated_deleted_some_clipped_inBbox_fulldelta_act_scaled_act.pkl"
    # output_file_path = "/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/random_cube_pose/cleaned/merged_pkl_data_action_in_world_frame_static_filtered_subsampled_action_shifted_regenerated_deleted_some_clipped_inBbox_fulldelta_act_scaled_act_epsilon.pkl"
    # add_epsilon_to_near_edge_actions(input_file_path=input_file_path, output_file_path=output_file_path) # /home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/same_cube_pose/sim_dataset_replayed_and_verified_delta_act_CLEANED_no_jump_regenerate_FINAL_list_corrected_masks_delta_gripper_scaled_no_edge.pkl
    # visualize_data(input_file_path, 'gripper', episode_indices=[0,1,24], data_source='action')
    # if uou look at graph, zoom in when gripper is closing the spikes and you can choose a vlue. ex: TRIGGER_THRESHOLD = 0.005
    # input_file_path = "/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/random_cube_pose/cleaned/merged_pkl_data_action_in_world_frame_static_filtered_subsampled_action_shifted_regenerated_deleted_some_clipped_inBbox_fulldelta_act_scaled_act.pkl"
    # output_file_path = "/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/random_cube_pose/cleaned/merged_pkl_data_action_in_world_frame_static_filtered_subsampled_action_shifted_regenerated_deleted_some_clipped_inBbox_fulldelta_act_scaled_act_binarized.pkl"
    # TRIGGER_THRESHOLD = 0.005
    # binarize_gripper_deltas(trigger_threshhold=TRIGGER_THRESHOLD, input_file_path=input_file_path, output_file_path=output_file_path)
    # visualize_data(output_file_path, 'gripper', episode_indices=[i for i in range(10, 20)], data_source='action')
    # ===================================================================
    
    # Compute action ranges for better random exploration
    # input_file_path = "/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/random_cube_pose/cleaned/merged_pkl_data_action_in_world_frame_static_filtered_subsampled_action_shifted_regenerated_deleted_some_clipped_inBbox_fulldelta_act_scaled_act_binarized_epsilon_regenerated_deleted_some.pkl"
    # original_ds_with_act_in_sim_world_frame = "/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/random_cube_pose/cleaned/merged_pkl_data_action_in_world_frame_clipped_inBbox.pkl"
    # compute_action_ranges(input_file_path)
    # pass

    # input_file_path = "/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/random_cube_pose/cleaned/merged_pkl_data_action_in_world_frame_static_filtered_subsampled_action_shifted_regenerated_deleted_some_clipped_inBbox_fulldelta_act_scaled_act_binarized.pkl"
    # output_file_path = "/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/random_cube_pose/cleaned/merged_pkl_data_action_in_world_frame_static_filtered_subsampled_action_shifted_regenerated_deleted_some_clipped_inBbox_fulldelta_act_scaled_act_binarized_epsilon.pkl"
    # add_epsilon_to_near_edge_actions(input_file_path=input_file_path, output_file_path=output_file_path) # /home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/same_cube_pose/sim_dataset_replayed_and_verified_delta_act_CLEANED_no_jump_regenerate_FINAL_list_corrected_masks_delta_gripper_scaled_no_edge.pkl

    # Just Visualization
    # file_path = "/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/random_cube_pose/cleaned/merged_pkl_data_action_in_world_frame_static_filtered_subsampled_action_shifted_regenerated_deleted_some_fulldelta_act_scaled_act_epsilon_binarized_regenerated_deleted_episodes_epsilon.pkl"
    # unscaled_file_path = "/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/random_cube_pose/cleaned/merged_pkl_data_action_in_world_frame_static_filtered_subsampled_action_shifted_regenerated_deleted_some_fulldelta_act_scaled_act_epsilon.pkl"
    # visualize_data("/home/qte9489/personal_abhi/temp/hil-serl/trossen_arm_mujoco/trossen_arm_mujoco/dataset_utils/delta_action_ds/franka_lift_cube_image_20_trajs.pkl", 'gripper', episode_indices=[0], data_source='action')



    # ================== RL SPECIFIC CLEANING: CLIP AND CLEAN WITHIN BOUNDS ==================
    # ------->>>>>> Do this First and then follow the gripper delta action conversion steps next <<<<<<<--------
    # Define Bounds. MAKE SURE SAME AS THE ONE"S YOU VISUALIZED EARLIER in ./trossen_arm_mujoco/trossen_arm_mujoco/dataset_utils/visualize_bbox.py
    # LEFT_CARTESIAN_BOUNDS = (
    #     np.array([-0.25, -0.13, 0.05]),  # MINS
    #     np.array([-0.17,  0.10, 0.25])   # MAXS
    # )

    # RIGHT_CARTESIAN_BOUNDS = (
    #     np.array([-0.10, -0.10, 0.0025]), # MINS
    #     np.array([ 0.25,  0.32, 0.35])   # MAXS
    # )
    
    # original_ds_with_act_in_sim_world_frame = "/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/random_cube_pose/cleaned/merged_pkl_data_action_in_world_frame_static_filtered_subsampled_action_shifted_regenerated_deleted_some.pkl"
    # original_ds_clipped_within_bounds = "/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/random_cube_pose/cleaned/merged_pkl_data_action_in_world_frame_static_filtered_subsampled_action_shifted_regenerated_deleted_some_clipped_inBbox.pkl"

    # clip_and_clean_dataset_within_bounds(original_ds_with_act_in_sim_world_frame, original_ds_clipped_within_bounds, LEFT_CARTESIAN_BOUNDS, RIGHT_CARTESIAN_BOUNDS)


    # Visualize binary gripper values convert back to origiginal physical scale as per env.step function
    file_path = "/home/qte9489/personal_abhi/temp/hil-serl/train_data_sets/test1/only_right_arm_data_regenerated_deleted.pkl"
    # visualize_absolute_gripper_trajectory(file_path, episode_indices=[0,1,2,3,4,5], action_scale_gripper=0.005)
    visualize_data(file_path, 'gripper', episode_indices=[0], data_source='action')
