import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.spatial.transform import Rotation as R
from typing import Union, List
from pathlib import Path
import copy
from trossen_arm_mujoco.ee_transforms import action_14d_robot_to_world_aa


def quaternion_to_angle_axis(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to angle-axis representation.
    
    Args:
        quat: Quaternion [w, x, y, z]
    
    Returns:
        Angle-axis vector [rx, ry, rz]
    """
    # Convert to scipy format [x, y, z, w]
    q_scipy = np.array([quat[1], quat[2], quat[3], quat[0]])
    rot = R.from_quat(q_scipy)
    return rot.as_rotvec()

def wrap_angle(delta_rot: np.ndarray) -> np.ndarray:
    """
    Wraps angle differences to be between -pi and pi.
    This prevents the robot from doing a 360 spin for a small move.
    """
    return (delta_rot + np.pi) % (2 * np.pi) - np.pi

# --- Your existing data loading functions ---
def load_transitions(file_path: str) -> list:
    print(f"Loading data from {file_path}...")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    # Handle different formats
    if isinstance(data, dict) and 'transitions' in data:
        return data['transitions']
    elif isinstance(data, list):
        # Check if it's a nested list (common bug from regenerate function)
        if len(data) > 0 and isinstance(data[0], list):
            print(f"Warning: Detected nested list structure. Unwrapping...")
            return data[0]
        # Check if first element looks like a transition dict
        elif len(data) > 0 and isinstance(data[0], dict) and 'actions' in data[0]:
            return data
        else:
            raise ValueError(f"List format unexpected. First element type: {type(data[0]) if len(data) > 0 else 'empty'}")
    else:
        raise ValueError(f"Unexpected data format in pickle file. Type: {type(data)}")

def get_episode_boundaries(transitions: list) -> list:
    boundaries = []
    start = 0
    for i, t in enumerate(transitions):
        if t['dones']:
            boundaries.append((start, i + 1))
            start = i + 1
    if start < len(transitions):
        boundaries.append((start, len(transitions)))
    return boundaries

def fix_obs_structure(raw_obs):
    """
    Transforms an observation dictionary to match the target SERL format:
    
    1. Top-level keys: 'state', 'cam_high', 'cam_low', ..., 'full_observation'.
    2. 'state' shape: (1, 16).
    3. Camera shapes: (1, 128, 128, 3).
    4. 'full_observation': Kept as a dict, but 'images' key is REMOVED from it.
    """
    if not isinstance(raw_obs, dict):
        return {}

    new_obs = {}

    # --- 1. Fix State: Ensure (1, 16) ---
    # It might be in 'state' or nested in 'full_observation' -> 'state'
    # We prioritize the top-level 'state' if it exists.
    if 'state' in raw_obs:
        state = np.array(raw_obs['state'], dtype=np.float64)
        if state.ndim == 1:
            state = state[None, :]  # (16,) -> (1, 16)
        new_obs['state'] = state

    # --- 2. Fix Images: Flatten to top level & fix shape ---
    # We look for images in 'images' dict or top level
    source_images = {}
    
    # Check top-level 'images' dict
    if 'images' in raw_obs and isinstance(raw_obs['images'], dict):
        source_images.update(raw_obs['images'])
            
    # Check top-level keys (e.g. cam_high already flat)
    for k, v in raw_obs.items():
        if k.startswith('cam_'):
            source_images[k] = v

    # Add flattened images to new_obs with shape (1, 128, 128, 3)
    for cam_name, img_data in source_images.items():
        img = np.array(img_data, dtype=np.uint8)
        if img.ndim == 3:
            img = img[None, ...]  # (H, W, 3) -> (1, H, W, 3)
        new_obs[cam_name] = img

    # --- 3. Handle full_observation (Keep it, but remove images) ---
    if 'full_observation' in raw_obs and isinstance(raw_obs['full_observation'], dict):
        # Deep copy so we don't modify the original during iteration if needed
        full_obs_clean = copy.deepcopy(raw_obs['full_observation'])
        
        # POP the images dict
        if 'images' in full_obs_clean:
            del full_obs_clean['images']
            
        new_obs['full_observation'] = full_obs_clean

    return new_obs

def fix_pickle_files(folder_path: str):
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Error: Folder {folder} not found.")
        return

    pkl_files = sorted(list(folder.glob("*.pkl")))
    if not pkl_files:
        print(f"No .pkl files found in {folder}")
        return

    print(f"Processing {len(pkl_files)} files in {folder}...")
    
    for i, pkl_file in enumerate(pkl_files):
        print(f"[{i+1}/{len(pkl_files)}] Fixing {pkl_file.name}...", end=" ")
        
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)

            if not isinstance(data, list):
                print("Skipped (not a list)")
                continue

            fixed_transitions = []
            
            for t in data:
                new_t = {}

                # 1. Fix Observations
                raw_obs = t.get('observations', t.get('observation', {}))
                new_t['observations'] = fix_obs_structure(raw_obs)

                # 2. Fix Next Observations
                raw_next = t.get('next_observations', t.get('next_observation', {}))
                new_t['next_observations'] = fix_obs_structure(raw_next)

                # 3. Fix Actions
                raw_act = t.get('actions', t.get('action'))
                if raw_act is not None:
                    act = np.array(raw_act, dtype=np.float32)
                    if act.ndim > 1:
                        act = act.flatten()
                    new_t['actions'] = act #  of shape (14, )

                # 4. Fix Scalars
                new_t['rewards'] = float(t.get('rewards', t.get('reward', 0.0)))
                new_t['dones'] = bool(t.get('dones', t.get('done', False)))

                if 'masks' in t:
                    new_t['masks'] = float(t['masks'])
                else:
                    new_t['masks'] = 0.0 if new_t['dones'] else 1.0
                
                # 5. Convert actions from leader robot coord frame aa to sim world frame aa
                # t["actions"] is in robot frame aa
                new_t['actions'] = action_14d_robot_to_world_aa(t['actions'])
                fixed_transitions.append(new_t)

            # Overwrite file
            with open(pkl_file, 'wb') as f:
                pickle.dump(fixed_transitions, f)
            
            print("Done.")

        except Exception as e:
            print(f"Error: {e}")


def pretty_print_obs(obs, indent=0):
    pad = " " * indent
    print("\n=== Observation Dump ===" if indent == 0 else "")
    for k, v in obs.items():
        print(f"\n{pad}Key: {k}")
        if isinstance(v, dict):
            print(f"{pad}  Type: dict")
            # recurse into nested dict
            pretty_print_obs(v, indent=indent+4)
        elif isinstance(v, np.ndarray):
            print(f"{pad}  Type: {type(v)}")
            print(f"{pad}  Shape: {v.shape}")
            print(f"{pad}  Dtype: {v.dtype}")
            if "cam" in k.lower():
                print(f"{pad}  Values: <skipped for camera data>")
            else:
                print(f"{pad}  Values:\n{pad}{v}")
        else:
            print(f"{pad}  Type: {type(v)}")
            print(f"{pad}  Value:\n{pad}{v}")
    if indent == 0:
        print("\n========================\n")

if __name__ == "__main__":
    # Example usage for conversion
    folder_to_convert = "/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/random_cube_pose/serl_pkl_files"
    fix_pickle_files(folder_to_convert)

    # file_path = "/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/same_cube_pose/sim_dataset_binarized_gripper.pkl"
    # pretty_print_obs(load_transitions(file_path)[0])