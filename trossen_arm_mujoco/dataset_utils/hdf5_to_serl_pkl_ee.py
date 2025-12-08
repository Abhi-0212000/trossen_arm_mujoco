#!/usr/bin/env python3
"""
HDF5 to SERL Pickle Converter (EE Control Mode)
================================================

Converts HDF5 episode recordings from sim_data_collector to SERL pickle format.
This is for END-EFFECTOR (EE) control mode only.

NO normalization, NO clipping - just raw data conversion.

=== ROTATION REPRESENTATIONS ===

This script supports multiple rotation representations for both Action and State:

1. **angle_axis** (Default, 3D):
   - Compact 3D vector where direction is axis and magnitude is angle (radians).
   - Good for small rotations, but has singularities at 0 and 2pi.
   - Used by default in Trossen teleop.

2. **quat** (Quaternion, 4D):
   - [w, x, y, z] or [x, y, z, w] (MuJoCo uses [w, x, y, z]).
   - No singularities, continuous.
   - Double cover issue (q and -q represent same rotation) - handled by normalization.

3. **euler** (Euler Angles, 3D):
   - [roll, pitch, yaw].
   - Intuitive but suffers from Gimbal Lock.
   - Not recommended for learning complex 3D rotations.

4. **ortho6d** (6D Continuous, 6D):
    Justification: "On the Continuity of Rotation Representations in Neural Networks" by Zhou et al. (CVPR 2019).
   - First two columns of the rotation matrix (flattened).
   - Continuous, unique, no singularities.
   - Best for learning 3D rotations (Zhou et al., CVPR 2019).
   - Computed from Quaternions or Rotation Matrices.

=== DATA FORMAT ===

ACTION (Variable Dim) - What the policy outputs:
    [L_pos(3), L_rot(N), L_grip(1), R_pos(3), R_rot(N), R_grip(1)]
    
    - N depends on rotation mode: 3 (angle_axis/euler), 4 (quat), 6 (ortho6d).
    - Total Dim: 8 (3+3+1+...) to 14 (3+6+1+...) per arm.

STATE (Variable Dim) - What the sim robots report back:
    [L_pos(3), L_rot(N), L_grip(1), R_pos(3), R_rot(N), R_grip(1)]
    
    - Matches action structure for consistency.

IMAGES (4 cameras, 128x128x3 each):
    - cam_high, cam_low, cam_left_wrist, cam_right_wrist


Position & Gripper:
    Always the same. We just take L_pos, R_pos and L_grip, R_grip directly. No variations.
Rotation (The only thing that changes):
    Angle-Axis (14D total):
        Action: Taken directly from action_angle_axis in HDF5.
        State: Taken directly from robot0_eef_angle_axis in HDF5.
    Quaternion (16D total):
        Action: Taken directly from action in HDF5.
        State: Taken directly from robot0_eef_quat in HDF5.
    Euler (14D total):
        Action: Not available in HDF5. The script currently raises an error because we would need to convert action_angle_axis -> Euler mathematically (which is messy due to gimbal lock).
        State: Taken directly from robot0_eef_euler in HDF5.
    Ortho6D (20D total):
        Action: We take action (Quat) -> Convert to Ortho6D on the fly.
        State: We take robot0_eef_quat -> Convert to Ortho6D on the fly.


=== USAGE ===

    # Default (Angle-Axis)
    python -m trossen_arm_mujoco.dataset_utils.hdf5_to_serl_pkl_ee ./sim_recordings_ee ./output
    
    # Use Quaternions (16D total action)
    python -m trossen_arm_mujoco.dataset_utils.hdf5_to_serl_pkl_ee ./sim_recordings_ee ./output --rot_mode quat
    
    # Use 6D Rotation (20D total action)
    python -m trossen_arm_mujoco.dataset_utils.hdf5_to_serl_pkl_ee ./sim_recordings_ee ./output --rot_mode ortho6d
"""

import h5py
import numpy as np
import pickle
import argparse
import cv2
from pathlib import Path
import logging
from scipy.spatial.transform import Rotation as R

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def quat_to_ortho6d(quat):
    """
    Convert quaternion [w, x, y, z] to Ortho6D [r11, r21, r31, r12, r22, r32].
    
    Args:
        quat: (N, 4) array in [w, x, y, z] format (MuJoCo standard)
    Returns:
        ortho6d: (N, 6) array
    """
    # Scipy expects [x, y, z, w]
    # MuJoCo provides [w, x, y, z]
    quat_scipy = np.concatenate([quat[:, 1:], quat[:, 0:1]], axis=1)
    
    rot = R.from_quat(quat_scipy)
    matrix = rot.as_matrix()  # (N, 3, 3)
    
    # Take first two columns: R[:, 0] and R[:, 1]
    # Flatten them: [r11, r21, r31, r12, r22, r32]
    ortho6d = matrix[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)
    return ortho6d


def load_episode(h5_path: Path, rot_mode: str = 'angle_axis') -> dict:
    """Load a single HDF5 episode file."""
    with h5py.File(h5_path, 'r') as f:
        obs = f['observations']
        
        # --- LOAD RAW DATA ---
        # Positions (L, R)
        eef_pos = np.array(obs['robot0_eef_pos'])           # (N, 6) [L_pos(3), R_pos(3)]
        
        # Grippers (L, R)
        gripper = np.array(obs['robot0_gripper_qpos'])      # (N, 2) [L_grip, R_grip]
        
        # Rotations (L, R) - Load all available to convert
        # Note: 'action' usually contains quat, 'action_angle_axis' contains aa
        
        # --- PROCESS ROTATION (ACTION & STATE) ---
        
        if rot_mode == 'angle_axis':
            # Action
            if 'action_angle_axis' in f:
                actions = np.array(f['action_angle_axis'])  # (N, 14)
            else:
                # Fallback: Convert from quat action if needed, or warn
                logger.warning(f"  No action_angle_axis, using raw action (check format!)")
                actions = np.array(f['action'])
            
            # State
            eef_rot = np.array(obs['robot0_eef_angle_axis']) # (N, 6)
            
        elif rot_mode == 'quat':
            # Action
            if 'action' in f:
                actions = np.array(f['action'])  # (N, 16) usually [pos, quat, grip...]
            else:
                raise ValueError("Action (quat) not found in HDF5")
                
            # State
            eef_rot = np.array(obs['robot0_eef_quat'])       # (N, 8) [L_quat(4), R_quat(4)]
            
        elif rot_mode == 'euler':
            # Action: Need to convert or find euler action (usually not stored directly)
            # For now, let's assume we convert from angle_axis action if available
            if 'action_angle_axis' in f:
                # Extract AA -> Convert to Euler
                raw_act = np.array(f['action_angle_axis'])
                # ... conversion logic would go here ...
                # For simplicity, let's use state euler for now or raise error if not implemented
                raise NotImplementedError("Euler action conversion not yet implemented fully.")
            
            # State
            eef_rot = np.array(obs['robot0_eef_euler'])      # (N, 6)
            
        elif rot_mode == 'ortho6d':
            # Action: Convert from Quat Action
            if 'action' in f:
                raw_act = np.array(f['action']) # (N, 16) [L_p(3), L_q(4), L_g(1), R_p(3), R_q(4), R_g(1)]
                
                # Extract components
                l_pos = raw_act[:, 0:3]
                l_quat = raw_act[:, 3:7]
                l_grip = raw_act[:, 7:8]
                
                r_pos = raw_act[:, 8:11]
                r_quat = raw_act[:, 11:15]
                r_grip = raw_act[:, 15:16]
                
                # Convert Quats to Ortho6D
                l_ortho = quat_to_ortho6d(l_quat)
                r_ortho = quat_to_ortho6d(r_quat)
                
                # Reassemble: [L_p(3), L_o(6), L_g(1), R_p(3), R_o(6), R_g(1)]
                actions = np.concatenate([l_pos, l_ortho, l_grip, r_pos, r_ortho, r_grip], axis=1)
            else:
                raise ValueError("Action (quat) not found for Ortho6D conversion")
            
            # State: Convert from Quat State
            raw_quat = np.array(obs['robot0_eef_quat']) # (N, 8) [L_q(4), R_q(4)]
            l_ortho_s = quat_to_ortho6d(raw_quat[:, 0:4])
            r_ortho_s = quat_to_ortho6d(raw_quat[:, 4:8])
            eef_rot = np.concatenate([l_ortho_s, r_ortho_s], axis=1) # (N, 12)

        else:
            raise ValueError(f"Unknown rotation mode: {rot_mode}")

        # --- ASSEMBLE STATE ---
        # [L_pos, L_rot, L_grip, R_pos, R_rot, R_grip]
        
        # Split rot into L/R
        rot_dim = eef_rot.shape[1] // 2
        l_rot = eef_rot[:, :rot_dim]
        r_rot = eef_rot[:, rot_dim:]
        
        state = np.concatenate([
            eef_pos[:, :3],   # L_pos (3)
            l_rot,            # L_rot (N)
            gripper[:, :1],   # L_grip (1)
            eef_pos[:, 3:],   # R_pos (3)
            r_rot,            # R_rot (N)
            gripper[:, 1:],   # R_grip (1)
        ], axis=1)
        
        # Images: resize from 480x640 to 128x128
        images = {}
        if 'images' in obs:
            for cam_name in obs['images'].keys():
                img_data = np.array(obs['images'][cam_name])  # (N, 480, 640, 3)
                resized = np.array([
                    cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
                    for img in img_data
                ])
                images[cam_name] = resized  # (N, 128, 128, 3)
        
        return {
            'name': h5_path.stem,
            'actions': actions,
            'state': state,
            'images': images
        }


def convert_to_serl(episodes: list) -> list:
    """Convert episodes to SERL transition format."""
    transitions = []
    
    for ep in episodes:
        N = len(ep['actions'])
        state_dim = ep['state'].shape[1]
        
        # Rewards: 0 everywhere, 1 at end
        rewards = np.zeros(N, dtype=np.float32)
        rewards[-1] = 1.0
        
        # Dones: False everywhere, True at end
        dones = np.zeros(N, dtype=bool)
        dones[-1] = True
        
        for t in range(N):
            # Current observation
            # Add time dimension (1,) for SERL ChunkingWrapper compatibility
            curr_state = ep['state'][t][np.newaxis, ...]  # (1, state_dim)
            curr_images = {k: v[t][np.newaxis, ...] for k, v in ep['images'].items()}  # (1, 128, 128, 3)
            
            # Next observation
            next_t = t if t == N - 1 else t + 1
            next_state = ep['state'][next_t][np.newaxis, ...]
            next_images = {k: v[next_t][np.newaxis, ...] for k, v in ep['images'].items()}
            
            curr_obs = {'state': curr_state, **curr_images}
            next_obs = {'state': next_state, **next_images}
            
            transitions.append({
                'observations': curr_obs,
                'next_observations': next_obs,
                'actions': ep['actions'][t],
                'rewards': rewards[t],
                'masks': 1.0,
                'dones': dones[t]
            })
    
    return transitions


def main():
    parser = argparse.ArgumentParser(
        description="Simple HDF5 to SERL pickle converter (no normalization)"
    )
    parser.add_argument("input", type=str, help="Input folder with HDF5 episodes")
    parser.add_argument("output", type=str, help="Output folder for pickle file")
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Max episodes to load (for testing)")
    parser.add_argument("--rot_mode", type=str, default="angle_axis",
                        choices=["angle_axis", "quat", "euler", "ortho6d"],
                        help="Rotation representation for Action and State")
    
    args = parser.parse_args()
    
    input_folder = Path(args.input)
    output_folder = Path(args.output)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Find HDF5 files
    h5_files = sorted(list(input_folder.glob("*.hdf5")) + list(input_folder.glob("*.h5")))
    
    if not h5_files:
        logger.error(f"❌ No HDF5 files found in {input_folder}")
        return 1
    
    if args.max_episodes:
        h5_files = h5_files[:args.max_episodes]
    
    # Load episodes
    logger.info(f"📂 Loading {len(h5_files)} episodes from {input_folder}")
    logger.info(f"   Rotation Mode: {args.rot_mode}")
    
    episodes = []
    for h5_path in h5_files:
        try:
            ep = load_episode(h5_path, rot_mode=args.rot_mode)
            episodes.append(ep)
            logger.info(f"   ✓ {h5_path.name}: {len(ep['actions'])} steps")
        except Exception as e:
            logger.warning(f"   ⚠️ Failed: {h5_path.name}: {e}")
    
    if not episodes:
        logger.error("❌ No episodes loaded!")
        return 1
    
    # Convert to SERL format
    logger.info(f"\n🔄 Converting to SERL format...")
    transitions = convert_to_serl(episodes)
    
    # Save
    pkl_filename = f"sim_dataset_ee_{args.rot_mode}.pkl"
    pkl_path = output_folder / pkl_filename
    logger.info(f"💾 Saving {len(transitions)} transitions to {pkl_path}")
    
    with open(pkl_path, 'wb') as f:
        pickle.dump(transitions, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Summary
    logger.info(f"\n✅ Done!")
    logger.info(f"   Episodes: {len(episodes)}")
    logger.info(f"   Transitions: {len(transitions)}")
    logger.info(f"   Action dim: {transitions[0]['actions'].shape}")
    logger.info(f"   State dim: {transitions[0]['observations']['state'].shape}")
    logger.info(f"   Image shape: {list(transitions[0]['observations'].values())[1].shape}")
    
    return 0


if __name__ == "__main__":
    exit(main())
