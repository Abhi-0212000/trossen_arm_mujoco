# Copyright 2025 Trossen Robotics
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#    * Neither the name of the copyright holder nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


from trossen_arm_mujoco.constants import START_ARM_POSE
from trossen_arm_mujoco.utils import (
    get_observation_base,
    make_sim_env,
    plot_observation_images,
    sample_box_pose,
    set_observation_images,
    pretty_print_obs,
)

import time
import collections
from typing import List, Optional
import pickle

from dm_control.mujoco.engine import Physics
from dm_control.suite import base
import numpy as np
from time import sleep
import dm_control.rl.control

from trossen_arm_mujoco.constants import (
    START_ARM_POSE,
    LEFT_MOCAP_MEAN_POSE, 
    RIGHT_MOCAP_MEAN_POSE,
    LEFT_GRIPPER_EPISODE0,
    RIGHT_GRIPPER_EPISODE0,
)
from trossen_arm_mujoco.utils import get_observation_base
from trossen_arm_mujoco.dataset_utils.delta_action.cleanup_utils import load_transitions, get_episode_boundaries

class TrossenAIStationaryEETask(base.Task):

    def __init__(
        self,
        random: Optional[int] = None,
        onscreen_render: bool = False,
        cam_list: List[str] = [],
    ):
        """
        Initialize the EE control task.
        
        Args:
            random: Random seed for environment variability
            onscreen_render: Whether to render camera images
            cam_list: List of camera names for observation capture
        """
        super().__init__(random=random)
        self.onscreen_render = onscreen_render
        self.cam_list = cam_list if cam_list else [
            "cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"
        ]
        self.counter = 0
        self._cube_pose_override = None  # For dataset replay

    def before_step(self, action: np.ndarray, physics: Physics) -> None:
        # print("[TrossenAIStationaryEETask.before_step] Processing EE action...")
        
        # Split action into left and right halves
        a_len = len(action) // 2  # 8
        action_left = action[:a_len]
        action_right = action[a_len:]
        
        # Set mocap positions and orientations
        # Left arm (mocap index 0)
        np.copyto(physics.data.mocap_pos[0], action_left[:3])
        np.copyto(physics.data.mocap_quat[0], action_left[3:7])
        
        # Right arm (mocap index 1)
        np.copyto(physics.data.mocap_pos[1], action_right[:3])
        np.copyto(physics.data.mocap_quat[1], action_right[3:7])

        # === GRIPPER CONTROL ===
        # Set gripper positions via actuators (not direct qpos)
        # At this point, grippers are ALWAYS in physical units [0.0, 0.044]
        # because the gym wrapper's step() already handled all normalization/denormalization
        #
        # Gripper assignment after split:
        #   action_left[7]  = LEFT robot gripper (was action[7] in full 16D array)
        #   action_right[7] = RIGHT robot gripper (was action[15] in full 16D array)
        GRIPPER_MIN = 0.0
        GRIPPER_MAX = 0.044
        
        left_grip_physical = np.clip(action_left[7], GRIPPER_MIN, GRIPPER_MAX)    # LEFT robot gripper
        right_grip_physical = np.clip(action_right[7], GRIPPER_MIN, GRIPPER_MAX)  # RIGHT robot gripper
        
        # Apply gripper commands to MuJoCo actuators
        # In EE control mode, physics.data.ctrl has ONLY 2 actuators (grippers):
        #   ctrl[0] = LEFT robot gripper motor
        #   ctrl[1] = RIGHT robot gripper motor
        physics.data.ctrl[0] = left_grip_physical   # LEFT robot gripper motor
        physics.data.ctrl[1] = right_grip_physical  # RIGHT robot gripper motor
        
        self.counter += 1
        # Note: We don't call super().before_step() for EE control
        # because mocap bodies handle arm positioning (not actuators)

    def initialize_robots(self, physics: Physics) -> None:
        print("[TrossenAIStationaryEETask.initialize_robots] Initializing robot state...")
        # Reset joint positions (arm joints only)
        # Using first 6 joints from each arm in START_ARM_POSE
        physics.named.data.qpos[:6] = START_ARM_POSE[:6]      # Left arm joints 0-5
        physics.named.data.qpos[8:14] = START_ARM_POSE[8:14]  # Right arm joints 8-13

        # Reset grippers via actuators (open position = 0.040)
        # This respects physics instead of directly setting qpos
        GRIPPER_OPEN = 0.040

        # OPTION 2: Use Episode 0 exact initial gripper positions (uncomment to test)
        physics.data.ctrl[0] = LEFT_GRIPPER_EPISODE0
        physics.data.ctrl[1] = RIGHT_GRIPPER_EPISODE0


    def initialize_episode(self, physics: Physics) -> None:
        print("[TrossenAIStationaryEETask.initialize_episode] Initializing episode...")
        self.counter = 0
        
        # === MOCAP RESET ===
        # CRITICAL: These MUST match the FK-computed poses from data collection
        
        # OPTION 1: Use averaged home poses from 27 teleoperation episodes (DEFAULT)
        np.copyto(physics.data.mocap_pos[0], LEFT_MOCAP_MEAN_POSE[:3])   # Left position
        np.copyto(physics.data.mocap_quat[0], LEFT_MOCAP_MEAN_POSE[3:])  # Left quaternion
        np.copyto(physics.data.mocap_pos[1], RIGHT_MOCAP_MEAN_POSE[:3])  # Right position
        np.copyto(physics.data.mocap_quat[1], RIGHT_MOCAP_MEAN_POSE[3:]) # Right quaternion
        
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics: Physics) -> np.ndarray:
        """Get environment state. Override in subclasses."""
        raise NotImplementedError

    def get_position(self, physics: Physics) -> np.ndarray:
        """Get current joint positions (16D: qpos[0:16])."""
        return physics.data.qpos.copy()[:16]

    def get_velocity(self, physics: Physics) -> np.ndarray:
        """Get current joint velocities (16D: qvel[0:16])."""
        return physics.data.qvel.copy()[:16]

    def get_observation(self, physics: Physics) -> collections.OrderedDict:

        # print("[TrossenAIStationaryEETask.get_observation] Gathering observation...")
        obs = get_observation_base(physics, self.cam_list, image_obs=bool(self.cam_list))

        obs["qpos"] = self.get_position(physics)
        obs["qvel"] = self.get_velocity(physics)
        
        # Cube pose (7D: position + quaternion)
        # qpos[16:23] = [x, y, z, qw, qx, qy, qz] from red_box_joint free joint
        obs["cube_pose"] = physics.data.qpos[16:23].copy()
        

        # Mocap poses (for EE control)
        obs["mocap_pose_left"] = np.concatenate([
            physics.data.mocap_pos[0].copy(),
            physics.data.mocap_quat[0].copy()
        ])
        obs["mocap_pose_right"] = np.concatenate([
            physics.data.mocap_pos[1].copy(),
            physics.data.mocap_quat[1].copy()
        ])
        obs["gripper_ctrl"] = physics.data.ctrl.copy()
    
        # print(f"[TrossenAIStationaryEETask.get_observation] Observation keys: {list(obs.keys())}")
        # self.pretty_print_obs(obs)
        return obs

    def get_reward(self, physics: Physics) -> float:
        """Compute reward. Override in subclasses."""
        raise NotImplementedError


class CubeStackingEE(TrossenAIStationaryEETask):
    """
    Cube stacking task with END-EFFECTOR control.
    
    ================================================================================
    ACTION SPACE (16D)
    ================================================================================
    
    +-------+-------------------+-------+-------------------+
    | Index | Component         | Index | Component         |
    +-------+-------------------+-------+-------------------+
    |  0-2  | L_Pos (3D)        |  8-10 | R_Pos (3D)        |
    |  3-6  | L_Quat (4D,wxyz)  | 11-14 | R_Quat (4D,wxyz)  |
    |   7   | L_Grip (1D)       |   15  | R_Grip (1D)       |
    +-------+-------------------+-------+-------------------+
    
    Uses: physics.data.mocap_pos, physics.data.mocap_quat
    XML Scene: trossen_ai_scene.xml
    """
    
    def __init__(
        self,
        random: int | None = None,
        onscreen_render: bool = False,
        cam_list: list[str] = [],
    ):
        super().__init__(
            random=random,
            onscreen_render=onscreen_render,
            cam_list=cam_list,
        )
        self.max_reward = 1

    def initialize_episode(self, physics: Physics) -> None:
        """
        Initializes the episode, resetting the robot's pose and cube position.
        
        CRITICAL: Must match EXACTLY the sequence used during data collection:
        1. Reset joints to START_ARM_POSE
        2. Randomize cube position (or use _cube_pose_override if set)
        3. Initialize mocap bodies to home position
        4. Run physics steps to let everything settle (matches "Reset Env" button behavior)
        
        Note: For dataset replay, set self._cube_pose_override before calling env.reset().
              For reproducible random placement, set np.random.seed() before env.reset().
        """
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            
            # Check if wrapper provided override (for dataset replay)
            if hasattr(self, '_cube_pose_override') and self._cube_pose_override is not None:
                cube_pose_to_use = self._cube_pose_override
                print(f"[CubeStackingEE] Using override cube_pose: {cube_pose_to_use[:3]}")
                self._cube_pose_override = None  # Clear after use
            else:
                cube_pose_to_use = sample_box_pose()
                print(f"[CubeStackingEE] Sampled random cube_pose: {cube_pose_to_use[:3]}")
            
            box_start_idx = physics.model.name2id("red_box_joint", "joint")
            np.copyto(physics.data.qpos[box_start_idx : box_start_idx + 7], cube_pose_to_use)
        
        # Initialize mocap bodies to home position
        # This matches the behavior of clicking "Home" or "Reset Env" button in data collection
        self.initialize_robots(physics)
        
        # CRITICAL: Let physics settle for ~25 steps (0.5 seconds at 50Hz)
        # During data collection, after "Reset Env", the code runs 25 steps at home position
        # to let IK controller and mocap constraints fully sync the simulation state
        # Without this, mocap bodies may not have fully moved the arms to home position yet
        for _ in range(25):
            physics.step()
        
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics: Physics) -> np.ndarray:
        """Get cube state (qpos[16:])."""
        return physics.data.qpos.copy()[16:]

    def get_reward(self, physics: Physics, verbose: bool = False) -> float:
            """
            Robust Staged Reward (Right Robot Only).
            Ignores Left Robot (assumes it is masked/frozen).
            
            Stages:
            1. Reach (0-1): Gripper -> Cube
            2. Grasp (1-2): Gripper Encloses Cube
            3. Lift  (2-3): Cube Height > Table
            4. Align (3-4): Cube -> Target XY
            5. Place (4-5): Cube on Target + Stable + Gripper Open
            """
            # ==========================
            # 1. CONSTANTS & GEOMETRY
            # ==========================
            # --- Blue Box Geometry (Derived from XML) ---
            # XML pos="0.0 0.27 0.02", size="0.07 0.07 0.15" (half-extents)
            BOX_X = 0.0
            BOX_Y = 0.27
            BOX_Z_CENTER = 0.02
            BOX_HALF_HEIGHT = 0.15
            
            # The physical top surface of the blue box
            TARGET_TOP_SURFACE = BOX_Z_CENTER + BOX_HALF_HEIGHT  # 0.02 + 0.15 = 0.17 meters

            # --- Target Definitions ---
            # We align XY first (Stage 4), then drop Z (Stage 5)
            TARGET_XY = np.array([BOX_X, BOX_Y]) 
            
            # --- Red Cube Geometry ---
            CUBE_HEIGHT_HALF = 0.0125  # size="0.0125"
            
            # --- Thresholds ---
            # Calculated Success Height (Where cube center should be when stacked)
            # 0.17 (Box Top) + 0.0125 (Cube Half) = 0.1825m
            STACKED_Z = TARGET_TOP_SURFACE + CUBE_HEIGHT_HALF 
            
            # Clearance Height (How high to lift before moving)
            # We want to be slightly above the box top (e.g. +3cm) before moving sideways 
            # to avoid hitting the edge.
            SAFE_LIFT_Z = TARGET_TOP_SURFACE + 0.03  # 0.20m
            
            # Gripper
            GRIPPER_OPEN_VAL = 0.03

            # ==========================
            # 2. STATE EXTRACTION
            # ==========================
            cube_pos = physics.data.qpos[16:19].copy()
            cube_vel = physics.data.qvel[16:19]
            cube_speed = np.linalg.norm(cube_vel)
            
            # Right Gripper Center
            c_l = physics.named.data.xpos['right/carriage_left']
            c_r = physics.named.data.xpos['right/carriage_right']
            gripper_mid = (c_l + c_r) / 2.0
            gripper_ctrl = physics.data.ctrl[1]
            
            if verbose:
                print(f"\n[REWARD DEBUG]")
                print(f"  Cube pos: {cube_pos}, speed: {cube_speed:.4f}")
                print(f"  Gripper mid: {gripper_mid}, ctrl: {gripper_ctrl:.4f}")

            # ==========================
            # 3. CONTACT CHECK (Must run every step)
            # ==========================
            # We use a set for O(1) lookups. This is fast.
            all_contacts = set()
            for i in range(physics.data.ncon):
                g1 = physics.model.id2name(physics.data.contact[i].geom1, 'geom')
                g2 = physics.model.id2name(physics.data.contact[i].geom2, 'geom')
                all_contacts.add(frozenset([g1, g2]))

            cube_geom = "subcube1" 
            gripper_geoms = ["right/gripper_follower_left", "right/gripper_follower_right"]
            target_geom = "table_box"

            is_touching_gripper = any(frozenset([cube_geom, g]) in all_contacts for g in gripper_geoms)
            is_touching_target = frozenset([cube_geom, target_geom]) in all_contacts

            # ==========================
            # 4. REWARD CALCULATION
            # ==========================
            reward = 0.0

            # --- Stage 1: Reach ---
            dist_reach = np.linalg.norm(gripper_mid - cube_pos)
            r_reach = 1.0 - np.tanh(4.0 * dist_reach)
            reward += r_reach
            if verbose: print(f"  Stage 1 (Reach): dist={dist_reach:.4f} -> r={r_reach:.4f}")

            # --- Stage 2: Grasp ---
            # Sweet spot: Cube is physically between the fingers (2cm to 7cm from base)
            dist_to_base = np.linalg.norm(cube_pos - gripper_mid)
            in_sweet_spot = (0.02 < dist_to_base < 0.07)
            
            r_grasp = 0.0
            # Gating: Only checking grasp if we are somewhat close to avoid weird artifacts
            if dist_reach < 0.1:
                if in_sweet_spot: r_grasp += 0.3
                if is_touching_gripper and gripper_ctrl > 0.002: r_grasp += 0.7
            reward += r_grasp
            if verbose: print(f"  Stage 2 (Grasp): sweet_spot={in_sweet_spot}, touching={is_touching_gripper} -> r={r_grasp:.4f}")

            # --- Stage 3: Lift ---
            # Gating: Must have good grasp score OR be resting on target (to preserve reward on placement)
            r_lift = 0.0
            if r_grasp > 0.5 or is_touching_target:
                # Map current Z range [0.0125, SAFE_LIFT_Z] to [0, 1]
                # Progress from Table Level -> Safe Lift Level
                lift_progress = (cube_pos[2] - CUBE_HEIGHT_HALF) / (SAFE_LIFT_Z - CUBE_HEIGHT_HALF)
                r_lift = np.clip(lift_progress, 0.0, 1.0)
                
                # If on target, we assume lift was successful and we are placing now,
                # so we force max lift reward to prevent penalty for lowering.
                if is_touching_target: r_lift = 1.0
            reward += r_lift
            if verbose: print(f"  Stage 3 (Lift): height={cube_pos[2]:.4f}, on_target={is_touching_target} -> r={r_lift:.4f}")

            # --- Stage 4: Align ---
            r_align = 0.0
            # Only allow alignment reward if we are high enough to not hit the box
            # OR if we have already landed on it.
            if cube_pos[2] > TARGET_TOP_SURFACE or is_touching_target:
                dist_xy = np.linalg.norm(cube_pos[:2] - TARGET_XY)
                r_align = 1.0 - np.tanh(4.0 * dist_xy)
            reward += r_align
            if verbose: print(f"  Stage 4 (Align): dist_xy={np.linalg.norm(cube_pos[:2] - TARGET_XY):.4f} -> r={r_align:.4f}")

            # --- Stage 5: Place ---
            r_place = 0.0
            if is_touching_target:
                r_place += 0.3 # Contact bonus
                if np.linalg.norm(cube_pos[:2] - TARGET_XY) < 0.05: r_place += 0.2 # Precision XY
                # Old: if gripper_ctrl > 0.02: r_place += 0.3
                # New: Boost to 1.5 so Release (1.5) > Grasp (1.0)
                if gripper_ctrl > 0.02: r_place += 1.5
                if cube_speed < 0.1: r_place += 0.2 # Stability
            reward += r_place
            if verbose: print(f"  Stage 5 (Place): on_target={is_touching_target} -> r={r_place:.4f}\n  TOTAL REWARD: {reward:.4f}\n")

            return reward

    def check_task_success(self, physics: Physics) -> bool:
        """
        Check if task is successfully completed by verifying all 5 stages.
        
        Success criteria (ALL must be satisfied):
        1. Reach: Gripper near cube (< 10cm)
        2. Grasp: Cube touching gripper + gripper closed
        3. Lift: Cube lifted above table (> 0.05m)
        4. Align: Cube XY aligned with target (< 5cm)
        5. Place: Cube on target box + stable + gripper open
        """
        # Constants from get_reward
        TARGET_XY = np.array([0.0, 0.27])
        TARGET_TOP_SURFACE = 0.17
        CUBE_HEIGHT_HALF = 0.0125
        
        # State extraction
        cube_pos = physics.data.qpos[16:19].copy()
        cube_vel = physics.data.qvel[16:19]
        cube_speed = np.linalg.norm(cube_vel)
        
        # Right Gripper
        c_l = physics.named.data.xpos['right/carriage_left']
        c_r = physics.named.data.xpos['right/carriage_right']
        gripper_mid = (c_l + c_r) / 2.0
        gripper_ctrl = physics.data.ctrl[1]
        
        # Contact check
        all_contacts = set()
        for i in range(physics.data.ncon):
            g1 = physics.model.id2name(physics.data.contact[i].geom1, 'geom')
            g2 = physics.model.id2name(physics.data.contact[i].geom2, 'geom')
            all_contacts.add(frozenset([g1, g2]))
        
        cube_geom = "subcube1"
        target_geom = "table_box"
        is_touching_target = frozenset([cube_geom, target_geom]) in all_contacts
        
        # Check all 5 stages
        # Stage 1: Reach - gripper was near cube at some point (assume true if later stages pass)
        dist_reach = np.linalg.norm(gripper_mid - cube_pos)
        stage1_reach = dist_reach < 0.1  # Close enough or already grasped
        
        # Stage 2: Grasp - not required at end, but helps validate
        # Skip for now since gripper should be open at success
        
        # Stage 3: Lift - cube must have been lifted (check current height or on target)
        stage3_lift = cube_pos[2] > (CUBE_HEIGHT_HALF + 0.03) or is_touching_target
        
        # Stage 4: Align - cube XY aligned with target
        dist_xy = np.linalg.norm(cube_pos[:2] - TARGET_XY)
        stage4_align = dist_xy < 0.05  # Within 5cm
        
        # Stage 5: Place - cube on target, stable, gripper open
        on_target = is_touching_target
        stable = cube_speed < 0.1
        gripper_open = gripper_ctrl > 0.02  # Open gripper
        stage5_place = on_target and stable and gripper_open
        
        # Success = ALL critical stages satisfied
        success = stage3_lift and stage4_align and stage5_place
        
        return success

"""
Gym wrapper for dm_control environments.
Returns RAW observations - normalization should be done in preprocessing.

Control Modes:
    - 'joint': 14D actions [L_Arm(6), L_Grip(1), R_Arm(6), R_Grip(1)]
    - 'ee': 16D actions [L_Pos(3), L_Quat(4), L_Grip(1), R_Pos(3), R_Quat(4), R_Grip(1)]
    - 'ee_angle_axis': 14D actions [L_Pos(3), L_AA(3), L_Grip(1), R_Pos(3), R_AA(3), R_Grip(1)]
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Optional, Tuple, Literal
import json
from pathlib import Path
from dm_control import composer
import cv2

from trossen_arm_mujoco.utils import plot_observation_images, set_observation_images
from trossen_arm_mujoco.constants import START_ARM_POSE
from trossen_arm_mujoco.ee_transforms import (
    angle_axis_to_quaternion, 
    quaternion_to_angle_axis,
    ortho6d_to_quaternion, 
    euler_to_quaternion, 
    quat_to_ortho6d,
    convert_dual_arm_angle_axis_to_quat_action,
    convert_dual_arm_ortho6d_to_quat_action,
    convert_dual_arm_euler_to_quat_action,
    transform_robot_to_world_frame
)


# --- LEFT ROBOT (Sentinel) ---
# Format: ([x_min, y_min, z_min], [x_max, y_max, z_max])
# Visualized in ./dataset_utils/visualize_bbox.py validate_action_sampling()
LEFT_CARTESIAN_BOUNDS = (
    np.array([-0.25, -0.13, 0.05]),  # MINS
    np.array([ -0.17,   0.1, 0.25])  # MAXS
)

# --- RIGHT ROBOT (Worker) ---
# Format: ([x_min, y_min, z_min], [x_max, y_max, z_max])
# Updated from dataset analysis in visualize_bbox.py
RIGHT_CARTESIAN_BOUNDS = (
    np.array([-0.10, -0.1, 0.0]),  # MINS  
    np.array([ 0.25,  0.32,  0.35])   # MAXS
)

# LEFT_CARTESIAN_BOUNDS = (
#     np.array([-0.255, -0.13, 0.05]),  
#     np.array([ -0.15,  0.1,  0.355])   # Pulled Max X back to -0.15 from -0.1
# )

# # --- RIGHT ROBOT (Worker) ---
# RIGHT_CARTESIAN_BOUNDS = (
#     # CHANGED X_MIN from -0.10 to 0.0 (Safety Buffer)
#     np.array([ 0.05,   -0.25, -0.02]),  
#     np.array([ 0.45,   0.35,  0.35])   
# )

import os
class SERLGymWrapper(gym.Env):
    """
    Gym wrapper for dm_control environments.
    
    Features:
    - Returns RAW observations (no normalization - do that in preprocessing)
    - Action denormalization from [-1,1] to physical joint limits
    - fake_env for fast startup (no MuJoCo initialization)
    - Automatic episode termination at max_episode_length
    - Optional data recording with GUI
    - Supports joint (14D), EE quaternion (16D), and EE angle-axis (14D) control
    
    Args:
        env: dm_control Environment instance
        state_obs_dim: State observation dimension (default 32)
        fake_env: If True, skip MuJoCo init for fast training (default False)
        image_obs: If True, include camera observations (default True)
        stats_path: Path to JSON file with normalization stats
        max_episode_length: Maximum steps per episode (default 400)
        arm_type: "widowx" or "viperx" for robot joint limits (default "viperx")
        control_mode: 'joint' (14D), 'ee' (16D), or 'ee_angle_axis' (14D) control mode
        action_dim: Action dimension (14 for joint/ee_angle_axis, 16 for ee)
        raw_actions: If True, bypass normalization/denormalization for teleop mode (default False)
    """
    
    def __init__(
        self,
        env: Optional[composer.Environment] = None,
        state_obs_dim: int = 8,
        fake_env: bool = False,
        image_obs: bool = True,
        max_episode_length: int = 400,
        arm_type: str = "viperx",
        onscreen_render: bool = False,
        cam_list: list = None,
        action_dim: int = 7,
        time_limit: float = float("inf"),
        action_scale = [1.0, 1.0, 0.005], # [position_scale, rotation_scale, gripper_scale]
        save_video: bool = False,
        video_path: str = None,
        control_mode = "delta",
        plot_name: str = "training_env",  # Unique name for matplotlib figure
    ):
        super().__init__()
        
        self.fake_env = fake_env
        self.image_obs = image_obs
        self.state_obs_dim = state_obs_dim
        self.max_episode_length = max_episode_length
        self.arm_type = arm_type
        self._steps = 0
        self.action_dim = action_dim
        self.time_limit = time_limit
        self.action_scale = action_scale

        self.random_seed = None
        # Visualization setup
        self.onscreen_render = onscreen_render
        self.cam_list = cam_list if cam_list is not None else ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]
        self.plot_name = plot_name  # Store plot name for figure management
        self.plt_imgs = None  # Matplotlib image handles for visualization
        self.plt_fig = None  # Matplotlib figure reference
        self._images_full_res = None  # Store full-res images (480x640) for visualization
        
        self.control_mode = control_mode

        # Video recording setup
        self.save_video = save_video
        self.video_path = video_path
        self._video_frames = []  # Buffer to store frames for video
        self._episode_reward = 0.0
        self._episode_number = 0
        self._video_run_dir = None  # Will be set on first video save

        self.action_space = spaces.Box(
            low=np.array([-0.5016695261001587, -0.515146017074585, -0.8840358257293701, -0.5616023540496826, -0.9998999834060669, -0.6274166703224182, -0.9998999834060669]),
            high=np.array([0.3446878492832184, 0.6354304552078247, 0.8687211275100708, 0.5261103510856628, 0.5571752786636353, 0.5912928581237793, 0.9998999834060669]),
            shape=(7,),
            dtype=np.float32
        )

        # Gym spaces - action space depends on control mode
        # self.action_space = spaces.Box(
        #     low=-1.0,
        #     high=1.0,
        #     shape=(self.action_dim,),
        #     dtype=np.float32
        # )
        
        # Observation space - Dict with state and optionally images
        state_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_obs_dim,),
            dtype=np.float32
        )
        
        obs_dict = {"state": state_space}
        if image_obs:
            # 4 cameras, 128x128 RGB images
            obs_dict["images"] = spaces.Dict({
                "cam_high": spaces.Box(0, 255, (128, 128, 3), dtype=np.uint8),
                "cam_low": spaces.Box(0, 255, (128, 128, 3), dtype=np.uint8),
                "cam_left_wrist": spaces.Box(0, 255, (128, 128, 3), dtype=np.uint8),
                "cam_right_wrist": spaces.Box(0, 255, (128, 128, 3), dtype=np.uint8),
            })
        self.observation_space = spaces.Dict(obs_dict)
        
        # Initialize environment (unless fake_env)
        self.env = env
        if not fake_env and env is None:
            raise ValueError("Must provide env unless fake_env=True")
        
        # For fake_env, create dummy observation
        if fake_env:
            self._dummy_obs = self._create_dummy_observation()
    
    def _create_dummy_observation(self) -> Dict:
        """Create dummy observation for fake_env mode."""
        obs = {
            'state': np.zeros(self.state_obs_dim, dtype=np.float32),
        }
        if self.image_obs:
            obs['images'] = {
                'cam_high': np.zeros((128, 128, 3), dtype=np.uint8),
                'cam_low': np.zeros((128, 128, 3), dtype=np.uint8),
                'cam_left_wrist': np.zeros((128, 128, 3), dtype=np.uint8),
                'cam_right_wrist': np.zeros((128, 128, 3), dtype=np.uint8),
            }
        return obs
    


    def _process_observation(self, dm_obs: Dict) -> Dict:

        state = np.concatenate([
            dm_obs['mocap_pose_right'],
            np.array([dm_obs['gripper_ctrl'][1]]),
        ]) # 8D state
        
        # Get images BEFORE popping them
        images_dict = dm_obs.get('images', {})
        
        # Store full resolution (480x640) for visualization
        if self.onscreen_render:
            self._images_full_res = images_dict
        
        obs = {'state': state.astype(np.float64)}
        obs["cube_pose"] = dm_obs["cube_pose"].astype(np.float64)
        
        # Add images if enabled
        if self.image_obs:
            # Resize to 128x128 for SERL policy
            resized_images = {}
            for cam_name, img in images_dict.items():
                resized_images[cam_name] = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
            obs['images'] = resized_images
        
        return obs
    
    def _update_visualization(self):
        """Update visualization window with full-resolution images."""
        if not self.onscreen_render or self._images_full_res is None:
            return
        
        viz_obs = {'images': self._images_full_res}
        
        if self.plt_imgs is None:
            # First time - create the plot
            import matplotlib.pyplot as plt
            self.plt_imgs = plot_observation_images(viz_obs, self.cam_list)
            # Get the figure that was just created and set its title
            self.plt_fig = plt.gcf()
            self.plt_fig.canvas.manager.set_window_title(self.plot_name)
        else:
            # Update existing plot
            self.plt_imgs = set_observation_images(viz_obs, self.plt_imgs, self.cam_list)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict, Dict]:

        print("[SERLGymWrapper] Resetting environment...")
        if seed is not None and self.random_seed is None:
            np.random.seed(seed)
            self.random_seed = seed
            print(f"[SERLGymWrapper] Set random seed to: {seed}")
        
        self._steps = 0
        self._episode_reward = 0.0
        self._video_frames = []  # Clear video buffer for new episode
        
        if self.fake_env:
            return self._dummy_obs, {}
        
        # Extract cube_pose from options if provided (for dataset replay)
        if options is not None and 'cube_pose' in options:
            cube_pose = options['cube_pose']
            print(f"[SERLGymWrapper] Received cube_pose from options: {cube_pose[:3]}")
            # Store it in task so initialize_episode() can use it
            self.env.task._cube_pose_override = cube_pose
        
        # Reset dm_control environment (normal flow, handles everything properly)
        dm_timestep = self.env.reset()

        obs = self._process_observation(dm_timestep.observation)
        # print(f"obs keys after reset: {list(obs.keys())}")
        # print(f"State observation after reset: {obs['state']}, type: {type(obs['state'])}, shape: {obs['state'].shape}, dtype: {obs['state'].dtype}")
        
        # print("[SERLGymWrapper] Updating visualization on reset...")
        self._update_visualization()
        # print("[SERLGymWrapper] Visualization updated.")
        print("[SERLGymWrapper] Environment reset complete. returning below obs")
        # self.pretty_print_obs(obs)
        # print(f"obs keys before pop dm obs: {list(obs.keys())}")
        # print(f"after pop dm obs keys: {list(obs.keys())}")
        # self.pretty_print_obs(obs)
        return obs, {}

    def step(
        self,
        daction: np.ndarray
    ) -> Tuple[Dict, float, bool, bool, Dict]:

        # print(f"[SERLGymWrapper] Step {self._steps}: Received action: {daction.tolist()}")
        if self.fake_env:
            print("[SERLGymWrapper] Fake env mode: skipping step execution.")
            self._steps += 1
            truncated = self._steps >= self.max_episode_length
            done = truncated
            return self._dummy_obs, 0.0, done, truncated, {}

        if self.control_mode == "delta":
            # Make action writable (JAX returns read-only arrays)
            daction = np.array(daction, copy=True)
            
            # =================== RIGHT ROBOT STUFF: START=========================================
            # Scale actions
            daction[0:3] = daction[0:3] * self.action_scale[0]   # Right position
            daction[3:6] = daction[3:6] * self.action_scale[1] # Right rotation
            daction[6] = daction[6] * self.action_scale[2]       # Right gripper delta
            # Get Current Right State
            current_right_quat = self.env.physics.data.mocap_quat[1].copy()
            current_right_pos = self.env.physics.data.mocap_pos[1].copy()
            current_right_aa = quaternion_to_angle_axis(current_right_quat)
            current_right_grip = self.env.physics.data.ctrl[1].copy()

            # Apply Deltas (Right Arm Only)
            target_right_pos = current_right_pos + daction[0:3]
            target_right_pos = np.clip(target_right_pos, RIGHT_CARTESIAN_BOUNDS[0], RIGHT_CARTESIAN_BOUNDS[1])
            target_right_aa  = current_right_aa  + daction[3:6]
            target_right_quat = angle_axis_to_quaternion(target_right_aa)
            
            # Gripper
            target_right_grip = np.clip(current_right_grip + daction[6], 0.0, 0.044)
            # =================== RIGHT ROBOT STUFF: CLOSE=========================================

            # Construct final absolute action
            # Note: We just send current Left state (no change) + New Right state
            # Assuming env.step takes concatenated absolute pose
            # We need the current Left State to pass it back in "noop"
            curr_l_pos = self.env.physics.data.mocap_pos[0]
            curr_l_quat = self.env.physics.data.mocap_quat[0]
            curr_l_grip = self.env.physics.data.ctrl[0]

            action_to_take = np.concatenate([
                curr_l_pos, curr_l_quat, [curr_l_grip], # Left (No Op)
                target_right_pos, target_right_quat, [target_right_grip] # Right (Active)
            ])

        elif self.control_mode == "teleop":
            # =========================================================
            # 1. HARDCODE LEFT ROBOT MASK (Safety)
            # =========================================================
            # 1. Left Arm: Clip Position to Workspace Bounds
            target_left_pos = LEFT_MOCAP_MEAN_POSE[0:3]
            target_left_quat = LEFT_MOCAP_MEAN_POSE[3:]
            
            # 2. Left Gripper: Clip to Physical Limits [0, 0.044]
            # Teleop usually sends absolute gripper position
            target_left_grip = np.array(LEFT_GRIPPER_EPISODE0)

            # 3. Right Arm: Clip Position to Workspace Bounds
            raw_right_pos = daction[0:3]
            target_right_pos = np.clip(raw_right_pos, RIGHT_CARTESIAN_BOUNDS[0], RIGHT_CARTESIAN_BOUNDS[1])
            target_right_aa  = daction[3:6]
            target_right_quat = angle_axis_to_quaternion(target_right_aa)

            # 4. Right Gripper: Clip to Physical Limits [0, 0.044]
            target_right_grip = np.clip(np.array([daction[6]]), 0.0, 0.044)
            
            # 5. Assemble Action
            action_to_take = np.concatenate([
                target_left_pos, target_left_quat, target_left_grip,
                target_right_pos, target_right_quat, target_right_grip
            ])
            # print(f"Teleop clipped action: {action_to_take.tolist()}")

        # 2. STEP PHYSICS
        dm_timestep = self.env.step(action_to_take)
        self._steps += 1

        # Process observation
        obs = self._process_observation(dm_timestep.observation)
        
        # Extract reward
        reward = dm_timestep.reward if dm_timestep.reward is not None else 0.0
        
        # ============ SERL TERMINATION LOGIC (matches Franka env) ============
        # CRITICAL MATH FOR THIS ENVIRONMENT:
        # - control_timestep = 0.02s (50 Hz control frequency)
        # - max_episode_length = 167 steps
        # - time_limit = 20.0s
        # 
        # Episode Duration Calculation:
        #   167 steps × 0.02s/step = 3.34 seconds of simulation time
        # 
        # Termination Analysis:
        #   Since 3.34s << 20.0s, the condition (sim_time >= time_limit) will NEVER be True!
        #   Episodes will ALWAYS end via truncated=True at step 167, never via terminated=True.
        # 
        # Expected Behavior:
        #   - Step 167: truncated=True, terminated=False, done=True
        #   - This produces masks=1.0 throughout the episode (correct for RLPD!)
        #   - Matches Franka env pattern: time truncation, not task completion
        # ======================================================================
        
        # 1. Check time limit (simulation physics time, not step count)
        #    dm_control uses physics.data.time (seconds of simulation time)
        sim_time = self.env.physics.data.time
        terminated = sim_time >= self.time_limit  # Will be False (3.34s < 20.0s)
        
        # 2. Check max steps (for truncation signal - artificial episode cutoff)
        #    This is separate from termination
        truncated = self._steps >= self.max_episode_length  # Will be True at step 167
        
        # 3. Construct done signal for episode boundary (used by replay buffer)
        #    Actor will use: masks = 1.0 - terminated (not 1.0 - done!)
        #    Since terminated=False always, masks=1.0 always (correct!)
        done = terminated or truncated
        # ======================================================================
        
        # Task success detection: Simple reward threshold
        # Based on reward function: Stage 5 completion gives ~4.0+ reward
        # Success = reward > 3.9 (cube on target, gripper open, stable)
        SUCCESS_REWARD_THRESHOLD = 4.35
        is_success = reward > SUCCESS_REWARD_THRESHOLD
        
        # Update episode stats
        self._episode_reward += reward
        
        # Add task-specific info (RecordEpisodeStatistics will add episode stats)
        # Always include last_step_reward for debugging/analysis
        info = {
            "is_success": is_success,
            "last_step_reward": float(reward),  # Current step's reward
        }
        
        if done:
            if terminated and not truncated:
                info["termination_reason"] = "time_limit_exceeded"
                reason = "TIME_LIMIT"
            elif truncated and not terminated:
                info["termination_reason"] = "max_steps_reached"
                reason = "TRUNCATED"
            else:
                info["termination_reason"] = "both"
                reason = "BOTH"
            
            # NOTE: Do NOT add info["episode"] here - RecordEpisodeStatistics wrapper will add it
            # RecordEpisodeStatistics adds: info["episode"] = {"r": total_reward, "l": episode_length, "t": time}
            # We just provide task-specific info like is_success and last_step_reward
            print(f"[SERLGymWrapper] Episode ended at step {self._steps}: {reason}, sim_time={sim_time:.2f}s, episode_reward={self._episode_reward:.4f}, is_success={is_success}, last_step_reward={reward:.4f}")

        # Update viz and video
        self._update_visualization()
        if self.save_video and self._images_full_res is not None:
            self._video_frames.append(self._create_video_frame())
        if done and self.save_video and len(self._video_frames) > 0:
            self._save_episode_video(is_success, self._episode_reward)

        return obs, reward, terminated, truncated, info


    def render(self, mode: str = 'rgb_array'):
        """Render the environment (not implemented for dm_control)."""
        if self.fake_env:
            return None
        # dm_control rendering handled through camera observations
        return None
    
    def _create_video_frame(self) -> np.ndarray:
        """Create a video frame by arranging camera images in a grid."""
        if self._images_full_res is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Create 2x2 grid of camera views
        # Top row: cam_high, cam_low
        # Bottom row: cam_left_wrist, cam_right_wrist
        cam_high = self._images_full_res.get("cam_high", np.zeros((480, 640, 3), dtype=np.uint8))
        cam_low = self._images_full_res.get("cam_low", np.zeros((480, 640, 3), dtype=np.uint8))
        cam_left = self._images_full_res.get("cam_left_wrist", np.zeros((480, 640, 3), dtype=np.uint8))
        cam_right = self._images_full_res.get("cam_right_wrist", np.zeros((480, 640, 3), dtype=np.uint8))
        
        # Stack images in 2x2 grid
        top_row = np.hstack([cam_high, cam_low])
        bottom_row = np.hstack([cam_left, cam_right])
        frame = np.vstack([top_row, bottom_row])
        
        return frame
    def get_or_create_video_run_dir(self) -> str:
        """Get or create a versioned subdirectory for this evaluation run."""
        if self._video_run_dir is not None:
            return self._video_run_dir
        
        if self.video_path is None:
            return None
        
        try:
            # Ensure base directory exists
            os.makedirs(self.video_path, exist_ok=True)
            
            # Find existing version folders (v0, v1, v2, ...)
            existing_versions = []
            if os.path.exists(self.video_path):
                for item in os.listdir(self.video_path):
                    item_path = os.path.join(self.video_path, item)
                    # Check if it's a directory and starts with 'v' followed by digits
                    if os.path.isdir(item_path) and item.startswith('v'):
                        version_str = item[1:]  # Remove 'v' prefix
                        if version_str.isdigit():
                            existing_versions.append(int(version_str))
            
            # Get next version number
            next_version = max(existing_versions) + 1 if existing_versions else 0
            
            # Create new version directory
            self._video_run_dir = os.path.join(self.video_path, f"v{next_version}")
            os.makedirs(self._video_run_dir, exist_ok=True)
            print(f"[Video] Created video run directory: {self._video_run_dir}")
            
            return self._video_run_dir
            
        except Exception as e:
            print(f"[Video] Warning: Failed to create versioned directory: {e}")
            # Fallback to base path
            self._video_run_dir = self.video_path
            return self._video_run_dir
    
    def _save_episode_video(self, success: bool, total_reward: float):
        """Save collected frames as video with custom filename."""
        if not self._video_frames or self.video_path is None:
            return
        
        import imageio
        
        # Get or create versioned run directory
        run_dir = self.get_or_create_video_run_dir()
        if run_dir is None:
            return
        
        result_str = "success" if success else "fail"
        video_filename = f"eval_ep{self._episode_number:02d}_{result_str}_Rwrd{total_reward:.4f}.mp4"
        video_filepath = os.path.join(run_dir, video_filename)
        os.makedirs(self.video_path, exist_ok=True)
        
        try:
            # Save video at 20 FPS (matches 0.02s control timestep = 50Hz / 2.5 = 20fps for smooth playback)
            imageio.mimsave(video_filepath, self._video_frames, fps=20)
            print(f"[Video] Saved episode video: {video_filepath}")
        except Exception as e:
            print(f"[Video] Failed to save video: {e}")
        
        # Increment episode counter
        self._episode_number += 1
    
    def _save_debug_images(self):
        """Save the last 10 images from buffer to disk for debugging."""
        if not self._image_buffer:
            return
        
        print(f"\n[DEBUG] Saving last {len(self._image_buffer)} images to {self._debug_save_dir}/")
        
        metadata_list = []
        for i, frame_data in enumerate(self._image_buffer):
            step = frame_data['step']
            images = frame_data['images']
            
            # Save each camera view
            for cam_name, img in images.items():
                img_path = self._debug_save_dir / f"step_{step:06d}_{cam_name}.png"
                cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
            # Store metadata
            metadata_list.append({
                'step': step,
                'left_pos': frame_data['left_pos'].tolist(),
                'right_pos': frame_data['right_pos'].tolist(),
                'left_in_bounds': frame_data['left_in_bounds'],
                'right_in_bounds': frame_data['right_in_bounds'],
            })
        
        # Save metadata JSON
        metadata_path = self._debug_save_dir / f"metadata_{step:06d}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata_list, f, indent=2)
        
        print(f"[DEBUG] Saved images and metadata. Check {self._debug_save_dir}/metadata.json for positions.")
    
    def close(self):
        """Close the environment and cleanup visualization windows."""
        # Close matplotlib visualization if it exists
        if hasattr(self, 'plt_fig') and self.plt_fig is not None:
            import matplotlib.pyplot as plt
            # Close only THIS environment's figure by name
            plt.close(self.plt_fig)
            print(f"[SERLGymWrapper] Closed visualization window: {self.plot_name}")
            self.plt_fig = None
            self.plt_imgs = None
        
        # Close dm_control environment
        if hasattr(self, 'env') and self.env is not None:
            self.env.close()

    def reset_cube_pose_only(self):
        """Reset only the cube's pose in the environment."""
        if self.fake_env:
            print("[SERLGymWrapper] Fake env mode: skipping cube pose reset.")
            return
        
        # Reset cube position and orientation
        self.env.initialize_episode()
        print("[SERLGymWrapper] Cube pose reset.")



from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper


def check_states_match(state_dataset, state_env, tolerance=1e-3, verbose=True):
    """
    Compares two state vectors of shape (1, 16).
    
    Args:
        state_dataset (np.ndarray): State vector from the loaded dataset.
        state_env (np.ndarray): State vector returned by env.step().
        tolerance (float): Maximum allowed difference for L2 norm.
        verbose (bool): If True, prints detailed error stats.

    Returns:
        bool: True if states match within tolerance, False otherwise.
    """
    # 1. Ensure inputs are numpy arrays and flatten them to (16,) for easier math
    s_ds = np.array(state_dataset).flatten()
    s_env = np.array(state_env).flatten()

    # 2. Check shapes
    if s_ds.shape != s_env.shape:
        print(f"❌ Shape Mismatch! Dataset: {s_ds.shape}, Env: {s_env.shape}")
        return False

    # 3. Calculate Differences
    diff = s_ds - s_env
    
    # L2 Norm (Euclidean Distance) - Good for overall error
    l2_error = np.linalg.norm(diff)
    
    # Max Absolute Error - Good for catching single outlier values
    max_error = np.max(np.abs(diff))

    is_match = l2_error < tolerance

    if verbose:
        status = "✅ MATCH" if is_match else "❌ MISMATCH"
        # print(f"--- State Check: {status} ---")
        # print(f"L2 Error: {l2_error:.6f} | Max Abs Error: {max_error:.6f} | Tolerance: {tolerance}")
        
        if not is_match:
            # Print indices where error is high
            high_err_indices = np.where(np.abs(diff) > tolerance)[0]
            print(f"Indices with high error: {high_err_indices}")
            print(f"Dataset values: {s_ds[high_err_indices]}")
            print(f"Env values:     {s_env[high_err_indices]}")
            print("---------------------------------")

    return is_match

import copy
from tqdm import tqdm  # Recommended for progress tracking
import numpy as np
import copy
import pickle
from tqdm import tqdm

def regenerate_ds_state_from_obs(pkl_file_path: str, output_file_path: str):
    """
    Regenerate dataset with updated environment physics and termination logic.
    
    Strategy:
    1. Calculate max_episode_length = ceil(90th percentile + 20)
    2. For each episode:
       - Replay actions from dataset
       - If episode < max_length: extend with last action until truncated=True
       - If episode = max_length: just replay normally
       - If episode > max_length: truncate and track for manual review
    3. CRITICAL: masks = 1 - terminated (NOT 1 - done!)
       - terminated checks physics time (always False for our settings)
       - done = terminated OR truncated (episode boundary signal)
       - masks = 1.0 always (correct for RLPD bootstrapping)
    """
    CONTROL_TIMESTEP = 0.02
    PHYSICS_TIMESTEP = 0.002
    cam_list = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]
    onscreen_render = False

    # Load and analyze dataset
    transitions = load_transitions(pkl_file_path)
    episode_boundaries = get_episode_boundaries(transitions)
    episode_lengths = [end - start for start, end in episode_boundaries]
    
    # Calculate Max Episode Length (round up to clean number)
    percentile_90 = np.percentile(episode_lengths, 90.0)
    max_episode_length = int(np.ceil(percentile_90 + 20))
    
    print(f"\n{'='*80}")
    print(f"DATASET STATS:")
    print(f"  Episodes: {len(episode_boundaries)} | Min: {min(episode_lengths)} | Max: {max(episode_lengths)}")
    print(f"  Mean: {np.mean(episode_lengths):.1f} | Median: {np.median(episode_lengths):.1f}")
    print(f"  90th percentile: {percentile_90:.1f} → max_episode_length = {max_episode_length}")
    print(f"\nEXPECTED BEHAVIOR:")
    print(f"  Episode duration: {max_episode_length} steps × {CONTROL_TIMESTEP}s = {max_episode_length * CONTROL_TIMESTEP:.2f}s")
    print(f"  Time limit: 20.0s → terminated will be FALSE, truncated will be TRUE")
    print(f"  Result: masks = 1 - terminated = 1.0 (correct for RLPD!)")
    print(f"{'='*80}\n")

    dm_env = make_sim_env(
        CubeStackingEE,
        task_name="sim_transfer_cube",
        onscreen_render=onscreen_render,
        cam_list=cam_list,
        control_timestep=CONTROL_TIMESTEP, 
        physics_timestep=PHYSICS_TIMESTEP,
    )
    
    gym_env = SERLGymWrapper(
        env=dm_env,
        state_obs_dim=8,  # RIGHT ARM ONLY
        onscreen_render=onscreen_render,
        cam_list=cam_list,
        action_dim=7,  # RIGHT ARM ONLY
        time_limit=20.0,
        control_mode="delta",
        max_episode_length=max_episode_length, 
        action_scale=[0.025, 0.1, 0.005], 
    )
    gym_env = SERLObsWrapper(gym_env)
    gym_env = ChunkingWrapper(gym_env, obs_horizon=1, act_exec_horizon=None)
    
    new_transitions = []
    episodes_truncated = []  # Episodes where we cut off data (len > max_length)
    
    # --- Main Replay Loop ---
    for ep_idx, (start, end) in enumerate(tqdm(episode_boundaries, desc="Replaying Episodes")):
        original_episode = transitions[start:end]
        original_length = len(original_episode)
        
        # Track episodes that will lose data
        if original_length > max_episode_length:
            episodes_truncated.append((ep_idx, original_length))
        
        # Extract cube_pose from first transition
        cube_pose_from_dataset = original_episode[0]['observations']['full_observation']['cube_pose']
        
        current_obs, _ = gym_env.reset(options={'cube_pose': cube_pose_from_dataset})
        step_count = 0
        last_action = None
        last_trans_template = None
        
        # --- Phase 1: Replay actions from dataset (up to max_episode_length) ---
        num_actions_to_replay = min(original_length, max_episode_length)
        
        for i in range(num_actions_to_replay):
            original_trans = original_episode[i]
            
            # Deep copy to preserve metadata (language, etc.)
            new_trans = copy.deepcopy(original_trans)
            
            # Overwrite observations with current env state
            new_trans['observations']['state'] = current_obs['state'].copy()
            for cam in cam_list:
                if cam in current_obs:
                    new_trans['observations'][cam] = current_obs[cam].copy()
            
            # Execute action in environment
            action = original_trans['actions'].copy()
            next_obs, reward, terminated, truncated, info = gym_env.step(action)
            
            # Save for potential extension
            last_action = action
            last_trans_template = new_trans
            
            # Overwrite next_observations with actual env output
            new_trans['next_observations']['state'] = next_obs['state'].copy()
            for cam in cam_list:
                if cam in next_obs:
                    new_trans['next_observations'][cam] = next_obs[cam].copy()
            
            # CRITICAL: Update done/masks correctly
            # done = terminated OR truncated (episode boundary)
            # masks = 1 - terminated (bootstrapping signal)
            done = terminated or truncated
            new_trans['rewards'] = reward
            new_trans['dones'] = done
            new_trans['masks'] = 1.0 - terminated  # NOT 1 - done!
            
            new_transitions.append(new_trans)
            current_obs = next_obs
            step_count += 1
            
            # If truncated=True, episode ended naturally
            if truncated:
                break
        
        # --- Phase 2: Extend short episodes with last action ---
        if step_count < max_episode_length and last_action is not None:
            while step_count < max_episode_length:
                # Reuse last transition structure
                extended_trans = copy.deepcopy(last_trans_template)
                
                # Current observations
                extended_trans['observations']['state'] = current_obs['state'].copy()
                for cam in cam_list:
                    extended_trans['observations'][cam] = current_obs[cam].copy()
                
                # Execute last action (hover/hold)
                next_obs, reward, terminated, truncated, info = gym_env.step(last_action)
                
                # Next observations
                extended_trans['next_observations']['state'] = next_obs['state'].copy()
                for cam in cam_list:
                    extended_trans['next_observations'][cam] = next_obs[cam].copy()
                
                # Update metadata
                extended_trans['actions'] = last_action.copy()
                extended_trans['rewards'] = reward
                done = terminated or truncated
                extended_trans['dones'] = done
                extended_trans['masks'] = 1.0 - terminated  # Always 1.0 for our case
                
                new_transitions.append(extended_trans)
                current_obs = next_obs
                step_count += 1
                
                # Stop when truncated (should happen at max_episode_length)
                if truncated:
                    break

    # --- Save Result ---
    print(f"\n{'='*80}")
    print(f"REPLAY COMPLETE!")
    print(f"  Original transitions: {len(transitions)}")
    print(f"  New transitions: {len(new_transitions)}")
    print(f"  Episodes truncated (data loss): {len(episodes_truncated)}")
    if episodes_truncated:
        print(f"\n  Episodes with data loss (original_length > {max_episode_length}):")
        for ep_idx, orig_len in episodes_truncated:
            print(f"    Episode {ep_idx}: {orig_len} steps (lost {orig_len - max_episode_length} steps)")
        print(f"\n  ⚠️  VERIFY THESE EPISODES with test_data_replay_sim_env()!")
        print(f"      If cube not on blue box, delete episode manually.")
    print(f"{'='*80}")
    
    with open(output_file_path, 'wb') as f:
        pickle.dump(new_transitions, f)
    print(f"\nSaved regenerated dataset to: {output_file_path}")
    print(f"{'='*80}")

def test_data_replay_sim_env(pkl_file_path, episode_indices=None):
    """
    Replay episodes from the dataset.
    
    Args:
        episode_indices: int, list of ints, or None
            - int: replay single episode (e.g., 13)
            - list: replay multiple episodes (e.g., [0, 5, 13])
            - None: replay all episodes
    """
    transitions = load_transitions(pkl_file_path)
    # state_check_transitions = load_transitions(state_check_file_path)
    print(f"Loaded {len(transitions)} transitions from dataset.")
    episode_boundaries = get_episode_boundaries(transitions)
    print(f"Found {len(episode_boundaries)} episodes in the dataset.")
    print(f"max episode length: {max(end - start for start, end in episode_boundaries)}")
    print(f"All lengths of episodes: {[end - start for start, end in episode_boundaries]}")
    time.sleep(1.0)
    print(f"shape of state in first transition: {np.array(transitions[0]['observations']['state']).shape}")
    print(f"shape of action in first transition: {np.array(transitions[0]['actions']).shape}")

    CONTROL_TIMESTEP = 0.02  # 50 Hz control frequency
    PHYSICS_TIMESTEP = 0.002  # 10 substeps per control step

    onscreen_render = True  # Set to False for faster checking
    cam_list = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]
    dm_env = make_sim_env(
        CubeStackingEE,
        task_name="sim_transfer_cube",
        onscreen_render=onscreen_render,
        cam_list=cam_list,
        control_timestep=CONTROL_TIMESTEP, # control freq = 1/control_timestep = 50 Hz
        physics_timestep=PHYSICS_TIMESTEP, # n_substeps = control_timestep / physics_timestep = 10
    )
    print("Created dm_env")
    gym_env = SERLGymWrapper(
        env = dm_env,
        state_obs_dim = 8,
        onscreen_render = onscreen_render,
        cam_list = cam_list,
        action_dim = 7,
        time_limit = 20.0,
        control_mode = "delta",
        max_episode_length = 180,
        action_scale = [0.025, 0.1, 0.005],  # Position, Rotation, Gripper (was 0.05, too large!)
    )
    gym_env = SERLObsWrapper(gym_env)
    gym_env = ChunkingWrapper(gym_env, obs_horizon=1, act_exec_horizon=None)
    print("Created gym_env")

    # Check first observation of each episode against reset
    obs, _ = gym_env.reset()
    state_after_reset = obs['state']
    print(f"Reset state: {state_after_reset}")

    all_matches = []
    for ep_idx, (start, end) in enumerate(episode_boundaries):
        episode_transitions = transitions[start:end]
        if len(episode_transitions) > 0:
            state_from_1st_transition = episode_transitions[0]['observations']['state']
            match = check_states_match(state_after_reset, state_from_1st_transition, tolerance=1e-3, verbose=False)
            all_matches.append(match)
            if not match:
                print(f"Episode {ep_idx}: MISMATCH")
                print(f"  Dataset first state: {state_from_1st_transition}")
            else:
                print(f"Episode {ep_idx}: MATCH")

    print(f"Summary: {sum(all_matches)}/{len(all_matches)} episodes match reset state.")

    # Determine which episodes to replay
    if episode_indices is None:
        # Replay all episodes
        episodes_to_replay = list(range(len(episode_boundaries)))
        print(f"Replaying ALL {len(episodes_to_replay)} episodes.")
    elif isinstance(episode_indices, int):
        # Replay single episode
        episodes_to_replay = [episode_indices]
        print(f"Replaying single episode {episode_indices}.")
    elif isinstance(episode_indices, list):
        # Replay list of episodes
        episodes_to_replay = episode_indices
        print(f"Replaying {len(episodes_to_replay)} episodes: {episodes_to_replay}")
    else:
        raise ValueError("episode_indices must be None, int, or list of ints")

    # Now replay the specified episodes
    for ep_to_replay in episodes_to_replay:
        if ep_to_replay >= len(episode_boundaries):
            print(f"Episode {ep_to_replay} out of range. Skipping.")
            continue
            
        start, end = episode_boundaries[ep_to_replay]
        episode_transitions = transitions[start:end]
        print(f"\n{'='*80}")
        print(f"Replaying Episode {ep_to_replay} with {len(episode_transitions)} transitions.")
        print(f"{'='*80}")

        # pretty_print_obs(episode_transitions[0]['observations'], indent=4)
        state_check_results = []
        
        # Extract cube_pose from first transition to hardcode it during reset
        cube_pose_from_dataset = episode_transitions[0]['observations']['full_observation']['cube_pose']
        print(f"\n[Dataset Replay] Using cube_pose from episode {ep_to_replay}: {cube_pose_from_dataset[:3]}")
        time.sleep(1.0)
        # Reset with hardcoded cube pose
        obs, _ = gym_env.reset(options={'cube_pose': cube_pose_from_dataset})
        state_after_reset = obs['state']
        state_from_1st_transition = episode_transitions[0]['observations']['state']
        print(f"state after reset: {state_after_reset}")
        print(f"state from 1st transition: {state_from_1st_transition}")
        match = check_states_match(state_after_reset, state_from_1st_transition, tolerance=1e-3, verbose=True)
        state_check_results.append(match)

        start_time = time.time()
        print(f"Starting episode replay at wall-clock time: {start_time}")

        for t, transition in enumerate(episode_transitions[:]):  # Limit to first 10 steps for speed
            action = transition['actions']
            obs, reward, done, truncated, info = gym_env.step(action)
            # time.sleep(0.5)
            state_from_obs = obs['state']
            state_from_ds_to_check = transition['next_observations']['state']
            match = check_states_match(state_from_ds_to_check, state_from_obs, tolerance=1e-3, verbose=True)
            state_check_results.append(match)
            print(f"Step {t}: reward={reward}, done={done}")
            if done:
                print("Episode ended early.")
                break

        end_time = time.time()
        total_wall_time = end_time - start_time
        num_steps = len(episode_transitions)
        expected_sim_time = num_steps * CONTROL_TIMESTEP  # control_timestep = 0.02
        print(f"Episode {ep_to_replay} completed in {total_wall_time:.2f} seconds wall-clock time")
        print(f"Number of steps: {num_steps}")
        print(f"Expected simulation time: {expected_sim_time:.2f} seconds (at {1/CONTROL_TIMESTEP:.2f} Hz control)")
        print(f"Control frequency verification: {num_steps / total_wall_time:.2f} Hz (actual steps/second)")
        print(f"State match results for episode {ep_to_replay}: {state_check_results} and its length is {len(state_check_results)}")


def test_ee_sim_env():
    onscreen_render = True
    cam_list = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]
    dm_env = make_sim_env(
        CubeStackingEE,
        task_name="sim_transfer_cube",
        onscreen_render=onscreen_render,
        cam_list=cam_list,
        control_timestep=0.02,
        physics_timestep=0.002,
    )
    print("Created dm_env")
    gym_env = SERLGymWrapper(
        env = dm_env,
        state_obs_dim = 16,
        onscreen_render = True,
        cam_list = cam_list,
        action_dim = 14,
        time_limit = 20.0

    )
    obs, _ = gym_env.reset()
    for t in range(1000):
        action = np.random.uniform(-0.1, 0.1, 14)
        obs, reward, done, truncated, info = gym_env.step(action)
        sleep(0.5)


def delete_specific_episodes_and_save(input_file_path: str = None, output_file_path: str = None, episodes_to_delete: list = None):

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
    final_data = kept_transitions
    
    print(f"Saving to {output_file_path}...")
    with open(output_file_path, 'wb') as f:
        pickle.dump(final_data, f)
    print("Done.")


if __name__ == "__main__":
    # --- Test data replay ---
    # INPUT_FILE_PATH = "/home/qte9489/personal_abhi/temp/hil-serl/trossen_arm_mujoco/trossen_arm_mujoco/tasks/RL/Right_robot_stacking_cube/only_right_arm_data_regenerated.pkl"
    # test_data_replay_sim_env(INPUT_FILE_PATH, episode_indices=[0, 1, 4, 5, 9, 11, 13])
    
    # --- Regenerate dataset from observations ---
    # Auto-set max_episode_length based on dataset (max length + 100 buffer)
    INPUT_FILE_PATH = "/home/qte9489/personal_abhi/temp/hil-serl/examples/experiments/cube_stacking_gym/only_right_arm_data_regenerated_final.pkl"
    OUTPUT_FILE_PATH = "/home/qte9489/personal_abhi/temp/hil-serl/examples/experiments/cube_stacking_gym/only_right_arm_data_regenerated_final.pkl"
    regenerate_ds_state_from_obs(INPUT_FILE_PATH, OUTPUT_FILE_PATH)

    # Delete specific episodes from dataset
    # INPUT_FILE_PATH = "/home/qte9489/personal_abhi/temp/hil-serl/trossen_arm_mujoco/trossen_arm_mujoco/tasks/RL/Right_robot_stacking_cube/only_right_arm_data_regenerated.pkl"
    # OUTPUT_FILE_PATH = "/home/qte9489/personal_abhi/temp/hil-serl/trossen_arm_mujoco/trossen_arm_mujoco/tasks/RL/Right_robot_stacking_cube/only_right_arm_data_regenerated_final.pkl"
    # delete_specific_episodes_and_save(INPUT_FILE_PATH, OUTPUT_FILE_PATH, episodes_to_delete=[1, 4, 13])