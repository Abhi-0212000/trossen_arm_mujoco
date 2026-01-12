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


import collections
from typing import List, Optional

from dm_control.mujoco.engine import Physics
from dm_control.suite import base
import numpy as np
from time import sleep
import dm_control.rl.control

from trossen_arm_mujoco.constants import (
    START_ARM_POSE,
    START_ARM_POSE_MEAN,
    LEFT_MOCAP_HOME_POSE,
    RIGHT_MOCAP_HOME_POSE,
    LEFT_MOCAP_EPISODE0_POSE,
    RIGHT_MOCAP_EPISODE0_POSE,
    LEFT_MOCAP_MEAN_POSE, 
    RIGHT_MOCAP_MEAN_POSE,
    LEFT_GRIPPER_EPISODE0,
    RIGHT_GRIPPER_EPISODE0,
)
from trossen_arm_mujoco.utils import get_observation_base


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
        # === GRIPPER INITIALIZATION ===
        # OPTION 1: Use averaged home position from 27 episodes
        # physics.data.ctrl[0] = GRIPPER_OPEN  # Left gripper actuator
        # physics.data.ctrl[1] = GRIPPER_OPEN  # Right gripper actuator
        
        # OPTION 2: Use Episode 0 exact initial gripper positions (uncomment to test)
        physics.data.ctrl[0] = LEFT_GRIPPER_EPISODE0
        physics.data.ctrl[1] = RIGHT_GRIPPER_EPISODE0

        # === MOCAP INITIALIZATION ===
        # CRITICAL: These MUST match the FK-computed poses from data collection
        
        # OPTION 1: Use averaged home poses from 27 teleoperation episodes (DEFAULT)
        # np.copyto(physics.data.mocap_pos[0], LEFT_MOCAP_HOME_POSE[:3])   # Left position
        # np.copyto(physics.data.mocap_quat[0], LEFT_MOCAP_HOME_POSE[3:])  # Left quaternion
        # np.copyto(physics.data.mocap_pos[1], RIGHT_MOCAP_HOME_POSE[:3])  # Right position
        # np.copyto(physics.data.mocap_quat[1], RIGHT_MOCAP_HOME_POSE[3:]) # Right quaternion
        
        # OPTION 2: Use Episode 0 exact initial conditions for debugging (uncomment to test)
        # np.copyto(physics.data.mocap_pos[0], LEFT_MOCAP_MEAN_POSE[:3])
        # np.copyto(physics.data.mocap_quat[0], LEFT_MOCAP_MEAN_POSE[3:])
        # np.copyto(physics.data.mocap_pos[1], RIGHT_MOCAP_MEAN_POSE[:3])
        # np.copyto(physics.data.mocap_quat[1], RIGHT_MOCAP_MEAN_POSE[3:])

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
        
        # OPTION 2: Use Episode 0 exact initial conditions for debugging (uncomment to test)
        # np.copyto(physics.data.mocap_pos[0], LEFT_MOCAP_EPISODE0_POSE[:3])
        # np.copyto(physics.data.mocap_quat[0], LEFT_MOCAP_EPISODE0_POSE[3:])
        # np.copyto(physics.data.mocap_pos[1], RIGHT_MOCAP_EPISODE0_POSE[:3])
        # np.copyto(physics.data.mocap_quat[1], RIGHT_MOCAP_EPISODE0_POSE[3:])
        
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

    # SPARSE BINARY REWARD FUNCTION
    def get_reward(self, physics: Physics) -> float:
        """
        Strict Binary Reward: Stacking Success + Left Robot Discipline.
        
        Returns 1.0 ONLY if:
        1. Right Robot stacks cube perfectly (Position + Stability).
        2. Left Robot is COMPLETELY IDLE (Velocity < Threshold).
        
        Returns 0.0 if the Left Robot moves, even if the stacking is perfect.
        
        ================================================================================
        XML GEOMETRY REFERENCE (from trossen_ai_scene.xml):
        ================================================================================
        
        RED CUBE (to pick):
            - Body pos: (0.0, 0.0, 0.0125) - center of cube
            - Visual geom size: 0.0125 (half-extent) → 2.5cm x 2.5cm x 2.5cm cube
            - Subcubes: 4 collision geoms (subcube1-4)
            
        BLUE TARGET BOX (table_box): MODIFIED
            - Body pos: (0.0, 0.27, 0.02) - center of box (was y=0.22, then 0.35)
            - Geom size: (0.07, 0.07, 0.15) → 14cm x 14cm x 30cm box (was 0.1 → 20cm x 20cm)
            - Top surface Z = 0.02 (center) + 0.15 (half-height) = 0.17m
            
        For cube to be FULLY SUPPORTED on target:
            - Cube center must be within: target_half - cube_half = 0.07 - 0.0125 = 0.0575m (was 0.0875m)
            
        ================================================================================
        """
        # ==========================
        # 1. CONFIGURATION (FROM XML)
        # ==========================
        # --- Geometries (from trossen_ai_scene.xml) ---
        cube_geoms = {"subcube1", "subcube2", "subcube3", "subcube4"}
        target_geom = "table_box"
        gripper_geoms = {"right/gripper_follower_left", "right/gripper_follower_right"}
        source_table_geom = "table_collision"
        
        # --- Thresholds ---
        # Left arm EE position check (more robust than qvel which has settling noise)
        # 
        # IMPORTANT: This is the ACTUAL ROBOT BODY position, NOT the mocap command target!
        # 
        # How this position was determined:
        #   1. After env.reset(), query: physics.named.data.xpos['left/link_6']
        #   2. This returns the world-frame position of the left arm's end-effector link
        #   3. The robot is initialized with START_ARM_POSE_MEAN joint angles (from constants.py)
        #   4. Forward kinematics places left/link_6 at approximately [-0.3603, -0.019, 0.1875]
        #
        # Why NOT use mocap position [-0.2062, -0.019, 0.1835]?
        #   - Mocap is the COMMAND TARGET (where we tell the IK to move the arm)
        #   - The actual robot body position differs due to:
        #     a) Robot base offset: left/root is at [-0.4575, -0.019, 0.02] (from trossen_ai_bimanual.xml)
        #     b) Weld constraint between mocap_left and left/link_6
        #     c) IK solver positioning the kinematic chain
        #   - Using mocap position would cause false positives when mocap commands change
        #
        # To verify this value, run:
        #   physics.named.data.xpos['left/link_6'] after reset
        #
        LEFT_HOME_POS = np.array([-0.3603, -0.019, 0.1875])
        LEFT_EE_THRESHOLD = 0.10  # 10cm allowed deviation from home
        
        # Max allowed velocity for the cube (to ensure it's placed, not thrown)
        # 0.05 m/s = 5cm/s is reasonable for "at rest"
        CUBE_STABLE_THRESHOLD = 0.1  # 10cm/s (was 0.05)
        
        # --- Position checks (VERIFIED FROM XML) ---
        # Target box: pos="0.0 0.27 0.02", size="0.07 0.07 0.15" (MODIFIED: was pos y=0.22/0.35, size=0.1)
        TARGET_CENTER_X = 0.0
        TARGET_CENTER_Y = 0.27  # MODIFIED: was 0.22, then 0.35, now 0.27
        TARGET_TOP_Z = 0.17         # 0.02 (center) + 0.15 (half-height)
        
        # Red cube: size="0.0125" (half-extent) = 2.5cm cube
        CUBE_HALF_HEIGHT = 0.0125   # Half of 2.5cm cube
        CUBE_HALF_WIDTH = 0.0125
        
        # For cube to be fully supported on target:
        # Target half = 0.07m, Cube half = 0.0125m (MODIFIED: target was 0.1m)
        # Max offset = 0.07 - 0.0125 = 0.0575m for full support (MODIFIED: was 0.0875m)
        # We use this as alignment threshold (cube fully on top, no overhang)
        ALIGNMENT_THRESHOLD = 0.07  # 5.75cm - cube fully supported (MODIFIED: was 0.0875)
        
        # Z height thresholds (MUST MATCH relabel_hdf5_optimized.py)
        # When stacked, cube center Z should be at: TARGET_TOP_Z + CUBE_HALF = 0.17 + 0.0125 = 0.1825
        # Allow tolerance for physics settling
        MIN_STACK_Z = TARGET_TOP_Z + CUBE_HALF_HEIGHT - 0.04  # 0.1525 minimum
        MAX_STACK_Z = TARGET_TOP_Z + CUBE_HALF_HEIGHT + 0.04  # 0.2025 maximum
        
        # ==========================
        # 2. CHECK LEFT ROBOT (FAIL FAST) - Using EE Position
        # ==========================
        # Get left arm EE position from actual robot body (not mocap target)
        # left/link_6 is the end-effector link in the robot kinematic chain
        # This reflects the actual simulated robot position, not the mocap command
        left_ee_pos = physics.named.data.xpos["left/link_6"]
        
        # Calculate distance from home position
        left_deviation = np.linalg.norm(left_ee_pos - LEFT_HOME_POS)
        
        # VIOLATION CHECK: If left arm moved too far from home, immediate failure.
        if left_deviation > LEFT_EE_THRESHOLD:
            print(f"[REWARD] ❌ Left arm moved (deviation={left_deviation:.4f}m > {LEFT_EE_THRESHOLD}m) → Reward: 0.0")
            return 0.0

        # ==========================
        # 3. CHECK RIGHT ROBOT SUCCESS
        # ==========================
        
        # --- A. Contact Detection ---
        all_contact_pairs = set()
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, "geom")
            name_geom_2 = physics.model.id2name(id_geom_2, "geom")
            all_contact_pairs.add(frozenset([name_geom_1, name_geom_2]))

        # Check 1: Cube MUST be touching the target (blue box)
        touching_target = any(
            frozenset([cube, target_geom]) in all_contact_pairs
            for cube in cube_geoms
        )
        if not touching_target:
            # print(f"[REWARD DEBUG] ❌ Check 1 FAILED: Cube not touching target")
            # print(f"  All contacts: {all_contact_pairs}")
            # print(f"  Cube geoms: {cube_geoms}, Target: {target_geom}")
            # print(f"[REWARD] ❌ Cube not touching target → Reward: 0.0")
            return 0.0

        # Check 2: Cube must NOT be touching gripper (released)
        is_grasped = any(
            frozenset([cube, grip]) in all_contact_pairs
            for cube in cube_geoms
            for grip in gripper_geoms
        )
        if is_grasped:
            # print(f"[REWARD DEBUG] ❌ Check 2 FAILED: Cube still grasped")
            # print(f"  Gripper geoms: {gripper_geoms}")
            # print(f"[REWARD] ❌ Cube still grasped (not released) → Reward: 0.0")
            return 0.0
        
        # Check 3: Cube must NOT be touching source table anymore
        # (ensures it was actually picked up and placed, not just pushed)
        touching_source = any(
            frozenset([cube, source_table_geom]) in all_contact_pairs
            for cube in cube_geoms
        )
        if touching_source:
            # print(f"[REWARD DEBUG] ❌ Check 3 FAILED: Cube still touching source table")
            # print(f"  Source table geom: {source_table_geom}")
            # print(f"[REWARD] ❌ Cube still touching source table → Reward: 0.0")
            return 0.0

        # --- B. Position & Stability Checks ---
        # Cube position from free joint (starts at qpos[16])
        cube_pos = physics.data.qpos[16:19]  # [x, y, z]
        cube_vel = physics.data.qvel[16:19]  # [vx, vy, vz] - linear velocity
        cube_angvel = physics.data.qvel[19:22]  # [wx, wy, wz] - angular velocity
        
        # Check 4: Height - Cube center must be within valid stacking height range
        # Uses same thresholds as relabel_hdf5_optimized.py for consistency
        is_on_top = (cube_pos[2] >= MIN_STACK_Z) and (cube_pos[2] <= MAX_STACK_Z)
        if not is_on_top:
            # print(f"[REWARD] ❌ Cube height wrong (z={cube_pos[2]:.4f}, need {MIN_STACK_Z:.4f}-{MAX_STACK_Z:.4f}) → Reward: 0.0")
            return 0.0
        
        # Check 5: XY Alignment - Cube center within target bounds (fully supported)
        target_center = np.array([TARGET_CENTER_X, TARGET_CENTER_Y])
        xy_dist = np.linalg.norm(cube_pos[:2] - target_center)
        is_aligned = xy_dist <= ALIGNMENT_THRESHOLD
        if not is_aligned:
            # print(f"[REWARD] ❌ Cube XY misaligned (dist={xy_dist:.4f}m > {ALIGNMENT_THRESHOLD}m) → Reward: 0.0")
            return 0.0
        
        # Check 6: Stability - Cube must be at rest (not falling/sliding/spinning)
        linear_speed = np.linalg.norm(cube_vel)
        angular_speed = np.linalg.norm(cube_angvel)
        is_stable = (linear_speed < CUBE_STABLE_THRESHOLD) and (angular_speed < CUBE_STABLE_THRESHOLD)
        if not is_stable:
            # print(f"[REWARD] ❌ Cube unstable (lin_vel={linear_speed:.4f}, ang_vel={angular_speed:.4f}) → Reward: 0.0")
            return 0.0
        
        # check 7: Right gripper should be atleast 50% open
        right_gripper_pos = physics.data.ctrl[1]  # RIGHT robot gripper actuator position
        if right_gripper_pos < 0.022:
            return 0.0

        # ==========================
        # 4. SUCCESS!
        # ==========================
        # All checks passed: cube is on target, released, stable, and left arm is idle
        print(f"[REWARD] ✅✅✅ SUCCESS! All checks passed:")
        print(f"         ✓ Left arm idle (deviation={left_deviation:.4f}m)")
        print(f"         ✓ Cube touching target")
        print(f"         ✓ Cube released (not grasped)")
        print(f"         ✓ Cube lifted (not on source)")
        print(f"         ✓ Cube height correct (z={cube_pos[2]:.4f}m)")
        print(f"         ✓ Cube XY aligned (dist={xy_dist:.4f}m)")
        print(f"         ✓ Cube stable (lin={linear_speed:.4f}, ang={angular_speed:.4f})")
        print(f"         ✓ Right gripper open (pos={right_gripper_pos:.4f}m)")
        print(f"         → REWARD: 1.0")
        return 1.0





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
    np.array([ -0.125,   0.1, 0.355])  # MAXS
)

# --- RIGHT ROBOT (Worker) ---
# Format: ([x_min, y_min, z_min], [x_max, y_max, z_max])
# Visualized in ./dataset_utils/visualize_bbox.py validate_action_sampling()
RIGHT_CARTESIAN_BOUNDS = (
    np.array([-0.10, -0.15, -0.02]),  # MINS  
    np.array([ 0.275,  0.33,  0.35])   # MAXS
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
class SERLGymWrapper1(gym.Env):
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
        state_obs_dim: int = 16,
        fake_env: bool = False,
        image_obs: bool = True,
        max_episode_length: int = 400,
        arm_type: str = "viperx",
        onscreen_render: bool = False,
        cam_list: list = None,
        action_dim: int = 14,
        time_limit: float = float("inf"),
        action_scale = [1.0, 1.0, 0.005], # [position_scale, rotation_scale, gripper_scale] from stats analysis of /home/qte9489/personal_abhi/temp/serl/trossen_arm_mujoco/trossen_arm_mujoco/dataset_utils/abs_act_to_delta_act.py
        save_video: bool = False,
        video_path: str = None,
        control_mode = "delta",
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
        self.plt_imgs = None  # Matplotlib figure handles for visualization
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
            low=np.array([-0.003, -0.009, -0.25, -0.009, -0.02, -0.02, 1.0, -0.9, -0.4, -0.6, -0.6, -0.999, -0.999, -1.0]),
            high=np.array([0.002, 0.009, 0.0097, 0.008, 0.4, 0.02, 1.0, 0.6, 0.8, 0.999, 0.8, 0.999, 0.7, 1.0]),
            shape=(14,),
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
            dm_obs['mocap_pose_left'],
            np.array([dm_obs['gripper_ctrl'][0]]),
            dm_obs['mocap_pose_right'],
            np.array([dm_obs['gripper_ctrl'][1]]),
        ]) # 16D state for EE control mode
        
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
            self.plt_imgs = plot_observation_images(viz_obs, self.cam_list)
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
        print(f"obs keys after reset: {list(obs.keys())}")
        print(f"State observation after reset: {obs['state']}, type: {type(obs['state'])}, shape: {obs['state'].shape}, dtype: {obs['state'].dtype}")
        
        print("[SERLGymWrapper] Updating visualization on reset...")
        self._update_visualization()
        print("[SERLGymWrapper] Visualization updated.")
        print("[SERLGymWrapper] Environment reset complete. returning below obs")
        # self.pretty_print_obs(obs)
        print(f"obs keys before pop dm obs: {list(obs.keys())}")
        print(f"after pop dm obs keys: {list(obs.keys())}")
        # self.pretty_print_obs(obs)
        return obs, {}
  
    def _denormalize_gripper(self, normalized_grip: float) -> float:
        """
        Convert normalized gripper [-1, +1] to physical units [0, 0.044].
        
        Args:
            normalized_grip: Gripper value in [-1, +1] range
        
        Returns:
            Physical gripper value in [0, 0.044] range
        """
        GRIPPER_MIN = 0.0
        GRIPPER_MAX = 0.044
        # Clip and denormalize: physical = (normalized + 1) / 2 * range + min
        normalized_grip = np.clip(normalized_grip, -1.0, 1.0)
        physical_grip = (((normalized_grip + 1.0) * (GRIPPER_MAX - GRIPPER_MIN)) / 2.0 )+ GRIPPER_MIN
        return physical_grip

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

            # 1. GET CURRENT STATE (Concat into 7D: Pos(3) + RotAA(3) + Grip(1))
            # Note: We need the Gripper state too for Delta Gripper control.
            if self._steps % 250 == 0: print(f"original daction: {daction.tolist()}")
            # Scaling back to original physical units
            daction[0:3] = daction[0:3] * self.action_scale[0]     # Left position
            daction[3:6] = daction[3:6] * self.action_scale[1]     # Left rotation
            daction[6] = daction[6] * self.action_scale[2]         # Left gripper delta
            daction[7:10] = daction[7:10] * self.action_scale[0]   # Right position
            daction[10:13] = daction[10:13] * self.action_scale[1] # Right rotation
            daction[13] = daction[13] * self.action_scale[2]       # Right gripper delta
            if self._steps % 250 == 0: print(f"scaled daction: {daction.tolist()}")

            current_left_quat = self.env.physics.data.mocap_quat[0].copy() # [w,x,y,z]
            current_left_pos = self.env.physics.data.mocap_pos[0].copy()
            current_left_aa = quaternion_to_angle_axis(current_left_quat)
            current_left_grip = self.env.physics.data.ctrl[0].copy()

            current_right_quat = self.env.physics.data.mocap_quat[1].copy()
            current_right_pos = self.env.physics.data.mocap_pos[1].copy()
            current_right_aa = quaternion_to_angle_axis(current_right_quat)
            current_right_grip = self.env.physics.data.ctrl[1].copy()

            if self._steps % 250 == 0: print(f"[CURRENT STATE] Left pos: {current_left_pos}, Right pos: {current_right_pos}")
            if self._steps % 250 == 0: print(f"[BOUNDS] Left: min={LEFT_CARTESIAN_BOUNDS[0]}, max={LEFT_CARTESIAN_BOUNDS[1]}")
            if self._steps % 250 == 0: print(f"[BOUNDS] Right: min={RIGHT_CARTESIAN_BOUNDS[0]}, max={RIGHT_CARTESIAN_BOUNDS[1]}")

            # 2. APPLY DELTAS (For Pose Only)
            # daction structure: [L_dPos(3), L_dRot(3), L_Grip(1), R_dPos(3), R_dRot(3), R_Grip(1)]
            
            # Left Arm
            target_left_pos_before_clip = current_left_pos + daction[0:3]
            if self._steps % 250 == 0: print(f"[LEFT] Delta: {daction[0:3]}, Target before clip: {target_left_pos_before_clip}")
            target_left_pos = np.clip(target_left_pos_before_clip, *LEFT_CARTESIAN_BOUNDS)
            left_clipped = not np.allclose(target_left_pos_before_clip, target_left_pos, atol=1e-6)
            if left_clipped:
                if self._steps % 250 == 0: print(f"[CLIP WARNING] ⚠️  LEFT EE clipped! Before: {target_left_pos_before_clip}, After: {target_left_pos}")
            else:
                if self._steps % 250 == 0: print(f"[LEFT] ✓ No clipping needed, target: {target_left_pos}")
            target_left_aa  = current_left_aa  + daction[3:6]  # absolute rotation
            
            # Right Arm
            target_right_pos_before_clip = current_right_pos + daction[7:10]
            if self._steps % 250 == 0: print(f"[RIGHT] Delta: {daction[7:10]}, Target before clip: {target_right_pos_before_clip}")
            target_right_pos = np.clip(target_right_pos_before_clip, *RIGHT_CARTESIAN_BOUNDS)
            # target_right_pos = target_right_pos_before_clip  # DISABLE CLIPPING FOR RIGHT ARM
            right_clipped = not np.allclose(target_right_pos_before_clip, target_right_pos, atol=1e-6)
            if right_clipped:
                if self._steps % 250 == 0: print(f"[CLIP WARNING] ⚠️  RIGHT EE clipped! Before: {target_right_pos_before_clip}, After: {target_right_pos}")
            else:
                if self._steps % 250 == 0: print(f"[RIGHT] ✓ No clipping needed, target: {target_right_pos}")
            target_right_aa  = current_right_aa  + daction[10:13]  # absolute rotation
            # 3. CONVERT ROTATION BACK TO QUATERNION
            # MuJoCo needs [w, x, y, z], Scipy gives [x, y, z, w]
            target_left_quat = angle_axis_to_quaternion(target_left_aa)
            
            target_right_quat = angle_axis_to_quaternion(target_right_aa)

            # 4. HANDLE GRIPPERS (Delta Control)
            # daction[6] and daction[13] are now delta values in physical units
            # Add delta to current gripper position and clip to valid range [0, 0.044]
            if self._steps % 250 == 0: print(f"[GRIPPER] Left current: {current_left_grip}, Delta: {daction[6]}")
            if self._steps % 250 == 0: print(f"[GRIPPER] Right current: {current_right_grip}, Delta: {daction[13]}")
            target_left_grip = np.array([current_left_grip + daction[6]])
            target_right_grip = np.array([current_right_grip + daction[13]])
            if self._steps % 250 == 0: print(f"[GRIPPER] Left target (before clip): {target_left_grip}")
            if self._steps % 250 == 0: print(f"[GRIPPER] Right target (before clip): {target_right_grip}")
            
            target_left_grip_final = np.clip(target_left_grip, 0.0, 0.044)  # No extra np.array wrap
            target_right_grip_final = np.clip(target_right_grip, 0.0, 0.044)  # No extra np.array wrap
            if self._steps % 250 == 0: print(f"[GRIPPER] Left target (after clip): {target_left_grip_final}")
            if self._steps % 250 == 0: print(f"[GRIPPER] Right target (after clip): {target_right_grip_final}")
            action_to_take = np.concatenate([
                target_left_pos, target_left_quat, target_left_grip_final,
                target_right_pos, target_right_quat, target_right_grip_final
            ])
            
            if self._steps % 250 == 0: print(f"[FINAL TARGETS] Left: {target_left_pos}, Right: {target_right_pos}")
            if self._steps % 250 == 0: print(f"[CLIPPING SUMMARY] Left clipped: {left_clipped}, Right clipped: {right_clipped}")
            if self._steps % 250 == 0: print(f"action to take (physical units): {action_to_take.tolist()}")
        elif self.control_mode == "teleop":
            # print("[SERLGymWrapper] Non-delta control mode selected.")
            # Direct action mode (no delta)
            target_left_pos = daction[0:3]
            target_left_aa  = daction[3:6]
            target_right_pos = daction[7:10]
            target_right_aa  = daction[10:13]
            target_left_quat = angle_axis_to_quaternion(target_left_aa)
            target_right_quat = angle_axis_to_quaternion(target_right_aa)
            
            # target_left_world = transform_robot_to_world_frame(target_left_pos, target_left_quat, robot_name="left")
            # target_right_world = transform_robot_to_world_frame(target_right_pos, target_right_quat, robot_name="right")
            
            target_left = np.concatenate([target_left_pos, target_left_quat, np.array([daction[6]])])
            target_right = np.concatenate([target_right_pos, target_right_quat, np.array([daction[13]])])
            action_to_take = np.concatenate([
                target_left,
                target_right
            ])
            # print(f"action to take (physical units): {action_to_take.tolist()}")

        # Step dm_control environment
        dm_timestep = self.env.step(action_to_take)
        
        # Verify control frequency
        # print(f"Step {self._steps}: Physics time = {self.env.physics.time()}, Physics timestep = {self.env.physics.model.opt.timestep}")
        
        # Verify EE positions after step
        actual_left_pos = self.env.physics.data.mocap_pos[0].copy()
        actual_right_pos = self.env.physics.data.mocap_pos[1].copy()
        left_within = np.all(actual_left_pos >= LEFT_CARTESIAN_BOUNDS[0]) and np.all(actual_left_pos <= LEFT_CARTESIAN_BOUNDS[1])
        right_within = np.all(actual_right_pos >= RIGHT_CARTESIAN_BOUNDS[0]) and np.all(actual_right_pos <= RIGHT_CARTESIAN_BOUNDS[1])
        
        if not left_within:
            if self._steps % 250 == 0: print(f"[BOUNDS VIOLATION] LEFT EE OUT OF BOUNDS! Pos: {actual_left_pos}")
            if self._steps % 250 == 0: print(f"                   Bounds: min={LEFT_CARTESIAN_BOUNDS[0]}, max={LEFT_CARTESIAN_BOUNDS[1]}")
        if not right_within:
            if self._steps % 250 == 0: print(f"[BOUNDS VIOLATION] RIGHT EE OUT OF BOUNDS! Pos: {actual_right_pos}")
            if self._steps % 250 == 0: print(f"                   Bounds: min={RIGHT_CARTESIAN_BOUNDS[0]}, max={RIGHT_CARTESIAN_BOUNDS[1]}")
        
        # Process observation
        obs = self._process_observation(dm_timestep.observation)

        # === DISABLE LEFT ARM VIOLATION CHECK ===
        # We now mask left arm actions to prevent movement, so no need to terminate episodes
        # The agent can now explore and learn the right arm stacking task
        
        # Extract reward and termination
        reward = dm_timestep.reward if dm_timestep.reward is not None else 0.0
        
        # Check truncation (max steps)
        self._steps += 1
        truncated = self._steps >= self.max_episode_length
        
        # Task success: terminate if reward >= 1.0 (cube stacked successfully)
        terminated = reward >= 1.0
        is_success = terminated  # Success only if task completed (reward >= 1.0)
        
        # For SERL compatibility: done = terminated OR truncated
        done = terminated or truncated
        
        # Build info dict for data collection tracking
        info = {"is_success": is_success}
        if done:
            if terminated:
                info["termination_reason"] = "task_success"
            else:
                info["termination_reason"] = "max_steps_reached"
        
        # Debug: print when episode ends
        if done:
            reason = "SUCCESS (reward >= 1.0)" if terminated else f"TRUNCATED (max_steps={self.max_episode_length})"
            print(f"[DEBUG] Episode done at step {self._steps}: {reason}, reward={reward:.4f}")


        self._update_visualization()
        
        # Collect frame for video if enabled
        if self.save_video and self._images_full_res is not None:
            # Use cam_high for video (or create a grid of all cameras)
            frame = self._create_video_frame()
            self._video_frames.append(frame)
        
        # Accumulate reward for episode
        self._episode_reward += reward
        
        # Save video when episode ends
        if done and self.save_video and len(self._video_frames) > 0:
            self._save_episode_video(terminated, self._episode_reward)
        
        # print(f"State obs from mujoco: {obs['state']}")
        # print("Printing observation at step end: \n")
        # self.pretty_print_obs(obs)
        return obs, reward, done, truncated, info
    
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
        if hasattr(self, 'plt_imgs') and self.plt_imgs is not None:
            import matplotlib.pyplot as plt
            plt.close('all')  # Close all matplotlib figures
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


import pickle
import time

def load_transitions(file_path: str) -> list:
    print(f"Loading data from {file_path}...")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, dict) and 'transitions' in data:
        print("Data is a dict with 'transitions' key.")
        return data['transitions']
    elif isinstance(data, list):
        print("Data is a list of transitions.")
        return data
    else:
        raise ValueError("Unexpected data format in pickle file.")

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
        print(f"--- State Check: {status} ---")
        print(f"L2 Error: {l2_error:.6f} | Max Abs Error: {max_error:.6f} | Tolerance: {tolerance}")
        
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

def regenerate_ds_state_from_obs(pkl_file_path: str, output_file_path: str):
    # Simulation Config
    CONTROL_TIMESTEP = 0.02
    PHYSICS_TIMESTEP = 0.002
    cam_list = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]
    onscreen_render = False # Set to False for faster processing, True to watch

    # --- Load Data ---
    transitions = load_transitions(pkl_file_path)
    episode_boundaries = get_episode_boundaries(transitions)
    print(f"Found {len(episode_boundaries)} episodes in the dataset.")
    import time
    time.sleep(2.0)
    print(f"Loaded {len(transitions)} transitions comprising {len(episode_boundaries)} episodes.")

    dm_env = make_sim_env(
        CubeStackingEE,
        task_name="sim_transfer_cube",
        onscreen_render=onscreen_render,
        cam_list=cam_list,
        control_timestep=CONTROL_TIMESTEP, # control freq = 1/control_timestep = 50 Hz
        physics_timestep=PHYSICS_TIMESTEP, # n_substeps = control_timestep / physics_timestep = 10
    )
    print("Created dm_env")
    gym_env = SERLGymWrapper1(
        env = dm_env,
        state_obs_dim = 16,
        onscreen_render = onscreen_render,
        cam_list = cam_list,
        action_dim = 14,
        time_limit = 20.0,
        control_mode = "delta",
        action_scale = [0.025, 0.1, 0.02],  # Position, Rotation, Gripper (was 0.05, too large!)
    )
    gym_env = SERLObsWrapper(gym_env)
    gym_env = ChunkingWrapper(gym_env, obs_horizon=1, act_exec_horizon=None)
    
    new_transitions = []

    episode_idx_without_reward1_or_done = []
    
    # --- Main Replay Loop ---
    for ep_idx, (start, end) in enumerate(tqdm(episode_boundaries, desc="Replaying Episodes")):
    
        # Slice out the current episode's original data
        original_episode = transitions[start:end]

        # Extract cube_pose from first transition to hardcode it during reset
        cube_pose_from_dataset = original_episode[0]['observations']['full_observation']['cube_pose']
        print(f"\n[Dataset Replay] Using cube_pose from episode {ep_idx}: {cube_pose_from_dataset[:3]}")
        # Reset with hardcoded cube pose
        current_obs, _ = gym_env.reset(options={'cube_pose': cube_pose_from_dataset})
        time.sleep(2.0)
        last_action = None
        last_reward = 0.0
        
        for i, original_trans in enumerate(original_episode):
            # Create a Deep Copy to avoid shared memory reference bugs
            # We discard the old 'observations' and 'next_observations' data here essentially
            new_trans = copy.deepcopy(original_trans)
            
            # 2. Overwrite 'observations' with the actual current simulator state
            new_trans['observations']['state'] = current_obs['state']
            for cam in cam_list:
                if cam in current_obs:
                    new_trans['observations'][cam] = current_obs[cam]

            # 3. Step the environment using the ORIGINAL action
            action = original_trans['actions']
            next_obs, reward, done, truncated, info = gym_env.step(action)
            
            # 4. Overwrite 'next_observations', reward, done
            new_trans['next_observations']['state'] = next_obs['state']
            for cam in cam_list:
                if cam in next_obs:
                    new_trans['next_observations'][cam] = next_obs[cam]
            
            new_trans['rewards'] = reward
            new_trans['dones'] = done
            
            # Append to our new master list
            new_transitions.append(new_trans)
            
            # Update loop variables
            current_obs = next_obs
            last_action = action
            last_reward = reward
            
            # If the sim says we are done, we stop processing this episode's original transitions
            # (This handles cases where the sim might end earlier than the recording)
            if done:
                break

        # --- Retry Logic (Extension) ---
        # If we finished the recorded trajectory but reward is not 1.0 (success),
        # try repeating the last action up to 5 times.
        if last_reward < 1.0 and not done:
            # print(f"Episode {ep_idx} did not succeed. Retrying last action...")
            for retry_step in range(5):
                # Create a new transition for this extra step
                retry_trans = copy.deepcopy(new_transitions[-1])
                
                # Setup Observations (Current)
                retry_trans['observations']['state'] = current_obs['state']
                for cam in cam_list:
                    retry_trans['observations'][cam] = current_obs[cam]
                
                # Execute Last Action again
                next_obs, reward, done, truncated, info = gym_env.step(last_action)
                
                # Setup Next Observations
                retry_trans['next_observations']['state'] = next_obs['state']
                for cam in cam_list:
                    retry_trans['next_observations'][cam] = next_obs[cam]
                
                retry_trans['actions'] = last_action
                retry_trans['rewards'] = reward
                retry_trans['dones'] = done
                
                new_transitions.append(retry_trans)
                
                current_obs = next_obs
                
                if reward == 1.0 or done:
                    # print(f"  -> Success achieved on retry step {retry_step+1}")
                    break
            
            # CRITICAL FIX: If retry loop exhausted without success, force episode to end
            # This ensures every episode has a proper dones=True marker
            if new_transitions[-1]['dones'] == False:
                print(f"[WARNING] Episode {ep_idx} failed after {len(original_episode)} original + {retry_step+1} retry steps. Forcing done=True.")
                episode_idx_without_reward1_or_done.append(ep_idx)
                new_transitions[-1]['dones'] = True

    # --- Save Result ---
    # Save as list of transitions directly (not wrapped in extra list)
    print(f"Replay complete. Original count: {len(transitions)}, New count: {len(new_transitions)}")
    
    with open(output_file_path, 'wb') as f:
        pickle.dump(new_transitions, f)
    print(f"Saved replayed dataset to {output_file_path}")
    print(f"Episodes without reward=1.0 or done=True: {episode_idx_without_reward1_or_done}")


def test_data_replay_sim_env(pkl_file_path, episode_indices=None):
    """
    Replay episodes from the dataset.
    
    Args:
        episode_indices: int, list of ints, or None
            - int: replay single episode (e.g., 13)
            - list: replay multiple episodes (e.g., [0, 5, 13])
            - None: replay all episodes
    """
    # pkl_file_path = "/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/same_cube_pose/sim_dataset_binarized_gripper_regenerated_states_FINAL.pkl"
    # state_check_file_path = "/home/qte9489/personal_abhi/temp/serl/sim_recordings_ee/v1/same_cube_pose/act_ee_and_state_quat/sim_ds_act_ee_and_state_quat_processed_filtered.pkl"
    transitions = load_transitions(pkl_file_path)
    # state_check_transitions = load_transitions(state_check_file_path)
    print(f"Loaded {len(transitions)} transitions from dataset.")
    # print(f"Loaded {len(state_check_transitions)} transitions from state check dataset.")
    # return 
    episode_boundaries = get_episode_boundaries(transitions)
    print(f"Found {len(episode_boundaries)} episodes in the dataset.")
    import time
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
    gym_env = SERLGymWrapper1(
        env = dm_env,
        state_obs_dim = 16,
        onscreen_render = onscreen_render,
        cam_list = cam_list,
        action_dim = 14,
        time_limit = 20.0,
        control_mode = "delta",
        action_scale = [0.025, 0.1, 1],  # Position, Rotation, Gripper (was 0.05, too large!)
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

        pretty_print_obs(episode_transitions[0]['observations'], indent=4)
        state_check_results = []
        
        # Extract cube_pose from first transition to hardcode it during reset
        cube_pose_from_dataset = episode_transitions[0]['observations']['full_observation']['cube_pose']
        print(f"\n[Dataset Replay] Using cube_pose from episode {ep_to_replay}: {cube_pose_from_dataset[:3]}")
        time.sleep(2.0)
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
            import time
            # time.sleep(0.25)
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
    gym_env = SERLGymWrapper1(
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
    # --- CONFIGURATION ---
    # input_file_path = "/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/same_cube_pose/sim_dataset_binarized_gripper_regenerated_states.pkl"
    # output_file_path = "/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/same_cube_pose/sim_dataset_binarized_gripper_regenerated_states_FINAL.pkl"
    
    # LIST OF EPISODES TO DELETE (0-based indices)
    # Example: [0, 41, 5]
    # episodes_to_delete = [44] 
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
    final_data = kept_transitions
    
    print(f"Saving to {output_file_path}...")
    with open(output_file_path, 'wb') as f:
        pickle.dump(final_data, f)
    print("Done.")


if __name__ == "__main__":
    pkl_file_path = "/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/random_cube_pose/cleaned/merged_pkl_data_action_in_world_frame_static_filtered_subsampled_action_shifted_regenerated_deleted_some_fulldelta_act_scaled_act_epsilon_binarized_regenerated_deleted_episodes_epsilon.pkl"
    test_data_replay_sim_env(pkl_file_path, episode_indices=[0])
    
    # regenerae dataset base on input ds
    # pkl_file_path = "/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/random_cube_pose/cleaned/merged_pkl_data_action_in_world_frame_static_filtered_subsampled_action_shifted_regenerated_deleted_some_fulldelta_act_scaled_act_epsilon_binarized.pkl"
    # output_file_path = "/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/random_cube_pose/cleaned/merged_pkl_data_action_in_world_frame_static_filtered_subsampled_action_shifted_regenerated_deleted_some_fulldelta_act_scaled_act_epsilon_binarized_regenerated.pkl"
    # regenerate_ds_state_from_obs(pkl_file_path, output_file_path)

    # delete specific episodes
    # input_file_path = "/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/random_cube_pose/cleaned/merged_pkl_data_action_in_world_frame_static_filtered_subsampled_action_shifted_regenerated_deleted_some_fulldelta_act_scaled_act_epsilon_binarized_regenerated.pkl"
    # output_file_path = "/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/random_cube_pose/cleaned/merged_pkl_data_action_in_world_frame_static_filtered_subsampled_action_shifted_regenerated_deleted_some_fulldelta_act_scaled_act_epsilon_binarized_regenerated_deleted_episodes.pkl"
    # delete_specific_episodes_and_save(input_file_path=input_file_path, output_file_path=output_file_path, episodes_to_delete=[13, 43, 53, 57, 72, 74])