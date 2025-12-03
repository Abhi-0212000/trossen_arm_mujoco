"""
Base Task Classes for Trossen AI Bimanual Robot Simulation.

This module provides two base task classes for different control modes:
1. TrossenAIStationaryTask - Joint/Position Control
2. TrossenAIStationaryEETask - End-Effector (Cartesian) Control

================================================================================
CONTROL MODES OVERVIEW
================================================================================

JOINT CONTROL (TrossenAIStationaryTask):
    - Action: 14D [L_Arm(6), L_Grip(1), R_Arm(6), R_Grip(1)]
    - Control: Direct joint position targets via physics.data.ctrl
    - XML: trossen_ai_scene_joint.xml
    - Use Case: BC/RL training with joint-space actions

EE CONTROL (TrossenAIStationaryEETask):
    - Action: 16D [L_Pos(3), L_Quat(4), L_Grip(1), R_Pos(3), R_Quat(4), R_Grip(1)]
    - Control: Mocap bodies + weld constraints for IK
    - XML: trossen_ai_scene.xml (with mocap bodies and weld constraints)
    - Use Case: Teleoperation, Cartesian-space policies

================================================================================
QPOS INDEXING (16 joints + 7 cube = 23 total)
================================================================================
From trossen_ai_bimanual.xml / trossen_ai_joint.xml:

    Index | Joint Name                  | Description
    ------|-----------------------------|--------------------------
    0     | left/joint_0                | Left arm base rotation (Z)
    1     | left/joint_1                | Left arm shoulder (Y)
    2     | left/joint_2                | Left arm elbow (Y)
    3     | left/joint_3                | Left arm wrist1 (Y)
    4     | left/joint_4                | Left arm wrist2 (Z)
    5     | left/joint_5                | Left arm wrist3 (X)
    6     | left/right_carriage_joint   | Left gripper finger R (slide)
    7     | left/left_carriage_joint    | Left gripper finger L (slide)
    8     | right/joint_0               | Right arm base rotation (Z)
    9     | right/joint_1               | Right arm shoulder (Y)
    10    | right/joint_2               | Right arm elbow (Y)
    11    | right/joint_3               | Right arm wrist1 (Y)
    12    | right/joint_4               | Right arm wrist2 (Z)
    13    | right/joint_5               | Right arm wrist3 (X)
    14    | right/right_carriage_joint  | Right gripper finger R (slide)
    15    | right/left_carriage_joint   | Right gripper finger L (slide)
    16-22 | red_box_joint               | Cube free joint (x,y,z,qw,qx,qy,qz)

================================================================================
CTRL INDEXING (16 actuators for joint control)
================================================================================
From trossen_ai_joint.xml actuator section:

    Index | Actuator Name     | Controls
    ------|-------------------|---------------------------
    0     | left/joint_0      | Left arm joint 0
    1     | left/joint_1      | Left arm joint 1
    2     | left/joint_2      | Left arm joint 2
    3     | left/joint_3      | Left arm joint 3
    4     | left/joint_4      | Left arm joint 4
    5     | left/joint_5      | Left arm joint 5
    6     | left/joint_gripper| Left gripper (both fingers via equality)
    7     | (coupled)         | Left gripper coupled
    8     | right/joint_0     | Right arm joint 0
    9     | right/joint_1     | Right arm joint 1
    10    | right/joint_2     | Right arm joint 2
    11    | right/joint_3     | Right arm joint 3
    12    | right/joint_4     | Right arm joint 4
    13    | right/joint_5     | Right arm joint 5
    14    | right/joint_gripper| Right gripper (both fingers via equality)
    15    | (coupled)         | Right gripper coupled

================================================================================
MOCAP INDEXING (for EE control)
================================================================================
From trossen_ai_scene.xml:

    Index | Mocap Body   | Initial Position (world frame)
    ------|--------------|--------------------------------
    0     | mocap_left   | [-0.2062, -0.019, 0.1835]
    1     | mocap_right  | [0.2062, -0.019, 0.1835]

    Weld Constraints:
    - mocap_left  <-> left/link_6  (left end-effector)
    - mocap_right <-> right/link_6 (right end-effector)

================================================================================
ROBOT BASE POSITIONS (from trossen_ai_bimanual.xml)
================================================================================

    Robot | Position (world)        | Quaternion (wxyz)   | Notes
    ------|-------------------------|---------------------|------------------
    Left  | [-0.4575, -0.019, 0.02] | [1, 0, 0, 0]       | Identity rotation
    Right | [0.4575, -0.019, 0.02]  | [0, 0, 0, 1]       | 180° around Z-axis

================================================================================
CAMERA IDs
================================================================================

    ID | Camera Name      | Location
    ---|------------------|------------------
    0  | teleoperator_pov | Overhead view
    1  | cam_high         | High angle
    2  | cam_low          | Low angle (worm's eye)
    3  | cam_left_wrist   | Left end-effector
    4  | cam_right_wrist  | Right end-effector

================================================================================
"""

import collections
from typing import List, Optional

from dm_control.mujoco.engine import Physics
from dm_control.suite import base
import numpy as np

from trossen_arm_mujoco.constants import START_ARM_POSE, START_ARM_POSE_MEAN
from trossen_arm_mujoco.utils import get_observation_base


class TrossenAIStationaryTask(base.Task):
    """
    Base task for bimanual manipulation with JOINT/POSITION control.
    
    Action Format: 14D [L_Arm(6), L_Grip(1), R_Arm(6), R_Grip(1)]
    
    The action is processed in before_step():
    1. Split into left/right arm and gripper components
    2. Duplicate gripper value for both fingers (equality constraint in XML)
    3. Pass 16D action to physics.data.ctrl via super().before_step()
    
    Subclasses should implement:
    - initialize_episode(): Set initial robot pose and environment state
    - get_env_state(): Return task-specific environment state
    - get_reward(): Compute task-specific reward
    """

    def __init__(
        self,
        random: Optional[int] = None,
        onscreen_render: bool = False,
        cam_list: List[str] = [],
    ):
        """
        Initialize the joint control task.
        
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

    def before_step(self, action: np.ndarray, physics: Physics) -> None:
        """
        Process 14D action and apply to simulation.
        
        Action mapping:
            action[0:6]  -> Left arm joints (ctrl[0:6])
            action[6]    -> Left gripper (ctrl[6:8], duplicated)
            action[7:13] -> Right arm joints (ctrl[8:14])
            action[13]   -> Right gripper (ctrl[14:16], duplicated)
        
        Args:
            action: 14D array [L_Arm(6), L_Grip(1), R_Arm(6), R_Grip(1)]
            physics: MuJoCo physics instance
        """
        # Split action into components
        left_arm_action = action[:6]
        left_gripper_action = action[6]
        right_arm_action = action[7:13]
        right_gripper_action = action[13]

        # Duplicate gripper values for both fingers
        # (XML has equality constraint: left_carriage = right_carriage)
        full_left_gripper = [left_gripper_action, left_gripper_action]
        full_right_gripper = [right_gripper_action, right_gripper_action]

        # Concatenate to 16D ctrl array
        env_action = np.concatenate([
            left_arm_action,      # ctrl[0:6]
            full_left_gripper,    # ctrl[6:8]
            right_arm_action,     # ctrl[8:14]
            full_right_gripper,   # ctrl[14:16]
        ])
        
        self.counter += 1
        super().before_step(env_action, physics)

    def initialize_episode(self, physics: Physics) -> None:
        """Initialize episode state. Override in subclasses."""
        self.counter = 0
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics: Physics) -> np.ndarray:
        """Get environment state. Override in subclasses."""
        return physics.data.qpos.copy()

    def get_position(self, physics: Physics) -> np.ndarray:
        """Get current joint positions (16D: qpos[0:16])."""
        return physics.data.qpos.copy()[:16]

    def get_velocity(self, physics: Physics) -> np.ndarray:
        """Get current joint velocities (16D: qvel[0:16])."""
        return physics.data.qvel.copy()[:16]

    def get_observation(self, physics: Physics) -> collections.OrderedDict:
        """
        Get observation dictionary.
        
        Returns:
            OrderedDict with keys:
            - 'images': Dict of camera images (if onscreen_render=True)
            - 'qpos': 16D joint positions
            - 'qvel': 16D joint velocities
            - 'env_state': Task-specific environment state
            - 'cube_pose': 7D [x, y, z, qw, qx, qy, qz] cube position and orientation
        """
        obs = get_observation_base(physics, self.cam_list, on_screen_render=self.onscreen_render)
        obs["qpos"] = self.get_position(physics)
        obs["qvel"] = self.get_velocity(physics)
        obs["env_state"] = self.get_env_state(physics)
        
        # Cube pose (7D: position + quaternion)
        # qpos[16:23] = [x, y, z, qw, qx, qy, qz] from red_box_joint free joint
        obs["cube_pose"] = physics.data.qpos[16:23].copy()
        
        return obs

    def get_reward(self, physics: Physics) -> float:
        """Compute reward. Override in subclasses."""
        raise NotImplementedError


class TrossenAIStationaryEETask(base.Task):
    """
    Base task for bimanual manipulation with END-EFFECTOR (Cartesian) control.
    
    Action Format: 16D [L_Pos(3), L_Quat(4), L_Grip(1), R_Pos(3), R_Quat(4), R_Grip(1)]
    
    The action is processed in before_step():
    1. Split into left/right EE pose and gripper components
    2. Set mocap_pos and mocap_quat for each arm
    3. Set gripper qpos directly (not via ctrl)
    
    Control Mechanism:
    - Mocap bodies (mocap_left, mocap_right) are welded to end-effectors
    - MuJoCo's equality constraints enforce IK-like behavior
    - Gripper controlled via direct qpos manipulation
    
    Subclasses should implement:
    - initialize_episode(): Set initial robot pose and environment state
    - get_env_state(): Return task-specific environment state
    - get_reward(): Compute task-specific reward
    """

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

    def before_step(self, action: np.ndarray, physics: Physics) -> None:
        """
        Process 16D EE action and apply to simulation.
        
        Action mapping:
            action[0:3]   -> Left EE position (mocap_pos[0])
            action[3:7]   -> Left EE quaternion wxyz (mocap_quat[0])
            action[7]     -> Left gripper (qpos[6:8])
            action[8:11]  -> Right EE position (mocap_pos[1])
            action[11:15] -> Right EE quaternion wxyz (mocap_quat[1])
            action[15]    -> Right gripper (qpos[14:16])
        
        Args:
            action: 16D array [L_Pos(3), L_Quat(4), L_Grip(1), R_Pos(3), R_Quat(4), R_Grip(1)]
            physics: MuJoCo physics instance
        """
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

        # Set gripper positions directly via qpos
        # Left gripper: qpos[6] and qpos[7] (both fingers)
        physics.data.qpos[6] = action_left[7]
        physics.data.qpos[7] = action_left[7]
        
        # Right gripper: qpos[14] and qpos[15] (both fingers)
        physics.data.qpos[14] = action_right[7]
        physics.data.qpos[15] = action_right[7]
        
        self.counter += 1
        # Note: We don't call super().before_step() for EE control
        # because we're not using the actuator system

    def initialize_robots(self, physics: Physics) -> None:
        """
        Initialize robot joint positions and mocap bodies.
        
        This should be called at the start of each episode to:
        1. Reset arm joints to home position
        2. Align mocap bodies with current end-effector positions
        """
        # Reset joint positions (only arm joints, not grippers)
        # Using first 6 joints from each arm in START_ARM_POSE
        physics.named.data.qpos[:6] = START_ARM_POSE[:6]      # Left arm
        physics.named.data.qpos[8:14] = START_ARM_POSE[8:14]  # Right arm
        
        # Reset grippers to open position
        physics.data.qpos[6:8] = START_ARM_POSE[6:8]    # Left gripper
        physics.data.qpos[14:16] = START_ARM_POSE[14:16]  # Right gripper

        # Initialize mocap bodies to align with end-effectors
        # These positions are the home EE positions in world frame
        # (Computed from FK at home joint configuration)
        np.copyto(physics.data.mocap_pos[0], [-2.04248170e-01, -1.90390477e-02, 1.88026731e-01])
        np.copyto(physics.data.mocap_quat[0], [1, 0, 0, 0])  # Identity quaternion
        
        np.copyto(physics.data.mocap_pos[1], [2.05969129e-01, -1.97438376e-02, 1.88026731e-01])
        np.copyto(physics.data.mocap_quat[1], [1, 0, 0, 0])  # Identity quaternion

    def initialize_episode(self, physics: Physics) -> None:
        """Initialize episode state. Override in subclasses."""
        self.counter = 0
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
        """
        Get observation dictionary for EE control.
        
        Returns:
            OrderedDict with keys:
            - 'images': Dict of camera images (if onscreen_render=True)
            - 'qpos': 16D joint positions
            - 'qvel': 16D joint velocities
            - 'env_state': Task-specific environment state
            - 'cube_pose': 7D [x, y, z, qw, qx, qy, qz] cube position and orientation
            - 'mocap_pose_left': 7D [pos(3), quat(4)] left EE pose
            - 'mocap_pose_right': 7D [pos(3), quat(4)] right EE pose
            - 'gripper_ctrl': Current gripper control values
            
            Additional keys for SERL/IBRL compatibility:
            - 'robot0_eef_pos': 6D [L_pos(3), R_pos(3)]
            - 'robot0_eef_quat': 8D [L_quat(4), R_quat(4)]
            - 'robot0_gripper_qpos': 2D [L_grip, R_grip]
        """
        obs = get_observation_base(physics, self.cam_list, on_screen_render=self.onscreen_render)
        obs["qpos"] = self.get_position(physics)
        obs["qvel"] = self.get_velocity(physics)
        obs["env_state"] = self.get_env_state(physics)
        
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
        
        # SERL/IBRL compatible keys
        obs["robot0_eef_pos"] = np.concatenate([
            obs["mocap_pose_left"][:3],
            obs["mocap_pose_right"][:3]
        ])
        obs["robot0_eef_quat"] = np.concatenate([
            obs["mocap_pose_left"][3:],
            obs["mocap_pose_right"][3:]
        ])
        # Single gripper value per arm (not both fingers)
        obs["robot0_gripper_qpos"] = np.concatenate([
            obs["qpos"][6:7],    # Left gripper (one finger)
            obs["qpos"][14:15]   # Right gripper (one finger)
        ])
        
        return obs

    def get_reward(self, physics: Physics) -> float:
        """Compute reward. Override in subclasses."""
        raise NotImplementedError
