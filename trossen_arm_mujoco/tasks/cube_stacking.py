"""
Cube Stacking Task for Trossen AI Bimanual Manipulation.

This module provides task implementations for the cube stacking scenario:
- CubeStacking: Joint control version (14D actions)
- CubeStackingEE: End-effector control version (16D actions)
"""

from dm_control.mujoco.engine import Physics
import numpy as np

from trossen_arm_mujoco.constants import START_ARM_POSE_MEAN
from trossen_arm_mujoco.utils import sample_box_pose
from trossen_arm_mujoco.tasks.base_tasks import (
    TrossenAIStationaryTask,
    TrossenAIStationaryEETask,
)


class CubeStacking(TrossenAIStationaryTask):
    """
    Cube stacking task with joint control.
    
    Action space: 14D [L_Arm(6), L_Grip(1), R_Arm(6), R_Grip(1)]
    XML scene: trossen_ai_scene_joint.xml
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
        
        Note: For reproducible cube placement, set np.random.seed() before calling env.reset().
        The SERLGymWrapper does this automatically when you pass seed to reset().

        :param physics: The MuJoCo physics simulation instance.
        """
        # TODO Notice: this function does not randomize the env configuration. Instead, set
        # BOX_POSE from outside reset qpos, control and box position
        with physics.reset_context():
            # Reset the arm pose
            physics.named.data.qpos[:16] = START_ARM_POSE_MEAN

            # Randomize or sample the cube's initial position
            # (Uses global np.random state, which can be seeded via wrapper)
            cube_pose = sample_box_pose()
            print(f"cube_pose init: {cube_pose[:3]}")
            box_start_idx = physics.model.name2id("red_box_joint", "joint")
            np.copyto(physics.data.qpos[box_start_idx : box_start_idx + 7], cube_pose)


        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics: Physics) -> np.ndarray:
        """
        Retrieves the environment state related to the cube position.

        :param physics: The MuJoCo physics simulation instance.
        :return: The environment state.
        """
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics: Physics) -> float:
        """
        Shaped reward for cube pick-and-place task (joint control version).
        
        Uses robust contact detection with frozensets for order-agnostic matching.
        Joint scene uses single "red_box" geom.

        Reward structure (cumulative):
        - Base: 0
        - Grasping cube: +0.1 (cube touching gripper)
        - Lifting cube: +0.2 (cube off source table while grasped)
        - Distance bonus: +0.3 * (1 - normalized_distance) (closer to target = higher)
        - Cube above target: +0.2 (cube XY aligned with target, grasped)
        - SUCCESS: +1.0 (cube on target, released)

        :param physics: The MuJoCo physics simulation instance.
        :return: Reward value between 0 and 1
        """
        # Collect all contact pairs as frozensets (order-agnostic)
        all_contact_pairs = set()
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, "geom")
            name_geom_2 = physics.model.id2name(id_geom_2, "geom")
            all_contact_pairs.add(frozenset([name_geom_1, name_geom_2]))

        # Define gripper geoms
        gripper_geoms = {
            "right/gripper_follower_left", "right/gripper_follower_right"
        }
        
        # === Contact Detection ===
        # Check if cube is being grasped (touching gripper)
        cube_grasped = any(
            frozenset(["red_box", gripper_geom]) in all_contact_pairs
            for gripper_geom in gripper_geoms
        )
        
        # Check if cube is on the source table (table_collision)
        cube_on_source = frozenset(["red_box", "table_collision"]) in all_contact_pairs
        
        # Check if cube is on the target table (table_box = blue box)
        cube_on_target = frozenset(["red_box", "table_box"]) in all_contact_pairs
        
        # === Position-based metrics ===
        # Get cube position (from qpos, cube joint starts at index 16)
        cube_pos = physics.data.qpos[16:19].copy()
        
        # Get gripper positions (center of fingers)
        r_finger_l = physics.named.data.xpos['right/carriage_left']
        r_finger_r = physics.named.data.xpos['right/carriage_right']
        r_gripper_pos = (r_finger_l + r_finger_r) / 2.0
        
        # Distance to cube
        dist_r = np.linalg.norm(r_gripper_pos - cube_pos)
        min_dist = dist_r
        
        # Target position (blue table_box center, top surface)
        target_pos = np.array([0.0, 0.22, 0.17])  # z = 0.02 (base) + 0.15 (height)
        
        # Distance from cube to target (XY plane)
        xy_distance = np.linalg.norm(cube_pos[:2] - target_pos[:2])
        max_xy_distance = 0.3  # Approximate max reach
        normalized_distance = min(xy_distance / max_xy_distance, 1.0)
        
        # Check if cube is above target (XY aligned within threshold)
        xy_aligned = xy_distance < 0.05  # Within 5cm of target XY
        
        # === Reward Calculation ===
        reward = 0.0
        
        # Reaching reward (always active)
        reaching_reward = 0.1 * (1.0 - np.tanh(5.0 * min_dist))
        reward += reaching_reward
        
        # SUCCESS: Cube on target and released (highest priority, sparse)
        if cube_on_target and not cube_grasped:
            return 1.0
        
        # Shaping rewards (for learning progress)
        if cube_grasped:
            reward += 0.1  # Grasping
            
            if not cube_on_source:
                reward += 0.2  # Lifted off source
                
            # Distance-based bonus (closer to target = higher reward)
            distance_bonus = 0.3 * (1.0 - normalized_distance)
            reward += distance_bonus
            
            if xy_aligned:
                reward += 0.2  # Above target, ready to drop
        
        return reward


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
        
        Note: For reproducible cube placement, set np.random.seed() before calling env.reset().
        """
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE_MEAN
            cube_pose = sample_box_pose()
            print(f"cube_pose init: {cube_pose[:3]}")
            box_start_idx = physics.model.name2id("red_box_joint", "joint")
            np.copyto(physics.data.qpos[box_start_idx : box_start_idx + 7], cube_pose)
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics: Physics) -> np.ndarray:
        """Get cube state (qpos[16:])."""
        return physics.data.qpos.copy()[16:]

    def get_reward(self, physics: Physics) -> float:
        """
        Shaped reward for cube pick-and-place task.
        
        Uses robust contact detection for subcube geoms (subcube1-4) in EE scene.
        
        Reward structure (cumulative):
        - Base: 0
        - Grasping cube: +0.1 (cube touching gripper)
        - Lifting cube: +0.2 (cube off source table while grasped)
        - Distance bonus: +0.3 * (1 - normalized_distance) (closer to target = higher)
        - Cube above target: +0.2 (cube XY aligned with target, grasped)
        - SUCCESS: +1.0 (cube on target, released) -> Total possible: 1.0 (sparse for success)
        
        Returns:
            float: Reward value between 0 and 1
        """
        # Collect all contact pairs as frozensets (order-agnostic)
        all_contact_pairs = set()
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, "geom")
            name_geom_2 = physics.model.id2name(id_geom_2, "geom")
            all_contact_pairs.add(frozenset([name_geom_1, name_geom_2]))

        # Debug: Print all contacts involving cube or gripper
        cube_geom_names = {"subcube1", "subcube2", "subcube3", "subcube4"}
        gripper_geom_names = {
            "right/gripper_follower_left", "right/gripper_follower_right"
        }
        relevant_contacts = [pair for pair in all_contact_pairs 
                             if any(g in pair for g in cube_geom_names | gripper_geom_names)]
        if relevant_contacts:
            print(f"[REWARD DEBUG] Relevant contacts: {relevant_contacts}")

        # Define cube geoms (EE scene uses subcube1-4)
        cube_geoms = {"subcube1", "subcube2", "subcube3", "subcube4"}
        
        # Define gripper geoms
        gripper_geoms = {
            "right/gripper_follower_left", "right/gripper_follower_right"
        }
        
        # === Contact Detection ===
        # Check if cube is being grasped (touching gripper)
        cube_grasped = any(
            frozenset([cube_geom, gripper_geom]) in all_contact_pairs
            for cube_geom in cube_geoms
            for gripper_geom in gripper_geoms
        )
        
        # Check if cube is on the source table (table_collision)
        cube_on_source = any(
            frozenset([cube_geom, "table_collision"]) in all_contact_pairs
            for cube_geom in cube_geoms
        )
        
        # Check if cube is on the target table (table_box = blue box)
        cube_on_target = any(
            frozenset([cube_geom, "table_box"]) in all_contact_pairs
            for cube_geom in cube_geoms
        )
        
        # === Position-based metrics ===
        # Get cube position (from qpos, cube joint starts at index 16)
        cube_pos = physics.data.qpos[16:19].copy()
        
        # Get gripper positions (center of fingers)
        r_finger_l = physics.named.data.xpos['right/carriage_left']
        r_finger_r = physics.named.data.xpos['right/carriage_right']
        r_gripper_pos = (r_finger_l + r_finger_r) / 2.0
        
        # Distance to cube
        dist_r = np.linalg.norm(r_gripper_pos - cube_pos)
        min_dist = dist_r
        
        # Target position (blue table_box center, top surface)
        target_pos = np.array([0.0, 0.22, 0.17])  # z = 0.02 (base) + 0.15 (height)
        
        # Distance from cube to target (XY plane mainly matters for alignment)
        xy_distance = np.linalg.norm(cube_pos[:2] - target_pos[:2])
        max_xy_distance = 0.3  # Approximate max reach
        normalized_distance = min(xy_distance / max_xy_distance, 1.0)
        
        # Check if cube is above target (XY aligned within threshold)
        xy_aligned = xy_distance < 0.05  # Within 5cm of target XY
        
        # === Reward Calculation ===
        reward = 0.0
        
        # Reaching reward (always active to guide to cube)
        reaching_reward = 0.1 * (1.0 - np.tanh(5.0 * min_dist))
        reward += reaching_reward
        
        # SUCCESS: Cube on target and released (highest priority, sparse)
        if cube_on_target and not cube_grasped:
            print(f"--- Reward Debug Step ---")
            print(f"SUCCESS! Cube on target and released.")
            print(f"Total Reward: 1.0")
            print(f"-------------------------")
            return 1.0
        
        # Shaping rewards (for learning progress)
        if cube_grasped:
            reward += 0.1  # Grasping
            
            if not cube_on_source:
                reward += 0.2  # Lifted off source
                
            # Distance-based bonus (closer to target = higher reward)
            distance_bonus = 0.3 * (1.0 - normalized_distance)
            reward += distance_bonus
            
            if xy_aligned:
                reward += 0.2  # Above target, ready to drop

        # Debug print block
        print(f"--- Reward Debug Step ---")
        print(f"Cube Pos: {cube_pos}")
        print(f"R Gripper Pos: {r_gripper_pos}")
        print(f"Dist to Cube: {dist_r:.4f}")
        print(f"Reaching Reward: {reaching_reward:.4f}")
        print(f"Contacts: {relevant_contacts}")
        print(f"Grasped: {cube_grasped}, OnSource: {cube_on_source}, OnTarget: {cube_on_target}")
        print(f"XY Dist to Target: {xy_distance:.4f}, Aligned: {xy_aligned}")
        print(f"Total Reward: {reward:.4f}")
        print(f"-------------------------")
        
        return reward