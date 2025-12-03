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

        :param physics: The MuJoCo physics simulation instance.
        """
        # TODO Notice: this function does not randomize the env configuration. Instead, set
        # BOX_POSE from outside reset qpos, control and box position
        with physics.reset_context():
            # Reset the arm pose
            physics.named.data.qpos[:16] = START_ARM_POSE_MEAN

            # Randomize or sample the cube’s initial position
            cube_pose = sample_box_pose()
            print("cube_pose init : ", cube_pose)
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

    def get_reward(self, physics: Physics) -> int:
        """
        Computes the reward based on whether the cube has been transferred successfully.

        :param physics: The MuJoCo physics simulation instance.
        :return: The computed reward which is whether left gripper is holding the box
        """
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, "geom")
            name_geom_2 = physics.model.id2name(id_geom_2, "geom")
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_right_gripper = (
            "red_box",
            "right/gripper_follower_left",
        ) in all_contact_pairs
        touch_blue_table = (
            "red_box",
            "table_box",
        ) in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs

        reward = 0
        #if touch_right_gripper:
        #    reward = 1
        # lifted
        #if touch_right_gripper and not touch_table:
        #    reward = 2
        # attempted transfer
        #if touch_right_gripper and touch_blue_table: 
        #    return 3
        if touch_blue_table and not touch_right_gripper: 
            return 1
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
        """
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE_MEAN
            cube_pose = sample_box_pose()
            print("cube_pose init : ", cube_pose)
            box_start_idx = physics.model.name2id("red_box_joint", "joint")
            np.copyto(physics.data.qpos[box_start_idx : box_start_idx + 7], cube_pose)
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics: Physics) -> np.ndarray:
        """Get cube state (qpos[16:])."""
        return physics.data.qpos.copy()[16:]

    def get_reward(self, physics: Physics) -> int:
        """Reward: 1 if cube on blue table and not held by gripper."""
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, "geom")
            name_geom_2 = physics.model.id2name(id_geom_2, "geom")
            all_contact_pairs.append((name_geom_1, name_geom_2))

        touch_right_gripper = ("red_box", "right/gripper_follower_left") in all_contact_pairs
        touch_blue_table = ("red_box", "table_box") in all_contact_pairs

        if touch_blue_table and not touch_right_gripper:
            return 1
        return 0
        return reward