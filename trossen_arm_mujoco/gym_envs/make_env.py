"""
Convenience functions for creating SERL-compatible Trossen environments.

Control Modes:
    - 'joint': 14D actions [L_Arm(6), L_Grip(1), R_Arm(6), R_Grip(1)]
    - 'ee': 16D actions [L_Pos(3), L_Quat(4), L_Grip(1), R_Pos(3), R_Quat(4), R_Grip(1)]
"""

from dm_control import composer
from typing import Optional, Literal

from trossen_arm_mujoco.tasks.cube_stacking import CubeStacking, CubeStackingEE
from trossen_arm_mujoco.gym_envs import SERLGymWrapper
from trossen_arm_mujoco.utils import make_sim_env


def make_cube_stacking_env(
    fake_env: bool = False,
    image_obs: bool = True,
    stats_path: Optional[str] = None,
    max_episode_length: int = 400,
    onscreen_render: bool = False,
    cam_list: Optional[list] = None,
    arm_type: str = "widowx",
    recorder_mode: bool = False,
    data_save_path: str = "data_collection",
    data_save_every_n: int = 1,
    control_mode: Literal["joint", "ee"] = "joint",
) -> SERLGymWrapper:
    """
    Create a Cube Stacking environment.
    
    Args:
        fake_env: If True, skip MuJoCo initialization for fast training startup
        image_obs: If True, include camera observations
        stats_path: Path to JSON file with normalization stats.
                   If None: teleoperation mode (raw physical actions)
                   If set: policy eval mode (normalized actions, will denormalize)
        max_episode_length: Maximum steps per episode
        onscreen_render: If True, enable real-time rendering
        cam_list: List of camera names (default: all 4 cameras)
        arm_type: "widowx" or "viperx" for robot joint limits
        recorder_mode: If True, enable interactive data recording GUI
        data_save_path: Path to save recorded episodes
        data_save_every_n: Save data every N steps (1=50Hz, 5=10Hz, etc.)
        control_mode: 'joint' for 14D joint control, 'ee' for 16D end-effector control
    
    Returns:
        SERLGymWrapper instance wrapping dm_control environment
    
    Example:
        >>> # For training with fake_env
        >>> env = make_cube_stacking_env(fake_env=True, image_obs=False)
        >>> obs, info = env.reset()
        >>> obs, reward, done, truncated, info = env.step(action)
    """
    
    # Default camera list
    if cam_list is None:
        cam_list = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]
    
    # Select task class and XML based on control mode
    if control_mode == "joint":
        task_class = CubeStacking
        xml_file = "trossen_ai_scene_joint.xml"
        action_dim = 14
    elif control_mode == "ee":
        task_class = CubeStackingEE
        xml_file = "trossen_ai_scene.xml"
        action_dim = 16
    else:
        raise ValueError(f"control_mode must be 'joint' or 'ee', got '{control_mode}'")
    
    # Create dm_control environment (unless fake_env)
    dm_env = None
    if not fake_env:
        dm_env = make_sim_env(
            task_class=task_class,
            xml_file=xml_file,
            task_name="sim_transfer_cube",
            onscreen_render=onscreen_render,
            cam_list=cam_list,
        )
    
    # Wrap with Gym wrapper
    gym_env = SERLGymWrapper(
        env=dm_env,
        state_obs_dim=32,  # 16 qpos + 16 qvel
        fake_env=fake_env,
        image_obs=image_obs,
        stats_path=stats_path,
        max_episode_length=max_episode_length,
        arm_type=arm_type,
        onscreen_render=onscreen_render,
        cam_list=cam_list,
        recorder_mode=recorder_mode,
        data_save_path=data_save_path,
        data_save_every_n=data_save_every_n,
        control_mode=control_mode,
        action_dim=action_dim,
    )
    
    return gym_env


def make_cube_stacking_env_for_training(
    fake_env=True,
    image_obs=True,
    stats_path: Optional[str] = None,
) -> SERLGymWrapper:
    """
    Create environment optimized for training (fake_env, no images).
    
    Args:
        stats_path: Path to normalization stats JSON
    
    Returns:
        Fake environment for fast training
    """
    return make_cube_stacking_env(
        fake_env=True,
        image_obs=False,
        stats_path=stats_path,
    )


def make_cube_stacking_env_for_eval(
    stats_path: str,
    onscreen_render: bool = True,
    cam_list: Optional[list] = None,
    arm_type: str = "widowx",
    max_episode_length: int = 200,
    recorder_mode: bool = False,
    data_save_path: str = "data_collection",
    data_save_every_n: int = 1,
) -> SERLGymWrapper:
    """
    Create environment optimized for evaluation (real MuJoCo, with images).
    
    Args:
        stats_path: Path to normalization stats JSON (required for eval)
        onscreen_render: Enable real-time rendering
        data_save_every_n: Save data every N steps (1=50Hz, 5=10Hz)
    
    Returns:
        Real MuJoCo environment for evaluation
    """
    return make_cube_stacking_env(
        fake_env=False,
        image_obs=True,
        stats_path=stats_path,
        onscreen_render=onscreen_render,
        cam_list=cam_list,
        arm_type=arm_type,
        max_episode_length=max_episode_length,
        recorder_mode=recorder_mode,
        data_save_path=data_save_path,
        data_save_every_n=data_save_every_n,
    )
