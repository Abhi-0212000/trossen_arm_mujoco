"""
Gym wrapper for dm_control environments.
Returns RAW observations - normalization should be done in preprocessing.

Control Modes:
    - 'joint': 14D actions [L_Arm(6), L_Grip(1), R_Arm(6), R_Grip(1)]
    - 'ee': 16D actions [L_Pos(3), L_Quat(4), L_Grip(1), R_Pos(3), R_Quat(4), R_Grip(1)]
"""

import gym
from gym import spaces
import numpy as np
from typing import Dict, Optional, Tuple, Literal
import json
from pathlib import Path
from dm_control import composer
import cv2

from trossen_arm_mujoco.robot_limits import get_dual_arm_limits, get_dual_arm_position_limits
from trossen_arm_mujoco.utils import plot_observation_images, set_observation_images
from trossen_arm_mujoco.sim_data_collector import SimDataCollector


class SERLGymWrapper(gym.Env):
    """
    Gym wrapper for dm_control environments.
    
    Features:
    - Returns RAW observations (no normalization - do that in preprocessing)
    - Action denormalization from [-1,1] to physical joint limits
    - fake_env for fast startup (no MuJoCo initialization)
    - Automatic episode termination at max_episode_length
    - Optional data recording with GUI
    - Supports both joint control (14D) and EE control (16D)
    
    Args:
        env: dm_control Environment instance
        state_obs_dim: State observation dimension (default 32)
        fake_env: If True, skip MuJoCo init for fast training (default False)
        image_obs: If True, include camera observations (default True)
        stats_path: Path to JSON file with normalization stats
        max_episode_length: Maximum steps per episode (default 400)
        arm_type: "widowx" or "viperx" for robot joint limits (default "viperx")
        recorder_mode: If True, enable interactive data recording GUI (default False)
        data_save_path: Path to save recorded episodes (default "data_collection")
        data_save_every_n: Save data every N steps (1=50Hz, 5=10Hz)
        control_mode: 'joint' (14D) or 'ee' (16D) control mode
        action_dim: Action dimension (14 for joint, 16 for EE)
    """
    
    def __init__(
        self,
        env: Optional[composer.Environment] = None,
        state_obs_dim: int = 32,
        fake_env: bool = False,
        image_obs: bool = True,
        stats_path: Optional[str] = None,
        max_episode_length: int = 400,
        arm_type: str = "viperx",
        onscreen_render: bool = False,
        cam_list: list = None,
        recorder_mode: bool = False,
        data_save_path: str = "data_collection",
        data_save_every_n: int = 1,
        control_mode: Literal["joint", "ee"] = "joint",
        action_dim: int = 14,
    ):
        super().__init__()
        
        self.fake_env = fake_env
        self.image_obs = image_obs
        self.state_obs_dim = state_obs_dim
        self.max_episode_length = max_episode_length
        self.arm_type = arm_type
        self._steps = 0
        self.stats_path = stats_path
        self.control_mode = control_mode
        self.action_dim = action_dim
        
        # Visualization setup
        self.onscreen_render = onscreen_render
        self.cam_list = cam_list if cam_list is not None else ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]
        self.plt_imgs = None  # Matplotlib figure handles for visualization
        self._images_full_res = None  # Store full-res images (480x640) for visualization
        
        # Data Collector Setup
        self.recorder_mode = recorder_mode
        self.data_save_every_n = data_save_every_n
        self.data_collector = None
        if self.recorder_mode:
            # If recorder mode is on, we force onscreen_render to True to ensure we get images
            self.onscreen_render = True
            self.data_collector = SimDataCollector(
                env_name="trossen_arm_mujoco",
                save_dir=data_save_path,
                cam_list=self.cam_list,
                save_every_n=data_save_every_n
            )
        
        # Get robot-specific joint limits (for action denormalization only)
        self.action_min, self.action_max = get_dual_arm_limits(arm_type)
        self.action_range = self.action_max - self.action_min
        self.pos_min, self.pos_max = get_dual_arm_position_limits(arm_type)
        self.pos_range = self.pos_max - self.pos_min
        
        # Gym spaces - action space depends on control mode
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32
        )
        
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
    
    def _denormalize_action(self, normalized_action: np.ndarray) -> np.ndarray:
        """
        Convert normalized action [-1, 1] to physical units.
        
        Args:
            normalized_action: 14D action in [-1, 1] range
        
        Returns:
            Physical action in actual joint limits
        """
        # Clip to [-1, 1] for safety
        normalized_action = np.clip(normalized_action, -1.0, 1.0)
        # Min-max denormalization: action = (normalized + 1) / 2 * range + min
        physical_action = (normalized_action + 1.0) / 2.0 * self.action_range + self.action_min
        return physical_action
    
    def _process_observation(self, dm_obs: Dict) -> Dict:
        """
        Process dm_control observation into Gym format.
        
        Args:
            dm_obs: OrderedDict from dm_control task.get_observation()
                Joint mode keys:
                    - 'qpos': (16,) joint positions
                    - 'qvel': (16,) joint velocities
                    - 'images': {'cam_high': (480, 640, 3), ...}
                    - 'env_state': (7,) cube pose
                EE mode additional keys:
                    - 'mocap_pose_left': (7,) [pos(3), quat(4)]
                    - 'mocap_pose_right': (7,) [pos(3), quat(4)]
                    - 'robot0_eef_pos': (6,) [L_pos(3), R_pos(3)]
                    - 'robot0_eef_quat': (8,) [L_quat(4), R_quat(4)]
                    - 'robot0_gripper_qpos': (2,) [L_grip, R_grip]
        
        Returns:
            Dict with 'state', 'images' (128x128), and raw 'dm_obs' for recording
        """
        # Extract state (qpos + qvel)
        qpos = dm_obs['qpos']  # (16,)
        qvel = dm_obs['qvel']  # (16,)
        state = np.concatenate([qpos, qvel])  # (32,)
        
        obs = {'state': state.astype(np.float32)}
        
        # Store full dm_obs for data recording (includes EE fields if present)
        obs['dm_obs'] = dm_obs
        
        # Add images if enabled
        if self.image_obs:
            images_dict = dm_obs.get('images', {})
            
            # Store full resolution (480x640) for visualization
            if self.onscreen_render:
                self._images_full_res = images_dict
            
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
        """
        Reset the environment.
        
        Returns:
            observation: Dict with 'state' and 'images'
            info: Empty dict

        Flow:
            Your Code:
                env.reset()
                    ↓
            dm_control.Environment.reset():
                1. Reset physics simulation
                2. Call task.initialize_episode(physics)  ← Your CubeStacking code runs here
                3. Call task.get_observation(physics)     ← Your CubeStacking code runs here
                4. Return TimeStep(observation=obs)
                    ↓
            Back to your wrapper:
                dm_timestep = self.env.reset()
                obs = self._process_observation(dm_timestep.observation)
                return obs, {}
        """
        if seed is not None:
            np.random.seed(seed)
        
        self._steps = 0
        
        if self.fake_env:
            return self._dummy_obs, {}
        
        # Reset dm_control environment
        dm_timestep = self.env.reset()
        obs = self._process_observation(dm_timestep.observation)
        
        # Update visualization / Recorder
        if self.recorder_mode:
            self.data_collector.reset()
            self.data_collector.update_viz(self._images_full_res)
        else:
            self._update_visualization()
        
        return obs, {}
    
    def step(
        self,
        action: np.ndarray
    ) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Normalized action in [-1, 1] (14D)
        
        Returns:
            observation: Dict with 'state' and 'images'
            reward: Scalar reward
            terminated: Whether episode ended due to task completion
            truncated: Whether episode ended due to time limit
            info: Additional information
        """
        if self.fake_env:
            self._steps += 1
            truncated = self._steps >= self.max_episode_length
            done = truncated
            return self._dummy_obs, 0.0, done, truncated, {}
        
        # Convert action to physical units
        if self.stats_path is None:
            # No stats_path = teleoperation mode, actions already in physical units
            physical_action = action
        else:
            # stats_path set = policy eval mode, denormalize from [-1, 1]
            physical_action = self._denormalize_action(action)
        
        # Step dm_control environment
        dm_timestep = self.env.step(physical_action)
        
        # Process observation
        obs = self._process_observation(dm_timestep.observation)
        
        # Extract reward and termination
        reward = dm_timestep.reward if dm_timestep.reward is not None else 0.0
        
        # Check truncation (max steps)
        self._steps += 1
        truncated = self._steps >= self.max_episode_length
        
        # For BC training, no early termination
        terminated = False
        
        # For SERL compatibility: done = terminated OR truncated
        done = terminated or truncated

        # Update visualization / Record Data
        if self.recorder_mode:
            self.data_collector.step(
                obs, action, reward, done, {},
                full_res_images=self._images_full_res
            )
        else:
            self._update_visualization()
        
        return obs, reward, done, truncated, {}
    
    def render(self, mode: str = 'rgb_array'):
        """Render the environment (not implemented for dm_control)."""
        if self.fake_env:
            return None
        # dm_control rendering handled through camera observations
        return None
    
    def close(self):
        """Close the environment."""
        if hasattr(self, 'env') and self.env is not None:
            self.env.close()
