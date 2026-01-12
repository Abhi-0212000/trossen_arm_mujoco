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

import collections
import os

from dm_control import mujoco
from dm_control.mujoco import Physics
from dm_control.rl import control
from dm_control.suite import base
from matplotlib.image import AxesImage
import matplotlib.pyplot as plt
import numpy as np

from trossen_arm_mujoco.constants import ASSETS_DIR, DT


def sample_box_pose(seed: int = None, rng: np.random.Generator = None) -> np.ndarray:
    """
    Generate a random pose for a cube within predefined position ranges.
    
    The spawn area is constrained to:
    - Small centered area to ensure robot can reach
    - Avoid the blue target box (at y=0.22, size 0.1x0.1)
    - Stay within robot reach
    - Stay on the table surface
    
    Blue box: centered at (0, 0.22), extends from x=[-0.1, 0.1], y=[0.12, 0.32]
    Safe spawn area: small center zone, well away from blue box

    Args:
        seed: Optional random seed for reproducibility. If provided, creates a new RNG.
        rng: Optional numpy random Generator. If provided, uses this instead of seed.
             Takes precedence over seed if both are provided.
             If neither seed nor rng is provided, uses global np.random state.
    
    :return: A 7D array containing the sampled position ``[x, y, z, w, x, y, z]`` representing the
        cube's position and orientation as a quaternion.
    """
    # Visualized in ./dataset_utils/visualize_bbox.py validate_action_sampling()
    x_range = [0.03, 0.1]
    y_range = [-0.02, 0.1]
    z_range = [0.0125, 0.0125]

    ranges = np.vstack([x_range, y_range, z_range])
    
    # Choose random source based on arguments
    if rng is not None:
        # Use provided Generator
        print(f"[DEBUG] Sampling box pose with provided RNG")
        cube_position = rng.uniform(ranges[:, 0], ranges[:, 1])
    elif seed is not None:
        # Create seeded Generator for this call only
        print(f"[DEBUG] Sampling box pose with seed {seed}")
        local_rng = np.random.default_rng(seed)
        cube_position = local_rng.uniform(ranges[:, 0], ranges[:, 1])
    else:
        # Use global numpy random state (respects np.random.seed())
        cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    # cube_position = np.array([-3.76365857e-03,  1.35211311e-02,  1.13459624e-02])
    # cube_position = np.array([ 0.11381228, -0.00127775,  0.01345779])
    
    return np.concatenate([cube_position, cube_quat])


def get_observation_base(
    physics: Physics,
    cam_list: list[str],
    image_obs: bool = True,
) -> collections.OrderedDict:
    """
    Capture image observations from multiple cameras in the simulation.

    :param physics: The simulation physics instance.
    :param cam_list: List of camera names to capture images from.
    :param image_obs: Whether to capture images from cameras, defaults to ``True``.
    :return: A dictionary containing image observations.
    """
    obs: collections.OrderedDict = collections.OrderedDict()
    if image_obs and cam_list:
        obs["images"] = dict()
        for cam in cam_list:
            obs["images"][cam] = physics.render(height=480, width=640, camera_id=cam)
    return obs


def make_sim_env(
    task_class: base.Task,
    xml_file: str = "trossen_ai_scene.xml",
    task_name: str = "sim_transfer_cube",
    onscreen_render: bool = False,
    cam_list: list[str] = [],
    control_timestep: float = DT,
    physics_timestep: float = None,
):
    """
    Create a simulated environment for bimanual robotic manipulation.

    :param task_class: The task class for defining simulation behavior.
    :param xml_file: Path to the robot XML file, defaults to ``'trossen_ai_scene.xml'``.
    :param task_name: Name of the task, defaults to ``'sim_transfer_cube'``.
    :param onscreen_render: Whether to render the simulation on-screen, defaults to ``False``.
    :param cam_list: List of camera names to be used, defaults to ``[]``.
    :param control_timestep: The control timestep for the environment, defaults to ``DT``.
    :param physics_timestep: The physics timestep for the environment, defaults to ``None`` (automatic).
    :return: The simulated robot environment.
    """
    if "sim_transfer_cube" in task_name:
        assets_path = os.path.join(ASSETS_DIR, xml_file)
        physics = mujoco.Physics.from_xml_path(assets_path)
        # Override physics timestep if physics_timestep is provided
        if physics_timestep is not None and control_timestep is not None:
            physics.model.opt.timestep = physics_timestep
            n_sub_steps = int(control_timestep / physics_timestep)
        else:
            n_sub_steps = None
        task = task_class(
            random=False,
            onscreen_render=onscreen_render,
            cam_list=cam_list,
        )
    else:
        raise NotImplementedError(f"Task {task_name} is not implemented.")

    return control.Environment(
        physics,
        task,
        time_limit=20,
        control_timestep=None,
        n_sub_steps=n_sub_steps,
        flat_observation=False,
    )


def plot_observation_images(observation: dict, cam_list: list[str]) -> list[AxesImage]:
    """
    Plot observation images from multiple camera viewpoints.

    :param observation: The observation data containing images.
    :param cam_list: List of camera names used for capturing images.
    :return: A list of AxesImage objects for dynamic updates.
    """
    images = observation.get("images", {})

    # Define the layout based on the provided camera list
    num_cameras = len(cam_list)

    if num_cameras == 4:
        cols = 2
        rows = 2
    else:
        cols = min(3, num_cameras)  # Maximum of 3 columns
        rows = (num_cameras + cols - 1) // cols  # Compute rows dynamically
    _, axs = plt.subplots(rows, cols, figsize=(10, 10))
    axs = axs.flatten() if isinstance(axs, (list, np.ndarray)) else [axs]

    plt_imgs: list[AxesImage] = []
    titles = {
        "cam_high": "Camera High",
        "cam_low": "Camera Low",
        "cam_teleop": "Teleoperator POV",
        "cam_left_wrist": "Left Wrist Camera",
        "cam_right_wrist": "Right Wrist Camera",
    }

    for i, cam in enumerate(cam_list):
        if cam in images:
            plt_imgs.append(axs[i].imshow(images[cam]))
            axs[i].set_title(titles.get(cam, cam))

    for ax in axs:
        ax.axis("off")

    plt.ion()
    return plt_imgs


def set_observation_images(
    observation: dict,
    plt_imgs: list[AxesImage],
    cam_list: list[str],
) -> list[AxesImage]:
    """
    Update displayed observation images dynamically.

    :param observation: The observation data containing updated images.
    :param plt_imgs: A list of AxesImage objects for dynamic updates.
    :param cam_list: List of camera names.
    :return: Updated list of AxesImage objects for real-time visualization.
    """
    images = observation.get("images", {})

    # Update image data dynamically
    for i, cam in enumerate(cam_list):
        if cam in images and i < len(plt_imgs):
            plt_imgs[i].set_data(images[cam])

    plt.pause(0.02)
    return plt_imgs


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