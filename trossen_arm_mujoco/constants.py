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

from importlib.resources import files
import os

import numpy as np

ROOT_DIR = os.path.expanduser("~/.trossen/mujoco/data/")

### Simulated task configurations

SIM_TASK_CONFIGS = {
    "sim_transfer_cube": {
        "num_episodes": 1,
        "episode_len": 600,
        "onscreen_render": False,
        "inject_noise": False,
        "cam_names": ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"],
    }
}

### Simulation envs fixed constants
DT = 0.02
# HOME position: All joints at 0, grippers open
# CRITICAL: This MUST match the home position used in teleoperation and evaluation!
START_ARM_POSE = [
    0.0,
    0.0,  # Was π/12 - changed to match home position
    0.0,  # Was π/12 - changed to match home position
    0.0,
    0.0,
    0.0,
    0.04,  # Gripper open (was 0.044)
    0.04,
    0.0,
    0.0,  # Was π/12 - changed to match home position
    0.0,  # Was π/12 - changed to match home position
    0.0,
    0.0,
    0.0,
    0.04,  # Gripper open (was 0.044)
    0.04,
]

# CRITICAL: These values MUST match the training data starting position!
# Computed from dataset_latest_clean0.05_shift5_clip_normphys.pkl step 0
START_ARM_POSE_MEAN = [
    -0.000544, 0.005546, 0.014228, -0.004537, 0.000244, -0.002860, 0.043, 0.043,  # Left arm
    0.000190, 0.001273, 0.019751, -0.012350, -0.002097, -0.001226, 0.043, 0.043,  # Right arm
]

# Get the path to the assets directory
ASSETS_DIR = str(files("trossen_arm_mujoco").joinpath("assets"))

BOX_POSE = [None]

# ================================================================================
# Home EE Mocap Poses (for EE control mode)
# ================================================================================
# These are the actual FK-computed end-effector poses when joints are at START_ARM_POSE
# Averaged from 27 episodes of teleoperation data collection
# Format: [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]
#
# CRITICAL: These values MUST match the data collection setup!
# During teleoperation, physical leader robots at home position had these EE poses.
# If evaluation environment uses different values, policy will fail due to distribution shift.
# ================================================================================

# LEFT_MOCAP_HOME_POSE = np.array([
#     -2.04307659494e-01, -1.87919326432e-02, 1.88497987142e-01,  # position (x, y, z)
#     9.99995658840e-01, -1.72248687844e-03, -2.30759142861e-03, 6.24790001465e-04   # quaternion (w, x, y, z)
# ])

# RIGHT_MOCAP_HOME_POSE = np.array([
#     1.76713844743e-01, -1.37770213635e-02, 1.94197186865e-01,  # position (x, y, z)
#     9.99871576458e-01, -1.06892009132e-03, -7.53338116168e-03, -1.41044734067e-02   # quaternion (w, x, y, z)
# ])

# CORRECTED HOME POSES - Updated from actual joint home FK (2025-12-17)
# These are the ACTUAL cartesian positions when robots are at joint home (all zeros)
# Verified by moving leaders to joint home and reading get_cartesian_positions()
LEFT_MOCAP_HOME_POSE = np.array([
    2.53762e-01, -1.31132e-05, 1.63401e-01,  # position (x, y, z) - ACTUAL FK from joint home
    1, 0, 0, 0   # quaternion (w, x, y, z) - identity orientation
])

RIGHT_MOCAP_HOME_POSE = np.array([
    2.53646e-01, -1.31e-05, 1.64544e-01,  # position (x, y, z) - ACTUAL FK from joint home
    1, 0, 0, 0   # quaternion (w, x, y, z) - identity orientation
])

# ================================================================================
# Episode 0 Initial Conditions (for debugging/testing)
# ================================================================================
# These represent the EXACT initial state from Episode 0, Step 0 of the dataset
# Used to test if initial condition mismatch is causing policy evaluation failures
# Converted using ee_transforms.angle_axis_to_quaternion()

# LEFT_MOCAP_EPISODE0_POSE = np.array([
#     -2.04296003397e-01, -1.89684428286e-02, 1.88309666511e-01,  # position (x, y, z)
#     9.99996259153e-01, -1.62293989576e-03, -2.19378864476e-03, 1.87180835266e-04  # quaternion (w, x, y, z)
# ])

# RIGHT_MOCAP_EPISODE0_POSE = np.array([
#     1.73631667058e-01, -1.23482533578e-02, 1.80266549116e-01,  # position (x, y, z)
#     9.93336774906e-01, -3.30368108273e-02, -1.08268850695e-01, -2.16443230299e-02  # quaternion (w, x, y, z)
# ])

# # Gripper positions for Episode 0
# LEFT_GRIPPER_EPISODE0 = 4.00028793474e-02
# RIGHT_GRIPPER_EPISODE0 = 3.99405725230e-02

LEFT_MOCAP_EPISODE0_POSE = np.array([
    -2.05507200000e-01,
    -1.95774000000e-02,
    1.85379780000e-01,
    9.99913447041e-01,
    2.68637249282e-04,
    1.30647180569e-02,
    -1.52951087057e-03
])

RIGHT_MOCAP_EPISODE0_POSE = np.array([
    2.05389540000e-01,
    -1.92951100000e-02,
    1.97761500000e-01,
    9.99986822437e-01,
    -4.73947918163e-04,
    5.05416279937e-03,
    7.65351638159e-04
])


LEFT_MOCAP_MEAN_POSE = np.array([
    -0.20546554, -0.01945821,  0.18549709,  1.0000000e+00,
  0.0000000e+00,  0.0000000e+00,  0.0000000e+00
])

RIGHT_MOCAP_MEAN_POSE = np.array([ 0.20441881, -0.01706709,  0.19527784,  1.0000000e+00,
  0.0000000e+00,  0.0000000e+00,  0.0000000e+00 ])

# Gripper positions for Episode 0
LEFT_GRIPPER_EPISODE0 = 3.99957900000e-02
RIGHT_GRIPPER_EPISODE0 = 3.99164200000e-02

# ================================================================================
# Ready Position Bounds for Randomization
# ================================================================================
# Define "ready" position workspace - a safe region above the table
# where robot can move without collisions before starting task

# ======================================================================
# 📍 RIGHT ROBOT READY POSE CAPTURED:
# ======================================================================
# Position (x, y, z):    [0.294823, -0.019731, 0.312455]
# Orientation (aa_x, aa_y, aa_z): [0.018458, 0.606561, -0.120023]
# Gripper: 0.039933

# Add to constants.py as:
# RIGHT_READY_CENTER = np.array([0.294823, -0.019731, 0.312455])

# Sample box around it (±0.05 in each axis):
RIGHT_READY_POS_MIN = np.array([0.244823, -0.069731, 0.262455])
RIGHT_READY_POS_MAX = np.array([0.344823, 0.030269, 0.362455])
# ======================================================================


def sample_random_ready_pose(arm='right', seed=42):
    """
    Sample a random "ready" pose for the specified arm.
    
    Args:
        arm: 'left' or 'right' 
        seed: Random seed for reproducibility (optional)
        
    Returns:
        np.ndarray: 7D pose [x, y, z, quat_w, quat_x, quat_y, quat_z]
    """
    if seed is not None:
        np.random.seed(seed)
    
    if arm == 'left':
        pos = np.random.uniform(LEFT_READY_POS_MIN, LEFT_READY_POS_MAX)
    elif arm == 'right':
        pos = np.random.uniform(RIGHT_READY_POS_MIN, RIGHT_READY_POS_MAX)
    else:
        raise ValueError(f"arm must be 'left' or 'right', got {arm}")
    
    # Keep orientation as identity (neutral, pointing down)
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    
    return np.concatenate([pos, quat])

def sample_random_ready_poses_both(seed=None):
    """
    Sample random ready poses for both arms.
    
    Args:
        seed: Random seed for reproducibility (optional)
        
    Returns:
        tuple: (left_pose, right_pose) each 7D [x, y, z, quat_w, quat_x, quat_y, quat_z]
    """
    if seed is not None:
        np.random.seed(seed)
    
    left_pose = sample_random_ready_pose('left')
    right_pose = sample_random_ready_pose('right')
    
    return left_pose, right_pose
