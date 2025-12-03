"""
Joint limits for Trossen Robotics arms.

These limits are used for:
1. Action denormalization (normalized [-1,1] → physical units)
2. State normalization (physical units → normalized for neural networks)
"""

import numpy as np

# WidowX-250 6DOF ARM LIMITS (radians)
# Based on: https://www.trossenrobotics.com/widowx-250-robot-arm-6dof.aspx
WIDOWX_JOINT_MIN = np.array([-3.05, 0.0, 0.0, -1.57, -1.57, -3.14])
WIDOWX_JOINT_MAX = np.array([3.05, 3.14, 2.35, 1.57, 1.57, 3.14])

# ViperX-300 6DOF ARM LIMITS (radians)
# Based on: https://www.trossenrobotics.com/viperx-300-robot-arm-6dof.aspx
VIPERX_JOINT_MIN = np.array([-3.14, -1.88, -2.15, -1.745, -2.617, -3.14])
VIPERX_JOINT_MAX = np.array([3.14, 1.88, 1.61, 1.745, 2.617, 3.14])

# GRIPPER LIMITS (meters)
GRIPPER_MIN = np.array([0.0])
GRIPPER_MAX = np.array([0.044])

# DUAL ARM ACTION LIMITS (14D: L_Arm(6) + L_Grip(1) + R_Arm(6) + R_Grip(1))
def get_dual_arm_limits(arm_type="viperx"):
    """
    Get action limits for dual arm setup.
    
    Args:
        arm_type: "widowx" or "viperx"
    
    Returns:
        tuple: (action_min, action_max) each of shape (14,)
    """
    if arm_type == "widowx":
        joint_min = WIDOWX_JOINT_MIN
        joint_max = WIDOWX_JOINT_MAX
    elif arm_type == "viperx":
        joint_min = VIPERX_JOINT_MIN
        joint_max = VIPERX_JOINT_MAX
    else:
        raise ValueError(f"Unknown arm_type: {arm_type}")
    
    # Build 14D action limits
    left_arm_min = joint_min
    left_grip_min = GRIPPER_MIN
    right_arm_min = joint_min
    right_grip_min = GRIPPER_MIN
    
    left_arm_max = joint_max
    left_grip_max = GRIPPER_MAX
    right_arm_max = joint_max
    right_grip_max = GRIPPER_MAX
    
    action_min = np.concatenate([left_arm_min, left_grip_min, right_arm_min, right_grip_min])
    action_max = np.concatenate([left_arm_max, left_grip_max, right_arm_max, right_grip_max])
    
    return action_min, action_max


# DUAL ARM POSITION LIMITS (16D: L_Arm(6) + L_Grip(2) + R_Arm(6) + R_Grip(2))
def get_dual_arm_position_limits(arm_type="viperx"):
    """
    Get position limits for dual arm qpos (16D with duplicated gripper values).
    
    Args:
        arm_type: "widowx" or "viperx"
    
    Returns:
        tuple: (pos_min, pos_max) each of shape (16,)
    """
    if arm_type == "widowx":
        joint_min = WIDOWX_JOINT_MIN
        joint_max = WIDOWX_JOINT_MAX
    elif arm_type == "viperx":
        joint_min = VIPERX_JOINT_MIN
        joint_max = VIPERX_JOINT_MAX
    else:
        raise ValueError(f"Unknown arm_type: {arm_type}")
    
    # Build 16D position limits (grippers have 2 values each)
    left_min = np.concatenate([joint_min, GRIPPER_MIN, GRIPPER_MIN])
    right_min = np.concatenate([joint_min, GRIPPER_MIN, GRIPPER_MIN])
    
    left_max = np.concatenate([joint_max, GRIPPER_MAX, GRIPPER_MAX])
    right_max = np.concatenate([joint_max, GRIPPER_MAX, GRIPPER_MAX])
    
    pos_min = np.concatenate([left_min, right_min])
    pos_max = np.concatenate([left_max, right_max])
    
    return pos_min, pos_max
