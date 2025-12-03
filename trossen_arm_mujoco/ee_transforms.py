"""
End-Effector Frame Transformation Utilities.

This module provides functions to transform end-effector poses between:
- Robot base frame (what leader robots report)
- World frame (what MuJoCo simulation expects)

================================================================================
COORDINATE FRAMES
================================================================================

WORLD FRAME (MuJoCo):
    - Origin at table center
    - X: Right (+) / Left (-)
    - Y: Forward (+) / Backward (-)
    - Z: Up (+) / Down (-)

ROBOT BASE FRAMES (from trossen_ai_bimanual.xml):
    Left Robot:
        - Base position: [-0.4575, -0.019, 0.02]
        - Base quaternion: [1, 0, 0, 0] (identity - no rotation)
        
    Right Robot:
        - Base position: [0.4575, -0.019, 0.02]
        - Base quaternion: [0, 0, 0, 1] (180° rotation around Z-axis)

================================================================================
LEADER ROBOT OUTPUT
================================================================================

driver.get_cartesian_positions() returns 6D: [x, y, z, rx, ry, rz]
    - Position (x, y, z) in robot's base frame (meters)
    - Orientation (rx, ry, rz) as angle-axis (radians)

driver.get_gripper_position() returns scalar (0 to 0.044 meters)

================================================================================
TRANSFORMATION FLOW
================================================================================

Teleoperation (Leader -> Sim):
    1. Get cartesian from leader: [x, y, z, rx, ry, rz] (robot frame)
    2. Convert angle-axis to quaternion: [w, x, y, z]
    3. Transform position to world frame
    4. Transform quaternion to world frame
    5. Combine into action: [pos(3), quat(4), gripper(1)]

Recording (Sim -> Dataset):
    - Save both robot frame and world frame data for flexibility

================================================================================
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Tuple


# Robot base positions and orientations from trossen_ai_bimanual.xml
LEFT_ROBOT_BASE_POS = np.array([-0.4575, -0.019, 0.02])
LEFT_ROBOT_BASE_QUAT = np.array([1.0, 0.0, 0.0, 0.0])  # Identity (wxyz)

RIGHT_ROBOT_BASE_POS = np.array([0.4575, -0.019, 0.02])
RIGHT_ROBOT_BASE_QUAT = np.array([0.0, 0.0, 0.0, 1.0])  # 180° around Z (wxyz)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions (Hamilton product).
    
    Args:
        q1: First quaternion [w, x, y, z]
        q2: Second quaternion [w, x, y, z]
    
    Returns:
        Product quaternion [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """
    Compute quaternion conjugate (inverse for unit quaternions).
    
    Args:
        q: Quaternion [w, x, y, z]
    
    Returns:
        Conjugate quaternion [w, -x, -y, -z]
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Rotate a vector by a unit quaternion.
    
    Uses the formula: v' = q * v * q^(-1)
    
    Args:
        q: Unit quaternion [w, x, y, z]
        v: 3D vector to rotate
    
    Returns:
        Rotated 3D vector
    """
    # Convert vector to pure quaternion (w=0)
    v_quat = np.array([0.0, v[0], v[1], v[2]])
    
    # q * v * q^(-1)
    q_conj = quat_conjugate(q)
    result = quat_multiply(quat_multiply(q, v_quat), q_conj)
    
    return result[1:]  # Return xyz components


def angle_axis_to_quaternion(angle_axis: np.ndarray) -> np.ndarray:
    """
    Convert angle-axis representation to quaternion.
    
    The angle-axis format is a 3D vector where:
    - Direction = rotation axis
    - Magnitude = rotation angle (radians)
    
    Args:
        angle_axis: 3D angle-axis vector [rx, ry, rz]
    
    Returns:
        Quaternion [w, x, y, z]
    """
    angle = np.linalg.norm(angle_axis)
    
    if angle < 1e-10:
        # No rotation - return identity quaternion
        return np.array([1.0, 0.0, 0.0, 0.0])
    
    axis = angle_axis / angle
    
    # Use scipy for robust conversion
    rot = R.from_rotvec(axis * angle)
    q = rot.as_quat()  # Returns [x, y, z, w]
    
    # Convert to [w, x, y, z] format
    return np.array([q[3], q[0], q[1], q[2]])


def quaternion_to_angle_axis(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to angle-axis representation.
    
    Args:
        quat: Quaternion [w, x, y, z]
    
    Returns:
        Angle-axis vector [rx, ry, rz]
    """
    # Convert to scipy format [x, y, z, w]
    q_scipy = np.array([quat[1], quat[2], quat[3], quat[0]])
    rot = R.from_quat(q_scipy)
    return rot.as_rotvec()


def transform_robot_to_world_frame(
    pos_robot: np.ndarray,
    quat_robot: np.ndarray,
    robot_name: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform end-effector pose from robot base frame to world frame.
    
    Args:
        pos_robot: 3D position in robot's base frame
        quat_robot: Quaternion [w, x, y, z] in robot's base frame
        robot_name: 'left' or 'right'
    
    Returns:
        Tuple of (world_pos, world_quat)
        - world_pos: 3D position in world frame
        - world_quat: Quaternion [w, x, y, z] in world frame
    """
    if robot_name == "left":
        base_pos = LEFT_ROBOT_BASE_POS
        base_quat = LEFT_ROBOT_BASE_QUAT
    elif robot_name == "right":
        base_pos = RIGHT_ROBOT_BASE_POS
        base_quat = RIGHT_ROBOT_BASE_QUAT
    else:
        raise ValueError(f"robot_name must be 'left' or 'right', got '{robot_name}'")
    
    # Position: world_pos = base_pos + R(base_quat) * pos_robot
    pos_world = base_pos + quat_rotate_vector(base_quat, pos_robot)
    
    # Orientation: For the right robot, we need to handle the 180° Z rotation
    # The robot's orientation in world = base_quat * robot_quat
    quat_world = quat_robot.copy()
    
    if robot_name == "right":
        # The right robot is rotated 180° around Z, which inverts X and Y axes
        # To compensate, we negate the x and y components of the quaternion
        quat_world = np.array([quat_world[0], -quat_world[1], -quat_world[2], quat_world[3]])
    
    return pos_world, quat_world


def transform_world_to_robot_frame(
    pos_world: np.ndarray,
    quat_world: np.ndarray,
    robot_name: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform end-effector pose from world frame to robot base frame.
    
    This is the inverse of transform_robot_to_world_frame.
    
    Args:
        pos_world: 3D position in world frame
        quat_world: Quaternion [w, x, y, z] in world frame
        robot_name: 'left' or 'right'
    
    Returns:
        Tuple of (robot_pos, robot_quat)
        - robot_pos: 3D position in robot's base frame
        - robot_quat: Quaternion [w, x, y, z] in robot's base frame
    """
    if robot_name == "left":
        base_pos = LEFT_ROBOT_BASE_POS
        base_quat = LEFT_ROBOT_BASE_QUAT
    elif robot_name == "right":
        base_pos = RIGHT_ROBOT_BASE_POS
        base_quat = RIGHT_ROBOT_BASE_QUAT
    else:
        raise ValueError(f"robot_name must be 'left' or 'right', got '{robot_name}'")
    
    # Position: robot_pos = R(base_quat)^(-1) * (world_pos - base_pos)
    pos_offset = pos_world - base_pos
    base_quat_inv = quat_conjugate(base_quat)
    pos_robot = quat_rotate_vector(base_quat_inv, pos_offset)
    
    # Orientation
    quat_robot = quat_world.copy()
    
    if robot_name == "right":
        # Inverse of the forward transform
        quat_robot = np.array([quat_robot[0], -quat_robot[1], -quat_robot[2], quat_robot[3]])
    
    return pos_robot, quat_robot


def leader_cartesian_to_sim_action(
    left_cartesian: np.ndarray,
    left_gripper: float,
    right_cartesian: np.ndarray,
    right_gripper: float
) -> np.ndarray:
    """
    Convert leader robot cartesian outputs to simulation action.
    
    This is the main function for teleoperation:
    1. Takes raw outputs from leader robot drivers
    2. Converts angle-axis to quaternion
    3. Transforms from robot frame to world frame
    4. Returns 16D action for EE-controlled simulation
    
    Args:
        left_cartesian: 6D [x, y, z, rx, ry, rz] from left leader
        right_cartesian: 6D [x, y, z, rx, ry, rz] from right leader
        left_gripper: Gripper position (0 to 0.044) from left leader
        right_gripper: Gripper position (0 to 0.044) from right leader
    
    Returns:
        16D action: [L_Pos(3), L_Quat(4), L_Grip(1), R_Pos(3), R_Quat(4), R_Grip(1)]
    """
    # Convert angle-axis to quaternion
    left_quat = angle_axis_to_quaternion(left_cartesian[3:6])
    right_quat = angle_axis_to_quaternion(right_cartesian[3:6])
    
    # Transform to world frame
    left_pos_world, left_quat_world = transform_robot_to_world_frame(
        left_cartesian[:3], left_quat, "left"
    )
    right_pos_world, right_quat_world = transform_robot_to_world_frame(
        right_cartesian[:3], right_quat, "right"
    )
    
    # Combine into action
    action = np.concatenate([
        left_pos_world,      # [0:3]
        left_quat_world,     # [3:7]
        [left_gripper],      # [7]
        right_pos_world,     # [8:11]
        right_quat_world,    # [11:15]
        [right_gripper],     # [15]
    ])
    
    return action


def sim_mocap_to_robot_frame(
    mocap_pose_left: np.ndarray,
    mocap_pose_right: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert simulation mocap poses to robot frame coordinates.
    
    Useful for recording data in robot frame format.
    
    Args:
        mocap_pose_left: 7D [pos(3), quat(4)] left EE in world frame
        mocap_pose_right: 7D [pos(3), quat(4)] right EE in world frame
    
    Returns:
        Tuple of (left_robot_frame, right_robot_frame)
        Each is 7D [pos(3), quat(4)] in respective robot's base frame
    """
    left_pos, left_quat = transform_world_to_robot_frame(
        mocap_pose_left[:3], mocap_pose_left[3:7], "left"
    )
    right_pos, right_quat = transform_world_to_robot_frame(
        mocap_pose_right[:3], mocap_pose_right[3:7], "right"
    )
    
    left_robot_frame = np.concatenate([left_pos, left_quat])
    right_robot_frame = np.concatenate([right_pos, right_quat])
    
    return left_robot_frame, right_robot_frame
