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


def ortho6d_to_quaternion(ortho6d: np.ndarray) -> np.ndarray:
    """
    Convert 6D continuous rotation representation to quaternion.
    
    Args:
        ortho6d: (6,) array [r11, r21, r31, r12, r22, r32] (first two columns of rotation matrix)
    
    Returns:
        quaternion: (4,) array [w, x, y, z]
    """
    # Reshape to (3, 2)
    r_raw = ortho6d.reshape(3, 2, order='F') # Column-major reshape to match [col1, col2]
    # ortho6d = [r11, r21, r31, r12, r22, r32]
    #           |--- v1 ---|  |--- v2 ---|
    # r_raw[:, 0] = v1 = [r11, r21, r31]  (first column of R)
    # r_raw[:, 1] = v2 = [r12, r22, r32]  (second column of R)
    
    # Gram-Schmidt orthogonalization
    a1 = r_raw[:, 0]  # v1
    a2 = r_raw[:, 1]  # v2
    
    # u1 = v1 / ||v1||  (normalize first vector)
    b1 = a1 / np.linalg.norm(a1)

    # u2 = v2 - (u1^T * v2) * u1  (remove projection onto u1)
    # Then normalize
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)

    # u3 = u1 × u2  (cross product gives third orthonormal vector)
    b3 = np.cross(b1, b2)

    # Construct rotation matrix 3x3
    matrix = np.stack([b1, b2, b3], axis=1)
    
    # Convert to quaternion [x, y, z, w] -> [w, x, y, z]
    rot = R.from_matrix(matrix)
    quat_xyzw = rot.as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


def euler_to_quaternion(euler: np.ndarray) -> np.ndarray:
    """
    Convert Euler angles (XYZ) to quaternion.
    
    Args:
        euler: (3,) array [roll, pitch, yaw] in radians
    
    Returns:
        quaternion: (4,) array [w, x, y, z]
    """
    rot = R.from_euler('xyz', euler)
    quat_xyzw = rot.as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


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


def quaternion_to_euler(quat: np.ndarray, seq: str = 'xyz') -> np.ndarray:
    """
    Convert quaternion to Euler angles.
    
    Args:
        quat: Quaternion [w, x, y, z]
        seq: Euler angle sequence (default: 'xyz' = roll, pitch, yaw)
    
    Returns:
        Euler angles [roll, pitch, yaw] in radians
    """
    # Convert to scipy format [x, y, z, w]
    q_scipy = np.array([quat[1], quat[2], quat[3], quat[0]])
    rot = R.from_quat(q_scipy)
    return rot.as_euler(seq)


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


def action_14d_robot_to_world_aa(action_14d: np.ndarray) -> np.ndarray:
    """
    Convert 14D robot-frame action to 14D world-frame action (angle-axis format).
    
    This is for delta action conversion from recorded teleoperation data.
    
    Input format (14D, robot frame):
        [L_Pos(3), L_AA(3), L_Grip(1), R_Pos(3), R_AA(3), R_Grip(1)]
        
    Output format (14D, world frame):
        [L_Pos(3), L_AA(3), L_Grip(1), R_Pos(3), R_AA(3), R_Grip(1)]
    
    Args:
        action_14d: 14D action in robot frame with angle-axis rotations
        
    Returns:
        14D action in world frame with angle-axis rotations
    """
    # Extract components from 14D action
    left_pos_robot = action_14d[0:3]
    left_aa_robot = action_14d[3:6]
    left_grip = action_14d[6]
    right_pos_robot = action_14d[7:10]
    right_aa_robot = action_14d[10:13]
    right_grip = action_14d[13]
    
    # Convert angle-axis to quaternion for transformation
    left_quat_robot = angle_axis_to_quaternion(left_aa_robot)
    right_quat_robot = angle_axis_to_quaternion(right_aa_robot)
    
    # Transform to world frame
    left_pos_world, left_quat_world = transform_robot_to_world_frame(
        left_pos_robot, left_quat_robot, "left"
    )
    right_pos_world, right_quat_world = transform_robot_to_world_frame(
        right_pos_robot, right_quat_robot, "right"
    )
    
    # Convert back to angle-axis
    left_aa_world = quaternion_to_angle_axis(left_quat_world)
    right_aa_world = quaternion_to_angle_axis(right_quat_world)
    
    # Combine into 14D action in world frame
    return np.concatenate([
        left_pos_world,      # 3
        left_aa_world,       # 3
        [left_grip],         # 1
        right_pos_world,     # 3
        right_aa_world,      # 3
        [right_grip],        # 1
    ])

def state_16d_to_14d(state_16):
    """Converts 16D State (Quats) to 14D State (Axis-Angles) for comparison."""
    return np.concatenate([
        state_16[0:3], quaternion_to_angle_axis(state_16[3:7]), [state_16[7]],   # Left
        state_16[8:11], quaternion_to_angle_axis(state_16[11:15]), [state_16[15]] # Right
    ])


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

def quat_to_ortho6d(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to 6D rotation representation (ortho6d).
    
    Args:
        quat: (N, 4) array in [w, x, y, z] format (MuJoCo standard)
    Returns:
        ortho6d: (N, 6) array
    """
    # Scipy expects [x, y, z, w]
    # MuJoCo provides [w, x, y, z]
    if quat.ndim == 1:
        quat = quat.reshape(1, 4)
        
    quat_scipy = np.concatenate([quat[:, 1:], quat[:, 0:1]], axis=1)
    
    rot = R.from_quat(quat_scipy)
    matrix = rot.as_matrix()  # (N, 3, 3)
    
    # Take first two columns: R[:, 0] and R[:, 1]
    # Flatten them: [r11, r21, r31, r12, r22, r32]
    ortho6d = matrix[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)
    return ortho6d

def convert_dual_arm_angle_axis_to_quat_action(action_14d: np.ndarray) -> np.ndarray:
    """
    Convert 14D angle-axis EE action to 16D quaternion EE action for dual arm.
    
    Input format (14D):
        [L_Pos(3), L_AA(3), L_Grip(1), R_Pos(3), R_AA(3), R_Grip(1)]
        
    Output format (16D):
        [L_Pos(3), L_Quat(4), L_Grip(1), R_Pos(3), R_Quat(4), R_Grip(1)]
    """
    # Left arm: [0:3] pos, [3:6] angle-axis, [6] gripper
    left_pos = action_14d[0:3]
    left_aa = action_14d[3:6]
    left_grip = action_14d[6]
    left_quat = angle_axis_to_quaternion(left_aa)
    
    # Right arm: [7:10] pos, [10:13] angle-axis, [13] gripper
    right_pos = action_14d[7:10]
    right_aa = action_14d[10:13]
    right_grip = action_14d[13]
    right_quat = angle_axis_to_quaternion(right_aa)
    
    # Combine into 16D action
    action_16d = np.concatenate([
        left_pos,      # 3
        left_quat,     # 4
        [left_grip],   # 1
        right_pos,     # 3
        right_quat,    # 4
        [right_grip],  # 1
    ])
    return action_16d


def convert_dual_arm_ortho6d_to_quat_action(action_20d: np.ndarray) -> np.ndarray:
    """
    Convert 20D ortho6d EE action to 16D quaternion EE action for dual arm.
    
    Input format (20D):
        [L_Pos(3), L_Ortho(6), L_Grip(1), R_Pos(3), R_Ortho(6), R_Grip(1)]
        
    Output format (16D):
        [L_Pos(3), L_Quat(4), L_Grip(1), R_Pos(3), R_Quat(4), R_Grip(1)]
    """
    # Left arm: [0:3] pos, [3:9] ortho6d, [9] gripper
    left_pos = action_20d[0:3]
    left_ortho = action_20d[3:9]
    left_grip = action_20d[9]
    left_quat = ortho6d_to_quaternion(left_ortho)
    
    # Right arm: [10:13] pos, [13:19] ortho6d, [19] gripper
    right_pos = action_20d[10:13]
    right_ortho = action_20d[13:19]
    right_grip = action_20d[19]
    right_quat = ortho6d_to_quaternion(right_ortho)
    
    # Combine into 16D action
    action_16d = np.concatenate([
        left_pos,      # 3
        left_quat,     # 4
        [left_grip],   # 1
        right_pos,     # 3
        right_quat,    # 4
        [right_grip],  # 1
    ])
    return action_16d


def convert_dual_arm_euler_to_quat_action(action_14d: np.ndarray) -> np.ndarray:
    """
    Convert 14D euler EE action to 16D quaternion EE action for dual arm.
    
    Input format (14D):
        [L_Pos(3), L_Euler(3), L_Grip(1), R_Pos(3), R_Euler(3), R_Grip(1)]
        
    Output format (16D):
        [L_Pos(3), L_Quat(4), L_Grip(1), R_Pos(3), R_Quat(4), R_Grip(1)]
    """
    # Left arm: [0:3] pos, [3:6] euler, [6] gripper
    left_pos = action_14d[0:3]
    left_euler = action_14d[3:6]
    left_grip = action_14d[6]
    left_quat = euler_to_quaternion(left_euler)
    
    # Right arm: [7:10] pos, [10:13] euler, [13] gripper
    right_pos = action_14d[7:10]
    right_euler = action_14d[10:13]
    right_grip = action_14d[13]
    right_quat = euler_to_quaternion(right_euler)
    
    # Combine into 16D action
    action_16d = np.concatenate([
        left_pos,      # 3
        left_quat,     # 4
        [left_grip],   # 1
        right_pos,     # 3
        right_quat,    # 4
        [right_grip],  # 1
    ])
    return action_16d


if __name__ == "__main__":
    # Simple test cases can be added here for quick verification
    state_epi1 = [
        -0.2055072, -0.0195774,  0.18537978,  0.00053729,
         0.02613019, -0.00305911,  0.03999579,  0.20538954,
        -0.01929511,  0.1977615,  -0.0009479,   0.01010837,
         0.00153071,  0.03991642
    ]

    state_epi0 = state = [
        -0.20559927, -0.01957738,  0.18507987,  0.00053423,
        0.02841904, -0.00305973,  0.03999663,  0.20639778,
        -0.01873030,  0.20006111, -0.00097493,  0.02231704,
        -0.00151520,  0.03991173,
    ]


    action_16d = convert_dual_arm_angle_axis_to_quat_action(np.array(state_epi0))

    # Pretty print in copy-paste format
    print("[")
    for i, val in enumerate(action_16d):
        # format with scientific notation and commas
        print(f"    {val:.11e},")
    print("]")
    pass

