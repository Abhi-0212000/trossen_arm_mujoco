# Trossen Arm MuJoCo

## Overview

This package provides the necessary scripts and assets for simulating and training robotic policies using the Trossen AI kits in MuJoCo.
It includes URDFs, mesh models, and MuJoCo XML files for robot configuration, as well as Python scripts for policy execution, reward-based evaluation, data collection, and visualization.

This package supports two types of simulation environments:

1. End-Effector (EE) Controlled Simulation ([`ee_sim_env.py`](./trossen_arm_mujoco/ee_sim_env.py)): Uses motion capture bodies to move the arms
2. Joint-Controlled Simulation ([`sim_env.py`](./trossen_arm_mujoco/sim_env.py)): Uses joint position controllers

## Installation

First, clone this repository:

```bash
git clone https://github.com/TrossenRobotics/trossen_arm_mujoco.git
```

It is recommended to create a virtual environment before installing dependencies.
Create a Conda environment with Python 3.10 or above.

```bash
conda create --name trossen_mujoco_env python=3.10
```

After creation, activate the environment with:

```bash
conda activate trossen_mujoco_env
```

Install the package and required dependencies using:

```bash
cd trossen_arm_mujoco
pip install .
```

To verify the installation, run:

```bash
python trossen_arm_mujoco/ee_sim_env.py
```

If the simulation window appears, the setup was successful.

## 1. Assets ([`assets/`](./trossen_arm_mujoco/assets/))

This folder contains all required MuJoCo XML configuration files, URDF files, and mesh models for the simulation.

### Key Files:

- **`trossen_ai_bimanual.xml`** → Base robot definition with mocap bodies for end-effector control (includes dual arms, grippers, actuators).
- **`trossen_ai_joint.xml`** → Base robot definition with joint position controllers for hardware-like control.
- **`trossen_ai_scene.xml`** → Complete scene including `trossen_ai_bimanual.xml` + table + cameras + objects (uses mocap control).
- **`trossen_ai_scene_joint.xml`** → Complete scene including `trossen_ai_joint.xml` + table + cameras + objects (uses joint controllers).
- **`wxai_follower.urdf`** & **`wxai_follower.xml`** → URDF and XML descriptions of a single follower arm.
- **`meshes/`** → Contains STL and OBJ files for the robot components, including arms, grippers, cameras, and environmental objects.

### Motion Capture vs Joint-Controlled Environments:

- **Motion Capture** (`trossen_ai_scene.xml`): Uses mocap bodies to directly control end-effector positions. Actions are Cartesian coordinates (x,y,z, quaternion, gripper). Used with `ee_sim_env.py` for scripted trajectory generation.
- **Joint Control** (`trossen_ai_scene_joint.xml`): Uses position controllers for each joint, similar to real hardware. Actions are joint angles (6 joints + gripper per arm). Used with `sim_env.py` for realistic control and RL training.

## 2. Modules ([`trossen_arm_mujoco`](./trossen_arm_mujoco/))

This folder contains all Python modules necessary for running simulations, executing policies, recording episodes, and visualizing results.

### 2.1 Core Modules

- **`constants.py`**
  - Task configurations (episode length, cameras, number of episodes)
  - Fixed simulation parameters: `DT=0.02` (control timestep), `START_ARM_POSE` (initial joint positions)
  - Path to assets directory
  - `SIM_TASK_CONFIGS` dictionary for task-specific settings

- **`ee_sim_env.py`** (End-Effector Control Environment)
  - Loads `trossen_ai_scene.xml` (motion capture-based control)
  - **Classes**: `TrossenAIStationaryEETask` (base), `TransferCubeEETask` (transfer task)
  - **Action space**: 14D Cartesian commands `[x,y,z, quat(4), gripper] × 2 arms`
  - **Purpose**: Generate scripted trajectories, collect joint position data
  - The arms move by directly commanding mocap body positions

- **`sim_env.py`** (Joint Control Environment)
  - Loads `trossen_ai_scene_joint.xml` (position-controlled joints)
  - **Classes**: `TrossenAIStationaryTask` (base), `TransferCubeTask` (transfer task)
  - **Action space**: 14D joint commands `[6 joint angles + gripper] × 2 arms`
  - **Purpose**: Replay trajectories with realistic joint control, used for RL training
  - Uses position controllers similar to real hardware

- **`scripted_policy.py`**
  - **Classes**: `BasePolicy`, `PickAndTransferPolicy`
  - Defines pre-scripted movements for robot arms (e.g., pick and transfer cube)
  - Generates time-indexed waypoints with target positions, orientations, and gripper states
  - Used for demonstration data collection (not for RL policy learning)

- **`utils.py`**
  - **`sample_box_pose()`**: Randomize object positions within bounds
  - **`get_observation_base()`**: Capture multi-camera image observations
  - **`make_sim_env()`**: Create dm_control Environment wrapper
  - **`plot_observation_images()`**: Visualization helpers for debugging

### 2.2 Usage Summary

For **RL training** (e.g., SERL integration):
- Use `sim_env.py` (joint control) with `trossen_ai_scene_joint.xml`
- 14D action space: joint angles + gripper commands
- Wrap in gym.Env interface for standard RL algorithms

For **scripted demonstrations**:
- Use `ee_sim_env.py` (end-effector control) with `trossen_ai_scene.xml`
- Generate trajectories with `scripted_policy.py`
- Collect demonstration data for behavior cloning

## 3. How the Data Collection Works

The data collection process involves two simulation phases:

1. Running a scripted policy in `ee_sim_env.py` to record observations (joint positions).
2. Replaying the recorded joint positions in `sim_env.py` to capture full episode data.

### Step-by-Step Process

1. Run `record_sim_episodes.py`

    - Starts `ee_sim_env.py` and executes a scripted policy.
    - Captures observations in the form of joint positions.
    - Saves these joint positions for later replay.
    - Immediately replays the episode in sim_env.py using the recorded joint positions.
    - During replay, captures:
      - Camera feeds from 4 different viewpoints
      - Joint states (actual positions during execution)
      - Actions (input joint positions)
      - Reward values indicating success or failure

2. Save the Data

    - All observations and actions are stored in HDF5 format, with one file per episode.
    - Each episode is saved as `episode_X.hdf5` inside the `~/.trossen/mujoco/data/` folder.

3. Visualizing the Data

    - The stored HDF5 files can be converted into videos using `visualize_eps.py`.

4. Sim-to-real

    - Run `replay_episode_real.py`
    - This script:
      - Loads the joint position trajectory from a selected HDF5 file.
      - Sends commands to both arms using IP addresses (--left_ip, --right_ip).
      - Plays back the motions based on the saved trajectory.
      - Monitors position error between commanded and actual joint states.
      - Returns arms to home and sleep positions after execution.


## 4. Script Arguments Explanation

### a. record_sim_episodes.py

This script generates and saves demonstration episodes using a scripted policy in simulation. It supports both end-effector control (for task definition) and joint-space replay (for clean data collection), storing all observations in `.hdf5` format.

To generate and save simulation episodes, use:

```bash
python trossen_arm_mujoco/scripts/record_sim_episodes.py \
    --task_name sim_transfer_cube \
    --data_dir sim_transfer_cube \
    --num_episodes 5 \
    --onscreen_render
```
Arguments:

- `--task_name`: Name of the task (default: sim_transfer_cube).
- `--num_episodes`: Number of episodes to generate.
- `--data_dir`: Directory where episodes will be saved (required).
- `--root_dir`: Directory where the root is (optional). Default: `~/.trossen/mujoco/data/`
- `--episode_len`: Number of simulation steps of each episode.
- `--onscreen_render` : Enables on-screen rendering. Default: False (only true if explicitly set)
- `--inject_noise`: Injects noise into actions. Default: False (only true if explicitly set)
- `--cam_names`: Comma-separated list of camera names for image collection

**Note:**

- When you pass `--task_name`, the script will automatically load the corresponding configuration from constants.py.

- You can extend `SIM_TASK_CONFIGS` in `constants.py` to support new task configurations.

- All parameters loaded from `constants.py` can be individually overridden via command-line arguments.

### b. visualize_eps.py

To convert saved episodes to videos, run:

```bash
python trossen_arm_mujoco/scripts/visualize_eps.py \
    --data_dir sim_transfer_cube \
    --output_dir videos \
    --fps 50
```
Arguments:

- `--data_dir`: Directory containing .hdf5 files (required), relative to --root_dir if provided.
- `--root_dir`: Root path prefix for locating data_dir. Default: ~/.trossen/mujoco/data/
- `--output_dir`: Subdirectory inside data_dir where generated .mp4 videos will be saved. Default: videos
- `--fps`: Frames per second for the generated videos (default: 50)
- `--root_dir`: Directory where the root is (optional). Default: `~/.trossen/mujoco/`

**Note:** If you do not specify `--root_dir`, videos will be saved to `~/.trossen/mujoco/data/<data_dir>/<output_dir>`.
You can customize the output path by changing `--root_dir`, `--data_dir`, or `--output_dir` as needed.

### c. replay_episode_real.py

This script replays recorded joint-space episodes on real Trossen robotic arms using data saved in .hdf5 files.
It configures each arm, plays back the actions with a user-defined frame rate, and returns both arms to a safe rest pose after execution.

To perform sim to real, run:

```bash
python trossen_arm_mujoco/scripts/replay_episode_real.py \
    --data_dir sim_transfer_cube \
    --episode_idx 0 \
    --fps 10 \
    --left_ip 192.168.1.5 \
    --right_ip 192.168.1.4
```

Arguments:

- `--data_dir`: Directory containing `.hdf5` files (required).
- `--root_dir`: Directory where the root is (optional). Default: `~/.trossen/mujoco/data/`
- `--episode_idx`: Index of the episode to replay. Default: 0
- `--fps`: Playback frame rate (Hz). Controls the action replay speed. Default: 10
- `--left_ip` : IP address of the left Trossen arm. Default: 192.168.1.5
- `--right_ip`: 	IP address of the right Trossen arm. Default: 192.168.1.4

## Customization

### 1. Modifying Tasks

To create a custom task, modify `ee_sim_env.py` or `sim_env.py` and define a new subclass of `TrossenAIStationary(EE)Task`.
Implement:

- `initialize_episode(self, physics)`: Set up the initial environment state, including robot and object positions.
- `get_observation(self, physics)`: Define what data should be recorded as observations.
- `get_reward(self, physics)`: Implement the reward function to determine task success criteria.

### 2. Changing Policy Behavior

Modify `scripted_policy.py` to define new behavior for the robotic arms.
Update the trajectory generation logic in `PickAndTransferPolicy.generate_trajectory()` to create different movement patterns.

Each movement step in the trajectory is defined by:

- `t`: The time step at which the movement shall occur.
- `xyz`: The target position of the end effector in 3D space.
- `quat`: The target orientation of the end effector, represented as a quaternion.
- `gripper`: The target gripper finger position 0~0.044 where 0 is closed and 0.044 is fully open.

Example:

```python
def generate_trajectory(self, ts_first: TimeStep):
    self.left_trajectory = [
        {"t": 0, "xyz": [0, 0, 0.4], "quat": [1, 0, 0, 0], "gripper": 0},
        {"t": 100, "xyz": [0.1, 0, 0.3], "quat": [1, 0, 0, 0], "gripper": 0.044}
    ]
```

### 3. Adding New Environment Setups

The simulation uses XML files stored in the `assets/` directory. To introduce a new environment setup:

1. Create a new XML configuration file in `assets/` with desired object placements and constraints.
2. Modify `sim_env.py` to load the new environment by specifying the new XML file.
3. Update the scripted policies in `scripted_policy.py` to accommodate new task goals and constraints.

## Troubleshooting

If you encounter into Mesa Loader or `mujoco.FatalError: gladLoadGL error` errors:

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```


### `env.reset()` in `dm_control` which was used in `trossen_arm_mujoco` does the following in order.
```plaintext
env.reset()
   └─ physics.reset()
   └─ task.initialize_episode(physics)
   └─ obs = task.get_observation(physics)
   └─ return TimeStep.first(obs)
```
and `ts` looks like:
ts = env.reset()
```plaintext
ts.observation = {'qpos': [...], 'qvel': [...], 'images': {...}, ...}
ts.reward = 0.0 (initial)
ts.discount = 1.0
ts.step_type = StepType.FIRST
```

ts = env.step(action)
```plaintext
Internally:
1. physics.step() → advances MuJoCo simulation
2. task.get_observation(physics) → builds observation dict
3. task.get_reward(physics) → computes reward
4. returns TimeStep with observation, reward, discount, step_type
```

---

## Control Modes & Dataset Format

### Control Modes

| Mode | Action Dim | Action Format | XML Scene |
|------|------------|---------------|------------|
| `joint` | 14 | `[L_Arm(6), L_Grip(1), R_Arm(6), R_Grip(1)]` | `trossen_ai_scene_joint.xml` |
| `ee` | 16 | `[L_Pos(3), L_Quat(4), L_Grip(1), R_Pos(3), R_Quat(4), R_Grip(1)]` | `trossen_ai_scene.xml` |

### HDF5 Dataset Format

**Joint Mode:**
```
episode_X.hdf5
├── observations/
│   ├── qpos               (T, 16) float32   # Joint positions
│   ├── qvel               (T, 16) float32   # Joint velocities
│   ├── cube_pose          (T, 7) float32    # [x, y, z, qw, qx, qy, qz] cube pose
│   └── images/
│       ├── cam_high       (T, 480, 640, 3) uint8
│       ├── cam_low        (T, 480, 640, 3) uint8
│       ├── cam_left_wrist (T, 480, 640, 3) uint8
│       └── cam_right_wrist(T, 480, 640, 3) uint8
├── action                 (T, 14) float32   # [L_Arm(6), L_Grip(1), R_Arm(6), R_Grip(1)]
├── reward                 (T,) float32
└── done                   (T,) bool
```

**EE Mode (additional fields):**
```
episode_X.hdf5
├── observations/
│   ├── qpos               (T, 16) float32   # Joint positions
│   ├── qvel               (T, 16) float32   # Joint velocities
│   ├── cube_pose          (T, 7) float32    # [x, y, z, qw, qx, qy, qz] cube pose
│   ├── mocap_pose_left    (T, 7) float32    # [pos(3), quat(4)] world frame
│   ├── mocap_pose_right   (T, 7) float32    # [pos(3), quat(4)] world frame
│   ├── robot0_eef_pos     (T, 6) float32    # [L_pos(3), R_pos(3)]
│   ├── robot0_eef_quat    (T, 8) float32    # [L_quat(4), R_quat(4)]
│   ├── robot0_gripper_qpos(T, 2) float32    # [L_grip, R_grip]
│   └── images/
│       └── ... (same as joint mode)
├── action                 (T, 16) float32   # [L_Pos(3), L_Quat(4), L_Grip(1), R_...]
├── reward                 (T,) float32
└── done                   (T,) bool
```

---

## Simulation Timing & Frequency

### Quick Math

| Parameter | Value | Source |
|-----------|-------|--------|
| Physics timestep | 0.001s (1000 Hz) | `trossen_ai_bimanual.xml` → `<option timestep="0.001"/>` |
| Control timestep | 0.02s (50 Hz) | `constants.py` → `DT = 0.02` |
| Sub-steps per action | 20 | `control_dt / physics_dt = 0.02 / 0.001` |

### What Happens in `env.step(action)`

```
env.step(action)
   └─ for _ in range(20):      # n_sub_steps = control_dt / physics_dt
        task.before_step(action)  # Apply same action
        physics.step()            # 1ms physics update
   └─ task.get_observation()    # Render cameras, get state
   └─ return TimeStep
```

**Key Points:**
- Each `env.step()` call simulates **20ms of real time** (20 × 1ms physics steps)
- The action is **held constant** for all 20 sub-steps
- You get observations at **50 Hz** (every 20ms)
- Data recording at every step = 50 Hz data rate