# SERL-Compatible dm_control Architecture

This directory contains SERL-compatible Gym wrappers for dm_control environments, enabling proper integration between the dm_control physics simulation framework and SERL's training infrastructure.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│  SERL Training/Evaluation Scripts                       │
│  (async_drq_randomized.py, bc_policy.py, etc.)         │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│  SERLGymWrapper (gym.Env)                               │
│  - Action/observation normalization                     │
│  - fake_env mode for fast training                      │
│  - Stats loading for consistent normalization           │
│  - Episode termination handling                         │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│  dm_control Environment                                 │
│  (created by make_sim_env from utils)                  │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│  Task (base.Task)                                       │
│  - CubeStacking: Bimanual cube manipulation             │
│  - Other tasks can be added similarly                   │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│  MuJoCo Physics (mujoco.Physics)                        │
│  - trossen_ai_scene_joint.xml (16 control dims)         │
└─────────────────────────────────────────────────────────┘
```

## Files

### Core Implementation

- **`serl_gym_wrapper.py`**: Main Gym wrapper providing SERL compatibility
  - `SERLGymWrapper`: Wraps dm_control environments with Gym interface
  - Handles normalization (phys mode: min-max + z-score)
  - Supports `fake_env` for fast training startup
  - Automatic action denormalization (normalized [-1,1] → physical units)
  - Episode termination at `max_episode_length`

- **`make_env.py`**: Factory functions for easy environment creation
  - `make_cube_stacking_env()`: General-purpose factory
  - `make_cube_stacking_env_for_training()`: Optimized for training (fake_env, no images)
  - `make_cube_stacking_env_for_eval()`: Optimized for evaluation (real MuJoCo, with images)

### Task Definition

- **`../tasks/cube_stacking.py`**: CubeStacking task (dm_control Task)
  - Inherits from `TrossenAIStationaryTask` (defined in `sim_env.py`)
  - Handles 14D→16D action conversion in `before_step()`
  - Provides observation: 32D state (qpos[16] + qvel[16]) + 4 camera images
  - Implements reward logic for cube stacking

## Usage

### For Training (with fake_env)

```python
from trossen_arm_mujoco.gym_envs import make_cube_stacking_env_for_training

# Create fake environment for fast training
env = make_cube_stacking_env_for_training(
    stats_path="/path/to/norm_stats.json",  # Optional
)

# Use like any Gym environment
obs, info = env.reset()
action = policy(obs)  # Your policy
obs, reward, done, truncated, info = env.step(action)
```

### For Evaluation (with real MuJoCo)

```python
from trossen_arm_mujoco.gym_envs import make_cube_stacking_env_for_eval

# Create real MuJoCo environment for evaluation
env = make_cube_stacking_env_for_eval(
    stats_path="/path/to/norm_stats.json",  # Required for eval
    onscreen_render=True,  # Enable live rendering
)

# Evaluate policy
obs, info = env.reset()
for _ in range(400):
    action = policy(obs)
    obs, reward, done, truncated, info = env.step(action)
    if done:
        break
```

### Custom Configuration

```python
from trossen_arm_mujoco.gym_envs import make_cube_stacking_env

env = make_cube_stacking_env(
    fake_env=False,  # Use real MuJoCo
    image_obs=True,  # Include camera observations
    normalization_mode="phys",  # Min-max + z-score normalization
    stats_path="/path/to/norm_stats.json",
    max_episode_length=400,
    onscreen_render=False,
    cam_list=["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"],
)
```

## Features

### Normalization

**Phys Mode** (default):
- Positions (first 16 dims): Min-max normalization to [-1, 1]
- Velocities (last 16 dims): Z-score normalization using loaded stats
- Actions: Min-max denormalization from [-1, 1] to physical joint limits

**Z-score Mode**:
- All dimensions: Z-score normalization using loaded stats
- Requires `stats_path` with mean/std from training data

### Action Space

- **Input**: 14D normalized actions in [-1, 1]
  - Left arm (6): Joint angles
  - Left gripper (1): Gripper position
  - Right arm (6): Joint angles
  - Right gripper (1): Gripper position

- **Output to MuJoCo**: 16D physical actions
  - Automatically converts 14D→16D (duplicates gripper values)
  - Denormalizes to physical joint limits

### Observation Space

**State**: 32D continuous
- qpos (16): Joint positions
- qvel (16): Joint velocities

**Images** (optional): 4 cameras
- `cam_high`: (128, 128, 3)
- `cam_low`: (128, 128, 3)
- `cam_left_wrist`: (128, 128, 3)
- `cam_right_wrist`: (128, 128, 3)

### fake_env Mode

When `fake_env=True`:
- Skips MuJoCo initialization (fast startup for training)
- Returns dummy observations (zeros for state, blank images)
- Useful for training where only network architecture matters
- Episodes still respect `max_episode_length`

## Migration from Old Architecture

### Before (trossen_sim)

```python
from trossen_sim.envs import TrossenGymWrapper

env = TrossenGymWrapper(
    env_params={'max_episode_length': 400},
    fake_env=True,
    image_obs=False,
)
```

### After (trossen_arm_mujoco)

```python
from trossen_arm_mujoco.gym_envs import make_cube_stacking_env

env = make_cube_stacking_env(
    fake_env=True,
    image_obs=False,
    max_episode_length=400,
)
```

## Benefits Over Old Architecture

1. **Code Reuse**: Uses existing `trossen_arm_mujoco` infrastructure instead of duplicating logic
2. **Separation of Concerns**: Physics (dm_control) vs SERL compatibility (Gym wrapper)
3. **Extensibility**: Easy to add new tasks by subclassing `TrossenAIStationaryTask`
4. **Maintainability**: Changes to physics only need updates in one place
5. **Consistency**: Shares same dm_control framework with data collection pipeline

## Testing

Run the test suite:
```bash
cd trossen_arm_mujoco
python test_serl_wrapper.py
```

Run the example:
```bash
python example_usage.py
```

## Adding New Tasks

To add a new task:

1. Create task class in `trossen_arm_mujoco/tasks/your_task.py`:
```python
from trossen_arm_mujoco.sim_env import TrossenAIStationaryTask

class YourTask(TrossenAIStationaryTask):
    def initialize_episode(self, physics):
        # Reset logic
        super().initialize_episode(physics)
    
    def get_reward(self, physics):
        # Reward computation
        return reward
```

2. Create factory function in `gym_envs/make_env.py`:
```python
def make_your_task_env(...):
    dm_env = make_sim_env(
        task_class=YourTask,
        xml_file="trossen_ai_scene_joint.xml",
        ...
    )
    return SERLGymWrapper(env=dm_env, ...)
```

3. Export in `gym_envs/__init__.py`

## Troubleshooting

### XML File Issues

Make sure to use `trossen_ai_scene_joint.xml` which has 16 control dimensions. Other XML files may have different control dimensions:
- `trossen_ai_scene.xml`: nu=2 (wrong for joint control)
- `trossen_ai_scene_joint.xml`: nu=16 ✓ (correct)
- `trossen_ai_joint.xml`: nu=16 (no scene objects)

### Normalization Stats

For evaluation, always provide `stats_path` with normalization statistics from training data. Without stats, velocity normalization will just clip values which may not match training distribution.

### Episode Termination

Episodes automatically terminate when `steps >= max_episode_length`. The wrapper returns `done=True` and `truncated=True` at this point.

## References

- dm_control documentation: https://github.com/deepmind/dm_control
- SERL: https://github.com/rail-berkeley/serl
- Original trossen_arm_mujoco: `../sim_env.py`, `../utils.py`
