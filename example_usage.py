"""
Example: Using the new SERL-compatible dm_control architecture.

This demonstrates how to replace the old PureTrossenEnv with the new
CubeStacking task + SERLGymWrapper architecture.
"""

import numpy as np
from trossen_arm_mujoco.gym_envs import (
    make_cube_stacking_env_for_training,
    make_cube_stacking_env_for_eval,
)


def example_training_setup():
    """Example: Setting up environment for BC training."""
    print("=== Training Setup (with fake_env) ===\n")
    
    # For training, use fake_env to skip MuJoCo initialization
    # This makes training startup much faster
    env = make_cube_stacking_env_for_training(
        stats_path=None,  # Can provide path to normalization stats JSON
    )
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Max episode length: {env.max_episode_length}")
    print(f"Fake env: {env.fake_env}")
    print(f"Image obs: {env.image_obs}")
    
    # Reset and step
    obs, info = env.reset()
    print(f"\nObservation keys: {obs.keys()}")
    print(f"State shape: {obs['state'].shape}")
    
    # Sample random action
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    print(f"\nAfter step: done={done}, truncated={truncated}")
    
    env.close()


def example_evaluation_setup():
    """Example: Setting up environment for policy evaluation."""
    print("\n=== Evaluation Setup (with real MuJoCo) ===\n")
    
    # For evaluation, use real MuJoCo with images
    env = make_cube_stacking_env_for_eval(
        stats_path=None,  # MUST provide stats for real evaluation
        onscreen_render=False,  # Set True to see live rendering
    )
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Fake env: {env.fake_env}")
    print(f"Image obs: {env.image_obs}")
    
    # Reset
    obs, info = env.reset()
    print(f"\nObservation keys: {obs.keys()}")
    print(f"State shape: {obs['state'].shape}")
    
    if 'images' in obs:
        print(f"Image cameras: {list(obs['images'].keys())}")
        for cam_name, img in obs['images'].items():
            print(f"  {cam_name}: {img.shape}, dtype={img.dtype}")
    
    # Run a few steps
    print("\nRunning 5 steps...")
    for i in range(5):
        # In real scenario, replace this with policy output
        action = np.zeros(14)  # Zero action (stay still)
        obs, reward, done, truncated, info = env.step(action)
        print(f"  Step {i+1}: reward={reward}, done={done}")
        if done:
            break
    
    env.close()


def example_migration_guide():
    """Show how to migrate from old to new architecture."""
    print("\n=== Migration Guide ===\n")
    
    print("OLD CODE (trossen_sim):")
    print("""
    from trossen_sim.envs import TrossenGymWrapper
    
    env = TrossenGymWrapper(
        env_params={
            'max_episode_length': 400,
        },
        fake_env=True,
        image_obs=False,
        normalization_mode='phys',
    )
    """)
    
    print("\nNEW CODE (trossen_arm_mujoco):")
    print("""
    from trossen_arm_mujoco.gym_envs import make_cube_stacking_env
    
    env = make_cube_stacking_env(
        fake_env=True,
        image_obs=False,
        normalization_mode='phys',
        max_episode_length=400,
    )
    """)
    
    print("\nBenefits:")
    print("  ✓ Reuses existing dm_control infrastructure (trossen_arm_mujoco)")
    print("  ✓ Proper separation: physics (dm_control) vs SERL compat (Gym wrapper)")
    print("  ✓ Easier to add new tasks (just subclass CubeStacking)")
    print("  ✓ Cleaner code organization")


def main():
    """Run all examples."""
    print("=" * 60)
    print("SERL-compatible dm_control Architecture Examples")
    print("=" * 60)
    
    example_training_setup()
    example_evaluation_setup()
    example_migration_guide()
    
    print("\n" + "=" * 60)
    print("Done! Check the source code for more details.")
    print("=" * 60)


if __name__ == "__main__":
    main()
