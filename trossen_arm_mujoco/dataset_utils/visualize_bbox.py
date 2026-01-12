import mujoco
import mujoco.viewer
import numpy as np
import time
import pickle
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LEFT_CARTESIAN_BOUNDS = (
    np.array([-0.25, -0.13, 0.05]),  # MINS
    np.array([ -0.17,   0.1, 0.25])  # MAXS
)

# --- RIGHT ROBOT (Worker) ---
# Format: ([x_min, y_min, z_min], [x_max, y_max, z_max])
# Updated from dataset analysis in visualize_bbox.py
RIGHT_CARTESIAN_BOUNDS = (
    np.array([-0.10, -0.1, 0.0]),  # MINS  
    np.array([ 0.25,  0.32,  0.35])   # MAXS
)

# Home Positions (from dataset)
LEFT_HOME_POS = np.array([-0.20546554, -0.01945821,  0.18549709])
RIGHT_HOME_POS = np.array([ 0.20441881, -0.01706709,  0.19527784])

# Gripper positions for Episode 0
LEFT_GRIPPER = 3.99957900000e-02
RIGHT_GRIPPER = 3.99164200000e-02

# Cube spawn region (from utils.py sample_box_pose)
CUBE_SPAWN_BOUNDS = (
    np.array([0.03, -0.02, 0.0125]),   # MINS [x, y, z]
    np.array([ 0.1,  0.1, 0.0125])   # MAXS [x, y, z]
)

def draw_bbox(viewer, bounds_tuple, color):
    """Draw bounding box from (min_array, max_array) tuple format"""
    min_bounds, max_bounds = bounds_tuple
    x_min, y_min, z_min = min_bounds
    x_max, y_max, z_max = max_bounds
    
    center = (min_bounds + max_bounds) / 2
    size = (max_bounds - min_bounds) / 2

    # Check if we have space in the geom buffer
    if viewer.user_scn.ngeom < 1000:
        geom_id = viewer.user_scn.ngeom
        viewer.user_scn.ngeom += 1
        
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[geom_id],
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=size,
            pos=center,
            mat=np.eye(3).flatten(),
            rgba=color
        )

# --- LEFT ROBOT (Simple Box) ---
# Still just one box
LEFT_ZONES = [
    (np.array([-0.255, -0.13, 0.05]), np.array([-0.1, 0.1, 0.355]))
]

# --- RIGHT ROBOT (Complex "Polygon" Shape) ---
# We build this out of 2 overlapping boxes to make an L-shape
RIGHT_ZONES = [
    # BOX 1: The wide "Home" area (Safe zone near base)
    (np.array([-0.10, -0.25, -0.02]), np.array([0.20, 0.35, 0.35])),
    
    # BOX 2: The narrow "Reach" area (Reaching forward)
    # Notice X starts at 0.15 (overlap) but goes further to 0.45
    # But Y is tighter (-0.1 to 0.2) so it doesn't hit the sides
    (np.array([ 0.15, -0.10, -0.02]), np.array([0.45, 0.20, 0.35]))
]

def draw_zones(viewer, zones_list, color):
    """Draws all boxes in a zone list."""
    for min_bounds, max_bounds in zones_list:
        center = (min_bounds + max_bounds) / 2
        size = (max_bounds - min_bounds) / 2

        if viewer.user_scn.ngeom < 1000:
            geom_id = viewer.user_scn.ngeom
            viewer.user_scn.ngeom += 1
            
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[geom_id],
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=size,
                pos=center,
                mat=np.eye(3).flatten(),
                rgba=color
            )

def is_in_polygon(pos, zones_list):
    """Returns True if pos is inside ANY of the boxes."""
    for min_b, max_b in zones_list:
        if np.all(pos >= min_b) and np.all(pos <= max_b):
            return True
    return False

def draw_sphere(viewer, pos, radius, color):
    """Draw a sphere at given position"""
    if viewer.user_scn.ngeom < 1000:
        geom_id = viewer.user_scn.ngeom
        viewer.user_scn.ngeom += 1
        
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[geom_id],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[radius, radius, radius],
            pos=pos,
            mat=np.eye(3).flatten(),
            rgba=color
        )

def visualize_bboxes_and_mocap():
    """Visualize bounding boxes and mocap positions"""
    xml_path = os.path.join(os.path.dirname(__file__), "../assets/trossen_ai_scene.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mj_resetData(model, data)
        
        # 1. Teleport robots to home
        data.mocap_pos[0] = LEFT_HOME_POS
        data.mocap_pos[1] = RIGHT_HOME_POS
        data.ctrl[0] = LEFT_GRIPPER
        data.ctrl[1] = RIGHT_GRIPPER
        
        # 2. Step simulation briefly to settle
        for _ in range(500): 
            mujoco.mj_step(model, data)

        # 3. Enable transparency in viewer options
        viewer.opt.flags[mujoco.mjtRndFlag.mjRND_CULL_FACE] = 0 # Double sided
        
        print("\n=== VISUALIZER RUNNING ===")
        print(f"Left mocap: {data.mocap_pos[0]}")
        print(f"Right mocap: {data.mocap_pos[1]}")

        while viewer.is_running():
            # Step physics
            mujoco.mj_step(model, data)

            # CRITICAL FIX: Reset geom counter so we don't stack boxes infinitely
            viewer.user_scn.ngeom = 0

            # Add Transparent Geoms (Alpha 0.2)
            draw_bbox(viewer, LEFT_CARTESIAN_BOUNDS, color=[1.0, 0.0, 0.0, 0.2]) 
            draw_bbox(viewer, RIGHT_CARTESIAN_BOUNDS, color=[0.0, 1.0, 0.0, 0.2])
            draw_bbox(viewer, CUBE_SPAWN_BOUNDS, color=[1.0, 0.5, 0.0, 0.4])       # Orange - Cube spawn region
            
            # Add Mocap Spheres (Solid)
            # draw_sphere(viewer, data.mocap_pos[0], 0.03, [0.0, 1.0, 1.0, 1.0])  # Cyan
            # draw_sphere(viewer, data.mocap_pos[1], 0.03, [1.0, 1.0, 0.0, 1.0])  # Yellow
            
            # Sync viewer (this handles the actual scene update)
            viewer.sync()
            time.sleep(0.01)

def load_transitions(file_path: str) -> list:
    print(f"Loading data from {file_path}...")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict) and 'transitions' in data:
        return data['transitions']
    elif isinstance(data, list):
        return data
    else:
        raise ValueError("Unexpected data format.")

def check_dataset_bounds():
    file_path = "/home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/same_cube_pose/sim_dataset_binarized_gripper_regenerated_states_FINAL.pkl"
    
    try:
        transitions = load_transitions(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return

    # 1. Extract all positions into a numpy array for fast checking
    # Indices based on your description:
    # Left Pos: 0,1,2 | Right Pos: 8,9,10
    
    left_positions = []
    right_positions = []
    
    print("Extracting positions from dataset...")
    for t in transitions:
        # Obs is likely shape (1, 16) or (16,)
        obs = t['observations']['state'].flatten() 
        
        left_positions.append(obs[0:3])
        right_positions.append(obs[8:11])
        
    left_positions = np.array(left_positions)   # Shape (N, 3)
    right_positions = np.array(right_positions) # Shape (N, 3)
    
    print(f"Analyzed {len(left_positions)} steps.")
    print("-" * 60)

    # --- 2. CHECK LEFT ROBOT ---
    print("--- LEFT ROBOT CHECK ---")
    _validate_bounds(left_positions, LEFT_CARTESIAN_BOUNDS, "Left")

    print("\n" + "-" * 60)

    # --- 3. CHECK RIGHT ROBOT ---
    print("--- RIGHT ROBOT CHECK ---")
    _validate_bounds(right_positions, RIGHT_CARTESIAN_BOUNDS, "Right")

def _validate_bounds(positions, bounds_tuple, robot_name):
    """Helper to compare data min/max vs defined bounds."""
    # bounds_tuple is (min_array, max_array)
    defined_min, defined_max = bounds_tuple
    
    data_min = positions.min(axis=0)
    data_max = positions.max(axis=0)
    
    axes = ['X', 'Y', 'Z']
    violation = False
    
    print(f"{'Axis':<5} | {'Min (Data)':<12} | {'Min (Bound)':<12} | {'Max (Data)':<12} | {'Max (Bound)':<12} | {'Status'}")
    print("-" * 75)
    
    for i, axis in enumerate(axes):
        # Check Min
        min_ok = data_min[i] >= defined_min[i]
        # Check Max
        max_ok = data_max[i] <= defined_max[i]
        
        status = "OK"
        if not min_ok:
            status = f"LOW (Diff: {data_min[i] - defined_min[i]:.4f})"
            violation = True
        elif not max_ok:
            status = f"HIGH (Diff: {data_max[i] - defined_max[i]:.4f})"
            violation = True
            
        print(f"{axis:<5} | {data_min[i]:<12.4f} | {defined_min[i]:<12.4f} | {data_max[i]:<12.4f} | {defined_max[i]:<12.4f} | {status}")

    if violation:
        print(f"\n[WARNING] The {robot_name} robot dataset contains points OUTSIDE your defined bounds.")
        print("Recommendation: Widen the bounds slightly beyond the 'Min (Data)' and 'Max (Data)' columns above.")
    else:
        print(f"\n[SUCCESS] All {robot_name} robot data is contained within the bounds.")


def validate_action_sampling(num_samples=400):
    """Sample actions, step env, and visualize live with custom bounds."""

    print("\n" + "="*80)
    print("ACTION SAMPLING VALIDATION (WITH VISUALIZATION)")
    print("="*80)
    
    # Debug: Check what XML file is being loaded and initial solref
    import os
    from trossen_arm_mujoco.utils import ASSETS_DIR
    xml_path = os.path.join(ASSETS_DIR, "trossen_ai_scene.xml")
    print(f"\n[DEBUG] XML path: {xml_path}")
    print(f"[DEBUG] XML exists: {os.path.exists(xml_path)}")
    
    # Import the environment creation utilities
    from trossen_arm_mujoco.utils import make_sim_env
    from trossen_arm_mujoco.delta_ee_sim_env_serl_drq_BC import SERLGymWrapper1, CubeStackingEE
    from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
    from serl_launcher.wrappers.chunking import ChunkingWrapper
    
    # Create environment
    CONTROL_TIMESTEP = 0.02
    PHYSICS_TIMESTEP = 0.002
    cam_list = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]
    
    print("\nCreating environment...")
    dm_env = make_sim_env(
        CubeStackingEE,
        task_name="sim_transfer_cube",
        onscreen_render=True, # We use our own passive viewer, so keep this False
        cam_list=cam_list,
        control_timestep=CONTROL_TIMESTEP,
        physics_timestep=PHYSICS_TIMESTEP,
    )
    
    env = SERLGymWrapper1(
        env=dm_env,
        state_obs_dim=16,
        max_episode_length=200,
        onscreen_render=True,
        cam_list=cam_list,
        action_dim=14,
        time_limit=20.0,
        save_video=False,
        video_path=None,
    )
    
    env = SERLObsWrapper(env, proprio_keys=None)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    
    # --- EXTRACT MUJOCO OBJECTS FOR VIEWER ---
    # We need to dig through wrappers to get the raw dm_control physics
    base_env = env
    while hasattr(base_env, 'env') and not isinstance(base_env, SERLGymWrapper1):
        base_env = base_env.env
    
    # Get the physics object from the base env
    physics = base_env.env.physics
    
    # Extract raw MuJoCo model and data for the viewer
    # Note: .ptr gives the underlying C-structs required by mujoco.viewer
    model = physics.model.ptr
    data = physics.data.ptr

    # =========================================================================
    # [CRITICAL FIX]: OVERRIDE XML STIFFNESS AT RUNTIME (Targeted)
    # =========================================================================
    # print(f"\n[PHYSICS TWEAK] Inspecting {model.neq} equality constraints...")
    
    # for i in range(model.neq):
    #     # 1. Get body IDs and Names
    #     id1 = model.eq_obj1id[i]
    #     id2 = model.eq_obj2id[i]
        
    #     # mj_id2name handles the lookup (returns None if missing)
    #     name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, id1) or str(id1)
    #     name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, id2) or str(id2)
        
    #     # 2. Check if this is a "Mocap" constraint
    #     # We look for the word "mocap" in either body name
    #     is_mocap_constraint = ("mocap" in name1) or ("mocap" in name2)
        
    #     if is_mocap_constraint:
    #         old_solref = model.eq_solref[i].copy()
            
    #         # STIFFEN IT: 0.004 is 2x physics timestep (very rigid)
    #         model.eq_solref[i] = np.array([0.004, 1.0])
            
    #         print(f"  STIFFENED Constraint {i} [{name1} <--> {name2}]")
    #         print(f"     solref: {old_solref} -> {model.eq_solref[i]}")
    #     else:
    #         print(f"  Skipping Constraint {i} [{name1} <--> {name2}] (Not a Mocap weld)")

    # # Apply changes
    # mujoco.mj_forward(model, data)
    # =========================================================================

    print(f"Action space: {env.action_space}")
    
    # Reset environment
    obs, info = env.reset(seed=42)
    
    left_violations = 0
    right_violations = 0
    
    # --- LAUNCH VIEWER ---
    print("Launching Passive Viewer... (Check the popup window)")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        
        # Enable transparency/double-sided rendering
        viewer.opt.flags[mujoco.mjtRndFlag.mjRND_CULL_FACE] = 0
        
        print(f"\nSampling {num_samples} actions...")
        
        for i in range(num_samples):
            if not viewer.is_running():
                print("Viewer closed by user.")
                break

            # 1. Sample random action
            action = env.action_space.sample()
            
            # 2. Step environment (this moves the robot in 'data')
            obs, reward, terminated, truncated, info = env.step(action)
            import time
            time.sleep(3.0)
            # 3. Visualization Logic
            # IMPORTANT: Reset geom counter to prevent opacity stacking
            viewer.user_scn.ngeom = 0
            
            # Draw Transparent Safety Boxes
            # draw_bbox(viewer, LEFT_CARTESIAN_BOUNDS, color=[1.0, 0.0, 0.0, 0.2]) 
            # draw_bbox(viewer, RIGHT_CARTESIAN_BOUNDS, color=[0.0, 1.0, 0.0, 0.2])

            draw_bbox(viewer, CUBE_SPAWN_BOUNDS, color=[1.0, 0.5, 0.0, 1.0])       # Orange - Cube spawn region
            

            # Draw Mocap Spheres (Cyan/Yellow)
            # We access the positions directly from physics.data which stays synced
            # draw_sphere(viewer, physics.data.mocap_pos[0], 0.01, [0.0, 1.0, 1.0, 1.0]) 
            # draw_sphere(viewer, physics.data.mocap_pos[1], 0.01, [1.0, 1.0, 0.0, 1.0]) 
            
            # Sync the viewer to show the new state
            viewer.sync()
            
            # Slow down so we can see the movement
            # time.sleep(0.02) 
            
            # 4. Check Bounds (logic remains the same)
            left_pos = physics.data.mocap_pos[0].copy()
            right_pos = physics.data.mocap_pos[1].copy()
            
            left_in_bounds = np.all(left_pos >= LEFT_CARTESIAN_BOUNDS[0]) and np.all(left_pos <= LEFT_CARTESIAN_BOUNDS[1])
            right_in_bounds = np.all(right_pos >= RIGHT_CARTESIAN_BOUNDS[0]) and np.all(right_pos <= RIGHT_CARTESIAN_BOUNDS[1])
            
            if not left_in_bounds:
                left_violations += 1
                print(f"Step {i}: LEFT VIOLATION - pos={left_pos}")
            
            if not right_in_bounds:
                right_violations += 1
                print(f"Step {i}: RIGHT VIOLATION - pos={right_pos}")
            
            # Reset if episode ends
            if terminated or truncated:
                obs, info = env.reset()
                
    print("\n" + "="*80)
    print("VALIDATION FINISHED")
    print(f"Total steps: {num_samples}")
    print(f"Violations: Left={left_violations}, Right={right_violations}")
    print("="*80)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Visualize bounds and validate actions')
    parser.add_argument('--mode', choices=['visualize', 'check_dataset', 'validate_actions'], 
                       default='visualize', help='Mode to run')
    parser.add_argument('--num_samples', type=int, default=400, 
                       help='Number of action samples to test')
    args = parser.parse_args()
    
    if args.mode == 'visualize':
        print("Launching visualization...")
        visualize_bboxes_and_mocap()
    elif args.mode == 'check_dataset':
        print("Checking dataset bounds...")
        check_dataset_bounds()
    elif args.mode == 'validate_actions':
        print(f"Validating {args.num_samples} random actions...")
        validate_action_sampling(num_samples=args.num_samples)



"""
# 1. Visualize bounds and mocap positions (blue/yellow spheres)
python3 trossen_arm_mujoco/dataset_utils/visualize_bbox.py --mode visualize

# 2. Check demo dataset fits within bounds
python3 trossen_arm_mujoco/dataset_utils/visualize_bbox.py --mode check_dataset

# 3. Validate action sampling (what you just ran)
python3 trossen_arm_mujoco/dataset_utils/visualize_bbox.py --mode validate_actions --num_samples 400
"""



"""
Checking dataset bounds...
Loading data from /home/qte9489/personal_abhi/temp/hil-serl/delta_action_ds/v1/same_cube_pose/sim_dataset_binarized_gripper_regenerated_states_FINAL.pkl...
Extracting positions from dataset...
Analyzed 6752 steps.
------------------------------------------------------------
--- LEFT ROBOT CHECK ---
Axis  | Min (Data)   | Min (Bound)  | Max (Data)   | Max (Bound)  | Status
---------------------------------------------------------------------------
X     | -0.2055      | -0.2500      | -0.2054      | -0.1250      | OK
Y     | -0.0197      | -0.1300      | -0.0193      | 0.1000       | OK
Z     | 0.1756       | 0.0500       | 0.1857       | 0.3550       | OK

[SUCCESS] All Left robot data is contained within the bounds.

------------------------------------------------------------
--- RIGHT ROBOT CHECK ---
Axis  | Min (Data)   | Min (Bound)  | Max (Data)   | Max (Bound)  | Status
---------------------------------------------------------------------------
X     | -0.0496      | -0.1000      | 0.2055       | 0.2750       | OK
Y     | -0.0363      | -0.1500      | 0.3139       | 0.3300       | OK
Z     | -0.0138      | -0.0200      | 0.2615       | 0.3500       | OK

[SUCCESS] All Right robot data is contained within the bounds.
"""