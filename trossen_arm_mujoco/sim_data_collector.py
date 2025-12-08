"""
Sim Data Collector - Interactive GUI for recording simulation episodes.

=== HDF5 DATASET STRUCTURE in EE Mode===

Each HDF5 file (e.g., episode_0.hdf5) contains the following keys:

1. **action** (N, 16) [EE Mode]:
   - The RAW command sent to the robot.
   - Format: [L_Pos(3), L_Quat(4), L_Grip(1), R_Pos(3), R_Quat(4), R_Grip(1)]
   - Quaternions are [w, x, y, z] (MuJoCo convention).

2. **action_angle_axis** (N, 14) [EE Mode]:
   - A derived version of 'action' for convenience.
   - Converts Quaternions to Angle-Axis (3D vector).
   - Format: [L_Pos(3), L_AA(3), L_Grip(1), R_Pos(3), R_AA(3), R_Grip(1)]
   - Useful for training policies that output Angle-Axis instead of Quaternions.

3. **observations/** (Group):
   - Contains the STATE of the simulation (what happened).
   
   - **robot0_eef_pos** (N, 6): Actual EE Position [L_Pos(3), R_Pos(3)].
   - **robot0_eef_quat** (N, 8): Actual EE Orientation (Quat) [L_Quat(4), R_Quat(4)].
   - **robot0_eef_angle_axis** (N, 6): Actual EE Orientation (Angle-Axis).
   - **robot0_eef_euler** (N, 6): Actual EE Orientation (Euler).
   
   - **qpos** (N, 16): Joint positions.
   - **qvel** (N, 16): Joint velocities.
   - **images/**: Camera feeds (compressed).

=== KEY DISTINCTION ===
- **action_*** keys are what the LEADER (human/policy) *commanded*.
- **observations/robot0_eef_*** keys are where the FOLLOWER (sim robot) *actually was*.
- They should be very close, but not identical due to physics/tracking errors.

Dataset format and control modes documented in README.md:
    - Section: "Control Modes & Dataset Format"
    
Usage:
    python sim_data_collector.py --control_mode joint --save_dir ./data
    python sim_data_collector.py --control_mode ee --save_dir ./data
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from trossen_arm_mujoco.ee_transforms import quaternion_to_angle_axis


class SimDataCollector:
    """
    Data collector for simulation environments with interactive GUI.
    Records episodes to HDF5 files for BC/RL training.
    
    Args:
        env_name: Name of the environment
        save_dir: Directory to save HDF5 files
        cam_list: List of camera names
        save_every_n: Save data every N steps (1 = 50Hz, 5 = 10Hz, etc.)
        on_move_home: Callback function for "Move to Home" button (optional)
    """
    def __init__(self, env_name, save_dir, cam_list, save_every_n: int = 1, on_move_home=None):
        self.env_name = env_name
        self.save_dir = save_dir
        self.cam_list = cam_list
        self.save_every_n = save_every_n
        self.step_count = 0
        self.on_move_home = on_move_home  # Callback for home button
        self.on_reset_env = None  # Callback for reset button
        
        # Create save directory
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        # State
        self.is_recording = False
        self.episode_count = self._get_next_episode_idx()
        self.current_episode_data = self._init_buffer()
        
        # Visualization handles
        self.fig = None
        self.axs = None
        self.plt_imgs = None
        self.buttons = {}
        self.text_display = None
        
        # Setup visualizer
        self._setup_visualizer()

    def _get_next_episode_idx(self):
        """Find the next available episode index based on existing files."""
        existing_files = [f for f in os.listdir(self.save_dir) if f.endswith('.hdf5') and f.startswith('episode_')]
        if not existing_files:
            return 0
        indices = [int(f.split('_')[1].split('.')[0]) for f in existing_files]
        return max(indices) + 1

    def _init_buffer(self):
        """Initialize empty data buffer."""
        return {
            "observations": {
                "images": {cam: [] for cam in self.cam_list},
                # Other fields added dynamically from dm_obs
            },
            "action": [],           # 16D: [L_Pos(3), L_Quat(4), L_Grip(1), R_Pos(3), R_Quat(4), R_Grip(1)]
            "action_angle_axis": [], # 14D: [L_Pos(3), L_AA(3), L_Grip(1), R_Pos(3), R_AA(3), R_Grip(1)]
            "reward": [],
            "done": []
        }

    def _setup_visualizer(self):
        """Setup Matplotlib figure with camera feeds and control buttons."""
        # Create figure with extra space at bottom for controls
        num_cameras = len(self.cam_list)
        cols = 2 if num_cameras == 4 else min(3, num_cameras)
        rows = (num_cameras + cols - 1) // cols
        
        self.fig, self.axs = plt.subplots(rows, cols, figsize=(10, 12))
        self.fig.subplots_adjust(bottom=0.2)  # Make room for buttons
        
        self.axs = self.axs.flatten() if isinstance(self.axs, (list, np.ndarray)) else [self.axs]
        
        # Initialize images (will be updated in step)
        self.plt_imgs = []
        for ax in self.axs:
            ax.axis("off")
            # Create placeholder image
            img = ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
            self.plt_imgs.append(img)
            
        # Add GUI Elements
        # Position: [left, bottom, width, height]
        
        # Status Text
        self.text_ax = self.fig.add_axes([0.1, 0.12, 0.8, 0.05])
        self.text_ax.axis("off")
        self.text_display = self.text_ax.text(0.5, 0.5, self._get_status_text(), 
                                            ha="center", va="center", fontsize=12)

        # Start/Stop Button
        self.btn_record_ax = self.fig.add_axes([0.05, 0.05, 0.15, 0.05])
        self.btn_record = Button(self.btn_record_ax, 'Start Recording', color='lightgreen', hovercolor='0.975')
        self.btn_record.on_clicked(self._toggle_recording)

        # Reset Env Button
        self.btn_reset_ax = self.fig.add_axes([0.21, 0.05, 0.12, 0.05])
        self.btn_reset = Button(self.btn_reset_ax, 'Reset Env', color='lightblue', hovercolor='0.975')
        self.btn_reset.on_clicked(self._request_reset)

        # Delete Last Button
        self.btn_delete_ax = self.fig.add_axes([0.34, 0.05, 0.12, 0.05])
        self.btn_delete = Button(self.btn_delete_ax, 'Del Last', color='salmon', hovercolor='0.975')
        self.btn_delete.on_clicked(self._delete_last_episode)
        
        # Delete Specific ID (TextBox + Button)
        self.txt_delete_id_ax = self.fig.add_axes([0.52, 0.05, 0.08, 0.05])
        self.txt_delete_id = TextBox(self.txt_delete_id_ax, 'ID:', initial="0")
        self.delete_id_value = "0"  # Store the value separately
        self.txt_delete_id.on_submit(self._on_delete_id_change)
        
        self.btn_delete_id_ax = self.fig.add_axes([0.61, 0.05, 0.08, 0.05])
        self.btn_delete_id = Button(self.btn_delete_id_ax, 'Del ID', color='salmon', hovercolor='0.975')
        self.btn_delete_id.on_clicked(self._delete_specific_episode)
        
        # Move to Home Button
        self.btn_home_ax = self.fig.add_axes([0.70, 0.05, 0.12, 0.05])
        self.btn_home = Button(self.btn_home_ax, 'Home', color='lightyellow', hovercolor='0.975')
        self.btn_home.on_clicked(self._move_to_home)

        plt.ion()

    def _get_status_text(self):
        status = "RECORDING" if self.is_recording else "IDLE"
        frames = len(self.current_episode_data["action"]) if self.is_recording else 0
        freq = 50 // self.save_every_n
        return f"Status: {status} | Frames: {frames} | Next Ep: {self.episode_count} | {freq}Hz"

    def _update_status(self):
        self.text_display.set_text(self._get_status_text())
        self.btn_record.label.set_text("Stop Recording" if self.is_recording else "Start Recording")
        self.btn_record.color = 'salmon' if self.is_recording else 'lightgreen'
        self.fig.canvas.draw_idle()

    def _toggle_recording(self, event):
        self.is_recording = not self.is_recording
        if self.is_recording:
            print(f"Started recording episode {self.episode_count}")
            self.current_episode_data = self._init_buffer()
        else:
            print("Stopped recording")
            # If we have data, save it
            if len(self.current_episode_data["action"]) > 0:
                self._save_episode()
            else:
                print("No data recorded, discarding.")
        self._update_status()

    def _save_episode(self):
        filename = f"episode_{self.episode_count}.hdf5"
        filepath = os.path.join(self.save_dir, filename)
        
        print(f"Saving to {filepath}...")
        
        with h5py.File(filepath, "w") as root:
            root.attrs["sim"] = True
            root.attrs["env_name"] = self.env_name
            
            obs_grp = root.create_group("observations")
            img_grp = obs_grp.create_group("images")
            
            # Save images
            for cam_name in self.cam_list:
                imgs = np.array(self.current_episode_data["observations"]["images"][cam_name])
                if len(imgs) > 0:
                    img_grp.create_dataset(cam_name, data=imgs, compression="gzip")
            
            # Save all other observation fields (qpos, qvel, EE fields, etc.)
            for key, data in self.current_episode_data["observations"].items():
                if key == "images":
                    continue  # Already saved above
                if len(data) > 0:
                    obs_grp.create_dataset(key, data=np.array(data))
            
            # Save actions in both formats
            root.create_dataset("action", data=np.array(self.current_episode_data["action"]))
            if len(self.current_episode_data["action_angle_axis"]) > 0:
                root.create_dataset("action_angle_axis", data=np.array(self.current_episode_data["action_angle_axis"]))
            
            root.create_dataset("reward", data=np.array(self.current_episode_data["reward"]))
            root.create_dataset("done", data=np.array(self.current_episode_data["done"]))
            
        print(f"Saved episode {self.episode_count}")
        self.episode_count += 1
        self._update_status()

    def _delete_last_episode(self, event):
        target_idx = self.episode_count - 1
        self._delete_episode(target_idx)

    def _on_delete_id_change(self, text):
        """Called when user types in the ID text box."""
        self.delete_id_value = text

    def _delete_specific_episode(self, event):
        try:
            # Get the current text from the textbox
            target_idx = int(self.txt_delete_id.text.strip())
            print(f"[Delete] Requested deletion of episode {target_idx}")
            self._delete_episode(target_idx)
        except ValueError as e:
            print(f"Invalid episode ID: '{self.txt_delete_id.text}' - {e}")

    def _delete_episode(self, idx):
        filename = f"episode_{idx}.hdf5"
        filepath = os.path.join(self.save_dir, filename)
        
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"Deleted {filename}")
            # If we deleted the last one, decrement counter
            if idx == self.episode_count - 1:
                self.episode_count = idx
                self._update_status()
        else:
            print(f"File {filename} not found")
    
    def _move_to_home(self, event):
        """Handle Move to Home button click."""
        if self.on_move_home is not None:
            print("[Home] Moving robots to HOME position...")
            self.on_move_home()
        else:
            print("[Home] Home callback not set")
    
    def _request_reset(self, event):
        """Handle Reset Env button click."""
        if self.on_reset_env is not None:
            print("[Reset] Resetting environment...")
            self.on_reset_env()
        else:
            print("[Reset] Reset callback not set")
    
    def set_home_callback(self, callback):
        """Set the callback for Move to Home button."""
        self.on_move_home = callback
    
    def set_reset_callback(self, callback):
        """Set the callback for Reset Env button."""
        self.on_reset_env = callback

    def update_viz(self, full_res_images):
        """Update visualization only (e.g. at reset)."""
        viz_images = full_res_images if full_res_images is not None else {}
        for i, cam in enumerate(self.cam_list):
            if cam in viz_images and i < len(self.plt_imgs):
                self.plt_imgs[i].set_data(viz_images[cam])
        plt.pause(0.001)

    def step(self, obs, action, reward, done, info, full_res_images=None):
        """
        Called every environment step.
        Updates visualization and records data if recording (at configured frequency).
        
        Args:
            obs: Observation dict with:
                - 'state': (32,) qpos+qvel
                - 'images': resized images
                - 'dm_obs': raw dm_control observation (includes EE fields if present)
            action: Action taken
            reward: Reward received
            done: Done flag
            info: Info dict
            full_res_images: Optional dict of 480x640 images for visualization/recording
        """
        self.step_count += 1
        
        # Update visualization every step (50Hz)
        if self.step_count % 1 == 0:
            viz_images = full_res_images if full_res_images is not None else obs.get('images', {})
            for i, cam in enumerate(self.cam_list):
                if cam in viz_images and i < len(self.plt_imgs):
                    self.plt_imgs[i].set_data(viz_images[cam])
            
            plt.pause(0.001)  # Small pause to update GUI
        
        # Record data only every N steps
        if self.is_recording and (self.step_count % self.save_every_n == 0):
            # Save images (full resolution)
            save_images = full_res_images if full_res_images is not None else obs.get('images', {})
            for cam in self.cam_list:
                if cam in save_images:
                    self.current_episode_data["observations"]["images"][cam].append(save_images[cam])
            
            # Save all fields from dm_obs (qpos, qvel, EE fields, etc.)
            dm_obs = obs.get('dm_obs', {})
            for key, value in dm_obs.items():
                if key == 'images':
                    continue  # Already saved above
                if key not in self.current_episode_data["observations"]:
                    self.current_episode_data["observations"][key] = []
                self.current_episode_data["observations"][key].append(value)
            
            # Save action in quaternion format (16D for EE mode, 14D for joint mode)
            self.current_episode_data["action"].append(action)
            
            # Also save angle-axis format for EE mode (14D alternative to 16D quaternion)
            # This allows training with either representation without re-collecting data
            if len(action) == 16:  # EE mode with quaternions
                # Convert: [L_Pos(3), L_Quat(4), L_Grip(1), R_Pos(3), R_Quat(4), R_Grip(1)]
                #      to: [L_Pos(3), L_AA(3),   L_Grip(1), R_Pos(3), R_AA(3),   R_Grip(1)]
                left_aa = quaternion_to_angle_axis(action[3:7])
                right_aa = quaternion_to_angle_axis(action[11:15])
                action_aa = np.concatenate([
                    action[0:3], left_aa, action[7:8],    # Left: pos, aa, grip
                    action[8:11], right_aa, action[15:16] # Right: pos, aa, grip
                ])
                self.current_episode_data["action_angle_axis"].append(action_aa)
            else:  # Joint mode (14D) - no conversion needed
                self.current_episode_data["action_angle_axis"].append(action)
            
            self.current_episode_data["reward"].append(reward)
            self.current_episode_data["done"].append(done)
            
            # Update status to show frame count
            self._update_status()
            
            if done:
                print("Episode done.")

    def reset(self):
        """Called on environment reset."""
        # If we want to auto-start recording on reset, we could do it here.
        # For now, manual control is safer.
        pass


def main():
    """
    Main entry point for sim data collection with optional teleoperation.
    
    Usage (random actions - testing):
        python sim_data_collector.py --save_dir ./my_data
    
    Usage (teleoperation with real leader robots):
        python sim_data_collector.py --save_dir ./my_data \
            --leader_left_ip 192.168.1.4 --leader_right_ip 192.168.1.2
    """
    import argparse
    from trossen_arm_mujoco.gym_envs.make_env import make_cube_stacking_env
    
    parser = argparse.ArgumentParser(description="Sim Data Collector with GUI")
    parser.add_argument("--save_dir", type=str, default=None, 
                        help="Directory to save HDF5 episodes (default: ./sim_recordings_{control_mode})")
    parser.add_argument("--save_every_n", type=int, default=1,
                        help="Save every N steps (1=50Hz, 5=10Hz)")
    parser.add_argument("--max_episode_length", type=int, default=1000,
                        help="Max steps per episode")
    parser.add_argument("--arm_type", type=str, default="widowx",
                        choices=["widowx", "viperx"], help="Robot arm type")
    parser.add_argument("--control_mode", type=str, default="ee",
                        choices=["joint", "ee"], help="Control mode: joint (14D) or ee (16D)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible cube placement. If not set, uses random seed each reset.")
    # Teleoperation args
    parser.add_argument("--leader_left_ip", type=str, default="192.168.1.4",
                        help="Left leader robot IP (e.g., 192.168.1.4). If not set, uses random actions.")
    parser.add_argument("--leader_right_ip", type=str, default="192.168.1.2",
                        help="Right leader robot IP (e.g., 192.168.1.2). If not set, uses random actions.")
    
    args = parser.parse_args()
    
    # Check if teleoperation is enabled
    teleop_enabled = args.leader_left_ip is not None and args.leader_right_ip is not None
    
    # Set default save_dir based on control_mode if not provided
    if args.save_dir is None:
        args.save_dir = f"./sim_recordings_{args.control_mode}"
    
    print("=" * 60)
    print("SIM DATA COLLECTOR")
    print("=" * 60)
    print(f"  Save Dir: {args.save_dir}")
    print(f"  Save Frequency: {50 // args.save_every_n} Hz")
    print(f"  Max Episode Length: {args.max_episode_length}")
    print(f"  Arm Type: {args.arm_type}")
    print(f"  Control Mode: {args.control_mode} ({'14D' if args.control_mode == 'joint' else '16D'})")
    print(f"  Seed: {args.seed if args.seed is not None else 'random'}")
    if teleop_enabled:
        print(f"  Teleop Mode: ENABLED")
        print(f"    Left Leader IP: {args.leader_left_ip}")
        print(f"    Right Leader IP: {args.leader_right_ip}")
    else:
        print(f"  Teleop Mode: DISABLED (using random actions)")
    print("=" * 60)
    
    # Initialize leader robots if teleop enabled
    driver_left = None
    driver_right = None
    
    if teleop_enabled:
        try:
            import trossen_arm
            
            print("\n📡 Connecting to leader robots...")
            # 192.168.1.4 LEFT
            # 192.168.1.2 RIGHT
            # Left leader
            print(f"  Connecting to left leader at {args.leader_left_ip}...")
            driver_left = trossen_arm.TrossenArmDriver()
            driver_left.configure(
                trossen_arm.Model.wxai_v0,
                trossen_arm.StandardEndEffector.wxai_v0_leader,
                args.leader_left_ip,
                False
            )
            
            # Right leader
            print(f"  Connecting to right leader at {args.leader_right_ip}...")
            driver_right = trossen_arm.TrossenArmDriver()
            driver_right.configure(
                trossen_arm.Model.wxai_v0,
                trossen_arm.StandardEndEffector.wxai_v0_leader,
                args.leader_right_ip,
                False
            )
            
            print(f"✓ Both leaders connected")
            print(f"  Left: {driver_left.get_num_joints()} joints")
            print(f"  Right: {driver_right.get_num_joints()} joints")
            
            # Open grippers first (position mode)
            print("\n🔓 Opening grippers...")
            GRIPPER_OPEN = 0.04  # meters (fully open)
            
            driver_left.set_all_modes(trossen_arm.Mode.position)
            driver_right.set_all_modes(trossen_arm.Mode.position)
            
            # Set gripper to open position (index 6 is gripper)
            left_pos = np.array(driver_left.get_all_positions())
            right_pos = np.array(driver_right.get_all_positions())
            left_pos[6] = GRIPPER_OPEN
            right_pos[6] = GRIPPER_OPEN
            driver_left.set_all_positions(left_pos)
            driver_right.set_all_positions(right_pos)
            
            import time
            time.sleep(0.5)  # Wait for grippers to open
            print("✓ Grippers opened")
            
            # Set leaders to external effort mode (free to move)
            print("\n🎮 Setting leaders to external effort mode (free to move)...")
            zero_efforts = np.zeros(7)
            
            driver_left.set_all_modes(trossen_arm.Mode.external_effort)
            driver_left.set_all_external_efforts(zero_efforts, 0.0, False)
            
            driver_right.set_all_modes(trossen_arm.Mode.external_effort)
            driver_right.set_all_external_efforts(zero_efforts, 0.0, False)
            
            print("✓ Leaders are now FREE to move - start teleoperation!")
            
        except ImportError:
            print("❌ ERROR: trossen_arm module not found!")
            print("   Install it or run without --leader_*_ip flags for random actions.")
            return
        except Exception as e:
            print(f"❌ ERROR connecting to leaders: {e}")
            return
    
    # Create environment with recorder mode
    env = make_cube_stacking_env(
        fake_env=False,
        image_obs=True,
        onscreen_render=True,
        recorder_mode=True,
        data_save_path=args.save_dir,
        data_save_every_n=args.save_every_n,
        max_episode_length=args.max_episode_length,
        arm_type=args.arm_type,
        control_mode=args.control_mode,
    )
    
    # Flag for move_to_home request (to avoid event loop issues in matplotlib callback)
    move_home_requested = [False]  # Use list for mutable reference in closure
    reset_env_requested = [False]  # Flag for reset request
    
    def request_move_home():
        """Set flag to request move to home (called from GUI button)."""
        move_home_requested[0] = True
        print("[Home] Request queued - will execute on next loop iteration")
    
    def execute_move_to_home():
        """Actually move robots to home (called from main loop, not from callback)."""
        import trossen_arm
        import time
        
        GRIPPER_OPEN = 0.04  # meters
        HOME_ARM = np.zeros(6)  # All arm joints to 0
        
        home_left = np.concatenate([HOME_ARM, [GRIPPER_OPEN]])
        home_right = np.concatenate([HOME_ARM, [GRIPPER_OPEN]])
        
        try:
            # Switch leaders to position mode and move to home
            driver_left.set_all_modes(trossen_arm.Mode.position)
            driver_right.set_all_modes(trossen_arm.Mode.position)
            driver_left.set_all_positions(home_left)
            driver_right.set_all_positions(home_right)
            
            # Short wait for real robots to start moving
            time.sleep(0.5)
            
            # Move sim to home - use appropriate action based on control mode
            if args.control_mode == "joint":
                # Joint mode: 14D action
                home_action = np.concatenate([home_left, home_right])
            else:
                # EE mode: 16D action - use home EE positions
                # These are the home positions in world frame
                home_ee_left = np.array([-0.204, -0.019, 0.188, 1, 0, 0, 0, GRIPPER_OPEN])
                home_ee_right = np.array([0.206, -0.019, 0.188, 1, 0, 0, 0, GRIPPER_OPEN])
                home_action = np.concatenate([home_ee_left, home_ee_right])
            
            for _ in range(25):  # ~0.5 seconds at 50Hz
                env.step(home_action)
            
            # Set leaders back to external effort (free to move)
            zero_efforts = np.zeros(7)
            driver_left.set_all_modes(trossen_arm.Mode.external_effort)
            driver_left.set_all_external_efforts(zero_efforts, 0.0, False)
            driver_right.set_all_modes(trossen_arm.Mode.external_effort)
            driver_right.set_all_external_efforts(zero_efforts, 0.0, False)
            
            print("✓ Home done - leaders FREE")
            
        except Exception as e:
            print(f"❌ Error moving to home: {e}")
    
    # Set the home callback on the data collector (just sets the flag)
    if hasattr(env, 'data_collector') and env.data_collector is not None:
        if teleop_enabled:
            env.data_collector.set_home_callback(request_move_home)
        
        # Reset callback - works for both teleop and random mode
        def request_reset():
            reset_env_requested[0] = True
            print("[Reset] Request queued - will execute on next loop iteration")
        env.data_collector.set_reset_callback(request_reset)
    
    print("\n[INFO] Environment created. GUI should be visible.")
    print("[INFO] Use buttons to Start/Stop recording.")
    print("[INFO] Click 'Reset Env' to reset simulation (cube position, robot pose).")
    if teleop_enabled:
        print("[INFO] Click 'Home' to move real+sim robots to home position.")
    print("[INFO] Press Ctrl+C to exit.\n")
    
    # Main loop
    try:
        obs, info = env.reset(seed=args.seed)
        episode_done_printed = False  # Track if we already printed done message
        
        # If teleop enabled, sync sim robots to leader positions first
        if teleop_enabled:
            print("🔄 Syncing sim to leader positions...")
            try:
                if args.control_mode == "joint":
                    # Joint mode: use joint positions directly
                    left_state = np.array(driver_left.get_all_positions())
                    right_state = np.array(driver_right.get_all_positions())
                    initial_action = np.concatenate([left_state, right_state])  # 14D
                else:
                    # EE mode: get cartesian and transform
                    from trossen_arm_mujoco.ee_transforms import leader_cartesian_to_sim_action
                    left_cart = np.array(driver_left.get_cartesian_positions())
                    left_grip = driver_left.get_gripper_position()
                    right_cart = np.array(driver_right.get_cartesian_positions())
                    right_grip = driver_right.get_gripper_position()
                    initial_action = leader_cartesian_to_sim_action(
                        left_cart, left_grip, right_cart, right_grip
                    )  # 16D
                
                # Quick sync
                for _ in range(20):  # ~0.4 seconds at 50Hz
                    obs, _, _, _, _ = env.step(initial_action)
                
                print(f"✓ Synced ({args.control_mode} mode)")
            except Exception as e:
                print(f"⚠️  Sync failed: {e}")
        
        while True:
            # Check for reset request (from GUI button)
            if reset_env_requested[0]:
                reset_env_requested[0] = False
                episode_done_printed = False  # Reset the flag
                print("[Reset] Executing environment reset...")
                obs, info = env.reset(seed=args.seed)
                
                # If teleop, sync to leader positions after reset
                if teleop_enabled:
                    try:
                        if args.control_mode == "joint":
                            left_state = np.array(driver_left.get_all_positions())
                            right_state = np.array(driver_right.get_all_positions())
                            initial_action = np.concatenate([left_state, right_state])
                        else:
                            from trossen_arm_mujoco.ee_transforms import leader_cartesian_to_sim_action
                            left_cart = np.array(driver_left.get_cartesian_positions())
                            left_grip = driver_left.get_gripper_position()
                            right_cart = np.array(driver_right.get_cartesian_positions())
                            right_grip = driver_right.get_gripper_position()
                            initial_action = leader_cartesian_to_sim_action(
                                left_cart, left_grip, right_cart, right_grip
                            )
                        for _ in range(15):  # Quick sync
                            obs, _, _, _, _ = env.step(initial_action)
                        print("[Reset] Done - synced")
                    except Exception as e:
                        print(f"[Reset] Done (sync failed: {e})")
                else:
                    print("[Reset] Done")
                continue
            
            # Check for move_home request (from GUI button)
            if teleop_enabled and move_home_requested[0]:
                move_home_requested[0] = False
                execute_move_to_home()
                continue  # Skip this iteration, robot state may have changed
            
            if teleop_enabled:
                # Get leader positions and use as action
                try:
                    if args.control_mode == "joint":
                        # Joint mode: get joint positions (7D each: 6 arm + 1 gripper)
                        left_state = np.array(driver_left.get_all_positions())
                        right_state = np.array(driver_right.get_all_positions())
                        action = np.concatenate([left_state, right_state])  # 14D
                    else:
                        # EE mode: get cartesian positions and transform to world frame
                        from trossen_arm_mujoco.ee_transforms import leader_cartesian_to_sim_action
                        left_cart = np.array(driver_left.get_cartesian_positions())
                        left_grip = driver_left.get_gripper_position()
                        right_cart = np.array(driver_right.get_cartesian_positions())
                        right_grip = driver_right.get_gripper_position()
                        action = leader_cartesian_to_sim_action(
                            left_cart, left_grip, right_cart, right_grip
                        )  # 16D
                    
                except Exception as e:
                    print(f"⚠️  Error reading leader positions: {e}")
                    break
            else:
                # Random actions for testing
                action = env.action_space.sample()
            
            obs, reward, done, truncated, info = env.step(action)
            
            if done and not episode_done_printed:
                print("[INFO] Episode done! Click 'Reset Env' to start a new episode.")
                episode_done_printed = True
                
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user. Closing...")
    finally:
        # Cleanup
        if driver_left is not None:
            driver_left = None
            print("✓ Left leader disconnected")
        if driver_right is not None:
            driver_right = None
            print("✓ Right leader disconnected")
        env.close()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()
