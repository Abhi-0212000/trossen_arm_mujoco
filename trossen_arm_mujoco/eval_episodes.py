"""
Episode Evaluation/Playback Script

Two modes:
1. images: Play back recorded camera images from HDF5 (no sim needed)
2. replay: Apply recorded actions to simulation and compare

Usage:
    # Play recorded images only
    python eval_episodes.py --data_dir ./sim_recordings_joint --episode 0 --mode images
    
    # Replay actions in simulation
    python eval_episodes.py --data_dir ./sim_recordings_joint --episode 0 --mode replay
    
    # Play all episodes in sequence
    python eval_episodes.py --data_dir ./sim_recordings_joint --episode all --mode images
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import argparse
import time


class EpisodePlayer:
    """
    Interactive episode player with playback controls.
    """
    def __init__(self, data_dir: str, episode_idx: int, cam_list: list = None):
        self.data_dir = data_dir
        self.episode_idx = episode_idx
        self.cam_list = cam_list
        
        # Load episode data
        self.filepath = os.path.join(data_dir, f"episode_{episode_idx}.hdf5")
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Episode file not found: {self.filepath}")
        
        self._load_data()
        
        # Playback state
        self.current_frame = 0
        self.is_playing = False
        self.playback_speed = 1.0  # 1.0 = real-time (50 Hz)
        
        # Setup visualization
        self._setup_visualizer()
    
    def _load_data(self):
        """Load episode data from HDF5 file."""
        print(f"Loading {self.filepath}...")
        
        with h5py.File(self.filepath, "r") as f:
            # Get attributes
            self.env_name = f.attrs.get("env_name", "unknown")
            self.is_sim = f.attrs.get("sim", True)
            
            # Load observations
            obs_grp = f["observations"]
            
            # Load images
            self.images = {}
            if "images" in obs_grp:
                img_grp = obs_grp["images"]
                for cam_name in img_grp.keys():
                    self.images[cam_name] = img_grp[cam_name][:]
            
            # Auto-detect camera list if not provided
            if self.cam_list is None:
                self.cam_list = list(self.images.keys())
            
            # Load state data
            self.qpos = obs_grp["qpos"][:] if "qpos" in obs_grp else None
            self.qvel = obs_grp["qvel"][:] if "qvel" in obs_grp else None
            
            # Load EE data if present
            self.mocap_pose_left = obs_grp["mocap_pose_left"][:] if "mocap_pose_left" in obs_grp else None
            self.mocap_pose_right = obs_grp["mocap_pose_right"][:] if "mocap_pose_right" in obs_grp else None
            self.robot0_eef_pos = obs_grp["robot0_eef_pos"][:] if "robot0_eef_pos" in obs_grp else None
            self.robot0_eef_quat = obs_grp["robot0_eef_quat"][:] if "robot0_eef_quat" in obs_grp else None
            
            # Load action, reward, done
            self.actions = f["action"][:]
            self.rewards = f["reward"][:]
            self.dones = f["done"][:]
        
        self.num_frames = len(self.actions)
        self.control_mode = "ee" if self.actions.shape[1] == 16 else "joint"
        
        print(f"  Env: {self.env_name}")
        print(f"  Frames: {self.num_frames}")
        print(f"  Cameras: {self.cam_list}")
        print(f"  Control mode: {self.control_mode} ({self.actions.shape[1]}D)")
        print(f"  Action shape: {self.actions.shape}")
        if self.qpos is not None:
            print(f"  qpos shape: {self.qpos.shape}")
    
    def _setup_visualizer(self):
        """Setup matplotlib figure with playback controls."""
        num_cameras = len(self.cam_list)
        if num_cameras == 0:
            raise ValueError("No camera images found in episode!")
        
        cols = 2 if num_cameras == 4 else min(3, num_cameras)
        rows = (num_cameras + cols - 1) // cols
        
        self.fig, self.axs = plt.subplots(rows, cols, figsize=(12, 10))
        self.fig.subplots_adjust(bottom=0.25)
        
        self.axs = np.atleast_1d(self.axs).flatten()
        
        # Initialize camera images
        self.plt_imgs = []
        for i, cam in enumerate(self.cam_list):
            if i < len(self.axs):
                self.axs[i].set_title(cam)
                self.axs[i].axis("off")
                if cam in self.images and len(self.images[cam]) > 0:
                    img = self.axs[i].imshow(self.images[cam][0])
                else:
                    img = self.axs[i].imshow(np.zeros((480, 640, 3), dtype=np.uint8))
                self.plt_imgs.append(img)
        
        # Hide unused axes
        for i in range(num_cameras, len(self.axs)):
            self.axs[i].axis("off")
        
        # Status text
        self.text_ax = self.fig.add_axes([0.1, 0.18, 0.8, 0.04])
        self.text_ax.axis("off")
        self.status_text = self.text_ax.text(0.5, 0.5, self._get_status(), 
                                              ha="center", va="center", fontsize=11)
        
        # Frame slider
        self.slider_ax = self.fig.add_axes([0.15, 0.12, 0.7, 0.03])
        self.frame_slider = Slider(self.slider_ax, 'Frame', 0, self.num_frames - 1, 
                                   valinit=0, valstep=1)
        self.frame_slider.on_changed(self._on_slider_change)
        
        # Control buttons
        self.btn_play_ax = self.fig.add_axes([0.1, 0.05, 0.15, 0.05])
        self.btn_play = Button(self.btn_play_ax, 'Play', color='lightgreen')
        self.btn_play.on_clicked(self._toggle_play)
        
        self.btn_prev_ax = self.fig.add_axes([0.28, 0.05, 0.1, 0.05])
        self.btn_prev = Button(self.btn_prev_ax, '< Prev')
        self.btn_prev.on_clicked(self._prev_frame)
        
        self.btn_next_ax = self.fig.add_axes([0.40, 0.05, 0.1, 0.05])
        self.btn_next = Button(self.btn_next_ax, 'Next >')
        self.btn_next.on_clicked(self._next_frame)
        
        self.btn_speed_down_ax = self.fig.add_axes([0.55, 0.05, 0.1, 0.05])
        self.btn_speed_down = Button(self.btn_speed_down_ax, 'Slower')
        self.btn_speed_down.on_clicked(self._speed_down)
        
        self.btn_speed_up_ax = self.fig.add_axes([0.67, 0.05, 0.1, 0.05])
        self.btn_speed_up = Button(self.btn_speed_up_ax, 'Faster')
        self.btn_speed_up.on_clicked(self._speed_up)
        
        self.btn_reset_ax = self.fig.add_axes([0.80, 0.05, 0.1, 0.05])
        self.btn_reset = Button(self.btn_reset_ax, 'Reset', color='lightyellow')
        self.btn_reset.on_clicked(self._reset_playback)
        
        plt.ion()
    
    def _get_status(self):
        """Get status text."""
        play_status = "PLAYING" if self.is_playing else "PAUSED"
        reward = self.rewards[self.current_frame] if self.current_frame < len(self.rewards) else 0
        done = self.dones[self.current_frame] if self.current_frame < len(self.dones) else False
        
        status = f"Episode {self.episode_idx} | {play_status} | Frame: {self.current_frame}/{self.num_frames-1} | Speed: {self.playback_speed:.1f}x | R: {reward:.2f}"
        if done:
            status += " | DONE"
        return status
    
    def _update_display(self):
        """Update the display for current frame."""
        # Update images
        for i, cam in enumerate(self.cam_list):
            if cam in self.images and self.current_frame < len(self.images[cam]):
                self.plt_imgs[i].set_data(self.images[cam][self.current_frame])
        
        # Update slider (without triggering callback)
        self.frame_slider.eventson = False
        self.frame_slider.set_val(self.current_frame)
        self.frame_slider.eventson = True
        
        # Update status
        self.status_text.set_text(self._get_status())
        
        # Update play button text
        self.btn_play.label.set_text("Pause" if self.is_playing else "Play")
        self.btn_play.color = 'salmon' if self.is_playing else 'lightgreen'
        
        self.fig.canvas.draw_idle()
    
    def _on_slider_change(self, val):
        """Handle slider change."""
        self.current_frame = int(val)
        self._update_display()
    
    def _toggle_play(self, event):
        """Toggle play/pause."""
        self.is_playing = not self.is_playing
        self._update_display()
    
    def _prev_frame(self, event):
        """Go to previous frame."""
        self.current_frame = max(0, self.current_frame - 1)
        self._update_display()
    
    def _next_frame(self, event):
        """Go to next frame."""
        self.current_frame = min(self.num_frames - 1, self.current_frame + 1)
        self._update_display()
    
    def _speed_down(self, event):
        """Decrease playback speed."""
        self.playback_speed = max(0.1, self.playback_speed / 2)
        self._update_display()
    
    def _speed_up(self, event):
        """Increase playback speed."""
        self.playback_speed = min(10.0, self.playback_speed * 2)
        self._update_display()
    
    def _reset_playback(self, event):
        """Reset to beginning."""
        self.current_frame = 0
        self.is_playing = False
        self._update_display()
    
    def run(self):
        """Run the playback loop."""
        print("\n[Controls]")
        print("  Play/Pause: Click 'Play' button")
        print("  Navigate: Use slider or Prev/Next buttons")
        print("  Speed: Use Slower/Faster buttons")
        print("  Close: Close the window or Ctrl+C")
        print()
        
        self._update_display()
        
        try:
            while plt.fignum_exists(self.fig.number):
                if self.is_playing:
                    self.current_frame += 1
                    if self.current_frame >= self.num_frames:
                        self.current_frame = 0  # Loop
                    self._update_display()
                    
                    # Control playback speed (base rate is 50Hz)
                    delay = (1.0 / 50.0) / self.playback_speed
                    plt.pause(delay)
                else:
                    plt.pause(0.05)
        except KeyboardInterrupt:
            print("\n[INFO] Playback interrupted")
        
        plt.close(self.fig)


class EpisodeReplay:
    """
    Replay recorded actions in the simulation environment.
    """
    def __init__(self, data_dir: str, episode_idx: int, arm_type: str = "widowx"):
        self.data_dir = data_dir
        self.episode_idx = episode_idx
        self.arm_type = arm_type
        
        # Load episode data
        self.filepath = os.path.join(data_dir, f"episode_{episode_idx}.hdf5")
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Episode file not found: {self.filepath}")
        
        self._load_data()
        self._setup_env()
    
    def _load_data(self):
        """Load episode data."""
        print(f"Loading {self.filepath}...")
        
        with h5py.File(self.filepath, "r") as f:
            self.env_name = f.attrs.get("env_name", "unknown")
            
            # Load actions
            self.actions = f["action"][:]
            self.rewards = f["reward"][:]
            self.dones = f["done"][:]
            
            # Load recorded observations for comparison
            obs_grp = f["observations"]
            self.recorded_qpos = obs_grp["qpos"][:] if "qpos" in obs_grp else None
            
            # Load cube pose for perfect replay
            self.recorded_cube_pose = obs_grp["cube_pose"][:] if "cube_pose" in obs_grp else None
            
            # Load images
            self.recorded_images = {}
            if "images" in obs_grp:
                for cam in obs_grp["images"].keys():
                    self.recorded_images[cam] = obs_grp["images"][cam][:]
        
        self.num_frames = len(self.actions)
        self.control_mode = "ee" if self.actions.shape[1] == 16 else "joint"
        
        print(f"  Frames: {self.num_frames}")
        print(f"  Control mode: {self.control_mode}")
        if self.recorded_cube_pose is not None:
            print(f"  Cube pose: {self.recorded_cube_pose[0][:3]} (initial position)")
    
    def _setup_env(self):
        """Create simulation environment."""
        from trossen_arm_mujoco.gym_envs.make_env import make_cube_stacking_env
        
        print(f"\nCreating environment (control_mode={self.control_mode})...")
        
        self.env = make_cube_stacking_env(
            fake_env=False,
            image_obs=True,
            onscreen_render=True,
            recorder_mode=False,
            arm_type=self.arm_type,
            control_mode=self.control_mode,
        )
        
        self.cam_list = list(self.recorded_images.keys()) if self.recorded_images else ["cam_high"]
    
    def _reset_with_cube_pose(self):
        """Reset environment and set cube to recorded initial position."""
        obs, info = self.env.reset()
        
        # If we have recorded cube pose, set it to match the recording
        if self.recorded_cube_pose is not None and len(self.recorded_cube_pose) > 0:
            initial_cube_pose = self.recorded_cube_pose[0]  # First frame's cube pose
            
            # Access the underlying dm_control environment to set cube pose
            try:
                # SERLGymWrapper stores dm_control env as self.env (not self._env)
                # The wrapper itself is self.env, so we need self.env.unwrapped.env
                wrapper = self.env.unwrapped  # Get SERLGymWrapper
                dm_env = wrapper.env  # Get dm_control Environment
                physics = dm_env.physics
                
                # Set cube pose (qpos[16:23] = [x, y, z, qw, qx, qy, qz])
                physics.data.qpos[16:23] = initial_cube_pose
                physics.forward()  # Update physics state
                
                print(f"  Cube set to: [{initial_cube_pose[0]:.3f}, {initial_cube_pose[1]:.3f}, {initial_cube_pose[2]:.3f}]")
                
            except Exception as e:
                print(f"  Warning: Could not set cube pose: {e}")
        
        return obs, info
    
    def _setup_comparison_view(self):
        """Setup side-by-side comparison view."""
        # Create figure: top row = recorded, bottom row = replayed
        num_cameras = min(4, len(self.cam_list))
        
        self.fig, self.axs = plt.subplots(2, num_cameras, figsize=(14, 8))
        self.fig.subplots_adjust(bottom=0.15)
        
        if num_cameras == 1:
            self.axs = self.axs.reshape(2, 1)
        
        # Initialize images
        self.recorded_imgs = []
        self.replayed_imgs = []
        
        for i, cam in enumerate(self.cam_list[:num_cameras]):
            # Recorded (top row)
            self.axs[0, i].set_title(f"Recorded: {cam}")
            self.axs[0, i].axis("off")
            if cam in self.recorded_images and len(self.recorded_images[cam]) > 0:
                img = self.axs[0, i].imshow(self.recorded_images[cam][0])
            else:
                img = self.axs[0, i].imshow(np.zeros((480, 640, 3), dtype=np.uint8))
            self.recorded_imgs.append(img)
            
            # Replayed (bottom row)
            self.axs[1, i].set_title(f"Replayed: {cam}")
            self.axs[1, i].axis("off")
            img = self.axs[1, i].imshow(np.zeros((480, 640, 3), dtype=np.uint8))
            self.replayed_imgs.append(img)
        
        # Status text
        self.text_ax = self.fig.add_axes([0.1, 0.08, 0.8, 0.04])
        self.text_ax.axis("off")
        self.status_text = self.text_ax.text(0.5, 0.5, "Ready - Click 'Start' to begin", 
                                              ha="center", va="center", fontsize=11)
        
        # Single control button (changes state)
        self.btn_ctrl_ax = self.fig.add_axes([0.35, 0.02, 0.15, 0.04])
        self.btn_ctrl = Button(self.btn_ctrl_ax, 'Start', color='lightgreen')
        self.btn_ctrl.on_clicked(self._on_ctrl_click)
        
        # Restart button
        self.btn_restart_ax = self.fig.add_axes([0.52, 0.02, 0.15, 0.04])
        self.btn_restart = Button(self.btn_restart_ax, 'Restart', color='lightyellow')
        self.btn_restart.on_clicked(self._on_restart_click)
        
        # State: 'ready', 'playing', 'paused', 'finished'
        self.state = 'ready'
        self.current_frame = 0
        self.restart_requested = False
        
        plt.ion()
    
    def _update_button(self):
        """Update button text/color based on state."""
        if self.state == 'ready':
            self.btn_ctrl.label.set_text('Start')
            self.btn_ctrl.color = 'lightgreen'
        elif self.state == 'playing':
            self.btn_ctrl.label.set_text('Pause')
            self.btn_ctrl.color = 'salmon'
        elif self.state == 'paused':
            self.btn_ctrl.label.set_text('Continue')
            self.btn_ctrl.color = 'lightgreen'
        elif self.state == 'finished':
            self.btn_ctrl.label.set_text('Replay')
            self.btn_ctrl.color = 'lightblue'
        self.fig.canvas.draw_idle()
    
    def _on_ctrl_click(self, event):
        """Handle control button click."""
        if self.state in ('ready', 'paused'):
            self.state = 'playing'
        elif self.state == 'playing':
            self.state = 'paused'
        elif self.state == 'finished':
            self.restart_requested = True
        self._update_button()
    
    def _on_restart_click(self, event):
        """Handle restart button click."""
        self.restart_requested = True
    
    def run(self):
        """Run the replay comparison."""
        self._setup_comparison_view()
        
        print("\n[Replay Mode]")
        print("  Start/Pause/Continue: Control playback")
        print("  Restart: Reset env and replay from beginning")
        print("  Top row: Recorded images | Bottom row: Live replay")
        print()
        
        # Reset environment with recorded cube pose
        obs, _ = self._reset_with_cube_pose()
        
        try:
            while plt.fignum_exists(self.fig.number):
                # Handle restart request
                if self.restart_requested:
                    self.restart_requested = False
                    self.current_frame = 0
                    self.state = 'ready'
                    obs, _ = self._reset_with_cube_pose()
                    self.status_text.set_text("Reset - Click 'Start' to begin")
                    self._update_button()
                    plt.pause(0.05)
                    continue
                
                # Playing state
                if self.state == 'playing' and self.current_frame < self.num_frames:
                    action = self.actions[self.current_frame]
                    obs, reward, done, truncated, info = self.env.step(action)
                    
                    # Get live images
                    live_images = obs.get('dm_obs', {}).get('images', {})
                    if not live_images:
                        live_images = getattr(self.env, '_images_full_res', {})
                    
                    # Update display
                    for i, cam in enumerate(self.cam_list[:len(self.recorded_imgs)]):
                        if cam in self.recorded_images and self.current_frame < len(self.recorded_images[cam]):
                            self.recorded_imgs[i].set_data(self.recorded_images[cam][self.current_frame])
                        if cam in live_images:
                            self.replayed_imgs[i].set_data(live_images[cam])
                    
                    # Status with qpos diff
                    qpos_diff = ""
                    if self.recorded_qpos is not None and 'dm_obs' in obs:
                        live_qpos = obs['dm_obs'].get('qpos', None)
                        if live_qpos is not None and self.current_frame < len(self.recorded_qpos):
                            diff = np.abs(live_qpos - self.recorded_qpos[self.current_frame]).mean()
                            qpos_diff = f" | qpos MAE: {diff:.4f}"
                    
                    self.status_text.set_text(
                        f"Frame: {self.current_frame}/{self.num_frames-1} | "
                        f"R: {reward:.2f} (rec: {self.rewards[self.current_frame]:.2f}){qpos_diff}"
                    )
                    
                    self.current_frame += 1
                    
                    if self.current_frame >= self.num_frames:
                        self.state = 'finished'
                        self.status_text.set_text(f"Done! {self.num_frames} frames - Click 'Replay' or 'Restart'")
                        self._update_button()
                    
                    self.fig.canvas.draw_idle()
                    plt.pause(0.02)
                else:
                    plt.pause(0.05)
                    
        except KeyboardInterrupt:
            print("\n[INFO] Replay interrupted")
        finally:
            self.env.close()
            plt.close(self.fig)


def list_episodes(data_dir: str):
    """List all episodes in directory."""
    if not os.path.exists(data_dir):
        print(f"Directory not found: {data_dir}")
        return []
    
    episodes = []
    for f in sorted(os.listdir(data_dir)):
        if f.startswith("episode_") and f.endswith(".hdf5"):
            idx = int(f.split("_")[1].split(".")[0])
            filepath = os.path.join(data_dir, f)
            
            with h5py.File(filepath, "r") as hf:
                num_frames = len(hf["action"])
                action_dim = hf["action"].shape[1]
                
            episodes.append({
                "idx": idx,
                "file": f,
                "frames": num_frames,
                "action_dim": action_dim,
                "mode": "ee" if action_dim == 16 else "joint"
            })
    
    return episodes


def main():
    parser = argparse.ArgumentParser(description="Episode Evaluation/Playback")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing HDF5 episode files")
    parser.add_argument("--episode", type=str, default="0",
                        help="Episode index to play (or 'all' or 'list')")
    parser.add_argument("--mode", type=str, default="images",
                        choices=["images", "replay", "list"],
                        help="Mode: 'images' (playback), 'replay' (sim), 'list' (show episodes)")
    parser.add_argument("--arm_type", type=str, default="widowx",
                        choices=["widowx", "viperx"],
                        help="Robot arm type (for replay mode)")
    
    args = parser.parse_args()
    
    # List mode
    if args.mode == "list" or args.episode == "list":
        print(f"\nEpisodes in {args.data_dir}:")
        print("-" * 60)
        episodes = list_episodes(args.data_dir)
        if not episodes:
            print("  No episodes found")
        else:
            for ep in episodes:
                print(f"  Episode {ep['idx']:3d}: {ep['frames']:4d} frames | {ep['mode']} ({ep['action_dim']}D)")
        print("-" * 60)
        print(f"Total: {len(episodes)} episodes")
        return
    
    # Get episode index(es)
    if args.episode == "all":
        episodes = list_episodes(args.data_dir)
        episode_indices = [ep["idx"] for ep in episodes]
    else:
        episode_indices = [int(args.episode)]
    
    if not episode_indices:
        print("No episodes to play")
        return
    
    # Play episodes
    for ep_idx in episode_indices:
        print(f"\n{'='*60}")
        print(f"Episode {ep_idx}")
        print('='*60)
        
        try:
            if args.mode == "images":
                player = EpisodePlayer(args.data_dir, ep_idx)
                player.run()
            elif args.mode == "replay":
                replayer = EpisodeReplay(args.data_dir, ep_idx, args.arm_type)
                replayer.run()
        except FileNotFoundError as e:
            print(f"Error: {e}")
        except KeyboardInterrupt:
            print("\n[INFO] Stopped by user")
            break
    
    print("\n[INFO] Done")


if __name__ == "__main__":
    main()
