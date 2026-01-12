"""
Interactive episode cleanup and merge tool for SERL pickle datasets.

Modes:
1. Cleanup: Iterates through a folder of .pkl files, displays them, and deletes specific files.
2. Merge: Combines all .pkl files in a folder into one consolidated dataset.
"""

import pickle
import os
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np

# Import utils (ensure these are in your python path)
try:
    from trossen_arm_mujoco.dataset_utils.delta_action_ds.cleanup_utils import load_transitions, get_episode_boundaries, pretty_print_obs
except ImportError:
    # Fallback for standalone usage if utils aren't installed
    def load_transitions(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def get_episode_boundaries(transitions):
        # Basic boundary detection based on 'done' flag or assumption of 1 episode per file
        boundaries = []
        start = 0
        for i, t in enumerate(transitions):
            if t.get('done', False) or t.get('is_terminal', False):
                boundaries.append((start, i + 1))
                start = i + 1
        if start < len(transitions):
            boundaries.append((start, len(transitions)))
        return boundaries

def display_episode_images(episode_transitions: list, episode_idx: int, total_episodes: int, filename: str):
    """
    Play through all camera images from an episode frame by frame.
    """
    if not episode_transitions:
        print(f"Empty episode in {filename}")
        return

    # Get camera names from first transition
    first_obs = episode_transitions[0]['observations']
    cam_names = [k for k in first_obs.keys() if k != 'state']
    
    if not cam_names:
        print(f"No camera images found in {filename}")
        return
    
    n_steps = len(episode_transitions)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    print(f"\n{'='*60}")
    print(f"File: {filename}")
    print(f"Episode: {episode_idx + 1}/{total_episodes} inside file")
    print(f"Steps: {n_steps}")
    print(f"Cameras: {cam_names}")
    print(f"Close window to continue...")
    print(f"{'='*60}")
    
    # Initialize image plots
    img_plots = []
    for cam_idx, cam_name in enumerate(cam_names[:4]):  # Max 4 cameras
        if cam_idx < len(axes):
            img = first_obs[cam_name]
            # Handle (1, H, W, 3) vs (H, W, 3)
            if len(img.shape) == 4: img = img[0]
            
            im = axes[cam_idx].imshow(img)
            axes[cam_idx].set_title(f'{cam_name} (Step 0)', fontsize=10)
            axes[cam_idx].axis('off')
            img_plots.append((im, cam_name, cam_idx))
    
    # Hide unused subplots
    for idx in range(len(cam_names), len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(f'{filename} | Ep {episode_idx+1}/{total_episodes}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.ion()
    plt.show()
    
    # Play loop
    for step_idx in range(n_steps):
        if not plt.fignum_exists(fig.number): break
        
        obs = episode_transitions[step_idx]['observations']
        for im, cam_name, cam_idx in img_plots:
            img = obs[cam_name]
            if len(img.shape) == 4: img = img[0]
            im.set_data(img)
            axes[cam_idx].set_title(f'{cam_name} (Step {step_idx})', fontsize=10)
        
        plt.draw()
        plt.pause(0.001)
    
    # Wait for close
    while plt.fignum_exists(fig.number):
        plt.pause(0.1)
        plt.close(fig)

def cleanup_folder(folder_path: str):
    """
    Iterate over pickle files in a folder, review content, and delete bad files.
    """
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Error: Folder not found: {folder}")
        return

    # Get all .pkl files sorted
    files = sorted(list(folder.glob("*.pkl")))
    if not files:
        print("No .pkl files found in folder.")
        return

    print(f"Found {len(files)} pickle files in {folder}")
    files_to_delete = []

    for i, file_path in enumerate(files):
        print(f"\n[{i+1}/{len(files)}] Loading {file_path.name}...")
        
        try:
            transitions = load_transitions(file_path)
            print(f"  Loaded {len(transitions)} transitions and type is {type(transitions)}.")
            # print("each transition type is ", type(transitions[0]) if len(transitions) > 0 else "N/A")
            # pretty_print_obs(transitions[0], indent=4)
            boundaries = get_episode_boundaries(transitions)
        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")
            print("Marking for deletion (corrupt)?")
            if input("Delete corrupt file? (y/n): ").lower() == 'y':
                files_to_delete.append(file_path)
            continue

        if not boundaries:
            print("  No episodes found in file.")
        
        # Play all episodes in this file
        for ep_idx, (start, end) in enumerate(boundaries):
            display_episode_images(transitions[start:end], ep_idx, len(boundaries), file_path.name)
        
        # Ask to delete THE FILE
        while True:
            choice = input(f"DELETE file '{file_path.name}'? (y/n/q): ").strip().lower()
            if choice in ['y', 'yes']:
                files_to_delete.append(file_path)
                print(f"  X Marked '{file_path.name}' for deletion.")
                break
            elif choice in ['n', 'no']:
                print(f"  - Kept '{file_path.name}'.")
                break
            elif choice in ['q', 'quit']:
                print("Quitting review...")
                return  # Exit function, do not process remaining files
            
    # Final Execution
    if not files_to_delete:
        print("\nNo files marked for deletion.")
        return

    print(f"\n{'='*60}")
    print(f"SUMMARY: {len(files_to_delete)} files marked for deletion.")
    print(f"{'='*60}")
    for f in files_to_delete:
        print(f"  - {f.name}")
    
    confirm = input("\nPermanently delete these files? (y/n): ").strip().lower()
    if confirm in ['y', 'yes']:
        for f in files_to_delete:
            try:
                os.remove(f)
                print(f"Deleted {f.name}")
            except OSError as e:
                print(f"Error deleting {f.name}: {e}")
        print("Cleanup complete.")
    else:
        print("Deletion cancelled.")

def merge_folder(folder_path: str, output_file: str):
    """
    Load all .pkl files in a folder and save them as one giant list.
    """
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Error: Folder not found: {folder}")
        return

    files = sorted(list(folder.glob("*.pkl")))
    if not files:
        print("No .pkl files found.")
        return

    all_transitions = []
    total_files = 0

    print(f"Merging {len(files)} files from {folder}...")

    for file_path in files:
        try:
            # Skip the output file if it's in the same directory to avoid recursion loops
            if output_file and file_path.name == Path(output_file).name:
                continue
                
            data = load_transitions(file_path)
            if isinstance(data, list):
                all_transitions.extend(data)
                total_files += 1
                print(f"  + Loaded {file_path.name} ({len(data)} steps)")
            else:
                print(f"  ! Skipped {file_path.name} (not a list)")
        except Exception as e:
            print(f"  ! Error loading {file_path.name}: {e}")

    if not all_transitions:
        print("No transitions found to merge.")
        return

    print(f"\nTotal transitions: {len(all_transitions)}")
    
    out_path = Path(output_file) if output_file else folder / "merged_dataset.pkl"
    print(f"Saving to {out_path}...")
    
    with open(out_path, 'wb') as f:
        pickle.dump(all_transitions, f)
    
    print("✅ Merge complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool for Cleaning and Merging SERL Pickle Datasets")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing .pkl files")
    parser.add_argument("--mode", type=str, choices=['cleanup', 'merge'], required=True, help="Mode: 'cleanup' to review/delete, 'merge' to combine files")
    parser.add_argument("--output", type=str, help="Output path for the merged file (only used in merge mode)")

    args = parser.parse_args()

    if args.mode == 'cleanup':
        cleanup_folder(args.folder_path)
    elif args.mode == 'merge':
        merge_folder(args.folder_path, args.output)