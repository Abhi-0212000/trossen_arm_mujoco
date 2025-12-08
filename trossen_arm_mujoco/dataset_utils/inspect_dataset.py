#!/usr/bin/env python3
"""
Dataset Inspector - Analyze HDF5 episode files from sim data collection.

Usage:
    python -m trossen_arm_mujoco.dataset_utils.inspect_dataset /path/to/sim_recordings_ee
    
    # Or with options:
    python -m trossen_arm_mujoco.dataset_utils.inspect_dataset /path/to/recordings --episode 0 --verbose
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
from typing import Optional
import os


def format_shape(shape):
    """Format shape tuple for display."""
    return f"({', '.join(str(s) for s in shape)})"


def format_dtype(dtype):
    """Format dtype for display."""
    return str(dtype)


def is_image_data(data: np.ndarray) -> bool:
    """Check if data looks like image data (3D with last dim 3 or 4, or 4D batch)."""
    if data.ndim == 3 and data.shape[-1] in [1, 3, 4]:
        return True
    if data.ndim == 4 and data.shape[-1] in [1, 3, 4]:
        return True
    # Check for image-like values (0-255 range)
    if data.dtype == np.uint8:
        return True
    return False


def print_array_stats(name: str, data: np.ndarray, indent: int = 0, verbose: bool = False):
    """Print statistics for a numpy array."""
    prefix = "  " * indent
    shape_str = format_shape(data.shape)
    dtype_str = format_dtype(data.dtype)
    
    if is_image_data(data):
        # Image data - print pixel statistics
        print(f"{prefix}{name}:")
        print(f"{prefix}  Shape: {shape_str}")
        print(f"{prefix}  Dtype: {dtype_str}")
        print(f"{prefix}  Pixel Range: [{data.min()}, {data.max()}]")
        print(f"{prefix}  Mean: {data.mean():.2f}")
        if data.ndim == 4:
            print(f"{prefix}  Num Frames: {data.shape[0]}")
            print(f"{prefix}  Image Size: {data.shape[1]}x{data.shape[2]}x{data.shape[3]}")
        elif data.ndim == 3:
            print(f"{prefix}  Image Size: {data.shape[0]}x{data.shape[1]}x{data.shape[2]}")
    else:
        # Numeric data
        print(f"{prefix}{name}:")
        print(f"{prefix}  Shape: {shape_str}")
        print(f"{prefix}  Dtype: {dtype_str}")
        
        if data.size > 0:
            print(f"{prefix}  Range: [{data.min():.6f}, {data.max():.6f}]")
            print(f"{prefix}  Mean: {data.mean():.6f}")
            print(f"{prefix}  Std: {data.std():.6f}")
            
            # For small arrays, print first few values
            if verbose and data.size <= 50:
                if data.ndim == 1:
                    print(f"{prefix}  Values: {data}")
                elif data.ndim == 2 and data.shape[0] <= 5:
                    print(f"{prefix}  First rows:")
                    for i, row in enumerate(data[:5]):
                        print(f"{prefix}    [{i}]: {row}")
            elif verbose and data.ndim >= 1:
                # Print first and last few values
                print(f"{prefix}  First 3: {data.flatten()[:3]}")
                print(f"{prefix}  Last 3: {data.flatten()[-3:]}")


def inspect_group(group: h5py.Group, indent: int = 0, verbose: bool = False):
    """Recursively inspect an HDF5 group."""
    prefix = "  " * indent
    
    for key in group.keys():
        item = group[key]
        
        if isinstance(item, h5py.Group):
            print(f"{prefix}📁 {key}/")
            inspect_group(item, indent + 1, verbose)
        elif isinstance(item, h5py.Dataset):
            data = item[:]
            print_array_stats(f"📊 {key}", data, indent, verbose)
        print()  # Blank line between items


def inspect_hdf5_file(filepath: str, verbose: bool = False):
    """Inspect a single HDF5 file and print its structure."""
    print("=" * 70)
    print(f"📄 FILE: {filepath}")
    print("=" * 70)
    
    try:
        with h5py.File(filepath, 'r') as f:
            # Print top-level attributes
            if f.attrs:
                print("\n📋 ATTRIBUTES:")
                for attr_name, attr_value in f.attrs.items():
                    print(f"  {attr_name}: {attr_value}")
                print()
            
            # Print structure
            print("📂 STRUCTURE:")
            print("-" * 50)
            inspect_group(f, indent=0, verbose=verbose)
            
    except Exception as e:
        print(f"❌ Error reading file: {e}")


def get_episode_files(folder_path: str) -> list:
    """Get list of HDF5 episode files in a folder."""
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Look for .hdf5 or .h5 files
    files = list(folder.glob("*.hdf5")) + list(folder.glob("*.h5"))
    files = sorted(files)
    return files


def print_folder_summary(folder_path: str, files: list):
    """Print summary of the dataset folder."""
    print("=" * 70)
    print("📁 DATASET FOLDER SUMMARY")
    print("=" * 70)
    print(f"  Path: {folder_path}")
    print(f"  Total Episodes: {len(files)}")
    
    if files:
        # Get total size
        total_size = sum(f.stat().st_size for f in files)
        if total_size > 1e9:
            size_str = f"{total_size / 1e9:.2f} GB"
        elif total_size > 1e6:
            size_str = f"{total_size / 1e6:.2f} MB"
        else:
            size_str = f"{total_size / 1e3:.2f} KB"
        print(f"  Total Size: {size_str}")
        
        print(f"\n  Episode Files:")
        for i, f in enumerate(files[:10]):  # Show first 10
            size_mb = f.stat().st_size / 1e6
            print(f"    [{i}] {f.name} ({size_mb:.2f} MB)")
        if len(files) > 10:
            print(f"    ... and {len(files) - 10} more")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Inspect HDF5 dataset structure from sim recordings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Inspect folder and show first episode structure
    python -m trossen_arm_mujoco.dataset_utils.inspect_dataset ./sim_recordings_ee
    
    # Inspect specific episode
    python -m trossen_arm_mujoco.dataset_utils.inspect_dataset ./sim_recordings_ee --episode 2
    
    # Verbose mode (show sample values)
    python -m trossen_arm_mujoco.dataset_utils.inspect_dataset ./sim_recordings_ee -v
    
    # Inspect all episodes
    python -m trossen_arm_mujoco.dataset_utils.inspect_dataset ./sim_recordings_ee --all
        """
    )
    parser.add_argument("folder", type=str, help="Path to folder containing HDF5 episode files")
    parser.add_argument("--episode", "-e", type=int, default=0,
                        help="Episode index to inspect (default: 0 = first episode)")
    parser.add_argument("--all", "-a", action="store_true",
                        help="Inspect all episodes (not just one)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show sample values for small arrays")
    parser.add_argument("--summary-only", "-s", action="store_true",
                        help="Only show folder summary, don't inspect file structure")
    
    args = parser.parse_args()
    
    # Get episode files
    try:
        files = get_episode_files(args.folder)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return 1
    
    if not files:
        print(f"❌ No HDF5 files found in: {args.folder}")
        print("   Expected files with .hdf5 or .h5 extension")
        return 1
    
    # Print folder summary
    print_folder_summary(args.folder, files)
    
    if args.summary_only:
        return 0
    
    # Inspect episodes
    if args.all:
        for i, filepath in enumerate(files):
            print(f"\n{'#' * 70}")
            print(f"# EPISODE {i}")
            print(f"{'#' * 70}")
            inspect_hdf5_file(str(filepath), verbose=args.verbose)
    else:
        if args.episode >= len(files):
            print(f"❌ Episode {args.episode} not found. Available: 0-{len(files)-1}")
            return 1
        
        filepath = files[args.episode]
        inspect_hdf5_file(str(filepath), verbose=args.verbose)
    
    return 0


if __name__ == "__main__":
    exit(main())
