import sys
import argparse
from pathlib import Path
from typing import Optional
import os
import yaml

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run different methods of the project.")
    parser.add_argument("method_yaml", type=str, help="Path to a YAML configuration file (in root or config/).")
    return parser.parse_args()

 
def get_yaml_path(method_yaml: str, max_depth_up: int = 3) -> Path:
    """
    Searches recursively for the given YAML file in the current project directory,
    or up to `max_depth_up` levels above it.

    Args:
        method_yaml (str): Name of the YAML file to search for
        max_depth_up (int): How many parent levels to search above the current directory

    Returns:
        Path: Resolved path to the YAML file

    Exits:
        If not found, prints an error and exits the program.
    """
    root_dir = Path(".").resolve()
    target_filename = Path(method_yaml).name

    # 1. First try the current directory and its subdirs
    matches = list(root_dir.rglob(target_filename))
    if matches:
        print(f"[INFO] Found YAML config at: {matches[0]}")
        return matches[0]

    # 2. Try parent directories (up to max_depth_up)
    for level in range(1, max_depth_up + 1):
        try:
            search_root = root_dir.parents[level]
        except IndexError:
            break  # We've gone higher than root (e.g., /)

        matches = list(search_root.rglob(target_filename))
        if matches:
            print(f"[INFO] Found YAML config at: {matches[0]}")
            return matches[0]

    # 3. If nothing found
    print(f"[ERROR] Could not find '{method_yaml}' under current or any of {max_depth_up} parent levels.")
    sys.exit(1)

    

def find_file_anywhere(filename: str, max_depth_up: int = 3) -> Optional[Path]:
    """
    Searches recursively for the exact file (e.g., 'output_multilogbohm.json').
    Returns the most recently modified match.
    """

    root_dir = Path(".").resolve()

    def safe_rglob(directory: Path):
        try:
            return list(directory.rglob(filename))
        except (PermissionError, OSError) as e:
            print(f"[WARNING] Skipping inaccessible directory: {directory} — {e}")
            return []

    all_matches = safe_rglob(root_dir)

    for level in range(1, max_depth_up + 1):
        try:
            parent = root_dir.parents[level]
            all_matches.extend(safe_rglob(parent))
        except IndexError:
            break

    if not all_matches:
        print(f"[WARNING] Could not find file: {filename}")
        return None

    # Return most recently modified file
    return max(all_matches, key=lambda p: p.stat().st_mtime)
