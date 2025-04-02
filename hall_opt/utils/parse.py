import sys
import argparse
from pathlib import Path
from typing import Optional, List, Set
import os 
import time # Import time module for stat 

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



def find_file_anywhere(
    filename: str,
    start_dir: str = ".", # Where to start searching
    # --- MODIFICATION: Changed default max_depth_up from 3 to 1 ---
    max_depth_up: int = 1, # How many parent levels to check (0=start_dir, 1=start_dir + parent)
    # --- End MODIFICATION ---
    exclude_dirs: Optional[List[str]] = None # List of dir names like 'venv', '.venv'
) -> Optional[Path]:
    """
    Searches recursively for 'filename' starting from 'start_dir' and going up
    'max_depth_up' parent levels (default is 1 level up, total 2 levels searched: start + parent).
    Uses os.walk and explicitly skips directories named in 'exclude_dirs'.

    Returns the path to the most recently modified match found, or None.
    """
    if exclude_dirs is None:
        exclude_dirs = []
    # Use a set for faster checking: {'venv', '.venv', ...}
    exclude_set = set(exclude_dirs)

    matches: List[Path] = [] # To store paths of found files
    checked_roots: Set[Path] = set() # To avoid searching the same directory tree multiple times
    search_root_start = Path(start_dir).resolve() # Start from absolute path

    print(f"[INFO] Starting search for '{filename}' from '{search_root_start}'")
    # --- MODIFICATION: Adjusted print statement for clarity ---
    print(f"[INFO] Will search starting directory and up to {max_depth_up} parent level(s).")
    # --- End MODIFICATION ---
    if exclude_set:
        print(f"[INFO] Excluding directories named: {exclude_set}")

    # Loop through the start directory and specified number of parent levels
    # max_depth_up = 1 means loop runs for depth=0 (start_dir) and depth=1 (parent)
    for depth in range(max_depth_up + 1):
        # Resolve the actual root path for this depth level
        try:
            level_root = search_root_start
            # Navigate up 'depth' levels
            for _ in range(depth):
                parent = level_root.parent
                if parent == level_root: # Check if we hit the filesystem root
                    level_root = None # Signal we can't go further up
                    break
                level_root = parent
            # Skip if we couldn't navigate up or already searched this exact path
            if level_root is None or level_root in checked_roots:
                continue
        except Exception as e:
             print(f"[WARNING] Error navigating parent directories at depth {depth}: {e}")
             continue # Skip this depth level if navigation fails

        print(f"[DEBUG] Searching within directory tree: {level_root}")
        checked_roots.add(level_root) # Mark this root as searched

        # os.walk explores directory tree top-down
        # onerror handles permissions issues during walk
        try:
            for current_dir_path, sub_dir_names, file_names in os.walk(
                level_root, topdown=True, onerror=lambda e: print(f"[WARNING] os.walk error: {e}")
            ):
                # --- This is the key part for skipping ---
                # Modify sub_dir_names IN PLACE to remove excluded ones BEFORE os.walk visits them
                sub_dir_names[:] = [d_name for d_name in sub_dir_names if d_name not in exclude_set]

                # Check if our target filename is in the list of files for the current directory
                if filename in file_names:
                    try:
                        # Construct the full path
                        found_path = Path(current_dir_path) / filename
                        # Verify it's actually a file (not a dir) and accessible
                        if found_path.is_file():
                            matches.append(found_path)
                    except OSError as e:
                        # Handle potential errors accessing the file path
                        print(f"[WARNING] Error checking potential match {Path(current_dir_path) / filename}: {e}")
        except Exception as e:
            # Catch errors during the os.walk process itself for a given root
            print(f"[WARNING] Unexpected error during os.walk in {level_root}: {e}")


    # --- After searching all levels ---
    if not matches:
        # --- MODIFICATION: Adjusted print statement ---
        print(f"[WARNING] Could not find file '{filename}' (searched starting dir and up {max_depth_up} parent level(s), excluding {exclude_dirs}).")
        # --- End MODIFICATION ---
        return None

    # Remove duplicates if any were found via different root paths
    unique_matches = list(set(matches))
    print(f"[DEBUG] Found {len(unique_matches)} unique potential match(es) for '{filename}'.")

    # Find the most recently modified file among the unique matches
    latest_file: Optional[Path] = None
    latest_mtime: float = -1.0 # Initialize with a value older than any file time

    for p in unique_matches:
        try:
             # Use time.time() on stat result for clarity if preferred, os.stat().st_mtime works too
             mtime = p.stat().st_mtime # Get modification time
             if mtime > latest_mtime: # Check if this file is newer
                 latest_mtime = mtime
                 latest_file = p
        except OSError as e:
             print(f"[WARNING] Could not get modification time for file {p}: {e}")
        except ValueError:
             print(f"[WARNING] Invalid modification time found for {p}")

    # Log final result
    if latest_file:
        print(f"[DEBUG] Returning latest file found: {latest_file}")
    elif unique_matches: # Found matches but couldn't get valid time for any
         print(f"[WARNING] Found matches but failed to determine the latest due to errors. Returning None.")

    return latest_file