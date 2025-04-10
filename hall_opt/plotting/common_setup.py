import os
from pathlib import Path

import os
from pathlib import Path
from hall_opt.utils.data_loader import find_latest_results_dir  
import os
import re
from pathlib import Path
from typing import Optional, Dict, Union # Added Dict, Union
import json
import yaml
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from hall_opt.config.dict import Settings
from hall_opt.config.verifier import Settings
import re


def get_common_paths(settings, analysis_type: str) -> Dict[str, Optional[Path]]:
    """
    Get relevant paths for MAP or MCMC. Uses output_dir/map or output_dir/mcmc.
    """
    try:
        #  FIX: Adjust base path based on analysis type
        base_results_root = Path(settings.output_dir).resolve()
        if analysis_type == "map":
            base_results_root = base_results_root / "map"

        elif analysis_type == "mcmc":
            base_results_root = base_results_root / "mcmc"

        plots_root = Path(settings.plots.results_dir)
        metrics_root = base_results_root

        if not plots_root.is_absolute():
            plots_root = base_results_root / plots_root
        if not metrics_root.is_absolute():
            metrics_root = base_results_root / metrics_root

        base_name_pattern = f"{analysis_type}-results"

        print(f"[INFO] Searching in '{base_results_root}' for '{base_name_pattern}-N' folders...")
        latest_run_dir = find_latest_results_dir(base_results_root, base_name_pattern)

        paths = {
            "latest_results_dir": latest_run_dir,
            "parent_results_dir": base_results_root,
            "plots_dir": plots_root / analysis_type,
            "metrics_dir": metrics_root,
            "root_results_dir": base_results_root
        }

        for key in ["plots_dir", "metrics_dir"]:
            if paths[key]:
                paths[key].mkdir(parents=True, exist_ok=True)

        return paths

    except Exception as e:
        print(f"[ERROR] get_common_paths failed: {e}")
        return {
            "latest_results_dir": None,
            "parent_results_dir": None,
            "plots_dir": None,
            "metrics_dir": None,
            "root_results_dir": None,
        }


def find_latest_results_dir(base_dir: str, base_name: str) -> Optional[Path]:
    """
    Finds the results directory matching the pattern '{base_name}-N' inside
    'base_dir' with the highest number N.

    Args:
        base_dir: The parent directory containing the numbered result folders.
        base_name: The prefix of the result folders (e.g., "map-results").

    Returns:
        A Path object to the latest directory (highest N), or None if none found.
    """
    parent_dir = Path(base_dir)
    latest_num = -1
    latest_dir = None
    pattern = re.compile(rf"^{re.escape(base_name)}-(\d+)$") # Regex to match name-NUMBER

    if not parent_dir.is_dir():
        print(f"[WARNING] find_latest_results_dir: Base directory '{parent_dir}' does not exist.")
        return None

    try:
        for item in parent_dir.iterdir():
            if item.is_dir():
                match = pattern.match(item.name)
                if match:
                    num = int(match.group(1))
                    if num > latest_num:
                        latest_num = num
                        latest_dir = item
    except Exception as e:
         print(f"[WARNING] Error searching for latest results directory in '{parent_dir}': {e}")
         return None # Return None on error

    if latest_dir:
        print(f"[DEBUG] Found latest results directory: {latest_dir}")
    else:
         print(f"[INFO] No directories matching pattern '{base_name}-N' found in '{parent_dir}'.")

    return latest_dir


# --- Function to prompt user ---
def prompt_analysis_type() -> str:
    """Prompts the user to select MAP or MCMC."""
    print("Select analysis type to plot:")
    print("1. MAP")
    print("2. MCMC")

    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice == "1":
            return "map"
        elif choice == "2":
            return "mcmc"
        else:
            print("Invalid input. Please enter 1 for MAP or 2 for MCMC.")

def get_latest_analysis_type(settings: Settings) -> Optional[str]:
    """
    Determine whether 'map' or 'mcmc' was run most recently based on results folder timestamps.
    """

    output_dir = Path(settings.output_dir)

    map_dir = find_latest_results_dir(output_dir / "map", "map-results")
    mcmc_dir = find_latest_results_dir(output_dir / "mcmc", "mcmc-results")

    if not map_dir and not mcmc_dir:
        return None
    if map_dir and not mcmc_dir:
        return "map"
    if mcmc_dir and not map_dir:
        return "mcmc"

    # Compare modification times
    map_time = os.path.getmtime(map_dir)
    mcmc_time = os.path.getmtime(mcmc_dir)

    return "map" if map_time > mcmc_time else "mcmc"


def resolve_analysis_type(settings: Settings) -> str:
    """
    Determine analysis type (map or mcmc) based on settings flags.
    Falls back to latest results directory if both are False.
    """
    if settings.run_map:
        return "map"
    elif settings.run_mcmc:
        return "mcmc"
    else:
        detected = get_latest_analysis_type(settings)
        if detected:
            print(f"[INFO] Detected latest completed run as: {detected.upper()}")
            return detected
        return prompt_analysis_type()
    
from pathlib import Path

def interactive_plot_prompt(settings) -> tuple[str, Path]:
    """
    Interactive CLI prompt for plotting options.
    Returns:
        analysis_type (str): 'map' or 'mcmc'
        chosen_results_dir (Path): Path to result folder
    """
    print("\n[CLI] Would you like to plot the latest results? (y/n): ", end="")
    latest_choice = input().strip().lower()

    if latest_choice in ["y", "yes"]:
        # Infer latest analysis type from flags
        if settings.run_map:
            analysis_type = "map"
        elif settings.run_mcmc:
            analysis_type = "mcmc"
        else:
            analysis_type = prompt_analysis_type()

        paths = get_common_paths(settings, analysis_type)
        latest_dir = paths.get("latest_results_dir")

        if not latest_dir:
            raise FileNotFoundError(f"[ERROR] No latest {analysis_type.upper()} results directory found.")

        return analysis_type, latest_dir

    else:
        # Ask user for analysis type manually
        analysis_type = prompt_analysis_type()

        print(f"\n[CLI] Enter the exact name of the results folder to plot (e.g., {analysis_type}-results-3): ")
        folder_name = input(">> ").strip()

        base_dir = Path(settings.output_dir).resolve() / analysis_type
        results_dir = base_dir / folder_name

        if not results_dir.is_dir():
            raise FileNotFoundError(f"[ERROR] Specified folder not found: {results_dir}")

        return analysis_type, results_dir
