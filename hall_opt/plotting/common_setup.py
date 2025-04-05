import os
from pathlib import Path

import os
from pathlib import Path
from hall_opt.utils.data_loader import find_latest_results_dir  
import os
import re
from pathlib import Path
from typing import Optional, Dict, Union # Added Dict, Union

# --- Revised function to get paths ---
def get_common_paths(settings, analysis_type: str) -> Dict[str, Optional[Path]]:
    """
    Get relevant paths based on the analysis type (MAP or MCMC), using YAML settings.
    Finds the latest numbered results directory automatically.

    Args:
        settings: The loaded settings object.
        analysis_type: Either "map" or "mcmc".

    Returns:
        A dictionary containing Path objects for relevant directories.
        'latest_results_dir' will be None if no matching directory is found.
    """
    try:
        base_results_root = Path(settings.results_dir) # e.g., C:\...\model_error_uq_plasma
        plots_root = Path(settings.plotting.results_dir)  # e.g., relative path 'plots' or absolute path
        metrics_root = Path(settings.output_dir) # e.g., relative path 'metrics' or absolute path

        # make sure plots/metrics roots are absolute or relative to results_dir if not absolute
        if not plots_root.is_absolute():
             plots_root = base_results_root / plots_root
        if not metrics_root.is_absolute():
             metrics_root = base_results_root / metrics_root


        latest_run_dir: Optional[Path] = None
        search_base_dir: Optional[Path] = None
        base_name_pattern: Optional[str] = None

        if analysis_type == "map":
            search_base_dir = base_results_root / settings.map.subdir # e.g., C:\...\map_results\map
            base_name_pattern = settings.map.base_name # e.g., "map-results"
        elif analysis_type == "mcmc":
            search_base_dir = base_results_root / settings.mcmc.subdir # e.g., C:\...\mcmc_results\mcmc
            base_name_pattern = settings.mcmc.base_name # e.g., "mcmc-results"
        else:
            raise ValueError("Invalid analysis type. Choose 'map' or 'mcmc'.")

        if search_base_dir and base_name_pattern:
            print(f"[INFO] Searching for latest '{base_name_pattern}-N' in '{search_base_dir}'")
            latest_run_dir = find_latest_results_dir(search_base_dir, base_name_pattern)
        else:
             print("[ERROR] Could not determine search directory or base name pattern.")


        paths = {
            # The specific directory like 'map-results-1' or 'mcmc-run-5'
            "latest_results_dir": latest_run_dir,
             # The parent directory, e.g., 'map_results/map'
            "parent_results_dir": search_base_dir,
            # Saves plots under `plots/map` or `plots/mcmc` relative to plots_root
            "plots_dir": plots_root / analysis_type,
             # Where metrics are stored (potentially independent of specific run)
            "metrics_dir": metrics_root,
             # Include the root results dir as well
            "root_results_dir": base_results_root
        }

        # Create plot/metrics directories if they don't exist
        if paths["plots_dir"]:
            paths["plots_dir"].mkdir(parents=True, exist_ok=True)
        if paths["metrics_dir"]:
            paths["metrics_dir"].mkdir(parents=True, exist_ok=True)


        return paths

    except AttributeError as e:
        print(f"[ERROR] Missing setting attribute: {e}")
        # Return a dictionary with None values to prevent downstream crashes needing paths
        return {
            "latest_results_dir": None,
            "parent_results_dir": None,
            "plots_dir": None,
            "metrics_dir": None,
            "root_results_dir": None,
        }
    except Exception as e:
        print(f"[ERROR] Unexpected error in get_common_paths: {e}")
        # Return a dictionary with None values
        return {
            "latest_results_dir": None,
            "parent_results_dir": None,
            "plots_dir": None,
            "metrics_dir": None,
            "root_results_dir": None,
        }


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