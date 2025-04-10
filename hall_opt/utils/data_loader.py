import sys
import os
import json
import yaml
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from hall_opt.config.dict import Settings
from hall_opt.config.verifier import Settings
import re

def load_data(settings: Settings, analysis_type: str) -> pd.DataFrame:
    
    if analysis_type == "ground_truth":
        data_file = settings.postprocess.output_file.get("MultiLogBohm")
        print("[INFO] Ground truth data file path:", data_file)

        if not os.path.isfile(data_file):
            raise FileNotFoundError(f"[ERROR] Data file not found at {data_file}")

        return pd.read_csv(data_file)

    elif analysis_type == "map":


        base_results_root = Path(settings.output_dir).resolve() / "map"
        latest_dir = find_latest_results_dir(base_results_root, "map-results")

        if not latest_dir:
            raise FileNotFoundError(f"[ERROR] Could not find latest MAP results directory in {base_results_root}")

        opt_file = latest_dir / "optimization_result.json"
        log_file = latest_dir / "map_iteration_log.json"
        gen_data_file = latest_dir / "ground_truth" / "ground_truth_metrics.json"

        
        print(f"[INFO] Loading MAP optimization result from: {opt_file}")
        print(f"[INFO] Loading MAP iteration log from: {log_file}")
        print(f"[INFO] Loading Gen data metrics file from: {gen_data_file}")
        if not opt_file.is_file():
            raise FileNotFoundError(f"[ERROR] optimization_result.json not found at {opt_file}")

        if not log_file.is_file():
            raise FileNotFoundError(f"[ERROR] map_iteration_log.json not found at {log_file}")

        if not log_file.is_file():
            raise FileNotFoundError(f"[ERROR] gen_data_file.json not found at {gen_data_file}")

        # Load full MAP iteration trace as samples
        with open(log_file, "r") as f:
            all_iters = json.load(f)

        samples = pd.DataFrame(all_iters)
 
        if "c1_log" not in samples.columns or "alpha_log" not in samples.columns:
            raise ValueError("[ERROR] Required columns missing in map_iteration_log.json")

        return samples

    elif analysis_type == "mcmc":
        base_results_root = Path(settings.output_dir).resolve() / "mcmc"
        latest_dir = find_latest_results_dir(base_results_root, "mcmc-results")

        if not latest_dir:
            raise FileNotFoundError(f"[ERROR] Could not find latest MCMC results directory in {base_results_root}")

        data_file = latest_dir / "final_samples_log.csv"
        print(f"[INFO] Loading MCMC data from: {data_file}")

        if not data_file.is_file():
            raise FileNotFoundError(f"[ERROR] Data file not found at {data_file}")

        return pd.read_csv(data_file, header=None, names=["log_c1", "log_alpha"])

    else:
        raise ValueError("[ERROR] Invalid analysis type. Choose 'map', 'mcmc', or 'ground_truth'.")


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


