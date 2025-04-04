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
        data_file = settings.postprocess.output_file["MultiLogBohm"]

        print("Ground truth is loaded successfully")
    elif analysis_type == "map":
        data_file = os.path.join(settings.map.base_dir, "optimization_result.json")  
    elif analysis_type == "mcmc":
        data_file = os.path.join(settings.mcmc.output_dir, "final_samples.csv")  

    else:
        raise ValueError(" Invalid analysis type. Choose 'map' or 'mcmc'.")

    if not os.path.exists(data_file):
        raise FileNotFoundError(f" Error: Data file not found at {data_file}")

    if analysis_type == "map":
        with open(data_file, "r") as f:
            data = json.load(f)
        samples = pd.DataFrame([data])  #  Convert single MAP estimate to DataFrame
    else:  # MCMC
        samples = pd.read_csv(data_file, header=None, names=["log_c1", "log_alpha"])

    return samples

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


