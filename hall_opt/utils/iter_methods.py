import os
import json
import numpy as np
from pathlib import Path
from hall_opt.config.dict import Settings
import os

def get_next_results_dir(base_dir, base_name):
    """
    Finds the next available directory under `base_dir` by incrementing `base_name-N`.
    Example:
    - If `base_name="map-results"`, creates `map-results-1/`, `map-results-2/`, etc.
    - If `base_name="mcmc-results"`, creates `mcmc-results-1/`, `mcmc-results-2/`, etc.

    This function is **only called once per full run**.
    """
    base_dir = os.path.abspath(base_dir)  # Ensure it's absolute path
    os.makedirs(base_dir, exist_ok=True)  # Ensure base_dir exists
    i = 1
    while True:
        run_dir = os.path.join(base_dir, f"{base_name}-{i}")
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)  # Create the directory if not exists
            print(f"Created new run directory: {run_dir}")
            return run_dir
        i += 1  # Increment to the next available run

def get_next_filename(base_filename: str, directory: str, extension=".json") -> str:
    """
    Generate the next available filename inside `directory`.
        `get_next_filename("metrics", "hall_opt/results/mcmc/mcmc-results-1/iter_metrics/")`
        Returns: `hall_opt/results/mcmc/mcmc-results-1/iter_metrics/metrics_1.json`
    """
    Path(directory).mkdir(parents=True, exist_ok=True)

    i = 1
    while True:
        full_path = os.path.join(directory, f"{base_filename}_{i}{extension}")
        if not os.path.exists(full_path):
            return full_path  # Return unique file path
        i += 1
