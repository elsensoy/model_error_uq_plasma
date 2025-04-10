import os
import json
import numpy as np
from pathlib import Path
import os
from .save_data import subsample_data
from hall_opt.config.dict import Settings

def get_next_results_dir(base_dir: str, base_name: str) -> str:
    """
    Generate the next results directory dynamically based on `base_dir` from settings.

    - Example:
        `get_next_results_dir("hall_opt/results/mcmc", "mcmc-results")`
        Returns: `mcmc/mcmc-results-1/`
    """

    base_dir = os.path.abspath(base_dir)  #  absolute path
    Path(base_dir).mkdir(parents=True, exist_ok=True)  

    i = 1
    while True:
        dir_name = os.path.join(base_dir, f"{base_name}-{i}")
        if not os.path.exists(dir_name):
            Path(dir_name).mkdir(parents=True, exist_ok=True)  # Create directory ONCE per run
            print(f"Created results directory: {dir_name}")
            return dir_name  # Return path to new results folder
        i += 1

def get_next_filename(base_filename: str, directory: str, extension=".json") -> str:
    """
    Generate the next available filename inside `directory`, ensuring uniqueness.
    
    - Example:
        `get_next_filename("metrics", "hall_opt/results/mcmc/mcmc-results-1/iter_metrics/")`
        Returns: `mcmc-results-1/iter_metrics/metrics_1.json`
    """

    Path(directory).mkdir(parents=True, exist_ok=True)  # make sure directory exists

    i = 1
    while True:
        full_path = os.path.join(directory, f"{base_filename}_{i}{extension}")
        if not os.path.exists(full_path):
            return full_path  #  Return unique file path
        i += 1


