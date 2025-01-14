import os
import pickle
import json
import numpy as np


def get_next_results_dir(base_dir="mcmc/results", base_name="mcmc-results"):
  
    #Generate the next results directory with an enumerated suffix.
    # 'mcmc-results-1', 'mcmc-results-2', etc.
    #     Example structure:
    # results/
    # ├── mcmc-results-1/
    # ├── mcmc-results-2/
    # 
    base_dir = os.path.abspath(base_dir)  # Convert to an absolute path
    os.makedirs(base_dir, exist_ok=True)  # Ensure the base_dir exists
    i = 1
    while True:
        dir_name = os.path.join(base_dir, f"{base_name}-{i}")
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)  # Create the directory
            print(f"Created results directory: {dir_name}")
            return dir_name
        i += 1


def get_next_filename(base_filename, directory, extension=".csv"):

    i = 1
    full_path = os.path.join(directory, f"{base_filename}_{i}{extension}")
    while os.path.exists(full_path):
        i += 1
        full_path = os.path.join(directory, f"{base_filename}_{i}{extension}")
    return full_path


