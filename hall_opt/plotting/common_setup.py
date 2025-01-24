import os
import json
import pandas as pd
from pathlib import Path
#common utility file

def get_common_paths(settings, analysis_type):
    """
    Get relevant paths based on the analysis type (MAP or MCMC), using YAML settings.
    """
    base_dir = settings.general_settings["results_dir"]
    plots_base_dir = settings.plotting.plots_dir  # Use updated structure

    if analysis_type == "map":
        results_subdir = "map_results"
    elif analysis_type == "mcmc":
        results_subdir = "mcmc_results"
    else:
        raise ValueError("Invalid analysis type. Choose 'map' or 'mcmc'.")

    paths = {
        "results_dir": os.path.join(base_dir, results_subdir),
        "plots_dir": os.path.join(plots_base_dir, f"{analysis_type}"),
        "metrics_dir": os.path.join(base_dir, settings.plotting.metrics_subdir),
    }

    #  chech directories exist
    os.makedirs(paths["plots_dir"], exist_ok=True)
    os.makedirs(paths["metrics_dir"], exist_ok=True)

    return paths


def load_data(settings, analysis_type):
    """
     using paths from settings.
    """
    if analysis_type == "map":
        data_file = settings.optimization_params.map_params.final_map_params
    elif analysis_type == "mcmc":
        data_file = settings.optimization_params.mcmc_params.final_samples_file_log
    else:
        raise ValueError("Invalid analysis type. Choose 'map' or 'mcmc'.")

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Error: Data file not found at {data_file}")

    if analysis_type == "map":
        with open(data_file, 'r') as f:
            data = json.load(f)
        samples = pd.DataFrame([data])  # Convert single MAP estimate to DataFrame
    elif analysis_type == "mcmc":
        samples = pd.read_csv(data_file)

    return samples
