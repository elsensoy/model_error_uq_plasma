import os
from pathlib import Path

def get_common_paths(settings, analysis_type):
    """
    Get relevant paths based on the analysis type (MAP or MCMC), using YAML settings.
    """

    base_dir = settings.results_dir  # Root results directory
    plots_base_dir = settings.plots.plots_subdir  # Plots directory

    if analysis_type == "map":
        results_subdir = settings.map.output_dir
    elif analysis_type == "mcmc":
        results_subdir = settings.mcmc.output_dir
    else:
        raise ValueError("Invalid analysis type. Choose 'map' or 'mcmc'.")

    paths = {
        "results_dir": os.path.join(base_dir, results_subdir),
        "plots_dir": os.path.join(plots_base_dir, analysis_type),  # Saves plots under `plots/map` or `plots/mcmc`
        "metrics_dir": os.path.join(base_dir, settings.plots.metrics_subdir),  # Metrics directory
    }

    # make sure directories exist before saving plots
    os.makedirs(paths["plots_dir"], exist_ok=True)
    os.makedirs(paths["metrics_dir"], exist_ok=True)

    return paths
