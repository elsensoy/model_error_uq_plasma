import os
import json
import yaml
import pandas as pd
from pathlib import Path

# -----------------------------
# Load YAML plotting settings
# -----------------------------

def load_plotting_settings(yaml_path="plotting.yaml"):
    """Load plotting settings from YAML."""
    with open(yaml_path, 'r') as file:
        settings = yaml.safe_load(file)
    return settings

# Load settings once
plotting_settings = load_plotting_settings()

# Extract paths from YAML
base_results_dir = plotting_settings["plotting"]["base_results_dir"]
plots_dir = os.path.join(base_results_dir, run_dir, plotting_settings["plotting"]["plots_subdir"])
metrics_dir = os.path.join(base_results_dir, run_dir, plotting_settings["plotting"]["metrics_subdir"])

# Ensure necessary directories exist
os.makedirs(plots_dir, exist_ok=True)

# -----------------------------
# Helper Function
# -----------------------------

def check_file_exists(file_path):
    """Check if a file exists, raise an error if not."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

# -----------------------------
# Load MCMC Data
# -----------------------------

def load_data():
    """Load samples and related data from YAML-specified paths."""
    files = plotting_settings["plotting"]["files"]

    # File paths
    samples_path = os.path.join(base_results_dir, run_dir, files["samples"])
    truth_data_path = os.path.join(base_results_dir, files["truth_data"])
    pre_mcmc_data_path = os.path.join(base_results_dir, files["pre_mcmc_data"])
    initial_params_path = os.path.join(base_results_dir, files["initial_params"])

    # Check if files exist
    for path in [samples_path, truth_data_path, pre_mcmc_data_path, initial_params_path]:
        check_file_exists(path)

    # Load CSV sample data
    samples = pd.read_csv(samples_path, header=None, names=["log_c1", "log_alpha"])
    samples["c1"] = 10 ** samples["log_c1"]
    samples["alpha"] = 10 ** samples["log_alpha"]
    samples["c2"] = samples["c1"] * samples["alpha"]

    # Load JSON data
    with open(truth_data_path, 'r') as f:
        truth_data = json.load(f)
    with open(pre_mcmc_data_path, 'r') as f:
        pre_mcmc_data = json.load(f)
    with open(initial_params_path, 'r') as f:
        initial_params = json.load(f)

    return samples, truth_data, pre_mcmc_data, initial_params

def load_iteration_metrics():
    """Load all iteration metrics from JSON files."""
    metrics = []
    for file in sorted(os.listdir(metrics_dir)):
        if file.endswith(".json"):
            with open(os.path.join(metrics_dir, file), 'r') as f:
                metrics.append(json.load(f))
    return metrics


def get_common_paths():
    """Retrieve commonly used paths based on YAML settings."""
    return {
        "base_results_dir": base_results_dir,
        "plots_dir": plots_dir,
        "metrics_dir": metrics_dir,
    }

# -----------------------------
# Testing
# -----------------------------

if __name__ == "__main__":
    print("Base Results Directory:", base_results_dir)
    print("Run Directory:", run_dir)
    print("Plots Directory:", plots_dir)
    print("Metrics Directory:", metrics_dir)

    try:
        samples, truth_data, pre_mcmc_data, initial_params = load_data()
        print("MCMC samples loaded successfully.")
    except FileNotFoundError as e:
        print(e)

    try:
        iteration_metrics = load_iteration_metrics()
        print(f"Loaded {len(iteration_metrics)} iteration metrics.")
    except FileNotFoundError as e:
        print(e)
