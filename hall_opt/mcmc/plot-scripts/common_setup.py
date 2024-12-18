import os
import pandas as pd
import json

# Paths
base_results_dir = os.path.join("..", "..","results", "mcmc-results-7") #Update is required for each sampling.
plots_dir = os.path.join(base_results_dir, "plots-mcmc") #Will create a new dir that includes mcmc plots. Update is required after sampling.
metrics_dir = os.path.join(base_results_dir, "iteration_metrics")
os.makedirs(plots_dir, exist_ok=True)

# Helper function: Check file existence
def check_file_exists(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

# Load MCMC and related data
def load_data(base_results_dir=base_results_dir):
    """Loads MCMC samples, truth data, and initial parameter guess."""
    # File paths
    samples_path = os.path.join(base_results_dir, "/mnt/c/Users/MRover/elsensoy/model_error_uq_plasma/results/mcmc-results-7/final_mcmc_samples_1.csv") #no need to change
    truth_data_path = os.path.join(base_results_dir, "/mnt/c/Users/MRover/elsensoy/model_error_uq_plasma/results/mcmc-results-7/mcmc_observed_data_map.json")
    pre_mcmc_data_path = os.path.join(base_results_dir, "/mnt/c/Users/MRover/elsensoy/model_error_uq_plasma/results/mcmc-results-7/mcmc_pre_mcmc_initial.json")
    initial_params_path = os.path.join(base_results_dir, "/mnt/c/Users/MRover/elsensoy/model_error_uq_plasma/results/best_initial_guess_w_2_0 copy.json")

    # Check paths
    for path in [samples_path, truth_data_path, pre_mcmc_data_path, initial_params_path]:
        check_file_exists(path)

    # Load CSV data
    samples = pd.read_csv(samples_path, header=None, names=["log_v1", "log_alpha"])
    samples["v1"] = 10 ** samples["log_v1"]
    samples["alpha"] = 10 ** samples["log_alpha"]
    samples["v2"] = samples["v1"] * samples["alpha"]

    # Load JSON data
    with open(truth_data_path, 'r') as f:
        truth_data = json.load(f)

    with open(pre_mcmc_data_path, 'r') as f:
        pre_mcmc_data = json.load(f)

    with open(initial_params_path, 'r') as f:
        initial_params = json.load(f)

    return samples, truth_data, pre_mcmc_data, initial_params

# Load iteration metrics
def load_iteration_metrics(metrics_dir=metrics_dir):
    """Loads all iteration metrics from JSON files."""
    metrics = []
    for file in sorted(os.listdir(metrics_dir)):
        if file.endswith(".json"):
            with open(os.path.join(metrics_dir, file), 'r') as f:
                metrics.append(json.load(f))
    return metrics

def get_common_paths():
    return {
        "base_results_dir": base_results_dir, 
        "plots_dir": plots_dir,
        "metrics_dir": metrics_dir,
    }