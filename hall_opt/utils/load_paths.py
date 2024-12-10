import os
import json
from utils.mcmc_utils import load_json_data, load_optimized_params 


def resolve_results_paths(script_mcmc):

    print("Resolving file paths for results processing...")
    
    # Resolve the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(script_mcmc))
    path_dir = os.path.abspath(os.path.join(script_dir, "..", "hall_opt/results"))
    results_dir = get_next_results_dir(base_dir="results", base_name="mcmc-results")
    path_dir = os.path.abspath(os.path.join("..", "hall_opt/results"))
    initial_guess_path = os.path.join(path_dir, "best_initial_guess_w_2_0.json")


    # Ensure the results directory exists
    if not os.path.exists(path_dir):
        raise FileNotFoundError(f"Results directory not found: {path_dir}")

    # Define individual file paths
    initial_metrics_result = os.path.join(path_dir, "mcmc_pre_mcmc_initial.json")
    observed_data_file = os.path.join(path_dir, "mcmc_observed_data_map.json")
    initial_guess_path = os.path.join(path_dir, "best_initial_guess_w_2_0.json")

    # Verify that each file exists
    for file_path in [initial_metrics_result, observed_data_file, initial_guess_path]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")

    return {
        "initial_metrics_result": initial_metrics_result,
        "observed_data_file": observed_data_file,
        "initial_guess_path": initial_guess_path
    }
