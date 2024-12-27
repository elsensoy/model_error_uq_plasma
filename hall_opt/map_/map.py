import sys
import json
import math
import os
import time
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)  # Ensures hall_opt is at the top of the search path

# Add HallThruster Python API to sys.path
hallthruster_path = "/root/.julia/packages/HallThruster/J4Grt/python"
if hallthruster_path not in sys.path:
    sys.path.append(hallthruster_path)

print("Updated sys.path:", sys.path)

import hallthruster as het
from utils.save_data import load_json_data, subsample_data, save_results_to_json, save_parameters_linear, save_parameters_log
from config.simulation import simulation, config_spt_100, postprocess, config_multilogbohm, update_twozonebohm_config, run_simulation_with_config
from utils.statistics import log_likelihood, prior_logpdf, log_posterior

results_dir = "/mnt/c/Users/MRover/elsensoy/model_error_uq_plasma/hall_opt/map_/results-map"

def run_map(observed_data, config, simulation, postprocess, results_dir, final_params_file="final_parameters.json", ion_velocity_weight=2.0):
    """
    Perform MAP optimization using v1 and alpha, and save parameter evolution.
    """
    initial_guess = [-2, 0.5]  # Log-space initial guess for v1 and alpha
    iteration_counter = [0]  # Tracks iterations

    print(f"Running MAP optimization with initial guess (log-space): {initial_guess}")

    def bounds_penalty(v_log):
        penalty = 0
        if not (-5 <= v_log[0] <= 0):  # log(v1) bounds
            penalty += (v_log[0] - max(-5, min(v_log[0], 0))) ** 2
        if not (0 <= v_log[1] <= 3):  # log(alpha) bounds
            penalty += (v_log[1] - max(0, min(v_log[1], 3))) ** 2
        return penalty

    def neg_log_posterior_with_penalty(v_log):
        """
        Compute the negative log-posterior with bounds penalties.
        """
        log_posterior_value = log_posterior(
            v_log, observed_data, config.copy(), simulation, postprocess.copy(), ion_velocity_weight
        )
        return -log_posterior_value + bounds_penalty(v_log)

    def iteration_callback(v_log):
        """
        Callback function for each iteration during optimization.
        Saves parameters in both linear and log space to separate files.
        """
        iteration_counter[0] += 1

        # Convert log-space values to linear space
        v1_linear = float(np.exp(v_log[0]))
        alpha_linear = float(np.exp(v_log[1]))

        # Save parameters in linear space
        save_parameters_linear(
            iteration=iteration_counter[0],
            v1=v1_linear,
            alpha=alpha_linear,
            results_dir=results_dir,
            filename="parameters_linear.json"
        )

        # Save parameters in log space
        save_parameters_log(
            iteration=iteration_counter[0],
            v1_log=v_log[0],
            alpha_log=v_log[1],
            results_dir=results_dir,
            filename="parameters_log.json"
        )

        # Print status
        print(f"Iteration {iteration_counter[0]}: v1 = {v1_linear:.4f} (log: {v_log[0]:.4f}), "
              f"alpha = {alpha_linear:.4f} (log: {v_log[1]:.4f})")

    # Perform MAP optimization
    result = minimize(
        neg_log_posterior_with_penalty,
        initial_guess,
        method="Nelder-Mead",
        callback=iteration_callback,
        options={"maxfev": 5000, "fatol": 1e-3, "xatol": 1e-3}
    )

    if result.success:
        v1_opt = np.exp(result.x[0])
        alpha_opt = np.exp(result.x[1])
        print(f"Optimization succeeded: v1 = {v1_opt:.4f}, alpha = {alpha_opt:.4f}")

        # Save final parameters
        final_params_path = os.path.join(results_dir, final_params_file)
        final_params = {"v1": v1_opt, "alpha": alpha_opt}
        with open(final_params_path, 'w') as f:
            json.dump(final_params, f, indent=4)
        print(f"Final parameters saved to {final_params_path}")

        return v1_opt, alpha_opt
    else:
        print("MAP optimization failed.")
        return None, None
             
def main():
    # Step 1: Run ground truth simulation
    print("Running ground truth simulation (MultiLogBohm)...")
    ground_truth_postprocess = postprocess.copy()
    ground_truth_postprocess["output_file"] = os.path.join(results_dir, "ground_truth.json")

    ground_truth_solution = run_simulation_with_config(
        config_multilogbohm, simulation, ground_truth_postprocess, config_type="MultiLogBohm"
    )

    if not ground_truth_solution:
        print("Ground truth simulation failed. Exiting.")
        return

    # Extract observed data
    averaged_metrics = ground_truth_solution["output"]["average"]
    observed_data = {
        "thrust": averaged_metrics["thrust"],
        "discharge_current": averaged_metrics["discharge_current"],
        "ion_velocity": averaged_metrics["ui"][0],
        "z_normalized": averaged_metrics["z"]
    }

    print("Starting MAP optimization and saving parameter evolution...")

    # Step 2: Run MAP optimization
    v1_opt, alpha_opt = run_map(
        observed_data, config_spt_100, simulation, postprocess, results_dir
    )

    if v1_opt is None or alpha_opt is None:
        print("Optimization failed. Exiting.")
        return  # Exit if optimization fails

    print(f"Final optimized parameters: v1 = {v1_opt:.4f}, alpha = {alpha_opt:.4f}")

if __name__ == "__main__":
    main()