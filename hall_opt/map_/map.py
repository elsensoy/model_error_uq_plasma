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
from utils.save_data import load_json_data, subsample_data, save_results_to_json
from map_.simulation import simulation, config_spt_100, postprocess, config_multilogbohm, update_twozonebohm_config, run_simulation_with_config
from utils.posterior import log_likelihood, prior_logpdf, log_posterior

def run_map_single_initial_guess(observed_data, config, simulation, postprocess, ion_velocity_weight=2.0):
    """Run MAP optimization starting from a single initial guess."""

    initial_guess = [-2, 0.5]  # log(v1), log(alpha)

    print(f"Running MAP optimization with initial guess: {initial_guess}")

    # Bounds penalty to ensure parameters remain within feasible ranges
    def bounds_penalty(v_log):
        penalty = 0
        if not (-5 <= v_log[0] <= 0):  # log(v1) bounds
            penalty += (v_log[0] - max(-5, min(v_log[0], 0))) ** 2
        if not (0 <= v_log[1] <= 3):  # log(alpha) bounds
            penalty += (v_log[1] - max(0, min(v_log[1], 3))) ** 2
        return penalty

    def neg_log_posterior_with_penalty(v_log):
        log_posterior_value = log_posterior(
            v_log, observed_data, config.copy(), simulation, postprocess.copy(), ion_velocity_weight
        )
        return -log_posterior_value + bounds_penalty(v_log)

    # Perform MAP optimization
    result = minimize(
        neg_log_posterior_with_penalty,
        initial_guess,
        method="Nelder-Mead",
        options={"maxfev": 5000, "fatol": 1e-3, "xatol": 1e-3}
    )

    if result.success:
        v1_opt, alpha_opt = np.exp(result.x[0]), np.exp(result.x[1])
        v2_opt = alpha_opt * v1_opt
        print(f"Optimization succeeded: v1 = {v1_opt:.4f}, v2 = {v2_opt:.4f}")
        return v1_opt, v2_opt
    else:
        print("MAP optimization failed.")
        return None, None


def main():

    # Step 1: Run ground truth simulation
    print("Running ground truth simulation (MultiLogBohm)...")
    ground_truth_postprocess = postprocess.copy()
    ground_truth_postprocess["output_file"] = "ground_truth.json"

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
    save_file = "parameter_evolution.json"  # File to save sampled parameters

    # Step 2: Run MAP optimization
    v1_opt, v2_opt = run_map_single_initial_guess(
        observed_data, config_spt_100, simulation, postprocess, save_file
    )

    if v1_opt is not None and v2_opt is not None:
        print(f"Final optimized parameters: v1 = {v1_opt:.4f}, v2 = {v2_opt:.4f}")
    else:
        print("Optimization failed.")
if __name__ == "__main__":
    main()