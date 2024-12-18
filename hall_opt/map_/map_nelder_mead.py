import sys
import json
import math
import os
import time
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

# Add HallThruster Python API to the system path
sys.path.append("/root/.julia/packages/HallThruster/J4Grt/python")  # hallthruster python path 
import hallthruster as het

from map_.map_utils import save_iteration_metrics,load_json_data, subsample_data, save_results_to_json
from  map_.simulation import simulation, config_spt_100,postprocess, config_multilogbohm
from posterior import log_likelihood, prior_logpdf, log_posterior


def run_simulation_with_config(config, simulation, postprocess, config_type="MultiLogBohm"):
 
    config_copy = config.copy()  # Ensure the original config is not mutated
    input_data = {"config": config_copy, "simulation": simulation, "postprocess": postprocess}

    print(f"Running simulation with {config_type} configuration...")
    try:
        solution = het.run_simulation(input_data)
        if solution["output"]["retcode"] != "success":
            print(f"Simulation failed with retcode: {solution['output']['retcode']}")
            return None
        return solution
    except Exception as e:
        print(f"Error during simulation with {config_type}: {e}")
        return None

# #loss_historry parameter deleted
def compute_neg_log_posterior(v_log, observed_data, config, simulation, postprocess, sigma=0.08):
    v1 = float(np.exp(v_log[0]))
    alpha = np.exp(v_log[1])
    v2 = float(alpha * v1)

    config_copy = config.copy()  # for not mutating the original configuration
    config_copy["anom_model"] = {"type": "TwoZoneBohm", "c1": v1, "c2": v2}

    input_data = {"config": config_copy, "simulation": simulation, "postprocess": postprocess}

    try:
        solution = het.run_simulation(input_data)
        if solution["output"]["retcode"] != "success":
            return np.inf  # Penalize failed simulations
        avg = solution["output"]["average"]
        simulated_data = {
            "thrust": avg["thrust"],
            "discharge_current": avg["discharge_current"],
            "ion_velocity": avg.get("ui", [])
        }
        log_likelihood_value = log_likelihood(simulated_data, observed_data, postprocess)
        log_prior_value = prior_logpdf(v_log[0], v_log[1])
        return -(log_likelihood_value + log_prior_value)
    except Exception as e:
        print(f"Error during simulation in posterior: {e}")
        return np.inf

def update_twozonebohm_config(config, v1, v2):

    config_copy = config.copy()  # Ensure the original config is not mutated
    config_copy["anom_model"] = {"type": "TwoZoneBohm", "c1": v1, "c2": v2}
    return config_copy


def callback(v_log, iteration_counter, config, simulation, postprocess):
    iteration_counter[0] += 1
    v1 = float(np.exp(v_log[0]))
    alpha = np.exp(v_log[1])
    v2 = float(alpha * v1)

    config_copy = config.copy()  # Ensure a fresh copy
    config_copy["anom_model"] = {"type": "TwoZoneBohm", "c1": v1, "c2": v2}

    input_data = {"config": config_copy, "simulation": simulation, "postprocess": postprocess}
    try:
        solution = het.run_simulation(input_data)
        if solution["output"]["retcode"] == "success":
            metrics = solution["output"]["average"]
            iteration_metrics = {
                "thrust": metrics.get("thrust"),
                "discharge_current": metrics.get("discharge_current"),
                "ion_velocity": metrics.get("ui", []),
                "z_normalized": metrics.get("z", [])
            }
            save_results_to_json(iteration_metrics, f'iteration_metrics_{iteration_counter[0]}.json')
        else:
            print(f"Simulation failed at iteration {iteration_counter[0]}")
    except Exception as e:
        print(f"Error during callback simulation: {e}")

def run_map_single_initial_guess(observed_data, config, simulation, postprocess, ion_velocity_weight=2.0):

    initial_guess = [-2, 0.5] #initial guess 
    iteration_counter = [0]

    print(f"Running MAP optimization with initial guess: {initial_guess}")

    def bounds_penalty(v_log):
        penalty = 0
        if not (-5 <= v_log[0] <= 0):  # log(v1) bounds
            penalty += (v_log[0] - max(-5, min(v_log[0], 0))) ** 2
        if not (0 <= v_log[1] <= 3):  # log(alpha) bounds
            penalty += (v_log[1] - max(0, min(v_log[1], 3))) ** 2
        return penalty

    def iteration_callback(v_log):
        iteration_counter[0] += 1
        v1 = float(np.exp(v_log[0]))
        alpha = np.exp(v_log[1])
        v2 = float(np.exp(v_log[1]) * v1)

        print(f"Iteration {iteration_counter[0]}: v1 = {v1:.4f}, v2 = {v2:.4f}")

        # Update configuration
        config["anom_model"]["c1"] = v1
        config["anom_model"]["c2"] = v2
        print(f"Updated config: {config}")

        # Prepare input data
        input_data = {"config": config, "simulation": simulation, "postprocess": postprocess}
        print(f"Running simulation with input_data: {input_data}")

        # Run the simulation
        try:
            solution = het.run_simulation(input_data)
            if solution["output"]["retcode"] == "success":
                metrics = solution["output"]["average"]
                save_results_to_json(metrics, f'iteration_metrics_{iteration_counter[0]}.json')
            else:
                print(f"Simulation failed with retcode: {solution['output']['retcode']}")
        except Exception as e:
            print(f"Error during simulation: {e}")

    result = minimize(
        lambda v_log: bounds_penalty(v_log),
        initial_guess,
        method="Nelder-Mead",
        callback=iteration_callback,
        options={"maxfev": 5000, "fatol": 1e-3, "xatol": 1e-3}
    )

    if result.success:
        v1_opt, alpha_opt = np.exp(result.x[0]), np.exp(result.x[1])
        print(f"Optimization succeeded: v1 = {v1_opt}, v2 = {alpha_opt * v1_opt}")
        return v1_opt, alpha_opt * v1_opt
    else:
        print("MAP optimization failed.")
        return None, None
def main():
    # Run ground truth simulation (MultiLogBohm)
    ground_truth_postprocess = postprocess.copy()
    ground_truth_postprocess["output_file"] = "ground_truth.json"
    ground_truth_solution = run_simulation_with_config(
        config_multilogbohm, simulation, ground_truth_postprocess, config_type="MultiLogBohm"
    )

    # Extract observed data for optimization
    if ground_truth_solution:
        averaged_metrics = ground_truth_solution["output"]["average"]
        observed_data = {
            "thrust": averaged_metrics["thrust"],
            "discharge_current": averaged_metrics["discharge_current"],
            "ion_velocity": averaged_metrics["ui"][0],
            "z_normalized": averaged_metrics["z"]
        }
    else:
        print("Ground truth simulation failed. Exiting.")
        return

    # Run MAP optimization (TwoZoneBohm)
    initial_guess = [-2, 0.5]
    result = minimize(
        lambda v_log: compute_neg_log_posterior(
            v_log, observed_data, config_spt_100.copy(),  # Pass a copy of TwoZoneBohm config
            simulation, postprocess.copy()
        ),
        initial_guess,
        method="Nelder-Mead",
        options={"maxiter": 100}
    )

    if result.success:
        v1_opt, v2_opt = np.exp(result.x[0]), np.exp(result.x[1]) * np.exp(result.x[0])
        print(f"Optimized parameters: v1 = {v1_opt:.4f}, v2 = {v2_opt:.4f}")
    else:
        print("MAP optimization failed.")
        return

    # Run optimized simulation with updated TwoZoneBohm configuration
    optimized_postprocess = postprocess.copy()
    optimized_postprocess["output_file"] = "optimized_solution.json"
    optimized_config = update_twozonebohm_config(config_spt_100, v1_opt, v2_opt)
    optimized_solution = run_simulation_with_config(
        optimized_config, simulation, optimized_postprocess, config_type="TwoZoneBohm"
    )

    if optimized_solution:
        print("Optimized simulation completed successfully.")
    else:
        print("Optimized simulation failed.")

if __name__ == "__main__":
    main()
