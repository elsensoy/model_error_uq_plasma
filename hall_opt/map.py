import os
import json
import sys
import numpy as np
from scipy.optimize import minimize
from typing import Dict, Any
from hall_opt.config.load_settings import Settings, extract_anom_model
from hall_opt.utils.statistics import log_posterior


def run_map_workflow(
    observed_data: Dict[str, Any],
    settings: Settings,
    simulation: Dict[str, Any],
    results_dir: str,
):

    # Load initial guess
    try:
        initial_guess_path = settings.optimization_params["map_params"]["map_initial_guess_path"]
        with open(initial_guess_path, "r") as f:
            initial_guess = json.load(f)  # Example: [-2.0, 0.5]
    except Exception as e:
        print(f"Error loading initial guess from {initial_guess_path}: {e}")
        return None, None

    if not isinstance(initial_guess, list) or len(initial_guess) != 2:
        print(f"Invalid initial guess format in {initial_guess_path}. Expected a list of two values.")
        return None, None

    # Extract MAP parameters from settings
    map_params = settings.optimization_params["map_params"]
    method = map_params["method"]
    maxfev = map_params["maxfev"]
    fatol = float(map_params["fatol"])
    xatol = float(map_params["xatol"])
    final_params_file = map_params["final_map_params"]
    iteration_log_file = map_params["iteration_log_file"]

    iteration_counter = [0]  # Tracks iterations
    iteration_logs = []  # Store iteration logs

    print(f"Running MAP optimization with initial guess (log-space): {initial_guess}")

    def bounds_penalty(c_log):
        """
        Apply penalties for parameters outside bounds.
        """
        penalty = 0
        if not (-5 <= c_log[0] <= 0):  # log(c1) bounds
            penalty += (c_log[0] - max(-5, min(c_log[0], 0))) ** 2
        if not (0 <= c_log[1] <= 3):  # log(alpha) bounds
            penalty += (c_log[1] - max(0, min(c_log[1], 3))) ** 2
        return penalty

    def neg_log_posterior_with_penalty(c_log):
        """
        Compute the negative log-posterior with bounds penalties.
        """
        try:
            log_posterior_value = log_posterior(
                c_log, observed_data, settings=settings
            )
            return -log_posterior_value + bounds_penalty(c_log)
        except Exception as e:
            print(f"Error evaluating log-posterior: {e}")
            return np.inf

    def iteration_callback(c_log):
        """
        Callback function to save parameters and log progress at each iteration.
        """
        iteration_counter[0] += 1
        c1_log, alpha_log = c_log
        c1, alpha = np.exp(c1_log), np.exp(alpha_log)

        # Save iteration logs
        iteration_data = {
            "iteration": iteration_counter[0],
            "c1_log": c1_log,
            "alpha_log": alpha_log,
            "c1": c1,
            "alpha": alpha,
        }
        iteration_logs.append(iteration_data)

        # Save the log file after each iteration
        with open(iteration_log_file, "w") as log_file:
            json.dump(iteration_logs, log_file, indent=4)

        # Print progress
        print(f"Iteration {iteration_counter[0]}: c1 = {c1:.4f} (log: {c1_log:.4f}), "
              f"alpha = {alpha:.4f} (log: {alpha_log:.4f})")

    # Perform MAP optimization
    try:
        result = minimize(
            neg_log_posterior_with_penalty,
            initial_guess,
            method=method,
            callback=iteration_callback,
            options={"maxfev": maxfev, "fatol": fatol, "xatol": xatol}
        )
    except Exception as e:
        print(f"Error during optimization: {e}")
        return None, None

    if result.success:
        c1_opt = np.exp(result.x[0])
        alpha_opt = np.exp(result.x[1])
        print(f"Optimization succeeded: c1 = {c1_opt:.4f}, alpha = {alpha_opt:.4f}")

        # Save final parameters
        final_params_path = os.path.join(results_dir, final_params_file)
        optimized_param = {"c1": c1_opt, "alpha": alpha_opt}
        with open(final_params_path, "w") as f:
            json.dump(optimized_param, f, indent=4)
        print(f"Final parameters saved to {final_params_path}")

        return c1_opt, alpha_opt
    else:
        print("MAP optimization failed.")
        return None, None
