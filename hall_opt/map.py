import os
import json
import numpy as np
from scipy.optimize import minimize
from typing import Dict, Any
from pathlib import Path
from hall_opt.config.verifier import Settings
from hall_opt.utils.statistics import log_posterior
from hall_opt.utils.iter_methods import get_next_results_dir

def run_map_workflow(
    observed_data: Dict[str, Any],
    settings: Settings,
):
    """
    Runs the MAP optimization workflow using scipy.optimize and saves the last sampled parameters.
    """
    # Load initial guess
    map_settings = settings.map
    # Ensure the correct MAP results directory (e.g., map-results-1/, map-results-2/)
    settings.map.base_dir = get_next_results_dir(settings.map.results_dir, "map-results")
    try:
        with open(map_settings.map_initial_guess_file, "r") as f:
            initial_guess = json.load(f)  # Example: [-2.0, 0.5]
            print(f"Running MAP optimization with initial guess (log-space): {initial_guess}")
    except Exception as e:
        print(f" Error loading initial guess from {map_settings.map_initial_guess_file}: {e}")
        return None

    if not isinstance(initial_guess, list) or len(initial_guess) != 2:
        print(f" Invalid initial guess format in {map_settings.map_initial_guess_file}. Expected a list of two values.")
        return None

    iteration_counter = [0]  # Tracks iterations
    iteration_logs = []  # Store iteration logs

    def neg_log_posterior_with_penalty(c_log):
        """Compute the negative log-posterior with bounds penalties."""
        try:
            return -log_posterior(c_log, observed_data, settings=settings)
        except Exception as e:
            print(f" Error evaluating log-posterior: {e}")
            return np.inf

    def iteration_callback(c_log):
        """Callback function to save parameters and log progress at each iteration."""
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

        # Save iteration log file inside `map-results-N/`
        iteration_log_path = Path(settings.map.base_dir) / "map_iteration_log.json"
        with open(iteration_log_path, "w") as log_file:
            json.dump(iteration_logs, log_file, indent=4)

        print(f" Iteration {iteration_counter[0]}: c1 = {c1:.4f} (log: {c1_log:.4f}), "
              f"alpha = {alpha:.4f} (log: {alpha_log:.4f})")

    # Perform MAP optimization
    try:
        result = minimize(
            neg_log_posterior_with_penalty,
            initial_guess,
            method=map_settings.method,
            callback=iteration_callback,
            options={"maxfev": map_settings.maxfev, "fatol": map_settings.fatol, "xatol": map_settings.xatol}
        )
    except Exception as e:
        print(f" Error during optimization: {e}")
        return None

    if result.success:
        # Extract final MAP sample
        final_map_params = {
            "c1_log": result.x[0],
            "alpha_log": result.x[1],
            "c1": np.exp(result.x[0]),
            "alpha": np.exp(result.x[1]),
            "c2": np.exp(result.x[0]) * np.exp(result.x[1])
        }

        # Save final MAP sample to `final_map_params.json` inside `map-results-N/`
        final_map_params_path = Path(settings.map.base_dir) / "final_map_params.json"
        with open(final_map_params_path, "w") as f:
            json.dump(final_map_params, f, indent=4)

        final_general_map_path = Path(settings.map.final_map_params_file)
        with open(final_general_map_path, "w") as f:
            json.dump(final_map_params, f, indent=4)
    
        print(f"Final MAP parameters saved to {final_map_params_path} and {final_general_map_path}")
        
        return final_map_params
    else:
        print(" MAP optimization failed.")
        return None
