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
    yaml_file: str  # Pass YAML config file path
):
    """
    Runs the MAP optimization workflow using `scipy.optimize` and saves the last sampled parameters.
    """

    # Ensure the correct MAP results directory
    settings.map.base_dir = get_next_results_dir(settings.map.results_dir, "map-results")
    print(f"Using MAP results directory: {settings.map.base_dir}")

    # Load initial guess
    try:
        with open(settings.map.map_initial_guess_file, "r") as f:
            initial_guess = json.load(f)
            print(f"Running MAP optimization with initial guess: {initial_guess}")
    except Exception as e:
        print(f"ERROR: Failed to load initial guess: {e}")
        return None

    if not isinstance(initial_guess, list) or len(initial_guess) != 2:
        print(f"ERROR: Invalid initial guess format.")
        return None

    # Track iterations
    iteration_counter = [0]
    iteration_logs = []

    # File paths for saving MAP samples
    iteration_log_path = Path(settings.map.base_dir) / "map_iteration_log.json"
    checkpoint_path = Path(settings.map.base_dir) / "map_checkpoint.json"

    def neg_log_posterior_with_penalty(c_log):
        """Compute the negative log-posterior with bounds penalties."""
        try:
            return -log_posterior(
                c_log=c_log,
                observed_data=observed_data,
                settings=settings,  # Pass full settings
                yaml_file=yaml_file  # Pass YAML file path
            )
        except Exception as e:
            print(f"ERROR: Failed to evaluate log-posterior: {e}")
            return np.inf

    def iteration_callback(c_log):
        """Callback function to save parameters and log progress at each iteration."""
        iteration_counter[0] += 1
        c1_log, alpha_log = c_log
        c1, alpha = np.exp(c1_log), np.exp(alpha_log)
        c2 = c1 * alpha

        # Save each iteration in the log list
        iteration_data = {
            "iteration": iteration_counter[0],
            "c1_log": c1_log,
            "alpha_log": alpha_log,
            "c1": c1,
            "c2": c2,
        }
        iteration_logs.append(iteration_data)

        # Save checkpoint every 5 iterations
        if iteration_counter[0] % 5 == 0:
            try:
                with open(checkpoint_path, "w") as checkpoint_file:
                    json.dump({"latest_iteration": iteration_counter[0], "samples": iteration_logs}, checkpoint_file, indent=4)
                print(f"Checkpoint saved at iteration {iteration_counter[0]}")
            except Exception as e:
                print(f"ERROR: Failed to save checkpoint: {e}")

        print(f"Iteration {iteration_counter[0]}: c1 = {c1:.4f}, c2 = {c2:.4f}")

    # Perform MAP optimization
    try:
        result = minimize(
            neg_log_posterior_with_penalty,
            initial_guess,
            method=settings.map.method,
            callback=iteration_callback,
            options={"maxfev": settings.map.maxfev, "fatol": settings.map.fatol, "xatol": settings.map.xatol}
        )
    except Exception as e:
        print(f"ERROR: Optimization failed: {e}")
        return None

    # Save all iterations at the end (Ensures no missing iterations)
    try:
        with open(iteration_log_path, "w") as log_file:
            json.dump(iteration_logs, log_file, indent=4)
        print(f"All iterations saved to {iteration_log_path}")
    except Exception as e:
        print(f"ERROR: Failed to save final iteration logs: {e}")

    if result.success:
        # Extract final MAP sample
        final_map_params = {
            "c1_log": result.x[0],
            "alpha_log": result.x[1],
            "c1": np.exp(result.x[0]),
            "alpha": np.exp(result.x[1]),
            "c2": np.exp(result.x[0]) * np.exp(result.x[1])
        }

        # Save final MAP sample inside `map-results-N/`
        final_map_params_path = Path(settings.map.base_dir) / "final_map_params.json"
        with open(final_map_params_path, "w") as f:
            json.dump(final_map_params, f, indent=4)

        # Save final MAP sample to global results
        final_general_map_path = Path(settings.map.final_map_params_file)
        with open(final_general_map_path, "w") as f:
            json.dump(final_map_params, f, indent=4)

        print(f"Final MAP parameters saved to:")
        print(f"   - {final_map_params_path}")
        print(f"   - {final_general_map_path}")

        return final_map_params
    else:
        print("MAP optimization failed.")
        return None
