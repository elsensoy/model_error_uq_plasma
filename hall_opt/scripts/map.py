import os
import json
import numpy as np
from scipy.optimize import minimize
from typing import Dict, Any
from pathlib import Path
from hall_opt.config.verifier import Settings
from hall_opt.posterior.statistics import log_posterior
from hall_opt.utils.iter_methods import get_next_results_dir

def run_map_workflow(
    observed_data: Dict[str, Any],
    settings: Settings,
):
    # Ensure the correct MAP results directory

    settings.map.base_dir = get_next_results_dir(settings.map.output_dir, "map-results")
    print(f"Using MAP results directory: {settings.map.base_dir}")
    # Load initial guess
    map_settings= settings.map
    # Print entire map configuration for debugging
    print("DEBUG: MAP Configuration:")
    print(settings.map.model_dump())

    # Ensure `map` exists before accessing `initial_cs`
    if not hasattr(settings, "map") or settings.map is None:
        print("ERROR: `map` section is missing in settings!")
        return None

    # Access initial guess safely
    initial_guess = settings.map.initial_guess

    # Debug print
    print(f"DEBUG: Running MAP optimization with initial guess: {initial_guess}")

    # Track iterations
    iteration_counter = [0]
    iteration_logs = []

    def bounds_penalty(c_log):
        penalty = 0
        if not (-5 <= c_log[0] <= 0): 
            penalty += (c_log[0] - max(-5, min(c_log[0], 0))) ** 2 
        if not (0 <= c_log[1] <= 3): 
            penalty += (c_log[1] - max(0, min(c_log[1], 3))) ** 2 
        return penalty

    def neg_log_posterior_with_penalty(c_log):
        """Compute the negative log-posterior with bounds penalties."""
        try:

            loss = -log_posterior(
                c_log=c_log,
                observed_data=observed_data,
                settings=settings,
            ) + bounds_penalty(c_log)  # Add penalty term
            
            # print(f"Evaluating loss at c_log: {c_log}, loss: {loss}")

            return loss # We return a minimized "loss"
        except Exception as e:
            print(f"ERROR:  log-posterior: {e}")
            return np.inf

# Define paths for saving iteration logs and checkpoint
    iteration_log_path = Path(settings.map.base_dir) / "map_iteration_log.json"
    checkpoint_path = Path(settings.map.base_dir) / "map_checkpoint.json"

# Ensure the directory exists before writing files
    iteration_log_path.parent.mkdir(parents=True, exist_ok=True)

    def iteration_callback(c_log):
        """
        Callback function to save parameters and log progress at each iteration.
        """
        iteration_counter[0] += 1  # Increment iteration count
        c1_log, alpha_log = c_log
        c1, alpha = np.exp(c1_log), np.exp(alpha_log)  # Convert log-space to actual values

        # Store iteration data
        iteration_data = {
            "iteration": iteration_counter[0],
            "c1_log": c1_log,
            "alpha_log": alpha_log,
            "c1": c1,
            "alpha": alpha,
        }
        iteration_logs.append(iteration_data)

        # Save logs to JSON file after each iteration
        try:
            with open(iteration_log_path, "w") as log_file:
                json.dump(iteration_logs, log_file, indent=4)
            print(f"Saved iteration {iteration_counter[0]} to {iteration_log_path}")
        except Exception as e:
            print(f"WARNING: Failed to save iteration log: {e}")

        # Save checkpoint (latest parameters)
        try:
            checkpoint_data = {
                "last_iteration": iteration_counter[0],
                "last_c1_log": c1_log,
                "last_alpha_log": alpha_log,
                "last_c1": c1,
                "last_alpha": alpha,
            }
            with open(checkpoint_path, "w") as checkpoint_file:
                json.dump(checkpoint_data, checkpoint_file, indent=4)
            print(f"Checkpoint saved at {checkpoint_path}")
        except Exception as e:
            print(f"WARNING: Failed to save checkpoint: {e}")

        # Print iteration progress
        print(f"Iteration {iteration_counter[0]}: c1 = {c1:.4f} (log: {c1_log:.4f}), "
            f"alpha = {alpha:.4f} (log: {alpha_log:.4f})")

   # Perform MAP optimization
    try:
        result = minimize(
            neg_log_posterior_with_penalty, #--->first argument 
            initial_guess,
            method=map_settings.method,
            callback=iteration_callback,
            options={"maxfev": map_settings.maxfev, "fatol": map_settings.fatol, "xatol": map_settings.xatol}
        )
    except Exception as e:
        print(f"Error during optimization: {e}")
        return None, None

    if result.success:
        optimized_params = {"c1": np.exp(result.x[0]), "alpha": np.exp(result.x[1])}
        with open(map_settings.final_map_params_file, "w") as f:
            json.dump(optimized_params, f, indent=4)
        print(f"Final parameters saved to {map_settings.final_map_params_file}")
        return optimized_params
    else:
        print("MAP optimization failed.")
        return None, None