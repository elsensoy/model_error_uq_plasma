import os
import json
import numpy as np
from scipy.optimize import minimize
from typing import Dict, Any
from pathlib import Path
from ..config.verifier import Settings
from ..posterior.statistics import log_posterior
from ..utils.iter_methods import get_next_results_dir
from ..utils.iteration_logger import iteration_callback


def run_map_workflow(
    observed_data: Dict[str, Any],
    settings: Settings,
    yaml_file: str  # Pass YAML config file path
):
    #  the correct MAP results directory

    settings.map.base_dir = get_next_results_dir(settings.map.results_dir, "map-results")
    print(f"Using MAP results directory: {settings.map.base_dir}")
    # Load initial guess
    map_settings= settings.map
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
                yaml_file=yaml_file
            ) + bounds_penalty(c_log)  # Add penalty term
            
            # print(f"Evaluating loss at c_log: {c_log}, loss: {loss}")

            return loss # We return a minimized "loss"
        
        except Exception as e:
            print(f"ERROR: Failed to evaluate log-posterior: {e}")
            return np.inf

# # Define paths for saving iteration logs and checkpoint
    iteration_log_path = Path(settings.map.base_dir) / "map_iteration_log.json"
    checkpoint_path = Path(settings.map.base_dir) / "map_checkpoint.json"

#  the directory exists before writing files
    iteration_log_path.parent.mkdir(parents=True, exist_ok=True)

   # Perform MAP optimization
    try:
        result = minimize(
            neg_log_posterior_with_penalty,
            initial_guess,
            method=map_settings.method,
            callback=lambda c_log: iteration_callback(
                c_log, iteration_counter, iteration_logs, iteration_log_path, checkpoint_path
            ),
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