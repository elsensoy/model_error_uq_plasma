from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy.optimize import minimize
from hall_opt.utils.iter_methods import get_next_results_dir
from hall_opt.posterior.statistics import log_posterior
from hall_opt.config.verifier import Settings, get_valid_optimization_method

def run_map_workflow(
    observed_data: Optional[pd.DataFrame],
    settings: Settings,
):
    map_base_dir = Path(settings.output_dir) / "map"
    map_base_dir.mkdir(parents=True, exist_ok=True)
    settings.map.base_dir = get_next_results_dir(str(map_base_dir), "map-results")
  
    Path(settings.map.base_dir).mkdir(parents=True, exist_ok=True)
    print(f"[INFO] MAP results directory: {settings.map.base_dir}")

    # Initial guess
    initial_guess = settings.map.initial_guess
    print(f"[DEBUG] Initial guess: {initial_guess}")

    # Logs
    iteration_counter = [0]
    iteration_logs = []

    def bounds_penalty(c_log):
        penalty = 0
        if not (-5 <= c_log[0] <= 0):
            penalty += (c_log[0] - max(-5, min(c_log[0], 0)))**2
        if not (0 <= c_log[1] <= 3):
            penalty += (c_log[1] - max(0, min(c_log[1], 3)))**2
        return penalty

    def neg_log_posterior_with_penalty(c_log):
        try:
            loss = -log_posterior(
                c_log=c_log,
                observed_data=observed_data,
                settings=settings,
            ) + bounds_penalty(c_log)
            return loss
        except Exception as e:
            print(f"[ERROR] Failed to compute log-posterior: {e}")
            return np.inf

    # Iteration logging
    iteration_log_path = Path(settings.map.base_dir) / "map_iteration_log.json"
    checkpoint_path = Path(settings.map.base_dir) / "map_checkpoint.json"

    def iteration_callback(c_log):
        iteration_counter[0] += 1
        c1_log, alpha_log = c_log
        c1, alpha = np.exp(c1_log), np.exp(alpha_log)

        log_entry = {
            "iteration": iteration_counter[0],
            "c1_log": c1_log,
            "alpha_log": alpha_log,
            "c1": c1,
            "alpha": alpha
        }
        iteration_logs.append(log_entry)

        try:
            with open(iteration_log_path, "w") as f:
                json.dump(iteration_logs, f, indent=4)
        except Exception as e:
            print(f"[WARNING] Failed to save iteration log: {e}")

        try:
            with open(checkpoint_path, "w") as f:
                json.dump(log_entry, f, indent=4)
        except Exception as e:
            print(f"[WARNING] Failed to save checkpoint: {e}")

        print(f"[MAP] Iter {iteration_counter[0]} → c1: {c1:.4f}, alpha: {alpha:.4f}")

# Extract source YAML for debug
    yaml_file = settings.general.config_file
    if settings.map_settings:
        print(f"[DEBUG] User map_settings: {settings.map_settings.model_dump()}")
    else:
        print("[DEBUG] No map_settings provided in YAML.")


    # Prioritize user override in `map_settings` (from settings block in YAML)
    user_method = None
    if settings.map_settings and settings.map_settings.algorithm:
        user_method = settings.map_settings.algorithm
        print(f"[DEBUG] User provided optimization method override: {user_method}")
    else:
        user_method = settings.map.method
        print(f"[DEBUG] Using default method from map config: {user_method}")

    # Pass to validation function
    method = get_valid_optimization_method(user_method, source_yaml=yaml_file)


    # Run optimization
    try:
        result = minimize(
            neg_log_posterior_with_penalty,
            initial_guess,
            method=method,
            callback=iteration_callback,
            options={
                "maxfev": settings.map.maxfev,
                "fatol": settings.map.fatol,
                "xatol": settings.map.xatol
            }
        )
    except Exception as e:
        print(f"[ERROR] Optimization failed: {e}")
        return None

    if result.success:
        final_params = {
            "c1": float(np.exp(result.x[0])),
            "alpha": float(np.exp(result.x[1]))
        }
        try:
            with open(settings.map.final_map_params_file, "w") as f:
                json.dump(final_params, f, indent=4)
            print(f"[INFO] Final MAP parameters saved to: {settings.map.final_map_params_file}")
        except Exception as e:
            print(f"[WARNING] Could not save final parameters: {e}")
        return final_params
    else:
        print("[ERROR] MAP optimization failed.")
        return None
