from typing import Dict, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
from scipy.optimize import minimize
from hall_opt.utils.iter_methods import get_next_results_dir
from hall_opt.posterior.statistics import log_posterior
from hall_opt.config.verifier import Settings, get_valid_optimization_method
#assumed 
def run_map_workflow(
    observed_data: Optional[pd.DataFrame], # Or Union[pd.DataFrame, Dict]
    settings: Settings,
) -> Optional[Dict[str, float]]: # Return type: dictionary of non-log params or None
    """Runs the MAP optimization workflow and saves results."""

    # Setup and Objective Function Definitions 
    map_base_dir = Path(settings.output_dir) / "map"
    map_base_dir.mkdir(parents=True, exist_ok=True)
    settings.map.base_dir = get_next_results_dir(str(map_base_dir), "map-results")
    Path(settings.map.base_dir).mkdir(parents=True, exist_ok=True)
    print(f"[INFO] MAP results directory: {settings.map.base_dir}")

    #  Determine Initial Guess ( ASSUMING USER INPUT IS LOG SPACE)) 
    initial_guess_log = settings.map.initial_guess # Assume default initially
    log_source_info = f"default: {initial_guess_log}"

    user_cs = settings.map_settings.initial_cs if settings.map_settings else None
    if user_cs: # Attempt to use user input if provided
        try:
            cs_np = np.asarray(user_cs, dtype=float)
            if np.any(cs_np <= 0): raise ValueError("Values must be positive.") # Basic check
            initial_guess = cs_np.tolist()
            log_source_info = f"user (log converted): {initial_guess}"
        except (ValueError, TypeError) as e:
            # Failed: Keep default, update log message only
            log_source_info = f"default (user value {user_cs} invalid: {e}): {initial_guess}"

    print(f"[INFO] Using initial guess -> {log_source_info}")

    if initial_guess is None:
        print("[ERROR] No valid initial guess found (check default/user input). Aborting.")
        return None
    print(f"[DEBUG] Initial guess (log space): {initial_guess}")
    iteration_counter = [0]; iteration_logs = []

    def bounds_penalty(c_log): # (penalty func)
        penalty = 0.0; log_c1_min, log_c1_max = -5.0, 0.0; log_alpha_min, log_alpha_max = 0.0, 3.0
        if not (log_c1_min <= c_log[0] <= log_c1_max): penalty += 1e6 * (c_log[0] - np.clip(c_log[0], log_c1_min, log_c1_max))**2
        if not (log_alpha_min <= c_log[1] <= log_alpha_max): penalty += 1e6 * (c_log[1] - np.clip(c_log[1], log_alpha_min, log_alpha_max))**2
        return penalty
    
    def neg_log_posterior_with_penalty(c_log): # (objective func)
        c_log_np = np.asarray(c_log)
        try:
            log_prob = log_posterior(c_log=c_log_np, observed_data=observed_data, settings=settings)
            penalty = bounds_penalty(c_log_np)
            if not np.isfinite(log_prob): 
                return np.inf
            return -log_prob + penalty
        except Exception as e: print(f"[ERROR] Failed obj func at {c_log_np}: {e}"); return np.inf

    iteration_log_path = Path(settings.map.base_dir) / "map_iteration_log.json"
    checkpoint_path = Path(settings.map.base_dir) / "map_checkpoint.json"
    result_json_path = Path(settings.map.base_dir) / "optimization_result.json"


    def iteration_callback(current_params_log): # (callback)
        iteration_counter[0] += 1; c1_log, alpha_log = current_params_log
        c1 = np.exp(c1_log) if np.isfinite(c1_log) else float('nan'); alpha = np.exp(alpha_log) if np.isfinite(alpha_log) else float('nan')
        
        current_loss = neg_log_posterior_with_penalty(current_params_log)
        print(f"[MAP] Iter {iteration_counter[0]:<4} -> c1: {c1:.4f}, alpha: {alpha:.4f} | NegLogPost+Penalty: {current_loss:.4e}")
        log_entry = {"iteration": iteration_counter[0], "c1_log": c1_log, "alpha_log": alpha_log, "c1": c1, "alpha": alpha, "neg_log_posterior_penalty": current_loss}
        iteration_logs.append(log_entry)
        try:
            with open(iteration_log_path, "w") as f: json.dump(iteration_logs, f, indent=4)
        except Exception as e: print(f"[WARNING] Failed to save iteration log: {e}")
        try:
            with open(checkpoint_path, "w") as f: json.dump(log_entry, f, indent=4)
        except Exception as e: print(f"[WARNING] Failed to save checkpoint: {e}")


    # --- Determine Optimization Method and Options ---
    # User yaml file overrides 
    yaml_file = settings.general.config_file or "Unknown YAML"; user_method_override = None
    if settings.map_settings and settings.map_settings.algorithm: 
        user_method_override = settings.map_settings.algorithm
    method_to_validate = user_method_override if user_method_override else settings.map.method
    method = get_valid_optimization_method(method_to_validate, source_yaml=yaml_file)

    print(f"[INFO] Using optimization method: {method}")
    options = {"maxfev": settings.map.maxfev, "fatol": settings.map.fatol, "xatol": settings.map.xatol, 'disp': True}

    max_iterations = None
    if settings.map_settings and settings.map_settings.max_iter is not None: 
        max_iterations = settings.map_settings.max_iter
    elif settings.map and settings.map.max_iter is not None: 
        max_iterations = settings.map.max_iter

    if max_iterations is not None: 
        options['maxiter'] = int(max_iterations); print(f"[INFO] Using max_iter: {max_iterations}")
    
    else: 
        print(f"[WARNING] max_iter not defined. Using method default.")
    print(f"[DEBUG] Final options passed to minimize: {options}")

    # --- Run Optimization ---
 
    print(f"[INFO] Starting {method} optimization...")
    try:
        result = minimize(neg_log_posterior_with_penalty, initial_guess_log, method=method, callback=iteration_callback, options=options)
    except Exception as e: 
        print(f"[ERROR] Optimization call failed with exception: {e}"); return None

    # --- Process Results ---
    print("\n--- Optimization Result (SciPy Object) ---"); 
    print(result); 
    print("------------------------------------------\n")

    # --- Prepare data dictionary for saving
    print(f"[INFO] Preparing optimization result details for: {result_json_path}")
    result_dict_to_save = {}
    final_params_nonlog = None

    try:
        # Copy basic attributes
        result_dict_to_save['message'] = getattr(result, 'message', 'N/A')
        result_dict_to_save['success'] = bool(getattr(result, 'success', False))
        result_dict_to_save['status'] = int(getattr(result, 'status', -1))
        result_dict_to_save['fun'] = float(getattr(result, 'fun', float('nan')))
        result_dict_to_save['nit'] = int(getattr(result, 'nit', -1))
        result_dict_to_save['nfev'] = int(getattr(result, 'nfev', -1))

        # Process final parameters (log) and calculate non-log version
        if hasattr(result, 'x') and result.x is not None:
            final_params_log = result.x # Best parameters found (log space)
            result_dict_to_save['x_log'] = final_params_log.tolist()

            try: # Calculate non-log parameters
                 final_params_nonlog = {"c1": float(np.exp(final_params_log[0])), "alpha": float(np.exp(final_params_log[1]))}
                 result_dict_to_save['final_params_nonlog'] = final_params_nonlog
                 print(f"[INFO] Parameters calculated (non-log): c1={final_params_nonlog['c1']:.4f}, alpha={final_params_nonlog['alpha']:.4f}")
            except Exception as calc_e:
                 print(f"[WARNING] Failed to calculate non-log parameters: {calc_e}")
                 result_dict_to_save['final_params_nonlog'] = None; final_params_nonlog = None

        else: # Handle case where result.x is missing
            result_dict_to_save['x_log'] = None; result_dict_to_save['final_params_nonlog'] = None
   
        
        # Convert numpy arrays to lists ONLY if they are actually arrays (not None or error strings)
        result_dict_to_save['initial_guess'] = initial_guess_log.tolist() if isinstance(initial_guess, np.ndarray) else initial_guess
    


        # Handle method-specific outputs (e.g., final_simplex)
        if hasattr(result, 'final_simplex'):
            try:
                simplex_vertices, simplex_values = result.final_simplex
                result_dict_to_save['final_simplex'] = { 'vertices': simplex_vertices.tolist(), 'values': simplex_values.tolist() }
            except Exception as sim_e: result_dict_to_save['final_simplex'] = f"Error serializing: {sim_e}"

        # --- Save the combined results to ONE file ---
        result_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_json_path, "w") as f:
            json.dump(result_dict_to_save, f, indent=4) # Save the dict with all info
        print(f"[INFO] Successfully saved combined optimization results to {result_json_path}")

    except Exception as e:
        print(f"[WARNING] Failed to prepare or save optimization result object to JSON: {e}")
        return None # Return None if saving prep failed

    # --- Report outcome and determine return value ---
    if result.success: print("[INFO] MAP optimization successful (converged).")
    else: print(f"[ERROR] MAP optimization did NOT converge successfully. Status: {result.status}, Message: {result.message}")

    # Return non-log parameters if calculated, otherwise None
    return final_params_nonlog