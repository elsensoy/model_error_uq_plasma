import os
import json
import numpy as np
import pathlib
import argparse
from scipy.optimize import minimize
from hall_opt.config.settings_loader import Settings, load_yml_settings
from hall_opt.config.run_model import run_simulation_with_config
from utils.save_data import save_parameters_linear, save_parameters_log
from utils.statistics import log_posterior 

def run_map_workflow(
    observed_data,
    settings,
    results_dir,
    final_params_file="final_parameters.json",
):
    """
    Run MAP estimation workflow for TwoZoneBohm using c1 and c2 parameters.
    """
    # Load initial guess
    initial_guess_path = settings.initial_guess_path
    try:
        with open(initial_guess_path, 'r') as f:
            initial_guess = json.load(f)  # Example: [-2.0, 0.5]
    except Exception as e:
        print(f"Error loading initial guess from {initial_guess_path}: {e}")
        return None, None

    if not isinstance(initial_guess, list) or len(initial_guess) != 2:
        print(f"Invalid initial guess format in {initial_guess_path}. Expected a list of two values.")
        return None, None

    iteration_counter = [0]  # Tracks iterations

    print(f"Running MAP optimization with initial guess (log-space): {initial_guess}")

    def bounds_penalty(c_log):
        """
        Apply penalties for parameters outside bounds.
        """
        penalty = 0
        if not (-5 <= c_log[0] <= 0):  # log(c1) bounds
            penalty += (c_log[0] - max(-5, min(c_log[0], 0))) ** 2
        if not (0 <= c_log[1] <= 3):  # log(c2) bounds
            penalty += (c_log[1] - max(0, min(c_log[1], 3))) ** 2
        return penalty

    def neg_log_posterior_with_penalty(c_log):
        """
        Compute the negative log-posterior with bounds penalties.
        """
        log_posterior_value = log_posterior(
            c_log, observed_data, settings=settings
        )
        return -log_posterior_value + bounds_penalty(c_log)

    def iteration_callback(c_log):
        """
        Callback function to save parameters at each iteration.
        """
        iteration_counter[0] += 1

        # Convert log-space values to linear space
        c1_linear = float(np.exp(c_log[0]))
        c2_linear = float(np.exp(c_log[1]))

        # Save parameters in linear space
        save_parameters_linear(
            iteration=iteration_counter[0],
            c1=c1_linear,
            c2=c2_linear,
            results_dir=results_dir,
            filename="parameters_linear.json"
        )

        # Save parameters in log space
        save_parameters_log(
            iteration=iteration_counter[0],
            c1_log=c_log[0],
            c2_log=c_log[1],
            results_dir=results_dir,
            filename="parameters_log.json"
        )

        # Print status
        print(f"Iteration {iteration_counter[0]}: c1 = {c1_linear:.4f} (log: {c_log[0]:.4f}), "
              f"c2 = {c2_linear:.4f} (log: {c_log[1]:.4f})")

    # Perform MAP optimization
    result = minimize(
        neg_log_posterior_with_penalty,
        initial_guess,
        method="Nelder-Mead",
        callback=iteration_callback,
        options={"maxfev": 5000, "fatol": 1e-3, "xatol": 1e-3}
    )

    if result.success:
        c1_opt = np.exp(result.x[0])
        c2_opt = np.exp(result.x[1])
        print(f"Optimization succeeded: c1 = {c1_opt:.4f}, c2 = {c2_opt:.4f}")

        # Save final parameters
        final_params_path = os.path.join(results_dir, final_params_file)
        final_params = {"c1": c1_opt, "c2": c2_opt}
        with open(final_params_path, 'w') as f:
            json.dump(final_params, f, indent=4)
        print(f"Final parameters saved to {final_params_path}")

        return c1_opt, c2_opt
    else:
        print("MAP optimization failed.")
        return None, None
