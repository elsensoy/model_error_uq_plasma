import os
import json
import numpy as np
import pathlib
import argparse
from scipy.optimize import minimize
from config.settings_loader import Settings, load_yml_settings
from utils.save_data import save_parameters_linear, save_parameters_log
from config.simulation import update_twozonebohm_config, run_simulation_with_config
from utils.statistics import log_posterior 


def run_map_workflow(
    observed_data,
    settings,
    config_spt_100,
    simulation,
    postprocess,
    results_dir,
    final_params_file="final_parameters.json",
):
    """
    Run MAP estimation workflow for TwoZoneBohm.
    """
    initial_guess_path = settings.initial_guess_path
    try:
        with open(initial_guess_path, 'r') as f:
            initial_guess = json.load(f)  # [-2.0, 0.5]
    except Exception as e:
        print(f"Error loading initial guess from {initial_guess_path}: {e}")
        return None, None

    if not isinstance(initial_guess, list) or len(initial_guess) != 2:
        print(f"Invalid initial guess format in {initial_guess_path}. Expected a list of two values.")
        return None, None

    iteration_counter = [0]  # Tracks iterations

    print(f"Running MAP optimization with initial guess (log-space): {initial_guess}")

    def bounds_penalty(v_log):
        """
        Apply penalties for parameters outside bounds.
        """
        penalty = 0
        if not (-5 <= v_log[0] <= 0):  # log(v1) bounds
            penalty += (v_log[0] - max(-5, min(v_log[0], 0))) ** 2
        if not (0 <= v_log[1] <= 3):  # log(alpha) bounds
            penalty += (v_log[1] - max(0, min(v_log[1], 3))) ** 2
        return penalty

    def neg_log_posterior_with_penalty(v_log):
        """
        Compute the negative log-posterior with bounds penalties.
        """
        log_posterior_value = log_posterior(
            v_log, observed_data,settings=settings
        )
        return -log_posterior_value + bounds_penalty(v_log)

    def iteration_callback(v_log):
        """
        Callback function to save parameters at each iteration.
        """
        iteration_counter[0] += 1

        # Convert log-space values to linear space
        v1_linear = float(np.exp(v_log[0]))
        alpha_linear = float(np.exp(v_log[1]))

        # Save parameters in linear space
        save_parameters_linear(
            iteration=iteration_counter[0],
            v1=v1_linear,
            alpha=alpha_linear,
            results_dir=results_dir,
            filename="parameters_linear.json"
        )

        # Save parameters in log space
        save_parameters_log(
            iteration=iteration_counter[0],
            v1_log=v_log[0],
            alpha_log=v_log[1],
            results_dir=results_dir,
            filename="parameters_log.json"
        )

        # Print status
        print(f"Iteration {iteration_counter[0]}: v1 = {v1_linear:.4f} (log: {v_log[0]:.4f}), "
              f"alpha = {alpha_linear:.4f} (log: {v_log[1]:.4f})")

    # Perform MAP optimization
    result = minimize(
        neg_log_posterior_with_penalty,
        initial_guess,
        method="Nelder-Mead",
        callback=iteration_callback,
        options={"maxfev": 5000, "fatol": 1e-3, "xatol": 1e-3}
    )

    if result.success:
        v1_opt = np.exp(result.x[0])
        alpha_opt = np.exp(result.x[1])
        print(f"Optimization succeeded: v1 = {v1_opt:.4f}, alpha = {alpha_opt:.4f}")

        # Save final parameters
        final_params_path = os.path.join(results_dir, final_params_file)
        final_params = {"v1": v1_opt, "alpha": alpha_opt}
        with open(final_params_path, 'w') as f:
            json.dump(final_params, f, indent=4)
        print(f"Final parameters saved to {final_params_path}")

        return v1_opt, alpha_opt
    else:
        print("MAP optimization failed.")
        return None, None
