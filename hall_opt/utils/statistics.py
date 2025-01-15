import os
import json
import numpy as np
import pathlib
import sys
from scipy.optimize import minimize
from scipy.stats import norm
from hall_opt.config.loader import Settings, load_yml_settings, extract_anom_model
from hall_opt.config.run_model import run_simulation_with_config

# HallThruster Path Setup
hallthruster_path = "/home/elida/.julia/packages/HallThruster/tHQQa/python"
if hallthruster_path not in sys.path:
    sys.path.append(hallthruster_path)

import hallthruster as het


# -----------------------------
# 1. Prior
# -----------------------------
def prior_logpdf(c1_log: float, alpha_log: float) -> float:
    # Gaussian prior for c1_log
    prior1 = norm.logpdf(c1_log, loc=np.log10(1 / 160), scale=np.sqrt(2))
    # Uniform prior for alpha_log in range (0, 2]
    if alpha_log <= 0 or alpha_log > 2:
        return -np.inf
    prior2 = 0
    return prior1 + prior2


# -----------------------------
# 2. Likelihood
# -----------------------------
def log_likelihood(
    c_log: list[float], observed_data: dict, settings: Settings, sigma: float = 0.08
) -> float:

    c1_log, alpha_log = c_log
    c1, alpha = np.exp(c1_log), np.exp(alpha_log)
    c2 = c1 * alpha  # Calculate c2 from c1 and alpha

    # Update the TwoZoneBohm configuration with the new parameters
    try:
        twozonebohm_config = extract_anom_model(settings, model_type="TwoZoneBohm")
        twozonebohm_config["anom_model"].update({"c1": c1, "c2": c2})
    except Exception as e:
        print(f"Error updating TwoZoneBohm config: {e}")
        return -np.inf

    # Run the simulation
    solution = run_simulation_with_config(
        settings=settings,
        config=twozonebohm_config,  
        simulation=settings.simulation,      # Pass general simulation settings
        postprocess=settings.postprocess,    # Pass postprocessing settings
        model_type="TwoZoneBohm"
    )
    if solution is None:
        print("Simulation failed. Penalizing with -np.inf.")
        return -np.inf

    # Extract metrics from simulation results
    metrics = solution["output"].get("average", {})
    if not metrics or any(not np.isfinite(value) for value in metrics.values() if isinstance(value, (float, int))):
        print("Invalid or missing metrics in simulation output.")
        return -np.inf

    # Prepare simulated data
    simulated_data = {
        "thrust": metrics.get("thrust", 0),
        "discharge_current": metrics.get("discharge_current", 0),
        "ui": metrics.get("ui", []),
    }

    # Compute log-likelihood
    ion_velocity_weight = settings.general_settings.get("ion_velocity_weight", 1.0)
    log_likelihood_value = 0.0

    # Thrust and discharge current components
    for key in ["thrust", "discharge_current"]:
        if key in observed_data and key in simulated_data:
            residual = np.array(simulated_data[key]) - np.array(observed_data[key])
            log_likelihood_value += -0.5 * np.sum((residual / sigma) ** 2)

    # Ion velocity component
    if "ui" in simulated_data and "ion_velocity" in observed_data:
        simulated_ion_velocity = np.array(simulated_data["ui"][0], dtype=np.float64)
        observed_ion_velocity = np.array(observed_data["ion_velocity"], dtype=np.float64)
        print(f"Shape of simulated_ion_velocity: {simulated_ion_velocity.shape}")
        print(f"Shape of observed_ion_velocity: {observed_ion_velocity.shape}")

        if simulated_ion_velocity.shape == observed_ion_velocity.shape:
            residual = simulated_ion_velocity - observed_ion_velocity
            log_likelihood_value += -0.5 * np.sum((residual / (sigma / ion_velocity_weight)) ** 2)
        else:
            print("Mismatch in ion_velocity shapes. Penalizing with -np.inf.")
            return -np.inf

    return log_likelihood_value


# -----------------------------
# 3. Posterior
# -----------------------------
def log_posterior(
    c_log: list[float], observed_data: dict, settings: Settings
) -> float:

    c1_log, alpha_log = c_log

    # Compute prior
    log_prior_value = prior_logpdf(c1_log, alpha_log)
    if not np.isfinite(log_prior_value):
        return -np.inf

    # Compute likelihood
    log_likelihood_value = log_likelihood(c_log, observed_data, settings)
    if not np.isfinite(log_likelihood_value):
        return -np.inf

    return log_prior_value + log_likelihood_value
