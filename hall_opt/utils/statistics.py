import os
import json
import numpy as np
import pathlib
import sys
from scipy.optimize import minimize
from scipy.stats import norm
from hall_opt.config.settings_loader import Settings, load_yml_settings
from hall_opt.config.run_model import run_simulation_with_config

# HallThruster Path Setup
hallthruster_path = "/root/.julia/packages/HallThruster/J4Grt/python"
if hallthruster_path not in sys.path:
    sys.path.append(hallthruster_path)

import hallthruster as het

# Global Settings (loaded once and shared)
settings = None
# -----------------------------
# 1. Prior
# -----------------------------
def prior_logpdf(c1_log, c2_log):
    """
    Compute the prior probability for the given c1_log and c2_log values.
    """
    prior1 = norm.logpdf(c1_log, loc=np.log10(1 / 160), scale=np.sqrt(2))  # Gaussian prior
    prior2 = 0  # Uniform prior for c2_log
    if c2_log <= 0 or c2_log > 2:
        return -np.inf
    return prior1 + prior2

# -----------------------------
# 2. Likelihood
# -----------------------------
def log_likelihood(c_log, observed_data, settings, sigma=0.08):
    """
    Run simulation and compute the log-likelihood of the observed data given the simulated data.
    """
    c1_log, c2_log = c_log
    c1, c2 = np.exp(c1_log), np.exp(c2_log)

    # Update the TwoZoneBohm configuration with the new parameters
    twozonebohm_config = extract_anom_model(settings, model_type="TwoZoneBohm")
    twozonebohm_config["anom_model"].update({"c1": c1, "c2": c2})

    # Run simulation
    solution = run_simulation_with_config(
        config=twozonebohm_config,
        simulation=settings.simulation,
        postprocess=settings.postprocess,
        config_type="TwoZoneBohm",
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
        "z_coords": metrics.get("z_coords", []),
    }

    # Compute log-likelihood
    ion_velocity_weight = settings.ion_velocity_weight
    log_likelihood_value = 0.0

    # Thrust and discharge current components
    for key in ['thrust', 'discharge_current']:
        if key in observed_data and key in simulated_data:
            residual = np.array(simulated_data[key]) - np.array(observed_data[key])
            log_likelihood_value += -0.5 * np.sum((residual / sigma) ** 2)

    # Ion velocity component
    if "ui" in simulated_data and "ion_velocity" in observed_data and "z_coords" in simulated_data:
        simulated_ion_velocity = simulated_data["ui"][0]
        z_simulated = simulated_data["z_coords"]
        observed_ion_velocity = observed_data.get("ion_velocity", [])
        z_observed = observed_data.get("z_normalized", [])

        if len(z_observed) > 0 and len(simulated_ion_velocity) > 0 and len(z_simulated) > 0:
            interpolated_velocity = np.interp(z_observed, z_simulated, simulated_ion_velocity)
            residual = interpolated_velocity - np.array(observed_ion_velocity)
            log_likelihood_value += -0.5 * np.sum((residual / (sigma / ion_velocity_weight)) ** 2)

    return log_likelihood_value

# -----------------------------
# 3. Posterior
# -----------------------------
def log_posterior(c_log, observed_data, settings):
    """
    Compute the posterior value by combining prior and likelihood.
    """
    c1_log, c2_log = c_log

    # Compute prior
    log_prior_value = prior_logpdf(c1_log, c2_log)
    if not np.isfinite(log_prior_value):
        return -np.inf

    # Compute likelihood
    log_likelihood_value = log_likelihood(c_log, observed_data, settings)
    if not np.isfinite(log_likelihood_value):
        return -np.inf

    return log_prior_value + log_likelihood_value
