import json
import math
import os
import sys
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

# HallThruster Path Setup
hallthruster_path = "/root/.julia/packages/HallThruster/J4Grt/python"
if hallthruster_path not in sys.path:
    sys.path.append(hallthruster_path)

import hallthruster as het

# -----------------------------
# 1. Prior
# -----------------------------
def prior_logpdf(v1_log, alpha_log):
    """Gaussian prior on log10(c1) and uniform prior on log10(alpha)."""
    prior1 = norm.logpdf(v1_log, loc=np.log10(1/160), scale=np.sqrt(2))
    prior2 = 0 
    # if 0 < alpha_log <= 2 
    # else -np.inf  # Uniform prior on log10(alpha)
    return prior1 + prior2
# -----------------------------
# 2. Likelihood
# -----------------------------
def log_likelihood(simulated_data, observed_data, postprocess, sigma=0.08, ion_velocity_weight=2.0):
    """Compute the log-likelihood of the observed data given the simulated data."""
    log_likelihood_value = 0

    # Thrust and discharge current
    for key in ['thrust', 'discharge_current']:
        if key in observed_data and key in simulated_data:
            residual = np.array(simulated_data[key]) - np.array(observed_data[key])
            log_likelihood_value += -0.5 * np.sum((residual / sigma) ** 2)

    # Ion velocity (first array element for simulated 'ui')
    if "ui" in simulated_data and "ion_velocity" in observed_data:
        simulated_ion_velocity = np.array(simulated_data["ui"][0], dtype=np.float64)
        observed_ion_velocity = np.array(observed_data["ion_velocity"], dtype=np.float64)
        if simulated_ion_velocity.shape == observed_ion_velocity.shape:
            residual = simulated_ion_velocity - observed_ion_velocity
            # print(f"DEBUG: sigma = {sigma} (type: {type(sigma)})")
            # print(f"DEBUG: ion_velocity_weight = {ion_velocity_weight} (type: {type(ion_velocity_weight)})")

            log_likelihood_value += -0.5 * np.sum((residual / (sigma / ion_velocity_weight)) ** 2)
    return log_likelihood_value

# -----------------------------
# 3. Posterior
# -----------------------------
def log_posterior(v_log, observed_data, config, simulation, postprocess, ion_velocity_weight=2.0):
    """Compute the log-posterior using simulation results and observed data."""
    ion_velocity_weight=2.0
    v1_log, alpha_log = v_log
    v1 = np.exp(v1_log)
    alpha = np.exp(alpha_log)
    v2 = alpha * v1

    #Enforce physical constraint
    if v2 < v1:
        return -np.inf

    # Update configuration and run simulation
    config["anom_model"]["c1"], config["anom_model"]["c2"] = v1, v2
    input_data = {"config": config, "simulation": simulation, "postprocess": postprocess}
    solution = het.run_simulation(input_data)

    # if solution["output"]["retcode"] != "success":
    #     return -np.inf

    # Extract metrics
    metrics = solution["output"].get("average", {})
    simulated_data = {
        "thrust": metrics.get("thrust", 0),
        "discharge_current": metrics.get("discharge_current", 0),
        "ui": metrics.get("ui", [[0]])  # Default to empty ui array
    }

    # Compute posterior
    log_prior_value = prior_logpdf(v1_log, alpha_log)
    log_likelihood_value = log_likelihood(simulated_data, observed_data, postprocess, sigma=0.08, ion_velocity_weight=ion_velocity_weight)
    return log_prior_value + log_likelihood_value
