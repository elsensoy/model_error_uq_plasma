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
from config.simulation import simulation, config_spt_100, postprocess, config_multilogbohm, update_twozonebohm_config, run_simulation_with_config

# -----------------------------
# 1. Prior
# -----------------------------
def prior_logpdf(v1_log, alpha_log):
    prior1 = norm.logpdf(v1_log, loc=np.log10(1/160), scale=np.sqrt(2))  # Gaussian prior
    prior2 = 0  # Uniform prior for alpha_log
    if alpha_log <= 0 or alpha_log > 2:
        return -np.inf
    return prior1 + prior2

# -----------------------------
# 2. Likelihood
# -----------------------------
def interpolate_simulated_data(observed_z, simulated_z, simulated_u):
    """
    Interpolate simulated data onto the observed data grid.
    """
    return np.interp(observed_z, simulated_z, simulated_u)

def log_likelihood(simulated_data, observed_data, postprocess, sigma=0.08, ion_velocity_weight=2.0):
    """
    Compute the log-likelihood of the observed data given the simulated data.
    """
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

        # Ensure that both observed and simulated data are available for interpolation
        if len(z_observed) > 0 and len(simulated_ion_velocity) > 0 and len(z_simulated) > 0:
            # Interpolate simulated values to match observed z-coordinates
            interpolated_velocity = np.interp(z_observed, z_simulated, simulated_ion_velocity)
            residual = interpolated_velocity - np.array(observed_ion_velocity)
            log_likelihood_value += -0.5 * np.sum((residual / (sigma / ion_velocity_weight)) ** 2)

    return log_likelihood_value

# -----------------------------
# 3. Posterior
# -----------------------------\
def log_posterior(v_log, observed_data, config, simulation, postprocess, ion_velocity_weight=2.0):
    v1_log, alpha_log = v_log
    v1, alpha, v2 = np.exp(v1_log), np.exp(alpha_log), np.exp(v1_log + alpha_log)
    
    # Validate prior
    log_prior_value = prior_logpdf(v1_log, alpha_log)
    if not np.isfinite(log_prior_value):
        print(f"Invalid prior for v1_log={v1_log:.4f}, alpha_log={alpha_log:.4f}. Penalizing with -np.inf.")
        return -np.inf
    
    # Run simulation
    solution = run_simulation_with_config(
        update_twozonebohm_config(config, v1, v2), simulation, postprocess, config_type="TwoZoneBohm"
    )
    if solution is None:
        print("Simulation failed or invalid. Penalizing with -np.inf.")
        return -np.inf
    
    # Extract and validate metrics
    metrics = solution["output"].get("average", {})
    if not metrics:
        print("No metrics found in simulation output. Penalizing with -np.inf.")
        return -np.inf
    if any(not np.isfinite(value) for value in metrics.values() if isinstance(value, (float, int))):
        print(f"Invalid metrics in simulation output: {metrics}. Penalizing with -np.inf.")
        return -np.inf

    # Prepare simulated data (using your preferred logic)
    simulated_data = { 
        "thrust": metrics.get("thrust", 0),
        "discharge_current": metrics.get("discharge_current", 0),
        "ui": metrics.get("ui", []),  # Assuming "ui" is a list of velocities
        "z_coords": metrics.get("z_coords", []), #included for interpolation
    }

    # Compute log-likelihood
    log_likelihood_value = log_likelihood(simulated_data, observed_data, postprocess, ion_velocity_weight=ion_velocity_weight)
    if not np.isfinite(log_likelihood_value):
        print("Invalid likelihood. Penalizing with -np.inf.")
        return -np.inf

    print(f"v1={v1:.4f}, v2={v2:.4f}, Log Prior={log_prior_value:.4f}, Log Likelihood={log_likelihood_value:.4f}")
    return log_prior_value + log_likelihood_value
