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
        return -np.inf  # Reject invalid samples
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
# -----------------------------\
def log_posterior(v_log, observed_data, config, simulation, postprocess, ion_velocity_weight=2.0):
    """
    Compute the log-posterior while respecting the MCMC process by calculating the prior first.
    Penalize invalid outputs or failed simulations separately.
    """
    v1_log, alpha_log = v_log
    v1 = float(np.exp(v1_log))
    alpha = float(np.exp(alpha_log))
    v2 = alpha * v1

    # Prior calculation and validation
    log_prior_value = prior_logpdf(v1_log, alpha_log)
    if not np.isfinite(log_prior_value):
        print(f"Prior is invalid for v1_log={v1_log:.4f}, alpha_log={alpha_log:.4f}. Penalizing with -np.inf.")
        return -np.inf

    # Check physical constraint
    if v2 < v1:
        print(f"Constraint violated: v2={v2:.4f} < v1={v1:.4f}. Penalizing with -np.inf.")
        return -np.inf

    # Update configuration with v1 and v2
    updated_config = update_twozonebohm_config(config, v1, v2)

    # Run simulation
    solution = run_simulation_with_config(updated_config, simulation, postprocess, config_type="TwoZoneBohm")

    # Penalize simulation failure
    if solution is None:
        print("Simulation failed or detected invalid values. Penalizing with -np.inf.")
        return -np.inf

    try:
        # Extract metrics from simulation output
        metrics = solution["output"].get("average", {})
        if not metrics:
            print("No metrics found in simulation output. Penalizing with -np.inf.")
            return -np.inf

        # Check for NaN or Inf in metrics
        if any(
            not np.isfinite(value)
            for key, value in metrics.items()
            if isinstance(value, (float, int))
        ):
            print("Metrics contain NaN or Inf. Penalizing with -np.inf.")
            return -np.inf

        # Prepare simulated data
        simulated_data = {
            "thrust": metrics.get("thrust", 0),
            "discharge_current": metrics.get("discharge_current", 0),
            "ui": metrics.get("ui", []),
        }

        # Compute log-likelihood
        log_likelihood_value = log_likelihood(simulated_data, observed_data, postprocess, ion_velocity_weight)

        # Penalize invalid log-likelihood values
        if not np.isfinite(log_likelihood_value):
            print("Log-likelihood computation returned an invalid value. Penalizing with -np.inf.")
            return -np.inf

        print(f"v1={v1:.4f}, v2={v2:.4f}, Log Prior={log_prior_value:.4f}, Log Likelihood={log_likelihood_value:.4f}")
        return log_prior_value + log_likelihood_value

    except KeyError as e:
        print(f"KeyError during log-posterior calculation: {e}. Penalizing with -np.inf.")
        return -np.inf
    except Exception as e:
        print(f"Unexpected error during log-posterior calculation: {e}. Penalizing with -np.inf.")
        return -np.inf
