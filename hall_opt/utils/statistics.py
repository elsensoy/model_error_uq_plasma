import os
import json
import numpy as np
import pathlib
import sys
from scipy.optimize import minimize
from scipy.stats import norm

from hall_opt.config.settings_loader import Settings, load_yml_settings
from hall_opt.config.simulation import run_simulation_with_config, update_twozonebohm_config

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
def prior_logpdf(v1_log, alpha_log):
    prior1 = norm.logpdf(v1_log, loc=np.log10(1 / 160), scale=np.sqrt(2))  # Gaussian prior
    prior2 = 0  # Uniform prior for alpha_log
    if alpha_log <= 0 or alpha_log > 2:
        return -np.inf
    return prior1 + prior2

# -----------------------------
# 2. Likelihood
# -----------------------------
def log_likelihood(simulated_data, observed_data, sigma=0.08):
    """
    Compute the log-likelihood of the observed data given the simulated data.
    """
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


def log_posterior(v_log, observed_data, settings):
    v1_log, alpha_log = v_log
    v1, alpha, v2 = np.exp(v1_log), np.exp(alpha_log), np.exp(v1_log + alpha_log)

    log_prior_value = prior_logpdf(v1_log, alpha_log)
    if not np.isfinite(log_prior_value):
        return -np.inf

    solution = run_simulation_with_config(
        update_twozonebohm_config(settings.config_spt_100, v1, v2),
        settings.simulation,
        settings.postprocess,
        config_type="TwoZoneBohm"
    )
    if solution is None:
        return -np.inf

    metrics = solution["output"].get("average", {})
    if not metrics or any(not np.isfinite(value) for value in metrics.values() if isinstance(value, (float, int))):
        return -np.inf

    simulated_data = {
        "thrust": metrics.get("thrust", 0),
        "discharge_current": metrics.get("discharge_current", 0),
        "ui": metrics.get("ui", []),
        "z_coords": metrics.get("z_coords", []),
    }

    return log_prior_value + log_likelihood(simulated_data, observed_data, sigma=0.08)
