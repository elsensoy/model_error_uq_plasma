import os
import json
import numpy as np
import pathlib
from scipy.stats import norm
from hall_opt.config.dict import Settings
from hall_opt.config.verifier import load_yaml
from hall_opt.utils.data_loader import load_config
from hall_opt.config.run_model import run_model
from hall_opt.utils.save_posterior import save_metrics, save_posterior
# -----------------------------
# 1. Prior
# -----------------------------
def prior_logpdf(c1_log: float, alpha_log: float) -> float:
    """Computes the log prior probability of the parameters."""
    prior1 = norm.logpdf(c1_log, loc=np.log10(1 / 160), scale=np.sqrt(2))
    prior2 = 0 if 0 < alpha_log <= 2 else -np.inf  # Uniform prior
    return prior1 + prior2

# -----------------------------
# 2. Likelihood
# -----------------------------
def log_likelihood(c_log: list[float], observed_data: dict, settings: Settings, sigma: float = 0.08) -> float:
    """Computes log-likelihood by running the model with MCMC parameters."""

    # Print settings structure for debugging
    print(f"DEBUG: Settings Loaded = {settings}")
    print(f"DEBUG: config_settings = {settings.config_settings}")
    print(f"DEBUG: anom_model = {settings.config_settings.anom_model}")

    # Ensure the correct structure
    if not hasattr(settings, "config_settings"):
        print("ERROR: 'config_settings' missing from settings!")
        return -np.inf

    if "anom_model" not in settings.config_settings.__dict__:
        print("ERROR: 'anom_model' missing from config_settings!")
        return -np.inf

    if "TwoZoneBohm" not in settings.config_settings.anom_model:
        print("ERROR: 'TwoZoneBohm' missing from anom_model!")
        return -np.inf

    # Extract MCMC parameters
    c1_log, alpha_log = c_log
    c1, alpha = np.exp(c1_log), np.exp(alpha_log)
    c2 = c1 * alpha  

    # Update MCMC parameters in settings
    settings.config_settings.anom_model["TwoZoneBohm"]["c1"] = c1
    settings.config_settings.anom_model["TwoZoneBohm"]["c2"] = c2

    print(f"DEBUG: Updated model config: c1={c1}, c2={c2}")

    # Run the model simulation
    solution = run_model(
        settings=settings,
        config_settings=settings.config_settings,  # Use validated settings
        simulation=settings.simulation,
        postprocess=settings.postprocess,
        model_type="TwoZoneBohm"
    )

    if solution is None:
        print("Simulation failed. Returning -np.inf.")
        return -np.inf

    # Extract all metrics
    metrics = solution.get("output", {}).get("average", {})
    if not metrics:
        print("ERROR: Invalid or missing metrics in simulation output.")
        return -np.inf

    # Return all extracted metrics (thrust, discharge current, ion velocity, etc.)
    log_likelihood_value = 0.0
    for key in ["thrust", "discharge_current"]:
        if key in observed_data and key in metrics:
            residual = np.array(metrics[key]) - np.array(observed_data[key])
            log_likelihood_value += -0.5 * np.sum((residual / sigma) ** 2)

    if "ui" in metrics and "ion_velocity" in observed_data:
        simulated_ion_velocity = np.array(metrics["ui"][0], dtype=np.float64)
        observed_ion_velocity = np.array(observed_data["ion_velocity"], dtype=np.float64)

        if simulated_ion_velocity.shape == observed_ion_velocity.shape:
            residual = simulated_ion_velocity - observed_ion_velocity
            log_likelihood_value += -0.5 * np.sum((residual / sigma) ** 2)
        else:
            print("Mismatch in ion_velocity shapes. Penalizing with -np.inf.")
            return -np.inf

    return log_likelihood_value
# -----------------------------
# 3. Posterior (Only Save Posterior Value)
# -----------------------------
def log_posterior(c_log: list[float], observed_data: dict, settings: Settings, yaml_file: str) -> float:

    # Compute Prior
    log_prior_value = prior_logpdf(*c_log)
    if not np.isfinite(log_prior_value):
        return -np.inf  

    log_likelihood_value = log_likelihood(c_log, observed_data, settings, yaml_file)

    if not np.isfinite(log_likelihood_value):
        return -np.inf  

    # Compute Log Posterior
    log_posterior_value = log_prior_value + log_likelihood_value
    return log_posterior_value
