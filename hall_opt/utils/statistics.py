import os
import json
import numpy as np
import pathlib
import sys
from scipy.optimize import minimize
from scipy.stats import norm
from hall_opt.config.verifier import Settings, extract_anom_model
from hall_opt.config.run_model import run_model
from hall_opt.utils.iter_methods import get_next_filename, get_next_results_dir
from hall_opt.utils.save_data import save_results_to_json
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
def log_likelihood(
    c_log: list[float], observed_data: dict, settings: Settings, sigma: float = 0.08
) -> float:

    # Extract Parameters
    c1_log, alpha_log = c_log
    c1, alpha = np.exp(c1_log), np.exp(alpha_log)
    c2 = c1 * alpha  # Compute c2 from c1 and alpha

    # Ensure observed data is present
    if "ion_velocity" not in observed_data or observed_data["ion_velocity"] is None:
        print("ERROR: Observed ion velocity is missing!")
        return -np.inf  # Ensure it returns a float

    observed_ion_velocity = np.array(observed_data["ion_velocity"], dtype=np.float64)

    # Extract simulation configuration
    try:
        twozonebohm_config = extract_anom_model(settings, model_type="TwoZoneBohm")
        twozonebohm_config["anom_model"].update({"c1": c1, "c2": c2})
    except Exception as e:
        print(f"Error updating TwoZoneBohm config: {e}")
        return -np.inf  # Ensure it returns a float

    # Run Model Simulation
    solution = run_model(
        settings=settings,
        config_settings=twozonebohm_config,  
        simulation=settings.simulation,     
        postprocess=settings.postprocess,   
        model_type="TwoZoneBohm"
    )

    if solution is None:
        print("Simulation failed. Returning -np.inf.")
        return -np.inf  # Ensure it returns a float

    # Extract Output Metrics
    metrics = solution.get("output", {}).get("average", {})
    if not metrics or any(not np.isfinite(value) for value in metrics.values() if isinstance(value, (float, int))):
        print("Invalid or missing metrics in simulation output.")
        return -np.inf  # Ensure it returns a float

    simulated_ion_velocity = np.array(metrics.get("ui", [0])[0], dtype=np.float64)

    # Check for Matching Lengths
    if len(simulated_ion_velocity) != len(observed_ion_velocity):
        print("Mismatch in ion_velocity lengths. Penalizing with -np.inf.")
        return -np.inf  # Ensure it returns a float

    residual = simulated_ion_velocity - observed_ion_velocity
    log_likelihood_value = -0.5 * np.sum((residual / (sigma / settings.general.ion_velocity_weight)) ** 2)

    return log_likelihood_value

# -----------------------------
# 3. Posterior
# -----------------------------
def log_posterior(
    c_log: list[float], observed_data: dict, settings: Settings
) -> float:

    c1_log, alpha_log = c_log

    # Convert observed data lists to NumPy arrays
    observed_data = {
        key: np.array(value, dtype=np.float64) if isinstance(value, list) else value
        for key, value in observed_data.items()
    }

    # Compute prior
    log_prior_value = prior_logpdf(c1_log, alpha_log)
    if not np.isfinite(log_prior_value):
        return -np.inf

    # Compute likelihood
    log_likelihood_value = log_likelihood(c_log, observed_data, settings)
    if not np.isfinite(log_likelihood_value):
        return -np.inf

    return log_prior_value + log_likelihood_value
