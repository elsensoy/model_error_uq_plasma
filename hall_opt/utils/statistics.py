import os
import json
import numpy as np
import pathlib
import sys
from scipy.optimize import minimize
from scipy.stats import norm
from hall_opt.config.verifier import Settings, extract_anom_model
from hall_opt.config.run_model import run_model

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

    observed_ion_velocity = np.array(observed_data.get("ion_velocity"), dtype=np.float64)

    try:
        twozonebohm_config = extract_anom_model(settings, model_type="TwoZoneBohm")
        twozonebohm_config["anom_model"].update({"c1": c1, "c2": c2})
    except Exception as e:
        print(f"Error updating TwoZoneBohm config: {e}")
        return -np.inf

    solution = run_model(
        settings=settings,
        config=twozonebohm_config,  
        simulation=settings.simulation,     
        postprocess=settings.postprocess,   
        model_type="TwoZoneBohm"
    )

    if solution is None:
        print("Simulation failed. Penalizing with -np.inf.")
        return -np.inf

    metrics = solution["output"].get("average", {})
    if not metrics or any(not np.isfinite(value) for value in metrics.values() if isinstance(value, (float, int))):
        print("Invalid or missing metrics in simulation output.")
        return -np.inf

    simulated_data = {
        "thrust": metrics.get("thrust", 0),
        "discharge_current": metrics.get("discharge_current", 0),
        "ui": metrics.get("ui", []),
    }

    simulated_ion_velocity = np.array(simulated_data["ui"][0], dtype=np.float64)

    print(f"Type of simulated_ion_velocity: {type(simulated_ion_velocity)}, Length: {len(simulated_ion_velocity)}")
    print(f"Type of observed_ion_velocity: {type(observed_ion_velocity)}, Length: {len(observed_ion_velocity)}")

    # or iteration results
    # results_base_dir = settings.general_settings["results_dir"]
    # results_dir = get_next_results_dir(results_base_dir, base_name="mcmc-results")

    # iteration_metrics_dir = os.path.join(results_dir, "iteration_metrics")
    # os.makedirs(iteration_metrics_dir, exist_ok=True)
    
    # metrics_filename = get_next_filename("iteration_metrics", iteration_metrics_dir, ".json")

    iteration_results = {
        "c1": c1,
        "c2": c2,
        "thrust": simulated_data["thrust"],
        "discharge_current": simulated_data["discharge_current"],
        "ion_velocity": simulated_ion_velocity.tolist(),
    }
    # save_results_to_json(iteration_results, metrics_filename)
    # print(f"Saved iteration results to {metrics_filename}")
    
    if len(simulated_ion_velocity) == len(observed_ion_velocity):
        residual = simulated_ion_velocity - observed_ion_velocity
        log_likelihood_value = -0.5 * np.sum((residual / (sigma / settings.general_settings["ion_velocity_weight"])) ** 2)
    else:
        print("Mismatch in ion_velocity lengths. Penalizing with -np.inf.")
        return -np.inf

    return log_likelihood_value

# -----------------------------
# 3. Posterior
# -----------------------------
def log_posterior(
    c_log: list[float], observed_data: dict, settings: Settings
) -> float:

    c1_log, alpha_log = c_log

    # Convert observed data values to numpy arrays if they are lists
    for key in observed_data:
        if isinstance(observed_data[key], list):
            observed_data[key] = np.array(observed_data[key], dtype=np.float64)

    # Compute prior
    log_prior_value = prior_logpdf(c1_log, alpha_log)
    if not np.isfinite(log_prior_value):
        return -np.inf

    # Compute likelihood
    log_likelihood_value = log_likelihood(c_log, observed_data, settings)
    if not np.isfinite(log_likelihood_value):
        return -np.inf

    return log_prior_value + log_likelihood_value
