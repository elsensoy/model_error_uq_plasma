import juliacall
import tempfile
import json
import math
import os
import sys
import time
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

hallthruster_path = "/root/.julia/packages/HallThruster/J4Grt/python"
if hallthruster_path not in sys.path:
    sys.path.append(hallthruster_path)

import hallthruster as het
# -----------------------------
# 2. Prior and Posterior
# -----------------------------
def prior_logpdf(v1_log, alpha_log):
    # Gaussian prior on log10(c1)
    prior1 = norm.logpdf(v1_log, loc=np.log10(1/160), scale=np.sqrt(2))
    prior2 = 0  # log(1) for a uniform distribution in valid range
    # Uniform prior on log10(alpha) in [0, 2]
    if alpha_log <= 0 or alpha_log > 2:
        print(f"Invalid prior: log10(alpha)={alpha_log} is out of range [0, 2].")
        return -np.inf  # Reject invalid samples

    return prior1 + prior2


#     return log_likelihood_value
def log_likelihood(simulated_data, observed_data, postprocess, sigma=0.08, ion_velocity_weight=2.0):
    """Compute the log-likelihood of the observed data given the simulated data."""
    log_likelihood_value = 0

    # Thrust and discharge current
    for key in ['thrust', 'discharge_current']:
        if key in observed_data and key in simulated_data:
            simulated_metric = np.array(simulated_data[key])
            observed_metric = np.array(observed_data[key])
            residual = simulated_metric - observed_metric
            log_likelihood_value += -0.5 * np.sum((residual / sigma) ** 2)
        else:
            print(f"Warning: Key '{key}' not found in data.")

    # Ion velocity (using 'ui' for simulated_data)
    if "ui" in simulated_data and "ion_velocity" in observed_data:
        simulated_ion_velocity = np.array(simulated_data["ui"])
        observed_ion_velocity = np.array(observed_data["ion_velocity"])

        if simulated_ion_velocity.shape == observed_ion_velocity.shape:
            residual = simulated_ion_velocity - observed_ion_velocity
            log_likelihood_value += -0.5 * np.sum((residual / (sigma / ion_velocity_weight)) ** 2)
        else:
            print("Shapes are not compatible for subtraction.")
    else:
        print(f"Warning: Ion velocity data not found in simulation or observed data.")

    return log_likelihood_value

def log_posterior(v_log, observed_data, config, simulation, postprocess, ion_velocity_weight=2.0):
    """
    Compute the log-posterior using simulation results and observed data.
    """
    v1_log, alpha_log = v_log
    v1 = float(np.exp(v_log[0]))
    alpha = float(np.exp(v_log[1]))
    v2 = alpha * v1

    # Enforce physical constraint
    if v2 < v1:
        return -np.inf

    try:
        # Update the anomaly model in the configuration
        config["anom_model"]["c1"] = v1
        config["anom_model"]["c2"] = v2

        # Prepare input data for the simulation
        input_data = {"config": config, "simulation": simulation, "postprocess": postprocess}

        # Run the simulation
        solution = het.run_simulation(input_data)

        # Check simulation success
        if solution["output"]["retcode"] != "success":
            print(f"Simulation failed with retcode: {solution['output']['retcode']}")
            return -np.inf

        # Extract time-averaged metrics from the simulation output
        metrics = solution["output"].get("average", None)
        if not metrics:
            print("No 'average' key found in simulation output.")
            return -np.inf

        # Prepare simulated data for likelihood calculation
        simulated_data = {
            "thrust": metrics.get("thrust", 0),
            "discharge_current": metrics.get("discharge_current", 0),
            "ion_velocity": metrics.get("ui", [])
        }

        # Validate ion velocity shapes
        if "ion_velocity" in observed_data and simulated_data["ion_velocity"]:
            simulated_ion_velocity = np.array(simulated_data["ion_velocity"])
            observed_ion_velocity = np.array(observed_data["ion_velocity"])
            if simulated_ion_velocity.shape != observed_ion_velocity.shape:
                print(f"Shape mismatch: simulated {simulated_ion_velocity.shape}, observed {observed_ion_velocity.shape}")
                return -np.inf

        # Compute log-posterior
        log_prior_value = prior_logpdf(v1_log, alpha_log)
        log_likelihood_value = log_likelihood(simulated_data, observed_data, postprocess, ion_velocity_weight)
        log_posterior_value = log_prior_value + log_likelihood_value

        return log_posterior_value

    except Exception as e:
        print(f"Error during log-posterior calculation: {e}")
        return -np.inf

    v1_log, alpha_log = v_log
    v1 = float(np.exp(v_log[0]))
    alpha = float(np.exp(v_log[1]))
    v2 = alpha * v1

    # Enforce physical constraint
    if v2 < v1:
        return -np.inf

    try:
        # Update the anomaly model in the configuration
        config["anom_model"]["c1"] = v1
        config["anom_model"]["c2"] = v2

        # Prepare input data for the simulation
        input_data = {"config": config, "simulation": simulation, "postprocess": postprocess}

        # Run the simulation
        solution = het.run_simulation(input_data)

        # Check simulation success
        if solution["output"]["retcode"] != "success":
            print(f"Simulation failed with retcode: {solution['output']['retcode']}")
            return -np.inf

        # Extract time-averaged metrics from the simulation output
        metrics = solution["output"].get("average", {})
        if not metrics:
            print(f"No average metrics found in output: {solution['output']}")
            return -np.inf

        # Prepare simulated data for likelihood calculation
        simulated_data = {
            "thrust": metrics.get("thrust", 0),
            "discharge_current": metrics.get("discharge_current", 0),
            "ion_velocity": averaged_metrics["ui"][0]  
        }

        # Validate shapes of ion_velocity if present
        if "ion_velocity" in observed_data and simulated_data["ion_velocity"]:
            simulated_ion_velocity = np.array(simulated_data["ion_velocity"])
            observed_ion_velocity = np.array(observed_data["ion_velocity"])
            if simulated_ion_velocity.shape != observed_ion_velocity.shape:
                print(f"Shape mismatch: simulated {simulated_ion_velocity.shape}, observed {observed_ion_velocity.shape}")
                return -np.inf

        # Compute log-posterior
        log_prior_value = prior_logpdf(v1_log, alpha_log)
        log_likelihood_value = log_likelihood(simulated_data, observed_data, postprocess, ion_velocity_weight=ion_velocity_weight)
        log_posterior_value = log_prior_value + log_likelihood_value

        return log_posterior_value

    except Exception as e:
        print(f"Error during log-posterior calculation: {e}")
        return -np.inf