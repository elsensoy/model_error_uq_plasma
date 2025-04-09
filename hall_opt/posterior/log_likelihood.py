import numpy as np
import csv
from pathlib import Path
from scipy.stats import norm
from ..config.dict import Settings
from ..config.run_model import run_model
from ..utils.save_posterior import save_metrics

# -----------------------------
# 2. Likelihood
# -----------------------------
def log_likelihood(c_log: list[float], observed_data: dict, settings: Settings, sigma: float = 0.08) -> float:
    """Computes log-likelihood by running the model with specified method parameters."""
    # Extract MCMC/MAP parameters
    c1_log, alpha_log = c_log
    c1, alpha = np.exp(c1_log), np.exp(alpha_log)
    c2 = c1 * alpha  

    # Update MCMC parameters in settings
    if "TwoZoneBohm" in settings.config_settings.anom_model:
        settings.config_settings.anom_model["TwoZoneBohm"]["c1"] = c1
        settings.config_settings.anom_model["TwoZoneBohm"]["c2"] = c2
    else:
        print("[ERROR] Missing `TwoZoneBohm` model settings in configuration.")
        return -np.inf

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

    simulated_data = {
        "thrust": metrics.get("thrust", [0]),
        "discharge_current": metrics.get("discharge_current", 0),
        "z_normalized": metrics.get("z", []),
        "ion_velocity": metrics.get("ui", []) 
    }
    # Extract simulated ion velocity
    simulated_ion_velocity = np.array(metrics.get("ui", [0])[0], dtype=np.float64)
    ion_velocity_weight = settings.general.ion_velocity_weight
    log_likelihood_value = 0.0

    for key in ["thrust", "discharge_current"]:
        if key in observed_data and key in simulated_data:
            residual = np.array(simulated_data[key]) - np.array(observed_data[key])
            log_likelihood_value += -0.5 * np.sum((residual / sigma) ** 2)
            
    if "ui" in simulated_data and "ion_velocity" in observed_data:
        simulated_ion_velocity = np.array(simulated_data["ui"][0], dtype=np.float64)
        observed_ion_velocity = np.array(observed_data["ion_velocity"], dtype=np.float64)
        # print(f"Shape of simulated_ion_velocity: {simulated_ion_velocity.shape}")
        # print(f"Shape of observed_ion_velocity: {observed_ion_velocity.shape}")

        if simulated_ion_velocity.shape == observed_ion_velocity.shape:
            residual = simulated_ion_velocity - observed_ion_velocity
            log_likelihood_value += -0.5 * np.sum((residual / (sigma / ion_velocity_weight)) ** 2)
        else:
            print("Mismatch in ion_velocity shapes. Penalizing with -np.inf.")
            return -np.inf

    #  directory for saving metrics map/mcmc
    if settings.run_map:
        output_dir = settings.map.base_dir  
        print("DEBUG: Saving metrics in MAP results directory.")
    elif settings.run_mcmc:
        output_dir = settings.mcmc.base_dir   
        print("DEBUG: Saving metrics in MCMC results directory.")
    else:
        print("DEBUG: No MAP or MCMC executed")
        output_dir = settings.general.results_dir  # Default general directory

    save_metrics(
        settings=settings,
        extracted_metrics=simulated_data,
        output_dir=output_dir
    )
    return log_likelihood_value