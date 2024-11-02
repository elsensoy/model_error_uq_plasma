import os
import json
import numpy as np
from scipy.stats import norm
from MCMCIterators.samplers import DelayedRejectionAdaptiveMetropolis
from src.map_nelder_mead import hallthruster_jl_wrapper, config_multilogbohm

# Path to results directory
RESULTS_DIR = os.path.join("results-Nelder-Mead")
initial_guess_path = os.path.join(RESULTS_DIR, "nm_w_2.0_best_initial_guess_result.json")
observed_data_path = os.path.join(RESULTS_DIR, "nm_w_2.0_observed_data_map.json")

# -----------------------------
# 1.TwoZoneBohm Configuration
# -----------------------------
config_spt_100 = config_multilogbohm.copy()
config_spt_100['anom_model'] = 'TwoZoneBohm'

# -----------------------------
# 2. Helper Functions for Saving/Loading Results
# -----------------------------
# Load optimized parameters from JSON
def load_optimized_params(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data.get("v1"), data.get("v2")

def load_json_data(filepath):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {filepath}")
        return None

def save_metadata(metadata, filename="mcmc_metadata.json"):
    """Save metadata to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to {filename}")

def create_specific_config(config):
    specific_config = {
        "Thruster Model": "SPT-100 Hall Thruster",
        "Spatial Resolution and Grid": {
            "num_cells": config.get("num_cells", 100),
            "channel_length": config.get("channel_length", 0.025)
        },
        "Simulation Duration": {
            "duration_s": config.get("duration_s", 1e-3),
            "num_save": config.get("num_save", 100)
        },
        "Magnetic Field and Boundary Conditions": {
            "magnetic_field_file": config.get("magnetic_field_file", "bfield_spt100.csv"),
            "anode_potential": config.get("anode_potential", 300),
            "cathode_potential": config.get("cathode_potential", 0)
        },
        "Additional Model Configurations": {
            "Propellant and Wall Material": {
                "propellant": config.get("propellant", "Xenon"),
                "wall_material": config.get("wall_material", "BNSiO2")
            },
            "Ion and Neutral Temperatures": {
                "ion_temp_K": config.get("ion_temp_K", 1000),
                "neutral_temp_K": config.get("neutral_temp_K", 500),
                "neutral_velocity_m_s": config.get("neutral_velocity_m_s", 150)
            },
            "Anomalous Transport Coefficients": {
                "description": "Anomalous coefficients c(z) adjust the anomalous collision frequency.",
                "equation": r"\nu_{AN}(z) = c(z) \cdot \omega_{ce}(z)"
            }
        }
    }
    return specific_config

# -----------------------------
# 3. Prior and likelihood
# -----------------------------

# Define log-prior function
def prior_logpdf(v1_log, alpha_log):
    prior1 = norm.logpdf(v1_log, loc=np.log10(1/160), scale=np.sqrt(2))
    prior2 = norm.logpdf(alpha_log, loc=np.log10(1/16), scale=np.sqrt(2))
    return prior1 + prior2


# Define log-likelihood function with ion_velocity_weight 2.0
def log_likelihood(simulated_data, observed_data, sigma_thrust=0.08, sigma_discharge=0.08, sigma_velocity=0.08, ion_velocity_weight=2.0):
    log_likelihood_value = 0
    for key, sigma in zip(['thrust', 'discharge_current'], [sigma_thrust, sigma_discharge]):
        if key in observed_data and key in simulated_data:
            residual = np.array(simulated_data[key]) - np.array(observed_data[key])
            log_likelihood_value += -0.5 * np.sum((residual / sigma) ** 2) - len(residual) * np.log(sigma * np.sqrt(2 * np.pi))
        else:
            print(f"Warning: Key '{key}' not found in data.")
    if "ion_velocity" in observed_data and "ion_velocity" in simulated_data:
        residual = np.array(simulated_data["ion_velocity"]) - np.array(observed_data["ion_velocity"])
        log_likelihood_value += -0.5 * np.sum((residual / (sigma_velocity / ion_velocity_weight)) ** 2) - len(residual) * np.log((sigma_velocity / ion_velocity_weight) * np.sqrt(2 * np.pi))
    else:
        print(f"Warning: Ion velocity data not found in simulation or observed data.")
    return log_likelihood_value


# Define posterior function (log of posterior = log-likelihood + log-prior)
def log_posterior(v_log, observed_data, config, ion_velocity_weight=2.0):
    v1_log, alpha_log = v_log
    v1 = 10 ** v1_log
    alpha = 10 ** alpha_log
    v2 = alpha * v1
    simulated_result = hallthruster_jl_wrapper(v1, v2, config, use_time_averaged=True, save_every_n_grid_points=None)
    log_likelihood_value = log_likelihood(simulated_result, observed_data, ion_velocity_weight=ion_velocity_weight)
    log_prior_value = prior_logpdf(v1_log, alpha_log)
    return log_likelihood_value + log_prior_value

# -----------------------------
# 3. MCMC Step
# -----------------------------

def mcmc_inference(logpdf, initial_sample, iterations=1000, lower_bound=-5, upper_bound=3, save_interval=100, save_path=None):
    initial_cov = 0.1 * np.eye(len(initial_sample))
    sampler = DelayedRejectionAdaptiveMetropolis(
        logpdf, initial_sample, initial_cov, adapt_start=10, eps=1e-6,
        sd=2.4**2 / len(initial_sample), interval=10, level_scale=1e-1 
    )
    
    samples = []
    acceptances = 0
    checkpoint_data = {"samples": [], "acceptance_rate": 0}

    for i in range(iterations):
        try:
            result = next(sampler)
            sample = np.clip(result[0], lower_bound, upper_bound)
            accepted = result[1]
            samples.append(sample)
            if accepted:
                acceptances += 1

            # Save samples at intervals and update checkpoint data
            if (i + 1) % save_interval == 0:
                checkpoint_data["samples"] = samples
                checkpoint_data["acceptance_rate"] = acceptances / (i + 1)
                if save_path:
                    np.savetxt(save_path, np.array(samples), delimiter=',')
                    with open(f"{save_path}_checkpoint.json", 'w') as checkpoint_file:
                        json.dump(checkpoint_data, checkpoint_file)
                    print(f"Checkpoint saved at iteration {i + 1}.")

        except np.linalg.LinAlgError as e:
            print(f"Numerical error at iteration {i + 1}: {e}")
            continue
        except Exception as e:
            print(f"Unexpected error at iteration {i + 1}: {e}")
            break 

    # Final save of all samples
    if save_path:
        np.savetxt(save_path, np.array(samples), delimiter=',')
    acceptance_rate = acceptances / iterations
    return np.array(samples), acceptance_rate

# Run MCMC with JSON for optimized params
def run_mcmc_with_optimized_params(json_path, observed_data, config, ion_velocity_weight=2.0, iterations=1000):
    # Load optimized parameters as the initial guess
    v1_opt, v2_opt = load_optimized_params(json_path)
    v_log_initial = [np.log10(v1_opt), np.log10(v2_opt / v1_opt)]  # Convert v1 and alpha to log10 space

    # run MCMC sampling
    print("Running MCMC sampling based on loaded optimized parameters...")
    checkpoint_file = f"mcmc_samples_w_{ion_velocity_weight}_checkpoint.csv"
    
    samples, acceptance_rate = mcmc_inference(
        lambda v_log: log_posterior(v_log, observed_data, config, ion_velocity_weight=ion_velocity_weight),
        v_log_initial,
        iterations=iterations,
        lower_bound=-5,
        upper_bound=3,
        save_interval=100,
        save_path=checkpoint_file
    )
    
    print(f"MCMC sampling complete with acceptance rate: {acceptance_rate:.2f}")
    print(f"Samples saved to {checkpoint_file}")
    
    # Save final samples and metadata for analysis
    final_samples_file = f"final_mcmc_samples_w_{ion_velocity_weight}.csv"
    np.savetxt(final_samples_file, samples, delimiter=',')
    print(f"Final MCMC samples saved to {final_samples_file}")

    metadata = {
        "initial_guess": {"v1": v1_opt, "v2": v2_opt},
        "v_log_initial": v_log_initial,
        "iterations": iterations,
        "acceptance_rate": acceptance_rate,
        "ion_velocity_weight": ion_velocity_weight,
        "checkpoint_file": checkpoint_file,
        "final_samples_file": final_samples_file,
        "model": "TwoZoneBohm",
        "config": create_specific_config(config)
    }
    
    save_metadata(metadata, filename=f"mcmc_metadata_w_{ion_velocity_weight}.json")
    
# Main function
def main():
    initial_params = load_json_data(initial_guess_path)
    if initial_params is None:
        print("Failed to load initial parameters.")
        return

    observed_data = load_json_data(observed_data_path)
    if observed_data is None:
        print("Failed to load observed data.")
        return

    run_mcmc_with_optimized_params(
        json_path=initial_guess_path,
        observed_data=observed_data,
        config=config_spt_100,
        ion_velocity_weight=2.0,
        iterations=5000
    )

if __name__ == "__main__":
    main()








