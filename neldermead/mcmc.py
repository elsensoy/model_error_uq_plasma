import os
import json
import numpy as np
from scipy.stats import norm
from MCMCIterators.samplers import DelayedRejectionAdaptiveMetropolis
from neldermead.map_nelder_mead import hallthruster_jl_wrapper, config_multilogbohm,  run_simulation, run_multilogbohm_simulation


# MCMC results directory path
results_dir = os.path.join("..", "results-mcmc")
# Path to results directory
RESULTS_NELDERMEAD = os.path.join("..", "results-Nelder-Mead")

initial_guess_path = os.path.join(RESULTS_NELDERMEAD, "best_initial_guess_w_2_0.json")

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

def get_next_filename(base_filename, directory=results_dir, extension=".csv"):
    """
    Generate the next available filename with an incremented suffix in the specified directory.
    """
    i = 1
    # Ensure the file path includes the directory and extension
    full_path = os.path.join(directory, f"{base_filename}_{i}{extension}")
    while os.path.exists(full_path):
        i += 1
        full_path = os.path.join(directory, f"{base_filename}_{i}{extension}")
    return full_path

def save_metadata(metadata, filename="mcmc_metadata.json", directory=results_dir):
    """Save metadata to a JSON file with automatic filename incrementing."""
    # Use get_next_filename to determine a unique path in the directory
    full_path = get_next_filename(filename.split('.')[0], directory=directory, extension=".json")
    
    with open(full_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to {full_path}")


def subsample_data(data, step=10):
    """subsample the data by taking every nth element."""
    if isinstance(data, list):
        return data[::step]  # every nth element from the list
    return data 
def save_results_to_json(result_dict, filename, save_every_n_grid_points=10, subsample_for_saving=True):
    """
    save the results as a JSON file, ensure the directory exists.
    subsample only when saving and keep original data untouched for processing.
    """
    spatial_keys = ['ion_velocity', 'z_normalized']
    
    # create a copy to avoid modifying the original result_dict in memory
    result_dict_copy = result_dict.copy()

    for key in spatial_keys:
        if key in result_dict_copy:
            print(f"Original {key} data shape: {np.array(result_dict_copy[key]).shape}")
            
            # Subsample only for saving, if enabled
            if subsample_for_saving:
                if key == 'z_normalized' and len(result_dict_copy[key]) <= save_every_n_grid_points:
                    print(f"{key} already subsampled. Skipping subsampling.")
                else:
                    if key == 'z_normalized':
                        result_dict_copy[key] = subsample_data(result_dict_copy[key], save_every_n_grid_points)
                    else:
                        result_dict_copy[key] = [subsample_data(sublist, save_every_n_grid_points) for sublist in result_dict_copy[key]]
            
            print(f"Subsampled {key} data shape for saving: {np.array(result_dict_copy[key]).shape}")

    # Ensure the results directory exists
    results_dir = os.path.join("..", "results-mcmc")
    os.makedirs(results_dir, exist_ok=True)  # Create directory if it doesn't exist

    # Define the full path for the JSON file
    result_file_path = os.path.join(results_dir, filename)

    # Save the subsampled results to the file
    try:
        with open(result_file_path, 'w') as json_file:
            json.dump(result_dict_copy, json_file, indent=4)
        print(f"Results successfully saved to {result_file_path}")
    except Exception as e:
        print(f"Failed to save the results: {e}")

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



def log_likelihood(simulated_data, observed_data, sigma=0.08, ion_velocity_weight=10.0):
    """Compute the log-likelihood of the observed data given the simulated data."""
    log_likelihood_value = 0

    # DEBUG:Check the keys in the simulated and observed data
    print("Keys in simulated_data:", simulated_data.keys())
    print("Keys in observed_data:", observed_data.keys())

    # Thrust and discharge current are 1D arrays
    for key in ['thrust', 'discharge_current']:
        if key in observed_data and key in simulated_data:
            simulated_metric = np.array(simulated_data[key])
            observed_metric = np.array(observed_data[key])
            residual = simulated_metric - observed_metric
            log_likelihood_value += -0.5 * np.sum((residual / sigma) ** 2)
        else:
            print(f"Warning: Key '{key}' not found in data.")

    # ion velocity is 2D (space and time averaged), so we apply a lower weight
    if "ion_velocity" in observed_data and "ion_velocity" in simulated_data:
        simulated_ion_velocity = np.array(simulated_data["ion_velocity"])
        observed_ion_velocity = np.array(observed_data["ion_velocity"])

        # DEBUG: Print the shapes of both arrays
        # print(f"Shape of simulated_ion_velocity: {simulated_ion_velocity.shape}")
        # print(f"Shape of observed_ion_velocity: {observed_ion_velocity.shape}")

       
        if simulated_ion_velocity.shape == observed_ion_velocity.shape:
            residual = simulated_ion_velocity - observed_ion_velocity
            log_likelihood_value += -0.5 * np.sum((residual / (sigma / ion_velocity_weight)) ** 2)
        else:
            print("Shapes are not compatible for subtraction.")
    else:
        print(f"Warning: Ion velocity data not found in simulation or observed data.")

    return log_likelihood_value

def log_posterior(v_log, observed_data, config, ion_velocity_weight=2.0):
    v1_log, alpha_log = v_log
    v1 = 10 ** v1_log
    alpha = 10 ** alpha_log
    v2 = alpha * v1

    # Run the simulation with given parameters
    simulated_result = hallthruster_jl_wrapper(v1, v2, config, use_time_averaged=True, save_every_n_grid_points=None)

    # Convert all lists to numpy arrays
    for key in simulated_result:
        simulated_result[key] = np.array(simulated_result[key])

    # DEBUG: Confirm the conversion has taken effect
    # print("\nDEBUG: After conversion in log_posterior...")
    # for key, value in simulated_result.items():
    #     print(f"Key: {key}, Type: {type(value)}, Shape: {value.shape if hasattr(value, 'shape') else 'Not an array'}")

    # Calculate log-likelihood and log-prior
    log_likelihood_value = log_likelihood(simulated_result, observed_data, ion_velocity_weight=ion_velocity_weight)
    log_prior_value = prior_logpdf(v1_log, alpha_log)
    return log_likelihood_value + log_prior_value

# -----------------------------
# 3. MCMC Step
# -----------------------------

# Define the results directory path
def mcmc_inference(logpdf, initial_sample, iterations=100, lower_bound=-5, upper_bound=3, save_interval=10, base_path="mcmc_results"):
    # Ensure initial_sample is a numpy array
    initial_sample = np.array(initial_sample)
   
    print("\nDEBUG: Initial sample type and shape:")
    print(f"Type: {type(initial_sample)}, Shape: {initial_sample.shape}")

    initial_cov = 0.05 * np.eye(len(initial_sample))
    
    # Initialize the sampler
    sampler = DelayedRejectionAdaptiveMetropolis(
        logpdf, initial_sample, initial_cov, adapt_start=10, eps=1e-6,
        sd=2.4**2 / len(initial_sample), interval=10, level_scale=1e-1 
    )
    
    samples = []
    acceptances = 0
    checkpoint_data = {"samples": [], "acceptance_rate": 0}
    
    # Determine the save path based on the next available file in results_dir
    save_path = get_next_filename(base_path)

    # Main MCMC loop
    for i in range(iterations):
        try:
            result = next(sampler)
            sample = np.clip(result[0], lower_bound, upper_bound)
            accepted = result[1]

            # DEBUG: Check type and shape of sample
            print(f"\nDEBUG: Sample at iteration {i}, Type: {type(sample)}, Shape: {sample.shape if hasattr(sample, 'shape') else 'N/A'}")
            
            samples.append(sample)
            if accepted:
                acceptances += 1

            # Save samples at intervals
            if (i + 1) % save_interval == 0:
                checkpoint_data["samples"] = [s.tolist() for s in samples]
                checkpoint_data["acceptance_rate"] = acceptances / (i + 1)
                
                # Save to the generated path in results_dir
                np.savetxt(save_path, np.array(samples), delimiter=',')
                with open(f"{save_path}_checkpoint.json", 'w') as checkpoint_file:
                    json.dump(checkpoint_data, checkpoint_file)
                print(f"Checkpoint saved at iteration {i + 1}. Path: {save_path}")

        except np.linalg.LinAlgError as e:
            print(f"Numerical error at iteration {i + 1}: {e}")
            continue
        except Exception as e:
            print(f"Unexpected error at iteration {i + 1}: {e}")
            break 

    # Final save of all samples in results_dir
    np.savetxt(save_path, np.array(samples), delimiter=',')
    acceptance_rate = acceptances / iterations
    return np.array(samples), acceptance_rate

def run_mcmc_with_optimized_params(json_path, observed_data, config, ion_velocity_weight=2.0, iterations=100):
    # Load optimized parameters as the initial guess
    v1_opt, v2_opt = load_optimized_params(json_path)
    
    # Verify that parameters are loaded
    if v1_opt is None or v2_opt is None:
        print(f"Error: Optimized parameters could not be loaded. v1_opt: {v1_opt}, v2_opt: {v2_opt}")
        return  # Exit function if parameters are not loaded correctly

    try:
        # Convert v1 and alpha to log10 space
        v_log_initial = [np.log10(v1_opt), np.log10(v2_opt / v1_opt)]
        print("Initial log parameters:", v_log_initial)  # Debugging: Verify initial log values
    except Exception as e:
        print(f"Error calculating v_log_initial: {e}")
        return
    
    # Run MCMC sampling
    print("Running MCMC sampling based on loaded optimized parameters...")
    base_path = f"mcmc_samples_w_{ion_velocity_weight}"
    
    samples, acceptance_rate = mcmc_inference(
        lambda v_log: log_posterior(v_log, observed_data, config, ion_velocity_weight=ion_velocity_weight),
        v_log_initial,
        iterations=iterations,
        lower_bound=-5,
        upper_bound=3,
        save_interval=10,
        base_path=base_path
    )
    
    print(f"MCMC sampling complete with acceptance rate: {acceptance_rate:.2f}")
    
    # Save final samples and metadata for analysis in results_dir
    final_samples_file = get_next_filename(f"final_mcmc_samples_w_{ion_velocity_weight}")
    np.savetxt(final_samples_file, samples, delimiter=',')
    print(f"Final MCMC samples saved to {final_samples_file}")

    metadata = {
        "initial_guess": {"v1": v1_opt, "v2": v2_opt},
        "v_log_initial": v_log_initial,
        "iterations": iterations,
        "acceptance_rate": acceptance_rate,
        "ion_velocity_weight": ion_velocity_weight,
        "checkpoint_file": base_path,
        "final_samples_file": final_samples_file,
        "model": "TwoZoneBohm",
        "config": create_specific_config(config)
    }
    
    save_metadata(metadata, filename=os.path.join(results_dir, f"mcmc_metadata_w_{ion_velocity_weight}.json"))

# Main function
def main():
    # Load the initial guess parameters from the JSON file

    ion_velocity_weight = [2.0]
    for ion_velocity_weight in ion_velocity_weight:
        ground_truth_data = run_multilogbohm_simulation(config_multilogbohm, ion_velocity_weight)
        save_results_to_json(ground_truth_data, f'mcmc_w_{ion_velocity_weight}_observed_data_map.json', save_every_n_grid_points=10)
		
    v1_opt, v2_opt = load_optimized_params(initial_guess_path)
    if v1_opt is None or v2_opt is None:
        print("Failed to load initial guess parameters.")
        return

    # Run the simulation with the initial guess parameters
    print(f"Running initial simulation with v1: {v1_opt}, v2: {v2_opt}")
    initial_simulation_result = hallthruster_jl_wrapper(v1_opt, v2_opt, config_spt_100, use_time_averaged=True, save_every_n_grid_points=10)
    save_results_to_json(initial_simulation_result, f'mcmc_w_{ion_velocity_weight}_initial_mcmc.json', save_every_n_grid_points=10)
    print(f"Initial simulation result saved")

    # Load the observed data for MCMC (or use the simulation result directly)
    observed_data = ground_truth_data  # Using the result directly
    # or observed_data = load_json_data(os.path.join(RESULTS_DIR, observed_data_filename))

    # Run MCMC with the initial guess as the starting point
    print("Starting MCMC sampling...")
    run_mcmc_with_optimized_params(
        json_path=initial_guess_path,
        observed_data=observed_data,
        config=config_spt_100,
        ion_velocity_weight=2.0,
        iterations=100
    )

if __name__ == "__main__":
    main()









