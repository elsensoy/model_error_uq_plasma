import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hall_opt.map_nelder_mead import hallthruster_jl_wrapper, config_multilogbohm

# Directory to save plots
PLOTS_DIR = "plots-11-25-24"
os.makedirs(PLOTS_DIR, exist_ok=True) 
RESULTS_DIR = os.path.join("..", "mcmc-results-11-25-24")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Helper function to save results as JSON
def save_results_to_json(result_dict, filename, save_every_n_grid_points=10):
    result_dict_copy = result_dict.copy()
    for key in ['ion_velocity', 'z_normalized']:
        if key in result_dict_copy:
            if isinstance(result_dict_copy[key], list) and len(result_dict_copy[key]) > save_every_n_grid_points:
                result_dict_copy[key] = result_dict_copy[key][::save_every_n_grid_points]
            elif isinstance(result_dict_copy[key][0], list):  # For 2D data
                result_dict_copy[key] = [sublist[::save_every_n_grid_points] for sublist in result_dict_copy[key]]
    filepath = os.path.join(RESULTS_DIR, filename)
    with open(filepath, 'w') as f:
        json.dump(result_dict_copy, f, indent=4)
    print(f"Results saved to {filepath}")

# Load the MCMC samples, truth data, and simulation results
def load_data():
    # Load MCMC samples
    samples = pd.read_csv(os.path.join(RESULTS_DIR, "final_mcmc_samples_2_w_2.0_3.csv"), header=None)
    samples.columns = ["log_v1", "log_alpha"]
    samples["v1"] = 10 ** samples["log_v1"]
    samples["alpha"] = 10 ** samples["log_alpha"]
    samples["v2"] = samples["v1"] * samples["alpha"]

    # Load ground truth data
    with open(os.path.join(RESULTS_DIR, "mcmc_w_2.0_observed_data_map.json"), 'r') as f:
        truth_data = json.load(f)

    # Load pre-MCMC initial simulation results
    with open(os.path.join(RESULTS_DIR, "mcmc_w_2.0_initial_mcmc.json"), 'r') as f:
        pre_mcmc_target_data = json.load(f)

    return samples, truth_data, pre_mcmc_target_data

def plot_ion_velocity_comparison(truth_data, pre_mcmc_target_data, mcmc_simulation_result):
    """Plot and save predictive comparison for ion velocity."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot ion velocity for truth model
    ax.plot(truth_data["z_normalized"], truth_data["ion_velocity"][0], 
            label="Truth Model (MultiLogBohm)", color='blue', linestyle='-')

    # Plot ion velocity for initial guess
    ax.plot(pre_mcmc_target_data["z_normalized"], pre_mcmc_target_data["ion_velocity"][0], 
            label="Initial Guess (TwoZoneBohm)", color='purple', linestyle=':')

    # Plot ion velocity for MCMC posterior
    ax.plot(mcmc_simulation_result["z_normalized"], mcmc_simulation_result["ion_velocity"][0], 
            label="MCMC Posterior (TwoZoneBohm)", color='red', linestyle='--')

    # Add labels, title, and legend
    ax.set_title("Ion Velocity Comparison")
    ax.set_xlabel("Normalized Position (z_normalized)")
    ax.set_ylabel("Ion Velocity (m/s)")
    ax.legend()

    # Save the plot as a PNG file
    plot_path = os.path.join(PLOTS_DIR, "ion_velocity_comparison.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Ion velocity comparison plot saved to {plot_path}")

def main():
    
    samples, truth_data, pre_mcmc_target_data = load_data()

    last_sample = samples.iloc[-1]
    v1_mcmc, alpha_mcmc = last_sample["v1"], last_sample["alpha"]
    v2_mcmc = last_sample["v2"]
    print(f"Using MCMC parameters: v1 = {v1_mcmc}, alpha = {alpha_mcmc}, v2 = {v2_mcmc}")

    
    config = config_multilogbohm.copy()
    config['anom_model'] = 'TwoZoneBohm'
    mcmc_simulation_result = hallthruster_jl_wrapper(v1_mcmc, v2_mcmc, config, use_time_averaged=True)
    save_results_to_json(mcmc_simulation_result, "mcmc_metrics_3.json")

    
    plot_ion_velocity_comparison(truth_data, pre_mcmc_target_data, mcmc_simulation_result)
    # plot_predictive_comparison(samples, truth_data, pre_mcmc_target_data)

if __name__ == "__main__":
    main()
