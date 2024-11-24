import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neldermead.map_nelder_mead import hallthruster_jl_wrapper, config_multilogbohm

# Directory to save plots
PLOTS_DIR = "mcmc-plots-1123"
os.makedirs(PLOTS_DIR, exist_ok=True) 
RESULTS_DIR = os.path.join("..", "mcmc-results-11-23-24")
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
    samples = pd.read_csv(os.path.join(RESULTS_DIR, "mcmc_samples_w_2.0_checkpoint.csv"), header=None)
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

    # Optionally, you can show the plot (not recommended for non-interactive environments)
    # plt.show()
# def plot_predictive_comparison(samples, truth_data, pre_mcmc_target_data):
#     """Plot and save predictive comparison for thrust, discharge current, and ion velocity."""
#     posterior_means = samples[["v1", "v2"]].mean()
#     v1_mean, v2_mean = posterior_means["v1"], posterior_means["v2"]

#     fig, axes = plt.subplots(3, 1, figsize=(10, 12))

#     # Thrust Comparison
#     axes[0].plot(truth_data["thrust"], label="Truth Model (MultiLogBohm)", color='blue')
#     axes[0].axhline(y=v1_mean, color='red', linestyle='--', label="TwoZoneBohm (Posterior Mean)")
#     axes[0].axhline(y=pre_mcmc_target_data["thrust"], color='purple', linestyle=':', label="TwoZoneBohm (Initial Simulation)")
#     axes[0].set_title("Thrust Comparison")
#     axes[0].legend()

#     # Discharge Current Comparison
#     axes[1].plot(truth_data["discharge_current"], label="Truth Model (MultiLogBohm)", color='blue')
#     axes[1].axhline(y=v2_mean, color='red', linestyle='--', label="TwoZoneBohm (Posterior Mean)")
#     axes[1].axhline(y=pre_mcmc_target_data["discharge_current"], color='purple', linestyle=':', label="TwoZoneBohm (Initial Simulation)")
#     axes[1].set_title("Discharge Current Comparison")
#     axes[1].legend()

#     # Ion Velocity Comparison
#     if isinstance(truth_data["ion_velocity"], list):
#         ion_velocity_truth = np.array(truth_data["ion_velocity"])
#     else:
#         ion_velocity_truth = np.array(truth_data["ion_velocity"]).flatten()

#     ion_velocity_pre_mcmc = np.array(pre_mcmc_target_data["ion_velocity"]).flatten()

#     # Plot ion velocity comparison
#     axes[2].plot(range(len(ion_velocity_truth)), ion_velocity_truth, label="Truth Model (MultiLogBohm)", color='blue')
#     axes[2].plot(range(len(ion_velocity_truth)), [np.mean(ion_velocity_pre_mcmc)] * len(ion_velocity_truth), 
#                  label="TwoZoneBohm (Initial Simulation)", color='purple', linestyle=':')
#     axes[2].axhline(y=v1_mean, color='red', linestyle='--', label="TwoZoneBohm (Posterior Mean)")
#     axes[2].set_title("Ion Velocity Comparison")
#     axes[2].legend()

#     plt.tight_layout()
#     plt.savefig(os.path.join(PLOTS_DIR, "predictive_comparison.png"))
#     plt.show()

def main():
    # Load data and results
    samples, truth_data, pre_mcmc_target_data = load_data()

    # Extract parameters from the last MCMC iteration
    last_sample = samples.iloc[-1]
    v1_mcmc, alpha_mcmc = last_sample["v1"], last_sample["alpha"]
    v2_mcmc = last_sample["v2"]
    print(f"Using MCMC parameters: v1 = {v1_mcmc}, alpha = {alpha_mcmc}, v2 = {v2_mcmc}")

    # Run simulation with MCMC-derived parameters
    config = config_multilogbohm.copy()
    config['anom_model'] = 'TwoZoneBohm'
    mcmc_simulation_result = hallthruster_jl_wrapper(v1_mcmc, v2_mcmc, config, use_time_averaged=True)
    save_results_to_json(mcmc_simulation_result, "mcmc_w_2.0_mcmc_1.json")

    # Generate plots
    plot_ion_velocity_comparison(truth_data, pre_mcmc_target_data, mcmc_simulation_result)
    # plot_predictive_comparison(samples, truth_data, pre_mcmc_target_data)

if __name__ == "__main__":
    main()
