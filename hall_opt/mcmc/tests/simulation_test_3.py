import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# HallThruster Path Setup
hallthruster_path = "/root/.julia/packages/HallThruster/J4Grt/python"
if hallthruster_path not in sys.path:
    sys.path.append(hallthruster_path)

import hallthruster as het
from config.simulation import (
    simulation, config_spt_100, postprocess,
    update_twozonebohm_config, run_simulation_with_config
)

# Path to observed data file
observed_data_file = "/mnt/c/Users/MRover/elsensoy/model_error_uq_plasma/hall_opt/mcmc/results/mcmc_observed_data_map.json"

# Load observed data
with open(observed_data_file, "r") as f:
    observed_data = json.load(f)

# Extract observed ion velocity and z_normalized
observed_ion_velocity = observed_data.get("ion_velocity", [])
observed_z_normalized = observed_data.get("z_normalized", [])

# Check if observed data is valid
if not observed_ion_velocity or not observed_z_normalized:
    raise ValueError("Observed data for ion velocity or z_normalized is missing or empty.")

# Define test parameters
test_parameters = [
    {"v1_log": np.log10(1 / 160), "alpha_log": 0.5},
    {"v1_log": np.log10(1 / 200), "alpha_log": 1.0},
    {"v1_log": np.log10(1 / 100), "alpha_log": 2.0},
]

# Results storage
all_results = []

# Test each parameter set
for params in test_parameters:
    v1 = 10**params["v1_log"]
    alpha = 10**params["alpha_log"]
    v2 = alpha * v1

    # Update configuration for TwoZoneBohm model
    updated_config = update_twozonebohm_config(config_spt_100, v1, v2)
    simulation_result = run_simulation_with_config(
        updated_config, simulation, postprocess, config_type="TwoZoneBohm"
    )

    if simulation_result:
        # Extract metrics from simulation result
        metrics = simulation_result["output"]["average"]
        z_simulated = simulation_result["output"].get("z_coords", [])
        ion_velocity_simulated = metrics.get("ui", [[]])[0]

        # Interpolate simulation data to match observed positions
        if z_simulated and ion_velocity_simulated:
            interpolated_velocity = np.interp(
                observed_z_normalized, z_simulated, ion_velocity_simulated
            )
            all_results.append({
                "params": {"v1": v1, "v2": v2},
                "z_simulated": z_simulated,
                "ion_velocity_simulated": ion_velocity_simulated,
                "interpolated_velocity": interpolated_velocity
            })
        print(f"Test Case: {params}")
        print("Interpolated Velocity:", interpolated_velocity)
    else:
        print(f"Test Case: {params} failed.")

# Save ion velocity results
if all_results:
    results_file = "ion_velocity_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"Ion velocity results saved to {results_file}")

plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)

# Generate ion velocity plots
if all_results:
    for i, result in enumerate(all_results):
        plt.figure()
        z_simulated = result["z_simulated"]
        ion_velocity_simulated = result["ion_velocity_simulated"]
        interpolated_velocity = result["interpolated_velocity"]

        # Plot simulated and observed ion velocity profiles
        plt.plot(z_simulated, ion_velocity_simulated, label="Simulated Velocity")
        plt.scatter(observed_z_normalized, observed_ion_velocity,
                    color="red", label="Observed Velocity", zorder=5)
        plt.scatter(observed_z_normalized, interpolated_velocity,
                    color="blue", label="Interpolated Velocity", zorder=5)

        plt.xlabel("Normalized Axial Position (z)")
        plt.ylabel("Ion Velocity (m/s)")
        plt.title(f"Ion Velocity Profile (Test Case {i + 1})")
        plt.legend()
        plt.tight_layout()

        # Save plot
        plot_file = os.path.join(plots_dir, f"ion_velocity_test_{i + 1}.png")
        plt.savefig(plot_file, dpi=300)
        plt.close()
        print(f"Ion velocity plot saved to {plot_file}")
else:
    print("No results available for ion velocity plotting.")
