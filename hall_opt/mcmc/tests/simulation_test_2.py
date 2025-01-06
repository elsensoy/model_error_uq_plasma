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
        all_results.append({"params": {"v1": v1, "v2": v2}, "metrics": metrics})
        print(f"Test Case: {params}")
        print("Metrics:", json.dumps(metrics, indent=4))
    else:
        print(f"Test Case: {params} failed.")

# Save simulation keys to file
if simulation_result:
    with open("simulation_keys.txt", "w") as f:
        def save_keys(data, parent_key=""):
            if isinstance(data, dict):
                for key, value in data.items():
                    full_key = f"{parent_key}.{key}" if parent_key else key
                    f.write(full_key + "\n")
                    save_keys(value, full_key)
            elif isinstance(data, list):
                f.write(f"{parent_key} (list with {len(data)} items)\n")

        save_keys(simulation_result)
    print("Keys saved to simulation_keys.txt")
else:
    print("No valid simulation results to save keys.")

# Ensure plots directory exists
plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)

# Generate and save thrust vs. v1 plot
if all_results:
    thrust_values = [result["metrics"]["thrust"] for result in all_results]
    v1_values = [result["params"]["v1"] for result in all_results]

    plt.plot(v1_values, thrust_values, marker="o")
    plt.xlabel("v1")
    plt.ylabel("Thrust")
    plt.title("Thrust vs. v1")
    plt.tight_layout()

    plot_path = os.path.join(plots_dir, "thrust_sim_test.png")
    plt.savefig(plot_path, dpi=300)  # Save with high resolution
    plt.close()
    print(f"Plot saved as '{plot_path}'")
else:
    print("No results available for plotting.")
