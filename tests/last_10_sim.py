import os
import pandas as pd
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hall_opt.map_nelder_mead import hallthruster_jl_wrapper, config_multilogbohm

# Paths
checkpoint_dir = "../mcmc_samples_1/"
output_dir = "../results-mcmc/sim_check_last_10/"
os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

# Configuration for TwoZoneBohm
config_spt_100 = config_multilogbohm.copy()
config_spt_100['anom_model'] = 'TwoZoneBohm'

# Load checkpoint data
checkpoint_file = os.path.join(checkpoint_dir, "mcmc_samples_1_checkpoint_3.csv")
checkpoint_data = pd.read_csv(checkpoint_file, skiprows=1)

v1_column = checkpoint_data.iloc[:, 1]
v2_column = checkpoint_data.iloc[:, 2]

valid_samples = []
for v1, v2 in zip(reversed(v1_column), reversed(v2_column)):
    valid_samples.append((v1, v2))
    if len(valid_samples) == 10:  
        break

print(f"Found {len(valid_samples)} valid samples starting from the end.")

# Simulate and collect results for valid samples
simulation_results = []
for v1, v2 in valid_samples:
    # Run simulation using wrapper
    result = hallthruster_jl_wrapper(v1, v2, config_spt_100)
    if result is not None:
        simulation_results.append(result)

# Plot ion velocity predictions
plt.figure(figsize=(10, 6))
for res in simulation_results:
    plt.plot(res["z_normalized"], res["ion_velocity"][0], alpha=0.2)
plt.title("Ion Velocity Predictions for First 10 Valid Parameter Pairs (from End)")
plt.xlabel("Normalized Distance (z)")
plt.ylabel("Ion Velocity (m/s)")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "first_10_valid_ion_velocity_predictions.png"))
plt.show()

# Extract thrust and discharge current from simulation results
thrust_values = [res["thrust"][0] for res in simulation_results]
discharge_values = [res["discharge_current"][0] for res in simulation_results]
samples_labels = [f"Sample {i+1}" for i in range(len(thrust_values))]

# Plot thrust comparison
plt.figure(figsize=(10, 6))
plt.bar(samples_labels, thrust_values, color="blue", alpha=0.7, edgecolor="black")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Thrust (N)")
plt.title("Thrust Predictions for First 10 Valid Parameter Pairs")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "thrust_comparison_first_10.png"))
plt.show()

# Plot discharge current comparison
plt.figure(figsize=(10, 6))
plt.bar(samples_labels, discharge_values, color="green", alpha=0.7, edgecolor="black")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Discharge Current (A)")
plt.title("Discharge Current Predictions for First 10 Valid Parameter Pairs")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "discharge_comparison_first_10.png"))
plt.show()
