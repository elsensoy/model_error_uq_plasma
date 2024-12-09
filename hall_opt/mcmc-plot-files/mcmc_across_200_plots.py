import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#source ~/.venvs/pdm/bin/activate
from common_setup import load_data, get_common_paths, load_iteration_metrics

# Load data
samples, truth_data, pre_mcmc_data, initial_params = load_data()
iteration_metrics = load_iteration_metrics()
paths = get_common_paths()
plots_dir = paths["plots_dir"]

# Verify Observed and Initial Data
observed_thrust = truth_data["thrust"][0]
initial_thrust = pre_mcmc_data["thrust"][0]
observed_discharge_current = truth_data["discharge_current"][0]
initial_discharge_current = pre_mcmc_data["discharge_current"][0]
observed_ion_velocity = truth_data["ion_velocity"][0]
initial_ion_velocity = pre_mcmc_data["ion_velocity"][0]
z_normalized = truth_data["z_normalized"]

# Extract metrics for plots
thrust_values = [metric["thrust"][0] for metric in iteration_metrics]
discharge_values = [metric["discharge_current"][0] for metric in iteration_metrics]
ion_velocity_values = [metric["ion_velocity"][0] for metric in iteration_metrics]

# Compute means and last sample values
mean_thrust = np.mean(thrust_values)
last_thrust = thrust_values[-1]
mean_discharge = np.mean(discharge_values)
last_discharge = discharge_values[-1]


# Histogram: Thrust Predictions
plt.figure(figsize=(10, 6))
plt.hist(thrust_values, bins=30, alpha=0.7, color="blue", label="MCMC Thrust Predictions")
# Add observed thrust line
plt.axvline(observed_thrust, color="red", linestyle="--", label=f"Observed: {observed_thrust:.3f}")
# Add initial thrust line
plt.axvline(initial_thrust, color="green", linestyle="--", label=f"Initial: {initial_thrust:.3f}")
# Add final (last sample) thrust line
plt.axvline(last_thrust, color="orange", linestyle="--", label=f"Final (Last Sample): {last_thrust:.3f}")
# Add mean thrust line
plt.axvline(mean_thrust, color="purple", linestyle="--", label=f"Mean: {mean_thrust:.3f}")

# Customize the plot
plt.xlabel("Thrust (N)")
plt.ylabel("Frequency")
plt.title("Thrust Predictions Histogram")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save and display the plot
plt.savefig(os.path.join(plots_dir, "thrust_histogram_200.png"))
plt.show()

# 2. Histogram: Discharge Current Predictions
plt.figure(figsize=(10, 6))
plt.hist(discharge_values, bins=30, alpha=0.7, color="purple", label="MCMC Discharge Current Predictions")
plt.axvline(observed_discharge_current, color="red", linestyle="--", label=f"Observed: {observed_discharge_current:.2f}")
plt.axvline(initial_discharge_current, color="green", linestyle="--", label=f"Initial: {initial_discharge_current:.2f}")
plt.axvline(last_discharge, color="orange", linestyle="--", label=f"Final (Last MCMC): {last_discharge:.2f}")
# Add mean discharge line
plt.axvline(mean_discharge, color="purple", linestyle="--", label=f"Mean: {mean_discharge:.3f}")

plt.xlabel("Discharge Current (A)")
plt.ylabel("Frequency")
plt.title("Discharge Current Predictions Histogram")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "discharge_current_histogram_200.png"))
plt.show()

# 3. Ion Velocity Predictions
plt.figure(figsize=(12, 8))

# (for density visualization)
for ion_velocity in ion_velocity_values:
    plt.plot(z_normalized, ion_velocity, alpha=0.5, color="purple")

# Calculate and plot mean ion velocity across 200 iterations
mean_ion_velocity = np.mean(ion_velocity_values, axis=0)
plt.plot(z_normalized, mean_ion_velocity, color="orange", linestyle="--", linewidth=2, 
         label=f"Ion Velocity MCMC (Mean): {np.mean(mean_ion_velocity):.2f}")

# Observed Ion Velocity
plt.plot(z_normalized, observed_ion_velocity, color="maroon", linewidth=2, 
         label=f"Observed Ion Velocity (Mean: {np.mean(observed_ion_velocity):.2f})")
plt.scatter(z_normalized, observed_ion_velocity, color="red", label="Observed Data")

# Initial Ion Velocity
plt.plot(z_normalized, initial_ion_velocity, color="teal", linestyle="--", linewidth=2, 
         label=f"Initial Ion Velocity (Mean: {np.mean(initial_ion_velocity):.2f})")
plt.scatter(z_normalized, initial_ion_velocity, color="green", label="Initial Data")

# Annotate initial and final points
first_z, last_z = z_normalized[0], z_normalized[-1]
plt.text(first_z, observed_ion_velocity[0] + 200, f"{observed_ion_velocity[0]:.2f}", color="red", fontsize=8)
plt.text(last_z, observed_ion_velocity[-1] + 200, f"{observed_ion_velocity[-1]:.2f}", color="red", fontsize=8)
plt.text(first_z, initial_ion_velocity[0] + 500, f"{initial_ion_velocity[0]:.2f}", color="green", fontsize=8)
plt.text(last_z, initial_ion_velocity[-1] + 500, f"{initial_ion_velocity[-1]:.2f}", color="green", fontsize=8)

# Annotate last ion velocity
plt.text(last_z, mean_ion_velocity[-1] - 500, f"{mean_ion_velocity[-1]:.2f}", color="orange", fontsize=8)

# Add labels, title, and legend
plt.xlabel("Normalized Distance (z)")
plt.ylabel("Ion Velocity (m/s)")
plt.title("Ion Velocity Predictions across 200 Iterations")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "ion_velocity_predictions_200.png"))
plt.show()
