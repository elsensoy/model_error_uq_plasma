import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from common_setup import load_data, get_common_paths, load_iteration_metrics

# Load data
samples, truth_data, pre_mcmc_data, initial_params = load_data()
iteration_metrics = load_iteration_metrics()
paths = get_common_paths()
plots_dir = paths["plots_dir"]

# Extract scalar values
observed_thrust = truth_data["thrust"]
initial_thrust = pre_mcmc_data["thrust"]
observed_discharge_current = truth_data["discharge_current"]
initial_discharge_current = pre_mcmc_data["discharge_current"]

# Extract arrays for z_normalized and ion_velocity
z_normalized = np.array(truth_data["z_normalized"])  # Ensure as numpy array
observed_ion_velocity = np.array(truth_data["ion_velocity"])  # Ensure as numpy array

# Validate dimensions
if z_normalized.shape != observed_ion_velocity.shape:
    raise ValueError(
        f"Mismatch in dimensions: z_normalized has shape {z_normalized.shape}, "
        f"but observed_ion_velocity has shape {observed_ion_velocity.shape}."
    )

# Extract metrics for MCMC iteration plots
thrust_values = [metric["thrust"] for metric in iteration_metrics]
discharge_values = [metric["discharge_current"] for metric in iteration_metrics]
ion_velocity_values = [np.array(metric["ion_velocity"]) for metric in iteration_metrics]

# Ensure metrics are not empty
if not thrust_values or not discharge_values or not ion_velocity_values:
    raise ValueError("Iteration metrics are empty. Check the input data.")

# Compute mean and last sample values
mean_thrust = np.mean(thrust_values)
last_thrust = thrust_values[-1]
mean_discharge = np.mean(discharge_values)
last_discharge = discharge_values[-1]
mean_ion_velocity = np.mean(ion_velocity_values, axis=0)
last_ion_velocity = ion_velocity_values[-1]

# Plot 1: Histogram of Thrust Predictions
plt.figure(figsize=(10, 6))
plt.hist(thrust_values, bins=30, alpha=0.7, color="blue", label="MCMC Thrust Predictions")
plt.axvline(observed_thrust, color="red", linestyle="--", label=f"Observed: {observed_thrust:.3f}")
plt.axvline(initial_thrust, color="green", linestyle="--", label=f"Initial: {initial_thrust:.3f}")
plt.axvline(last_thrust, color="orange", linestyle="--", label=f"Final (Last Sample): {last_thrust:.3f}")
plt.axvline(mean_thrust, color="purple", linestyle="--", label=f"Mean: {mean_thrust:.3f}")
plt.xlabel("Thrust (N)")
plt.ylabel("Frequency")
plt.title("Thrust Predictions Histogram")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "thrust_histogram.png"))
plt.close()

# Plot 2: Histogram of Discharge Current Predictions
plt.figure(figsize=(10, 6))
plt.hist(discharge_values, bins=30, alpha=0.7, color="purple", label="MCMC Discharge Current Predictions")
plt.axvline(observed_discharge_current, color="red", linestyle="--", label=f"Observed: {observed_discharge_current:.2f}")
plt.axvline(initial_discharge_current, color="green", linestyle="--", label=f"Initial: {initial_discharge_current:.2f}")
plt.axvline(last_discharge, color="orange", linestyle="--", label=f"Final (Last Sample): {last_discharge:.2f}")
plt.axvline(mean_discharge, color="purple", linestyle="--", label=f"Mean: {mean_discharge:.3f}")
plt.xlabel("Discharge Current (A)")
plt.ylabel("Frequency")
plt.title("Discharge Current Predictions Histogram")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "discharge_current_histogram.png"))
plt.close()

# Plot 3: Ion Velocity Predictions
plt.figure(figsize=(12, 8))

# Plot all ion velocity iterations (for density visualization)
for ion_velocity in ion_velocity_values:
    plt.plot(z_normalized, ion_velocity, alpha=0.2, color="purple", linestyle="-")

# Plot mean ion velocity
plt.plot(z_normalized, mean_ion_velocity, color="orange", linestyle="--", linewidth=2,
         label=f"Ion Velocity MCMC (Mean): {np.mean(mean_ion_velocity):.2f}")

# Plot last ion velocity sample
plt.plot(z_normalized, last_ion_velocity, color="blue", linestyle="--", linewidth=2,
         label=f"Ion Velocity MCMC (Last Sample): {np.mean(last_ion_velocity):.2f}")

# Plot observed ion velocity
plt.plot(z_normalized, observed_ion_velocity, color="red", linewidth=1,
         label=f"Observed Ion Velocity (Mean: {np.mean(observed_ion_velocity):.2f})")
plt.scatter(z_normalized, observed_ion_velocity, color="red", label="Observed Data")

# Annotate initial and final points
first_z, last_z = z_normalized[0], z_normalized[-1]
plt.text(first_z, observed_ion_velocity[0] + 200, f"{observed_ion_velocity[0]:.2f}", color="red", fontsize=8)
plt.text(last_z, observed_ion_velocity[-1] + 200, f"{observed_ion_velocity[-1]:.2f}", color="red", fontsize=8)
plt.text(last_z, mean_ion_velocity[-1] - 500, f"{mean_ion_velocity[-1]:.2f}", color="orange", fontsize=8)
plt.text(last_z, last_ion_velocity[-1] - 500, f"{last_ion_velocity[-1]:.2f}", color="blue", fontsize=8)

# Labels, legend, and grid
plt.xlabel("Normalized Distance (z)")
plt.ylabel("Ion Velocity (m/s)")
plt.title("Ion Velocity Predictions & Iterations")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "ion_velocity_predictions.png"))
plt.close()
