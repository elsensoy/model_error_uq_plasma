# import os
# import json
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path
# from typing import List


# def generate_mcmc_iteration_metric_plots(iter_metrics_dir: Path, save_dir: Path):
     
#     save_dir.mkdir(parents=True, exist_ok=True)

#     # --- Load all JSON files in order ---
#     metric_files = sorted(
#         [f for f in iter_metrics_dir.glob("metrics_*.json")],
#         key=lambda f: int(f.stem.split("_")[-1])
#     )

#     if not metric_files:
#         print(f"[ERROR] No metric files found in {iter_metrics_dir}")
#         return

#     thrust_vals = []
#     current_vals = []

#     for file in metric_files:
#         with open(file, "r") as f:
#             data = json.load(f)
#             thrust_vals.append(data.get("thrust"))
#             current_vals.append(data.get("discharge_current"))

#     iterations = list(range(1, len(thrust_vals) + 1))
# 	# Validate dimensions
# 	# if z_normalized.shape != observed_ion_velocity.shape:
# 	# 	raise ValueError(
# 	# 		f"Mismatch in dimensions: z_normalized has shape {z_normalized.shape}, "
# 	# 		f"but observed_ion_velocity has shape {observed_ion_velocity.shape}."
# 	# 	)

# 	# Extract metrics for MCMC iteration plots
# 	# Ensure metrics are not empty

# 	# Compute mean and last sample values
# 	mean_thrust = np.mean(thrust_values)
# 	last_thrust = thrust_values[-1]
# 	mean_discharge = np.mean(discharge_values)
# 	last_discharge = discharge_values[-1]
# 	mean_ion_velocity = np.mean(ion_velocity_values, axis=0)
# 	last_ion_velocity = ion_velocity_values[-1]

# 	# Plot 1: Histogram of Thrust Predictions
# 	plt.figure(figsize=(10, 6))
# 	plt.hist(thrust_values, bins=30, alpha=0.7, color="blue", label="MCMC Thrust Predictions")
# 	plt.axvline(observed_thrust, color="red", linestyle="--", label=f"Observed: {observed_thrust:.3f}")
# 	plt.axvline(initial_thrust, color="green", linestyle="--", label=f"Initial: {initial_thrust:.3f}")
# 	plt.axvline(last_thrust, color="orange", linestyle="--", label=f"Final (Last Sample): {last_thrust:.3f}")
# 	plt.axvline(mean_thrust, color="purple", linestyle="--", label=f"Mean: {mean_thrust:.3f}")
# 	plt.xlabel("Thrust (N)")
# 	plt.ylabel("Frequency")
# 	plt.title("Thrust Predictions Histogram")
# 	plt.legend()
# 	plt.grid(True)
# 	plt.tight_layout()
# 	plt.savefig(os.path.join(save_dir, "thrust_histogram.png"))
# 	plt.close()

# 	# Plot 2: Histogram of Discharge Current Predictions
# 	plt.figure(figsize=(10, 6))
# 	plt.hist(discharge_values, bins=30, alpha=0.7, color="purple", label="MCMC Discharge Current Predictions")
# 	plt.axvline(observed_discharge_current, color="red", linestyle="--", label=f"Observed: {observed_discharge_current:.2f}")
# 	plt.axvline(initial_discharge_current, color="green", linestyle="--", label=f"Initial: {initial_discharge_current:.2f}")
# 	plt.axvline(last_discharge, color="orange", linestyle="--", label=f"Final (Last Sample): {last_discharge:.2f}")
# 	plt.axvline(mean_discharge, color="purple", linestyle="--", label=f"Mean: {mean_discharge:.3f}")
# 	plt.xlabel("Discharge Current (A)")
# 	plt.ylabel("Frequency")
# 	plt.title("Discharge Current Predictions Histogram")
# 	plt.legend()
# 	plt.grid(True)
# 	plt.tight_layout()
# 	plt.savefig(os.path.join(save_dir, "discharge_current_histogram.png"))
# 	plt.close()

# 	# Plot 3: Ion Velocity Predictions
# 	plt.figure(figsize=(12, 8))

# 	# Plot all ion velocity iterations (for density visualization)
# 	for ion_velocity in ion_velocity_values:
# 		plt.plot(z_normalized, ion_velocity, alpha=0.2, color="purple", linestyle="-")

# 	# Plot mean ion velocity
# 	plt.plot(z_normalized, mean_ion_velocity, color="orange", linestyle="--", linewidth=2,
# 			label=f"Ion Velocity MCMC (Mean): {np.mean(mean_ion_velocity):.2f}")

# 	# Plot last ion velocity sample
# 	plt.plot(z_normalized, last_ion_velocity, color="blue", linestyle="--", linewidth=2,
# 			label=f"Ion Velocity MCMC (Last Sample): {np.mean(last_ion_velocity):.2f}")

# 	# Plot observed ion velocity
# 	plt.plot(z_normalized, observed_ion_velocity, color="red", linewidth=1,
# 			label=f"Observed Ion Velocity (Mean: {np.mean(observed_ion_velocity):.2f})")
# 	plt.scatter(z_normalized, observed_ion_velocity, color="red", label="Observed Data")

# 	# Annotate initial and final points
# 	first_z, last_z = z_normalized[0], z_normalized[-1]
# 	plt.text(first_z, observed_ion_velocity[0] + 200, f"{observed_ion_velocity[0]:.2f}", color="red", fontsize=8)
# 	plt.text(last_z, observed_ion_velocity[-1] + 200, f"{observed_ion_velocity[-1]:.2f}", color="red", fontsize=8)
# 	plt.text(last_z, mean_ion_velocity[-1] - 500, f"{mean_ion_velocity[-1]:.2f}", color="orange", fontsize=8)
# 	plt.text(last_z, last_ion_velocity[-1] - 500, f"{last_ion_velocity[-1]:.2f}", color="blue", fontsize=8)

# 	# Labels, legend, and grid
# 	plt.xlabel("Normalized Distance (z)")
# 	plt.ylabel("Ion Velocity (m/s)")
# 	plt.title("Ion Velocity Predictions & Iterations")
# 	plt.legend()
# 	plt.grid(True)
# 	plt.tight_layout()
#     ion_velocity_path = save_dir / "ion_velocity_predictions.png"
# 	plt.savefig( ion_velocity_path)
#     print(f"[INFO] Saved: {ion_velocity_path}")
#     plt.close()
