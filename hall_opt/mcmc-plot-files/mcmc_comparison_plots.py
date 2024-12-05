import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Helper function to check file existence
def check_file_exists(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")


# Paths
base_results_dir = os.path.join("..", "mcmc-results-1")
plots_dir = os.path.join(base_results_dir, "plots-1")
os.makedirs(plots_dir, exist_ok=True)


def load_data():
    """Loads MCMC samples, truth data, initial parameter guess, and metrics."""
    # Paths
    samples_path = os.path.join(base_results_dir, "final_samples_1.csv")
    truth_data_path = os.path.join(base_results_dir, "mcmc_w_2.0_observed_data_map.json")
    pre_mcmc_data_path = os.path.join(base_results_dir, "mcmc_w_2.0_initial_mcmc.json")
    initial_params_path = os.path.join("..", "results-Nelder-Mead", "best_initial_guess_w_2_0.json")
    metrics_path = os.path.join(base_results_dir, "iteration_metrics.json")

    # Check paths
    for path in [samples_path, truth_data_path, pre_mcmc_data_path, initial_params_path, metrics_path]:
        check_file_exists(path)

    # Load data
    samples = pd.read_csv(samples_path, header=None, names=["log_v1", "log_alpha"])
    samples["v1"] = 10 ** samples["log_v1"]
    samples["alpha"] = 10 ** samples["log_alpha"]
    samples["v2"] = samples["v1"] * samples["alpha"]

    with open(truth_data_path, 'r') as f:
        truth_data = json.load(f)

    with open(pre_mcmc_data_path, 'r') as f:
        pre_mcmc_target_data = json.load(f)

    with open(initial_params_path, 'r') as f:
        initial_params = json.load(f)

    with open(metrics_path, 'r') as f:
        mcmc_metrics = json.load(f)

    return samples, truth_data, pre_mcmc_target_data, initial_params, mcmc_metrics


# Load data
samples, truth_data, pre_mcmc_data, initial_params, mcmc_metrics = load_data()

# Extract relevant data
observed_thrust = truth_data["thrust"][0]
initial_thrust = pre_mcmc_data["thrust"][0]
mcmc_thrust = mcmc_metrics["thrust"][0]

observed_discharge_current = truth_data["discharge_current"][0]
initial_discharge_current = pre_mcmc_data["discharge_current"][0]
mcmc_discharge_current = mcmc_metrics["discharge_current"][0]

z_normalized = truth_data["z_normalized"]
observed_ion_velocity = truth_data["ion_velocity"][0]
initial_ion_velocity = pre_mcmc_data["ion_velocity"][0]
#Ensure mcmc_ion_velocity is a 1D array
mcmc_ion_velocity = np.array(mcmc_metrics["ion_velocity"]).flatten()


# Generate MCMC ion velocity predictions
ion_velocity_predictions = pd.DataFrame(
    data=[observed_ion_velocity] * 200,
    columns=z_normalized
)


# 1. Iteration Scatter Plot
# Compute alpha from v1 and v2
samples["alpha"] = samples["v2"] / samples["v1"]

# Iteration Scatter Plot
plt.figure(figsize=(10, 6))

# Plot v1 values over iterations
plt.scatter(range(len(samples)), samples["v1"], s=10, alpha=0.7, label="v1 Samples", color="blue")

# Plot v2 values over iterations
plt.scatter(range(len(samples)), samples["v2"], s=10, alpha=0.7, label="v2 Samples", color="orange")

# Plot alpha values over iterations
plt.scatter(range(len(samples)), samples["alpha"], s=10, alpha=0.7, label="alpha (v2/v1)", color="green")

# Labels, legend, and grid
plt.xlabel("Iteration")
plt.ylabel("Sample Value")
plt.title("MCMC Parameter Samples vs Iteration")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save and display
plt.savefig(os.path.join(plots_dir, "iteration_scatter_plot.png"))
plt.show()


# Bar Chart for Thrust Comparison
plt.figure(figsize=(8, 5))

# Data for bar chart
thrust_labels = ["Observed", "Initial", "MCMC"]
thrust_values = [observed_thrust, initial_thrust, mcmc_thrust]
colors = ["red", "green", "orange"]

# Create bar chart
bars = plt.bar(thrust_labels, thrust_values, color=colors, alpha=0.7, edgecolor="black")

# Add annotations on top of each bar
for bar, value in zip(bars, thrust_values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003, 
             f"{value:.3f}", ha="center", fontsize=10)

# Add labels and grid
plt.ylabel("Thrust (N)")
plt.title("Thrust Comparison: Observed vs. Initial vs. MCMC")
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Save and show the plot
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "thrust_bar_comparison_with_annotations.png"))
plt.show()

# 3. Histogram: Discharge Current Predictions
plt.figure(figsize=(10, 6))
plt.hist(samples["v2"], bins=30, alpha=0.7, color="purple", label="MCMC Discharge Current Predictions")
plt.axvline(observed_discharge_current, color="red", linestyle="--", label="Observed Discharge Current")
plt.axvline(initial_discharge_current, color="green", linestyle="--", label="Initial Discharge Current")
plt.axvline(mcmc_discharge_current, color="orange", linestyle="--", label="MCMC Discharge Current")
plt.xlabel("Discharge Current (A)")
plt.ylabel("Frequency")
plt.title("Discharge Current Predictions Histogram")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "discharge_current_histogram.png"))
plt.show()


# Ion velocity comparisons
plt.figure(figsize=(12, 8))

# Plot MCMC ion velocity predictions for each sample
for _, velocity in ion_velocity_predictions.iterrows():
    plt.plot(z_normalized, velocity, alpha=0.05, color="blue")  # Transparent lines for density

# Overlay observed ion velocity with dots
plt.plot(z_normalized, observed_ion_velocity, color="red", label="Observed Data", linewidth=2)
plt.scatter(z_normalized, observed_ion_velocity, color="red", label="Observed Data Points")

# Overlay initial ion velocity with dots
plt.plot(z_normalized, initial_ion_velocity, color="green", linestyle="--", label="Initial Metrics", linewidth=2)
plt.scatter(z_normalized, initial_ion_velocity, color="green", label="Initial Data Points")

# Overlay MCMC ion velocity with dots
plt.plot(z_normalized, mcmc_ion_velocity, color="orange", linestyle="--", label="MCMC Metrics", linewidth=2)
plt.scatter(z_normalized, mcmc_ion_velocity, color="orange", label="MCMC Data Points")

# Add labels, title, and legend
plt.xlabel("Normalized Distance (z)")
plt.ylabel("Ion Velocity (m/s)")
plt.title("Ion Velocity Predictions with Data Points and Annotations")
plt.legend()
plt.grid(True)

# Annotate the last points for each metric
last_z = z_normalized[-1]
plt.text(last_z, observed_ion_velocity[-1] + 200, f"{observed_ion_velocity[-1]:.2f}", 
         color="red", fontsize=8, ha="center")
plt.text(last_z, initial_ion_velocity[-1] + 200, f"{initial_ion_velocity[-1]:.2f}", 
         color="green", fontsize=8, ha="center")
plt.text(last_z, mcmc_ion_velocity[-1] - 500, f"{mcmc_ion_velocity[-1]:.2f}", 
         color="orange", fontsize=8, ha="center")

# Annotate the initial points for each metric
first_z = z_normalized[0]
plt.text(first_z, observed_ion_velocity[0] + 200, f"{observed_ion_velocity[0]:.2f}", 
         color="red", fontsize=8, ha="center")
plt.text(first_z, initial_ion_velocity[0] + 500, f"{initial_ion_velocity[0]:.2f}", 
         color="green", fontsize=8, ha="center")
plt.text(first_z, mcmc_ion_velocity[0] + 1000, f"{mcmc_ion_velocity[0]:.2f}", 
         color="orange", fontsize=8, ha="center")

# Save and display the plot
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "ion_velocity_predictions_with_annotations.png"))
plt.show()

# Data for bar chart
discharge_labels = ["Observed", "Initial", "MCMC"]
discharge_values = [observed_discharge_current, initial_discharge_current, mcmc_discharge_current]
colors = ["red", "green", "orange"]

# Create bar chart
plt.figure(figsize=(8, 5))
bars = plt.bar(discharge_labels, discharge_values, color=colors, alpha=0.7, edgecolor="black")

# Add annotations on top of each bar
for bar, value in zip(bars, discharge_values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, 
             f"{value:.2f}", ha="center", fontsize=10)

# Add labels and grid
plt.ylabel("Discharge Current (A)")
plt.title("Discharge Current Comparison: Observed vs. Initial vs. MCMC")
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Save and display the plot
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "discharge_current_bar_comparison_with_annotations.png"))
plt.show()


print(f"Plots saved in {plots_dir}")
