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
base_results_dir = os.path.join("..", "mcmc-results-12-3-24")
plots_dir = os.path.join(base_results_dir, "plots-12-3-24")
metrics_dir = os.path.join(base_results_dir, "iteration_metrics")
os.makedirs(plots_dir, exist_ok=True)

# Load Data
def load_data():
    """Loads MCMC samples, truth data, initial parameter guess, and metrics."""
    samples_path = os.path.join(base_results_dir, "final_mcmc_samples_12_3_w_2.0_2.csv")
    truth_data_path = os.path.join(base_results_dir, "mcmc_w_2.0_observed_data_map.json")
    pre_mcmc_data_path = os.path.join(base_results_dir, "mcmc_w_2.0_initial_mcmc.json")
    initial_params_path = os.path.join("..", "results-Nelder-Mead", "best_initial_guess_w_2_0.json")

    # Check paths
    for path in [samples_path, truth_data_path, pre_mcmc_data_path, initial_params_path]:
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

    return samples, truth_data, pre_mcmc_target_data, initial_params

# Load Metrics from JSON Files
def load_iteration_metrics(metrics_dir):
    """Load all metrics from individual iteration JSON files."""
    metrics = []
    for file in sorted(os.listdir(metrics_dir)):
        if file.endswith(".json"):
            with open(os.path.join(metrics_dir, file), 'r') as f:
                metrics.append(json.load(f))
    return metrics

# Load data
samples, truth_data, pre_mcmc_data, initial_params = load_data()
iteration_metrics = load_iteration_metrics(metrics_dir)

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
def plot_ion_velocity_deltas_all(z_normalized, observed_ion_velocity, initial_ion_velocity, ion_velocity_values, save_path):
    """
    Plot the delta (difference from observed ion velocity) for all MCMC iterations and the initial guess.
    """
    plt.figure(figsize=(12, 8))

    # Convert lists to numpy arrays
    z_normalized = np.array(z_normalized)
    observed_ion_velocity = np.array(observed_ion_velocity)
    initial_ion_velocity = np.array(initial_ion_velocity)

    # Debugging data shapes
    print(f"z_normalized shape: {z_normalized.shape}")
    print(f"observed_ion_velocity shape: {observed_ion_velocity.shape}")
    print(f"initial_ion_velocity shape: {initial_ion_velocity.shape}")
    print(f"Number of iterations: {len(ion_velocity_values)}")
    print(f"Shape of first iteration ion velocity: {np.array(ion_velocity_values[0]).shape}")

    # Calculate deltas
    initial_delta = initial_ion_velocity - observed_ion_velocity

    # Plot initial delta
    plt.plot(z_normalized, initial_delta, linestyle="--", color="purple", linewidth=1.5, label="Initial Delta")
    plt.scatter(z_normalized, initial_delta, color="purple", s=50, label="Initial Data Points")

    # Plot deltas for all MCMC iterations
    for i, iteration_velocity in enumerate(ion_velocity_values):
        iteration_velocity = np.array(iteration_velocity)  # Ensure iteration data is a numpy array
        iteration_delta = iteration_velocity - observed_ion_velocity
        # Label only the first iteration to avoid legend clutter
        label = "MCMC Iteration Deltas" if i == 0 else None
        plt.plot(z_normalized, iteration_delta, linestyle="-", color="teal", alpha=0.6, label=label)

    # Observed baseline
    plt.axhline(0, color="black", linestyle="--", linewidth=1.5, label="Observed Baseline")

    # Customize plot
    plt.title("Ion Velocity Delta Comparison (All Iterations)", fontsize=14)
    plt.xlabel("Normalized z", fontsize=12)
    plt.ylabel("Delta (Ion Velocity, m/s)", fontsize=12)
    plt.grid(True, linestyle=":", linewidth=0.7)
    plt.legend(fontsize=10, loc="upper right")
    plt.tight_layout()

    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the plot as PNG
    plt.savefig(save_path, format='png')
    plt.close()
    print(f"Ion velocity delta plot (all iterations) saved: {save_path}")
plot_ion_velocity_deltas_all(
    z_normalized,
    observed_ion_velocity,
    initial_ion_velocity,
    ion_velocity_values,
    save_path=os.path.join(plots_dir, "ion_velocity_delta_all_iterations.png")
)

def plot_mcmc_delta_bar_chart(deltas, labels, ylabel, title, save_path):
    """
    Plots a bar chart for MCMC deltas (difference from observed values).
    Simplified legend for clarity.
    """
    plt.figure(figsize=(10, 6))
    x_positions = np.arange(len(deltas))  # Positions for bars

    # Create the bar chart
    plt.bar(x_positions, deltas, color=["teal", "orange", "maroon"], alpha=0.8)

    # Annotate each bar with its value
    for i, delta in enumerate(deltas):
        plt.text(i, delta + 0.05 * max(abs(np.array(deltas))), f"{delta:.2f}", ha="center", fontsize=10)

    # Add observed baseline
    plt.axhline(0, color="black", linestyle="--", linewidth=1.5, label="Observed Baseline")

    # Customize labels and grid
    plt.xticks(x_positions, labels, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(axis="y", linestyle=":", alpha=0.7)
    plt.legend(fontsize=10, loc="upper right")
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path)
    plt.close()
    print(f"Delta bar chart saved: {save_path}")


# Compute thrust deltas
thrust_deltas = [
    np.mean(thrust_values) - observed_thrust,  # Mean delta
    thrust_values[-1] - observed_thrust,      # Last MCMC sample delta
    initial_thrust - observed_thrust          # Initial delta
]
thrust_labels = ["Mean (200 Iterations)", "Last Sample", "Initial"]
plot_mcmc_delta_bar_chart(
    thrust_deltas,
    thrust_labels,
    ylabel="Delta (Thrust, mN)",
    title="Thrust Delta Comparison",
    save_path=os.path.join(plots_dir, "thrust_delta_bar_chart.png")
)

# Compute discharge current deltas
discharge_deltas = [
    np.mean(discharge_values) - observed_discharge_current,  # Mean delta
    discharge_values[-1] - observed_discharge_current,      # Last MCMC sample delta
    initial_discharge_current - observed_discharge_current  # Initial delta
]
discharge_labels = ["Mean (200 Iterations)", "Last Sample", "Initial"]
plot_mcmc_delta_bar_chart(
    discharge_deltas,
    discharge_labels,
    ylabel="Delta (Discharge Current, A)",
    title="Discharge Current Delta Comparison",
    save_path=os.path.join(plots_dir, "discharge_current_delta_bar_chart.png")
)

def plot_initial_delta(z_normalized, observed_ion_velocity, initial_ion_velocity, save_path):
    """
    Plot the delta (difference from observed ion velocity) for the initial guess.
    """
    plt.figure(figsize=(12, 8))

    # Convert lists to numpy arrays
    z_normalized = np.array(z_normalized)
    observed_ion_velocity = np.array(observed_ion_velocity)
    initial_ion_velocity = np.array(initial_ion_velocity)

    # Calculate initial delta
    initial_delta = initial_ion_velocity - observed_ion_velocity

    # Plot initial delta
    plt.plot(z_normalized, initial_delta, linestyle="--", color="gray", linewidth=2, label="Initial Delta")
    plt.scatter(z_normalized, initial_delta, color="gray", s=50, label="Initial Data Points")

    # Observed baseline
    plt.axhline(0, color="black", linestyle="--", linewidth=1.5, label="Observed Baseline")

    plt.title("Ion Velocity Initial Delta", fontsize=14)
    plt.xlabel("Normalized z", fontsize=12)
    plt.ylabel("Delta (Ion Velocity, m/s)", fontsize=12)
    plt.grid(True, linestyle=":", linewidth=0.7)
    plt.legend(fontsize=10, loc="upper right")
    plt.tight_layout()

    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the plot
    plt.savefig(save_path, format='png')
    plt.close()
    print(f"Initial delta plot saved: {save_path}")

plot_initial_delta(
    z_normalized,
    observed_ion_velocity,
    initial_ion_velocity,
    save_path=os.path.join(plots_dir, "initial_delta_plot.png")
)
def plot_ion_velocity_with_last_values(
    z_normalized, observed_ion_velocity, initial_ion_velocity, ion_velocity_values, save_path
):

    plt.figure(figsize=(12, 8))

    # Convert lists to numpy arrays
    z_normalized = np.array(z_normalized)
    observed_ion_velocity = np.array(observed_ion_velocity)
    initial_ion_velocity = np.array(initial_ion_velocity)

    # Calculate deltas
    initial_delta = initial_ion_velocity - observed_ion_velocity
    last_iteration_velocity = np.array(ion_velocity_values[-1])
    last_iteration_delta = last_iteration_velocity - observed_ion_velocity

    # Plot initial delta
    plt.plot(z_normalized, initial_delta, linestyle="--", color="purple", linewidth=2, label="Initial Delta")
    plt.scatter(z_normalized, initial_delta, color="purple", s=50, label="Initial Data Points")

    # Plot deltas for all MCMC iterations
    for i, iteration_velocity in enumerate(ion_velocity_values):
        iteration_velocity = np.array(iteration_velocity)  # Ensure iteration data is a numpy array
        iteration_delta = iteration_velocity - observed_ion_velocity
        # Label only the first iteration to avoid legend clutter
        label = "MCMC Iteration Deltas" if i == 0 else None
        plt.plot(
            z_normalized, iteration_delta, linestyle="-", color="teal", alpha=0.6, label=label
        )

    # Highlight last iteration delta
    plt.plot(z_normalized, last_iteration_delta, linestyle="-", color="orange", linewidth=2, label="Last Iteration Delta")
    plt.scatter(z_normalized, last_iteration_delta, color="orange", s=50, label="Last Iteration Data Points")

    # Annotate last iteration delta points
    for x, y in zip(z_normalized, last_iteration_delta):
        plt.text(x, y, f"{y:.2f}", fontsize=8, color="orange", ha="center", va="bottom")

    # Observed baseline
    plt.axhline(0, color="black", linestyle="--", linewidth=1.5, label="Observed Baseline")

    # Customize plot
    plt.title("Ion Velocity Delta Comparison (Last Iteration Annotated)", fontsize=14)
    plt.xlabel("Normalized z", fontsize=12)
    plt.ylabel("Delta (Ion Velocity, m/s)", fontsize=12)
    plt.grid(True, linestyle=":", linewidth=0.7)
    plt.legend(fontsize=10, loc="upper right")
    plt.tight_layout()

    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the plot as PNG
    plt.savefig(save_path, format="png")
    plt.close()
    print(f"Ion velocity delta plot with last iteration values saved: {save_path}")


# Call the function for ion velocity delta plot with last iteration values annotated
plot_ion_velocity_with_last_values(
    z_normalized,
    observed_ion_velocity,
    initial_ion_velocity,
    ion_velocity_values,
    save_path=os.path.join(plots_dir, "ion_velocity_delta_last_iteration_annotated.png"),
)
