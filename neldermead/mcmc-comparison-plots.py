import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# Paths
results_dir = os.path.join("..", "mcmc-results-11-23-24")
plots_dir = os.path.join(results_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

def load_data():
    """Loads MCMC samples, truth data, initial parameter guess, and metrics."""
    # Load MCMC samples (iteration results)
    samples = pd.read_csv(
        "../results-mcmc/final_mcmc_samples_w_2.0_2.csv", 
        header=None,
        names=["log_v1", "log_alpha"]
    )
    samples["v1"] = 10 ** samples["log_v1"]
    samples["alpha"] = 10 ** samples["log_alpha"]
    samples["v2"] = samples["v1"] * samples["alpha"]

    # Load observed truth data
    RESULTS_DIR = os.path.join("..", "mcmc-results-11-23-24")
    with open(os.path.join(RESULTS_DIR, "mcmc_w_2.0_observed_data_map.json"), 'r') as f:
        truth_data = json.load(f)

    # Load pre-MCMC initial simulation data (MAP results)
    with open(os.path.join(RESULTS_DIR, "mcmc_w_2.0_initial_mcmc.json"), 'r') as f:
        pre_mcmc_target_data = json.load(f)

    # Load initial parameter values (Nelder-Mead optimized guesses)
    INITIAL_GUESS_PATH = os.path.join("..", "results-Nelder-Mead", "best_initial_guess_w_2_0.json")
    with open(INITIAL_GUESS_PATH, 'r') as f:
        initial_params = json.load(f)

    # Load final MCMC metrics (converged results)
    with open(os.path.join(RESULTS_DIR, "mcmc-metrics.json"), 'r') as f:
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
mcmc_ion_velocity = mcmc_metrics["ion_velocity"]

# Placeholder for MCMC ion velocity predictions (replace with actual logic)
ion_velocity_predictions = pd.DataFrame(
    data=[[observed_ion_velocity] for _ in range(200)], 
    columns=z_normalized
)

# 1. Iteration Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(range(len(samples)), samples["v1"], s=10, label="v1 Samples", alpha=0.7)
plt.scatter(range(len(samples)), samples["v2"], s=10, label="v2 Samples", alpha=0.7)
plt.xlabel("Iteration")
plt.ylabel("Sample Value")
plt.title("MCMC Parameter Samples vs Iteration")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plots_dir, "iteration_scatter_plot.png"))
plt.show()

# 2. Histogram: Thrust
plt.figure(figsize=(10, 6))
plt.hist(samples["v1"], bins=30, alpha=0.7, color="blue", label="MCMC Thrust Predictions")
plt.axvline(observed_thrust, color="red", linestyle="--", label="Observed Data")
plt.axvline(initial_thrust, color="green", linestyle="--", label="Initial Metrics")
plt.axvline(mcmc_thrust, color="orange", linestyle="--", label="MCMC Metrics")
plt.xlabel("Thrust (N)")
plt.ylabel("Frequency")
plt.title("Thrust Predictions Histogram")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plots_dir, "thrust_histogram.png"))
plt.show()

# 3. Histogram: Discharge Current
plt.figure(figsize=(10, 6))
plt.hist(samples["v2"], bins=30, alpha=0.7, color="purple", label="MCMC Discharge Current Predictions")
plt.axvline(observed_discharge_current, color="red", linestyle="--", label="Observed Data")
plt.axvline(initial_discharge_current, color="green", linestyle="--", label="Initial Metrics")
plt.axvline(mcmc_discharge_current, color="orange", linestyle="--", label="MCMC Metrics")
plt.xlabel("Discharge Current (A)")
plt.ylabel("Frequency")
plt.title("Discharge Current Predictions Histogram")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plots_dir, "discharge_current_histogram.png"))
plt.show()

# 4. Ion Velocity Predictions
plt.figure(figsize=(12, 8))
for _, velocity in ion_velocity_predictions.iterrows():
    plt.plot(z_normalized, velocity, alpha=0.1, color="blue")

# Overlay Observed, Initial, and MCMC Metrics
plt.plot(z_normalized, observed_ion_velocity, color="red", label="Observed Data")
plt.plot(z_normalized, initial_ion_velocity, color="green", linestyle="--", label="Initial Metrics")
plt.plot(z_normalized, mcmc_ion_velocity, color="orange", linestyle="--", label="MCMC Metrics")

plt.xlabel("Normalized Distance (z)")
plt.ylabel("Ion Velocity (m/s)")
plt.title("Ion Velocity Predictions")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plots_dir, "ion_velocity_predictions.png"))
plt.show()

print(f"Plots saved in {plots_dir}")
