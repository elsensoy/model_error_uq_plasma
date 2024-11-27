import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az

# Paths
base_results_dir = "../mcmc-results-11-25-24"
samples_path = os.path.join(base_results_dir, "final_mcmc_samples_2_w_2.0_2.csv")
truth_data_path = os.path.join(base_results_dir, "mcmc_w_2.0_observed_data_map.json")
pre_mcmc_data_path = os.path.join(base_results_dir, "mcmc_w_2.0_initial_mcmc.json")
initial_params_path = os.path.join("..", "results-Nelder-Mead", "best_initial_guess_w_2_0.json")
metrics_path = os.path.join(base_results_dir, "mcmc_metrics_3.json")
PLOTS_DIR = "plots-11-25-24"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Load Data
def load_data():
   
    # Load MCMC samples
    samples = pd.read_csv(samples_path, header=None)
    samples.columns = ["log_v1", "log_alpha"]
    samples["v1"] = 10 ** samples["log_v1"]
    samples["alpha"] = 10 ** samples["log_alpha"]
    samples["v2"] = samples["v1"] * samples["alpha"]

    # Load observed truth data
    with open(truth_data_path, 'r') as f:
        truth_data = json.load(f)

    # Load pre-MCMC initial simulation data
    with open(pre_mcmc_data_path, 'r') as f:
        pre_mcmc_target_data = json.load(f)

    # Load initial parameters
    with open(initial_params_path, 'r') as f:
        initial_params = json.load(f)

    # Load final MCMC metrics
    with open(metrics_path, 'r') as f:
        mcmc_metrics = json.load(f)

    return samples, truth_data, pre_mcmc_target_data, initial_params, mcmc_metrics


def plot_autocorrelation(samples):
    """Autocorrelation plots for MCMC samples."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Autocorrelation for log_v1
    az.plot_autocorr(samples["log_v1"].values, ax=axes[0])  # Convert Series to NumPy array
    axes[0].set_title("Autocorrelation for log(v1)")

    # Autocorrelation for log_alpha
    az.plot_autocorr(samples["log_alpha"].values, ax=axes[1])  # Convert Series to NumPy array
    axes[1].set_title("Autocorrelation for log(alpha)")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "autocorrelation_plots.png"))
    plt.close(fig)
    print("Autocorrelation plots saved as 'autocorrelation_plots.png'")


# Plotting Functions
def plot_trace(samples):
    """Trace plots for MCMC samples."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(samples["log_v1"], color='blue')
    axes[0].set_title("Trace Plot for log(v1)")

    axes[1].plot(samples["log_alpha"], color='green')
    axes[1].set_title("Trace Plot for log(alpha)")

    plt.xlabel("Iteration")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "trace_plots.png"))
    plt.close(fig)
    print("Trace plots saved as 'trace_plots.png'")

def plot_posterior(samples):
    """Posterior marginal distributions."""
    posterior = az.from_dict(posterior={"log_v1": samples["log_v1"], "log_alpha": samples["log_alpha"]})
    az.plot_posterior(posterior)
    plt.savefig(os.path.join(PLOTS_DIR, "posterior_marginals.png"))
    plt.close()
    print("Posterior marginals saved as 'posterior_marginals.png'")

def plot_pair(samples):
    """Pair plot for joint distributions."""
    sns.pairplot(samples[["log_v1", "log_alpha"]], kind="scatter", diag_kind="kde", corner=True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "pair_plot.png"))
    plt.close()
    print("Pair plot saved as 'pair_plot.png'")

def compare_with_observed(truth_data, pre_mcmc_target_data, mcmc_metrics):
    """Compare observed, initial, and MCMC metrics."""
    # Extract metrics
    observed_thrust = truth_data["thrust"][0]
    initial_thrust = pre_mcmc_target_data["thrust"][0]
    mcmc_thrust = mcmc_metrics["thrust"]

    observed_discharge = truth_data["discharge_current"][0]
    initial_discharge = pre_mcmc_target_data["discharge_current"][0]
    mcmc_discharge = mcmc_metrics["discharge_current"]

    # Compare thrust
    plt.figure(figsize=(8, 6))
    labels = ["Observed", "Initial", "MCMC"]
    thrust_values = [observed_thrust, initial_thrust, mcmc_thrust]
    plt.bar(labels, thrust_values, color=["red", "green", "blue"], alpha=0.7, edgecolor="black")
    plt.ylabel("Thrust (N)")
    plt.title("Thrust Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "thrust_comparison.png"))
    plt.close()
    print("Thrust comparison saved as 'thrust_comparison.png'")

    # Compare discharge current
    plt.figure(figsize=(8, 6))
    discharge_values = [observed_discharge, initial_discharge, mcmc_discharge]
    plt.bar(labels, discharge_values, color=["red", "green", "blue"], alpha=0.7, edgecolor="black")
    plt.ylabel("Discharge Current (A)")
    plt.title("Discharge Current Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "discharge_comparison.png"))
    plt.close()
    print("Discharge current comparison saved as 'discharge_comparison.png'")

# Main function
def main():
    # Load data
    samples, truth_data, pre_mcmc_target_data, initial_params, mcmc_metrics = load_data()

    # Generate plots
    plot_autocorrelation(samples)
    plot_trace(samples)
    plot_posterior(samples)
    plot_pair(samples)
    compare_with_observed(truth_data, pre_mcmc_target_data, mcmc_metrics)

if __name__ == "__main__":
    main()
