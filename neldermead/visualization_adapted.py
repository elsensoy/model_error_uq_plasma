import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import pandas as pd
import json

# Directory to save plots
PLOTS_DIR = "mcmc_plots-2"
os.makedirs(PLOTS_DIR, exist_ok=True)  # Ensure plot directory exists

def load_data():
    """Loads MCMC samples, truth data, and initial parameter guess."""
    # Load MCMC samples
    samples = pd.read_csv("final_mcmc_samples_w_2.0.csv", header=None)
    samples.columns = ["log_v1", "log_alpha"]
    samples["v1"] = 10 ** samples["log_v1"]
    samples["alpha"] = 10 ** samples["log_alpha"]
    samples["v2"] = samples["v1"] * samples["alpha"]

    # Load observed truth data
    RESULTS_DIR = os.path.join("..", "results-mcmc")
    with open(os.path.join(RESULTS_DIR, "mcmc_w_2.0_observed_data_map.json"), 'r') as f:
        truth_data = json.load(f)

    # Load pre-MCMC initial simulation data
    with open(os.path.join(RESULTS_DIR, "mcmc_w_2.0_initial_mcmc.json"), 'r') as f:
        pre_mcmc_target_data = json.load(f)

    # Load initial parameter values
    INITIAL_GUESS_PATH = os.path.join("..", "results-Nelder-Mead", "best_initial_guess_w_2_0.json")
    with open(INITIAL_GUESS_PATH, 'r') as f:
        initial_params = json.load(f)

    return samples, truth_data, pre_mcmc_target_data, initial_params

def plot_trace(samples, initial_params):
    """Plot and save trace plots for each parameter."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(samples["log_v1"], color='blue')
    axes[0].axhline(np.log10(initial_params["v1"]), color='red', linestyle='--', label="Initial Parameter Value")
    axes[0].set_title("Trace Plot for log(v1)")
    axes[0].legend()

    axes[1].plot(samples["log_alpha"], color='green')
    axes[1].axhline(np.log10(initial_params["v2"] / initial_params["v1"]), color='red', linestyle='--', label="Initial Parameter Value")
    axes[1].set_title("Trace Plot for log(alpha)")
    axes[1].legend()

    plt.xlabel("Iteration")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "trace_plots.png"))
    plt.close(fig)
    print("Trace plots saved as 'trace_plots.png'")

def plot_autocorrelation(samples):
    """Plot and save autocorrelation for each parameter."""
    # Convert samples to InferenceData format for ArviZ
    inference_data = az.from_dict(posterior={"log_v1": samples["log_v1"].values, "log_alpha": samples["log_alpha"].values})
    
    # Plot autocorrelations using ArviZ
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    az.plot_autocorr(inference_data, var_names=["log_v1"], ax=axes[0])
    axes[0].set_title("Autocorrelation for log(v1)")

    az.plot_autocorr(inference_data, var_names=["log_alpha"], ax=axes[1])
    axes[1].set_title("Autocorrelation for log(alpha)")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "autocorrelation_plots.png"))
    plt.close(fig)
    print("Autocorrelation plots saved as 'autocorrelation_plots.png'")

def plot_posterior_marginals(samples, initial_params):
    """Plot and save posterior marginal distributions with initial values."""
    posterior = az.from_dict(posterior={"log_v1": samples["log_v1"], "log_alpha": samples["log_alpha"]})
    az.plot_posterior(posterior)
    plt.savefig(os.path.join(PLOTS_DIR, "posterior_marginals.png"))
    plt.close()
    print("Posterior marginals saved as 'posterior_marginals.png'")

def plot_pair(samples):
    """Plot and save pair plot for joint distributions of samples."""
    posterior = az.from_dict(posterior={"log_v1": samples["log_v1"], "log_alpha": samples["log_alpha"]})
    az.plot_pair(posterior, var_names=["log_v1", "log_alpha"], kind='kde', marginals=True)
    plt.savefig(os.path.join(PLOTS_DIR, "pair_plot.png"))
    plt.close()
    print("Pair plot saved as 'pair_plot.png'")

def plot_predictive_comparison(samples, truth_data, pre_mcmc_target_data):
    """Plot and save predictive comparison for thrust, discharge current, and ion velocity."""
    posterior_means = samples[["v1", "v2"]].mean()
    v1_mean, v2_mean = posterior_means["v1"], posterior_means["v2"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # Thrust Comparison
    axes[0].plot(truth_data["thrust"], label="Truth Model (MultiLogBohm)", color='blue')
    axes[0].axhline(y=v1_mean, color='red', linestyle='--', label="TwoZoneBohm (Posterior Mean)")
    axes[0].axhline(y=pre_mcmc_target_data["thrust"], color='purple', linestyle=':', label="TwoZoneBohm (Initial Simulation)")
    axes[0].set_title("Thrust Comparison")
    axes[0].legend()

    # Discharge Current Comparison
    axes[1].plot(truth_data["discharge_current"], label="Truth Model (MultiLogBohm)", color='blue')
    axes[1].axhline(y=v2_mean, color='red', linestyle='--', label="TwoZoneBohm (Posterior Mean)")
    axes[1].axhline(y=pre_mcmc_target_data["discharge_current"], color='purple', linestyle=':', label="TwoZoneBohm (Initial Simulation)")
    axes[1].set_title("Discharge Current Comparison")
    axes[1].legend()

    # Ion Velocity Comparison
    axes[2].plot(truth_data["ion_velocity"], label="Truth Model (MultiLogBohm)", color='blue')
    axes[2].plot(range(len(truth_data["ion_velocity"])), [v1_mean] * len(truth_data["ion_velocity"]),
                 color='red', linestyle='--', label="TwoZoneBohm (Posterior Mean)")
    axes[2].plot(range(len(truth_data["ion_velocity"])), [pre_mcmc_target_data["ion_velocity"]] * len(truth_data["ion_velocity"]),
                 color='purple', linestyle=':', label="TwoZoneBohm (Initial Simulation)")
    axes[2].set_title("Ion Velocity Comparison")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "predictive_comparison.png"))
    plt.close(fig)
    print("Predictive comparison saved as 'predictive_comparison.png'")

def main():
    # Load data and initial settings
    samples, truth_data, pre_mcmc_target_data, initial_params = load_data()

    # Generate and save plots
    plot_trace(samples, initial_params)
    plot_autocorrelation(samples)
    plot_posterior_marginals(samples, initial_params)
    plot_pair(samples)
    plot_predictive_comparison(samples, truth_data, pre_mcmc_target_data)

if __name__ == "__main__":
    main()
