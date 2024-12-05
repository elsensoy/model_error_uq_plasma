import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az

# Paths
base_results_dir = "../home/elidasensoy/hall-project/mcmc-results-4"
samples_path = os.path.join(base_results_dir, "/home/elidasensoy/hall-project/mcmc-results-4/final_samples_1.csv")
truth_data_path = os.path.join(base_results_dir, "/home/elidasensoy/hall-project/mcmc-results-4/mcmc_observed_data_map.json")
pre_mcmc_data_path = os.path.join(base_results_dir, "/home/elidasensoy/hall-project/mcmc-results-4/mcmc_pre_mcmc_initial.json")
initial_params_path = os.path.join("..", "results-Nelder-Mead", "best_initial_guess_w_2_0.json")
# metrics_path = os.path.join(base_results_dir, "mcmc_metrics_3.json")
PLOTS_DIR = "/home/elidasensoy/hall-project/mcmc-results-4/plots-4"
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

    # # Load final MCMC metrics
    # with open(metrics_path, 'r') as f:
    #     mcmc_metrics = json.load(f)

    return samples, truth_data, pre_mcmc_target_data, initial_params

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

# Main function
def main():
    # Load data
    samples, truth_data, pre_mcmc_target_data, initial_params = load_data()

    # Generate plots
    plot_autocorrelation(samples)
    plot_trace(samples)
    plot_posterior(samples)
    plot_pair(samples)

if __name__ == "__main__":
    main()
