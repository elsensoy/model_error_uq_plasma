import os
import json
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt

# Set ArviZ style
az.style.use("arviz-doc")

# Directory to save plots
PLOTS_DIR = "mcmc-plots"
os.makedirs(PLOTS_DIR, exist_ok=True)  # Ensure plot directory exists

def load_data():
    """Loads MCMC samples, truth data, pre-MCMC target data, and initial parameter guess."""
    # Load MCMC samples
    samples = pd.read_csv("final_mcmc_samples_w_2.0.csv", header=None)
    samples.columns = ["log_v1", "log_alpha"]
    samples["v1"] = 10 ** samples["log_v1"]
    samples["alpha"] = 10 ** samples["log_alpha"]
    samples["v2"] = samples["v1"] * samples["alpha"]

    # Load observed truth data (MultiLogBohm model outputs)
    RESULTS_DIR = os.path.join("..", "results-mcmc")
    with open(os.path.join(RESULTS_DIR, "mcmc_w_2.0_observed_data_map.json"), 'r') as f:
        truth_data = json.load(f)

    # Load pre-MCMC TwoZoneBohm simulation results
    with open(os.path.join(RESULTS_DIR, "mcmc_w_2.0_initial_mcmc.json"), 'r') as f:
        pre_mcmc_target_data = json.load(f)

    # Load initial parameter values used for MCMC (TwoZoneBohm initial guess)
    NELDER_MEAD_DIR = os.path.join("..", "results-Nelder-Mead")
    initial_guess_path = os.path.join(NELDER_MEAD_DIR, "best_initial_guess_w_2_0.json")
    with open(initial_guess_path, 'r') as f:
        initial_params = json.load(f)

    return samples, truth_data, pre_mcmc_target_data, initial_params

def convert_to_inferencedata(samples):
    """Convert the samples DataFrame to ArviZ's InferenceData format."""
    return az.from_pandas(samples)

def plot_trace(idata, initial_params):
    """Trace plot using ArviZ."""
    fig = az.plot_trace(idata, var_names=["log_v1", "log_alpha"])
    fig[0][0].axhline(np.log10(initial_params["v1"]), color='red', linestyle='--', label="Initial Value (log_v1)")
    fig[1][0].axhline(np.log10(initial_params["v2"] / initial_params["v1"]), color='red', linestyle='--', label="Initial Value (log_alpha)")
    fig[0][0].legend()
    fig[1][0].legend()
    plt.savefig(os.path.join(PLOTS_DIR, "trace_plots.png"))
    plt.show()

def plot_posterior(idata, initial_params):
    """Posterior marginal distributions plot using ArviZ."""
    fig = az.plot_posterior(idata, var_names=["log_v1", "log_alpha"])
    fig[0].axvline(np.log10(initial_params["v1"]), color='red', linestyle='--', label="Initial Value (log(v1))")
    fig[1].axvline(np.log10(initial_params["v2"] / initial_params["v1"]), color='red', linestyle='--', label="Initial Value (log(alpha))")
    fig[0].legend()
    fig[1].legend()
    plt.savefig(os.path.join(PLOTS_DIR, "posterior_marginals.png"))
    plt.show()

def plot_autocorrelation(idata):
    """Autocorrelation plot using ArviZ."""
    fig = az.plot_autocorr(idata, var_names=["log_v1", "log_alpha"])
    plt.savefig(os.path.join(PLOTS_DIR, "autocorrelation_plots.png"))
    plt.show()

def plot_pair(idata):
    """Pair plot using ArviZ."""
    az.plot_pair(idata, var_names=["log_v1", "log_alpha"], kind="kde", marginals=True)
    plt.savefig(os.path.join(PLOTS_DIR, "pair_plot.png"))
    plt.show()

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
    if isinstance(truth_data["ion_velocity"], list):
        axes[2].plot(truth_data["ion_velocity"], label="Truth Model (MultiLogBohm)", color='blue')
    else:
        axes[2].plot(truth_data["ion_velocity"].flatten(), label="Truth Model (MultiLogBohm)", color='blue')

    axes[2].axhline(y=v1_mean, color='red', linestyle='--', label="TwoZoneBohm (Posterior Mean)")
    axes[2].axhline(y=np.mean(pre_mcmc_target_data["ion_velocity"]), color='purple', linestyle=':', label="TwoZoneBohm (Initial Simulation)")
    axes[2].set_title("Ion Velocity Comparison")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "predictive_comparison.png"))
    plt.show()

def plot_posterior_comparison(samples_truth, samples_bohm):
    """Plot posterior comparisons for true and TwoZoneBohm model values."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Apply log scaling
    log_samples_truth = np.log10(samples_truth)
    log_samples_bohm = np.log10(samples_bohm)

    # True Model
    az.plot_posterior({'log_v1': log_samples_truth[:, 0]}, ax=axes[0, 0])
    axes[0, 0].set_title('Log-Posterior of v1 (True Values)')
    az.plot_posterior({'log_alpha': log_samples_truth[:, 1]}, ax=axes[0, 1])
    axes[0, 1].set_title('Log-Posterior of alpha (True Values)')

    # TwoZoneBohm Model
    az.plot_posterior({'log_v1': log_samples_bohm[:, 0]}, ax=axes[1, 0])
    axes[1, 0].set_title('Log-Posterior of v1 (TwoZoneBohm Model)')
    az.plot_posterior({'log_alpha': log_samples_bohm[:, 1]}, ax=axes[1, 1])
    axes[1, 1].set_title('Log-Posterior of alpha (TwoZoneBohm Model)')

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust margins
    plt.savefig(os.path.join(PLOTS_DIR, "posterior_comparison_log_scaled.png"))
    plt.close()

def main():
    # Load data and initial settings
    samples, truth_data, pre_mcmc_target_data, initial_params = load_data()

    # Extract truth values for v1 and alpha from v1 and v2
    v1_truth = initial_params["v1"]
    alpha_truth = initial_params["v2"] / initial_params["v1"]

    # Convert truth values to log space if necessary
    log_v1_truth = np.log10(v1_truth)
    log_alpha_truth = np.log10(alpha_truth)

    # Convert samples to InferenceData format
    idata = convert_to_inferencedata(samples)

    # Generate and save plots
    plot_trace(idata, initial_params)
    plot_posterior(idata, initial_params)
    plot_autocorrelation(idata)
    plot_pair(idata)
    plot_predictive_comparison(samples, truth_data, pre_mcmc_target_data)

    # Posterior comparison with initial and true values
    samples_truth = np.column_stack([v1_truth, alpha_truth])
    samples_bohm = np.column_stack([samples["v1"].values, samples["alpha"].values])
    plot_posterior_comparison(samples_truth, samples_bohm)

if __name__ == "__main__":
    main()
