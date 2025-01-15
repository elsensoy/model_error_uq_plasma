import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from common_setup import load_data, get_common_paths, load_iteration_metrics

# Load data
samples, truth_data, pre_mcmc_data, initial_params = load_data()
iteration_metrics = load_iteration_metrics()
paths = get_common_paths()
plots_dir = paths["plots_dir"]

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
    plt.savefig(os.path.join(plots_dir, "autocorrelation_plots.png"))
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
    plt.savefig(os.path.join(plots_dir, "trace_plots.png"))
    plt.close(fig)
    print("Trace plots saved as 'trace_plots.png'")