import os
import json
import pandas as pd
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
from common_setup import load_data, get_common_paths

# Load data and paths
samples, truth_data, pre_mcmc_data, initial_params = load_data()
paths = get_common_paths()
plots_dir = paths["plots_dir"]

def plot_trace(log_v1, log_alpha, output_dir="plots"):

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot log_v1
    axes[0].plot(log_v1, color='blue', label="log(v1)")
    axes[0].set_title("Trace Plot for log(v1)")
    axes[0].set_ylabel("log(v1)")
    axes[0].grid(True)
    axes[0].legend()

    # Plot log_alpha
    axes[1].plot(log_alpha, color='green', label="log(alpha)")
    axes[1].set_title("Trace Plot for log(alpha)")
    axes[1].set_ylabel("log(alpha)")
    axes[1].set_xlabel("Iteration")
    axes[1].grid(True)
    axes[1].legend()

    # Adjust layout and save the plot
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "trace_plots.png")
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"Trace plots saved to {plot_path}")

def plot_arviz_trace(log_v1, log_alpha, output_dir="plots"):
    """
    Generate ArviZ trace plots for MCMC samples in log space, only showing trace plots.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Prepare the data for ArviZ
    inference_data = az.from_dict(
        posterior={"log_v1": np.array(log_v1), "log_alpha": np.array(log_alpha)}
    )

    # Generate trace plots (only trace, no posterior density)
    fig = az.plot_trace(
        inference_data,
        var_names=["log_v1", "log_alpha"],
        combined=True,
        kind="trace",  # Show only trace plots
        figsize=(12, 6),  
        compact=True,
    )

    # Save the figure
    output_path = os.path.join(output_dir, "arviz_trace_plots.png")
    plt.savefig(output_path, dpi=300)
    plt.close()  # Close the figure to free memory
    print(f"Trace plots saved to {output_path}")

def main():
    # Load data
    samples, truth_data, pre_mcmc_target_data, initial_params = load_data()

    # Log file path for MAP
    log_file = "/mnt/c/Users/MRover/elsensoy/model_error_uq_plasma/hall_opt/map_/results-map/parameters_log.json"
    
    output_dir = "plots"

    # Load iteration data from JSON
    with open(log_file, 'r') as file:
        data = json.load(file)

    # Extract log-space parameters
    log_v1 = [entry['v1_log'] for entry in data]
    log_alpha = [entry['alpha_log'] for entry in data]
    # Plot MAP iterations
    # plot_map_iterations(log_file, output_dir=plots_dir)
    plot_arviz_trace(log_v1, log_alpha, output_dir)
    plot_trace(log_v1, log_alpha, output_dir)

if __name__ == "__main__":
    main()
