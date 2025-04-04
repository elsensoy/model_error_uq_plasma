import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ..utils.data_loader import load_data
from ..config.dict import Settings

def generate_plots(settings: Settings):
    """
    Generate and save plots based on iteration metrics dynamically configured via YAML.
    """

    # Determine analysis type
    if settings.general.run_map:
        method = "map"
    elif settings.general.run_mcmc:
        method = "mcmc"
    elif settings.general.run_gen_data:
        method = "gen_data"
    else:
        print("ERROR: No valid method enabled! Skipping plots.")
        return
    
    print(f"Running plotting for {method.upper()}.")

    # Ensure plotting is enabled in YAML settings
    if not settings.plotting.enable:
        print("Plotting is disabled in settings.")
        return

    # Get paths from settings
    plots_dir = settings.plotting.save_dir.format(method=method)
    os.makedirs(plots_dir, exist_ok=True)

    # Load iteration data
    iteration_data_path = Path(f"hall_opt/results/{method}/iteration_log.json")
    if not iteration_data_path.exists():
        print(f"ERROR: Iteration log not found: {iteration_data_path}")
        return

    with open(iteration_data_path, "r") as file:
        iteration_metrics = json.load(file)

    if not iteration_metrics:
        print("ERROR: Iteration metrics are empty. Check input data.")
        return

    # Extract iteration values
    thrust_values = [metric["c1"] for metric in iteration_metrics]
    discharge_values = [metric["alpha"] for metric in iteration_metrics]

    # Ensure non-empty values
    if not thrust_values or not discharge_values:
        raise ValueError("ERROR: Missing iteration metrics.")

    # Compute statistics
    mean_thrust = np.mean(thrust_values)
    last_thrust = thrust_values[-1]
    mean_discharge = np.mean(discharge_values)
    last_discharge = discharge_values[-1]

    # Histogram: Thrust Predictions
    plt.figure(figsize=(10, 6))
    plt.hist(thrust_values, bins=settings.plotting.bins, alpha=0.7, color="blue", label="Thrust Predictions")
    plt.axvline(mean_thrust, color="purple", linestyle="--", label=f"Mean: {mean_thrust:.3f}")
    plt.axvline(last_thrust, color="orange", linestyle="--", label=f"Final (Last Sample): {last_thrust:.3f}")
    plt.xlabel(settings.plotting.thrust_label)
    plt.ylabel("Frequency")
    plt.title("Thrust Predictions Histogram")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "thrust_histogram.png"))
    if settings.plotting.show:
        plt.show()
    plt.close()

    # Histogram: Discharge Current Predictions
    plt.figure(figsize=(10, 6))
    plt.hist(discharge_values, bins=settings.plotting.bins, alpha=0.7, color="purple", label="Discharge Current Predictions")
    plt.axvline(mean_discharge, color="purple", linestyle="--", label=f"Mean: {mean_discharge:.3f}")
    plt.axvline(last_discharge, color="orange", linestyle="--", label=f"Final (Last Sample): {last_discharge:.3f}")
    plt.xlabel(settings.plotting.discharge_label)
    plt.ylabel("Frequency")
    plt.title("Discharge Current Predictions Histogram")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "discharge_current_histogram.png"))
    if settings.plotting.show:
        plt.show()
    plt.close()

    print(f"Plots saved in {plots_dir}")
