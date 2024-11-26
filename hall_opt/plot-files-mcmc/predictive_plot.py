import os
import numpy as np
import matplotlib.pyplot as plt
import json


def generate_predictive_plot_with_last_sample(z_normalized, observed_data, posterior_prediction, save_path):
    """
    Generate a predictive plot using the posterior prediction from the last MCMC sample.

    Args:
        z_normalized (array-like): Normalized distance values (x-axis).
        observed_data (array-like): Observed data points (y-axis).
        posterior_prediction (array-like): Posterior prediction values from the last MCMC sample.
        save_path (str): File path to save the generated plot.

    Returns:
        None
    """
    # Check for dimension mismatch
    if len(z_normalized) != len(posterior_prediction) or len(z_normalized) != len(observed_data):
        raise ValueError(
            f"Dimension mismatch: z_normalized ({len(z_normalized)}), "
            f"posterior_prediction ({len(posterior_prediction)}), "
            f"observed_data ({len(observed_data)})"
        )

    plt.figure(figsize=(10, 6))

    # Observed data
    plt.scatter(z_normalized, observed_data, color="red", label="Observed Data Points", zorder=5)
    plt.plot(z_normalized, observed_data, color="red", linestyle=":", label="Observed Data (MultiLogBohm)")

    # Posterior prediction
    plt.plot(z_normalized, posterior_prediction, color="blue", label="Posterior Prediction (Last Sample)", linewidth=2)

    # Add title and labels
    plt.xlabel("Normalized Distance (z)")
    plt.ylabel("Ion Velocity")
    plt.title("Predictive Plot with Observed Data")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # Save plot
    plt.savefig(save_path)
    print(f"Predictive plot saved as '{save_path}'.")
    plt.show()


if __name__ == "__main__":
    # File paths
    observed_data_path = "../results-mcmc/mcmc_w_2.0_observed_data_map.json"
    posterior_prediction_path = "../results-mcmc/posterior_prediction_last_sample.json"

    # Load observed data
    with open(observed_data_path, "r") as f:
        observed_data_content = json.load(f)
        z_normalized = np.array(observed_data_content["z_normalized"])
        observed_data = np.array(observed_data_content["ion_velocity"][0])

    # Load posterior prediction
    with open(posterior_prediction_path, "r") as f:
        posterior_prediction_content = json.load(f)
        posterior_prediction = np.array(posterior_prediction_content["posterior_prediction"])

    # Save path
    save_path = "../results-mcmc/predictive_plot_last_sample.png"

    # Generate the plot
    generate_predictive_plot_with_last_sample(z_normalized, observed_data, posterior_prediction, save_path)


