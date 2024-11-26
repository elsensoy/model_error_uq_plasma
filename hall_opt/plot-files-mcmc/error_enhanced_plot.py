import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error, mean_absolute_error
from map_nelder_mead import hallthruster_jl_wrapper, config_multilogbohm, save_results_to_json

config_spt_100 = config_multilogbohm.copy()
config_spt_100["anom_model"] = "TwoZoneBohm"

def subsample_data(data, step=10):
    """Subsample the data by taking every nth element."""
    if isinstance(data, list):
        return data[::step]  # Every nth element from the list
    return data

# def generate_posterior_prediction_last_sample(samples, config, save_filename, subsample_step=10):
#     """Generate posterior prediction using the last MCMC sample and save the result."""
#     # Extract the last sample
#     last_sample = samples.iloc[-1]
#     v1 = last_sample["v1"]
#     v2 = last_sample["v2"]

#     try:
#         # Run the TwoZoneBohm simulation with the last MCMC sample
#         print(f"Running TwoZoneBohm simulation with v1={v1}, v2={v2}...")
#         simulation_result = hallthruster_jl_wrapper(
#             v1, v2, 
#             config, 
#             use_time_averaged=True, 
#             save_every_n_grid_points=subsample_step
#         )

#         # Extract ion velocity predictions
#         ion_velocity = np.array(simulation_result["ion_velocity"])
#         posterior_prediction = np.mean(ion_velocity, axis=0)  # Mean over space/time

#         # Save the posterior prediction
#         save_results_to_json(
#             {
#                 "z_normalized": simulation_result["z_normalized"],
#                 "posterior_prediction": posterior_prediction.tolist()
#             },
#             save_filename,
#             save_every_n_grid_points=subsample_step
#         )

#         print(f"Posterior prediction saved to {save_filename}.")
#         return posterior_prediction, simulation_result["z_normalized"]

#     except Exception as e:
#         print(f"Error during TwoZoneBohm simulation for v1={v1}, v2={v2}: {e}")
#         return None, None
def generate_posterior_prediction_last_sample(samples, config, save_filename, subsample_step=10):
    """Generate posterior prediction using the last MCMC sample and save the result."""
    # Extract the last sample
    last_sample = samples.iloc[-1]
    v1 = last_sample["v1"]
    v2 = last_sample["v2"]

    try:
        # Run the TwoZoneBohm simulation with the last MCMC sample
        print(f"Running TwoZoneBohm simulation with v1={v1}, v2={v2}...")
        simulation_result = hallthruster_jl_wrapper(
            v1, v2, 
            config, 
            use_time_averaged=True, 
            save_every_n_grid_points=subsample_step
        )

        # Extract ion velocity predictions
        ion_velocity = np.array(simulation_result["ion_velocity"])
        posterior_prediction = np.mean(ion_velocity, axis=0)  # Take the mean over space/time

        # Subsample the posterior prediction and normalized distance
        z_normalized_subsampled = simulation_result["z_normalized"][::subsample_step]
        posterior_prediction_subsampled = posterior_prediction[::subsample_step]

        # Save the posterior prediction
        save_results_to_json(
            {
                "z_normalized": z_normalized_subsampled,
                "posterior_prediction": posterior_prediction_subsampled.tolist()
            },
            save_filename,
            save_every_n_grid_points=subsample_step,
            subsample_for_saving=False  # Subsampling already applied
        )

        print(f"Posterior prediction saved to {save_filename}.")
        return posterior_prediction_subsampled, z_normalized_subsampled

    except Exception as e:
        print(f"Error during TwoZoneBohm simulation for v1={v1}, v2={v2}: {e}")
        return None, None

def plot_comparison_with_error_and_truth(z_normalized, posterior_prediction, observed_data, initial_data):
    """
    Plot a comparison between observed data, initial (MAP) data, and posterior predictions.
    Correctly compute the model error bands to avoid artifacts at overlaps or crossings.
    """
    # Calculate model errors (difference at corresponding y-points)
    posterior_model_error_upper = np.maximum(posterior_prediction, observed_data)
    posterior_model_error_lower = np.minimum(posterior_prediction, observed_data)

    initial_model_error_upper = np.maximum(initial_data, observed_data)
    initial_model_error_lower = np.minimum(initial_data, observed_data)

    plt.figure(figsize=(10, 6))

    # Observed data
    plt.plot(
        z_normalized,
        observed_data,
        label="Observed Data (MultiLogBohm)",
        color="red",
        linestyle=":",
        linewidth=2,
    )
    plt.scatter(
        z_normalized,
        observed_data,
        color="red",
        label="Observed Data Points",
        zorder=5,
    )

    # Initial data (MAP)
    plt.plot(
        z_normalized,
        initial_data,
        label="Initial Guess Prediction (MAP)",
        color="black",
        linestyle="--",
        linewidth=1.5,
    )
    plt.scatter(
        z_normalized,
        initial_data,
        color="black",
        label="Initial Data Points (MAP)",
        zorder=5,
    )

    # Posterior predictions
    plt.plot(
        z_normalized,
        posterior_prediction,
        label="Posterior Prediction (Last Sample)",
        color="blue",
        linewidth=2,
    )

    # Posterior error band
    plt.fill_between(
        z_normalized,
        posterior_model_error_lower,
        posterior_model_error_upper,
        color="blue",
        alpha=0.3,
        label="Posterior Model Error Band",
    )

    # Initial error band
    plt.fill_between(
        z_normalized,
        initial_model_error_lower,
        initial_model_error_upper,
        color="purple",
        alpha=0.3,
        label="Initial Model Error Band",
    )

    plt.xlabel("Normalized Distance (z)")
    plt.ylabel("Ion Velocity")
    plt.title("Comparison of Predictions with Observed Data and Model Error Bands")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(
        "..", "neldermead/mcmc-plots-1123", "comparison_with_corrected_error_bands_y_points.png"
    )
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}.")
    plt.show()


def plot_comparison_with_subplot(z_normalized, observed_data, initial_data, posterior_prediction):
    """
    Plot comparison between initial predictions (MAP), posterior predictions, and observed data,
    with error bands calculated using point-wise differences (y-points only).
    """
    # Calculate model errors (difference at corresponding y-points)
    posterior_model_error_upper = np.maximum(posterior_prediction, observed_data)
    posterior_model_error_lower = np.minimum(posterior_prediction, observed_data)

    initial_model_error_upper = np.maximum(initial_data, observed_data)
    initial_model_error_lower = np.minimum(initial_data, observed_data)

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # Subplot 1: Initial Prediction vs Observed Data
    axes[0].plot(
        z_normalized,
        observed_data,
        label="Observed Data (MultiLogBohm)",
        color="red",
        linestyle=":",
        linewidth=2,
    )
    axes[0].scatter(
        z_normalized,
        observed_data,
        color="red",
        label="Observed Data Points",
        zorder=5,
    )
    axes[0].plot(
        z_normalized,
        initial_data,
        label="Initial Guess Prediction (MAP)",
        color="black",
        linestyle="--",
        linewidth=1.5,
    )
    axes[0].scatter(
        z_normalized,
        initial_data,
        color="black",
        label="Initial Data Points (MAP)",
        zorder=5,
    )
    axes[0].fill_between(
        z_normalized,
        initial_model_error_lower,
        initial_model_error_upper,
        color="purple",
        alpha=0.3,
        label="Initial Model Error Band",
    )
    axes[0].set_title("Initial Predictions vs Observed Data")
    axes[0].set_xlabel("Normalized Distance (z)")
    axes[0].set_ylabel("Ion Velocity")
    axes[0].legend()
    axes[0].grid()

    # Subplot 2: Posterior Prediction vs Observed Data
    axes[1].plot(
        z_normalized,
        observed_data,
        label="Observed Data (MultiLogBohm)",
        color="red",
        linestyle=":",
        linewidth=2,
    )
    axes[1].scatter(
        z_normalized,
        observed_data,
        color="red",
        label="Observed Data Points",
        zorder=5,
    )
    axes[1].plot(
        z_normalized,
        posterior_prediction,
        label="Posterior Prediction (Last Sample)",
        color="blue",
        linewidth=2,
    )
    axes[1].fill_between(
        z_normalized,
        posterior_model_error_lower,
        posterior_model_error_upper,
        color="blue",
        alpha=0.3,
        label="Posterior Model Error Band",
    )
    axes[1].set_title("Posterior Predictions vs Observed Data")
    axes[1].set_xlabel("Normalized Distance (z)")
    axes[1].legend()
    axes[1].grid()

    # Adjust layout and save
    plt.tight_layout()
    plot_path = os.path.join(
        "..", "neldermead/mcmc-plots-1123", "comparison_subplot_corrected_error_bands.png"
    )
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}.")
    plt.show()

def plot_comparison_with_corrected_error(
    z_normalized, observed_data, initial_data, posterior_prediction
):
    """
    Plot comparison between initial predictions (MAP), posterior predictions, and observed data,
    with corrected error bands that avoid artificial symmetry.
    """
    # Interpolation
    z_fine = np.linspace(z_normalized[0], z_normalized[-1], 500)
    observed_interp = interp1d(z_normalized, observed_data, kind="cubic")(z_fine)
    initial_interp = interp1d(z_normalized, initial_data, kind="cubic")(z_fine)
    posterior_interp = interp1d(z_normalized, posterior_prediction, kind="cubic")(z_fine)

    # Correctly calculate error bands
    initial_model_error_upper = np.maximum(initial_interp, observed_interp)
    initial_model_error_lower = np.minimum(initial_interp, observed_interp)

    posterior_model_error_upper = np.maximum(posterior_interp, observed_interp)
    posterior_model_error_lower = np.minimum(posterior_interp, observed_interp)

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # Subplot 1: Initial Prediction vs Observed Data
    axes[0].plot(
        z_fine,
        observed_interp,
        label="Observed Data (MultiLogBohm)",
        color="red",
        linestyle=":",
        linewidth=2,
    )
    axes[0].scatter(
        z_normalized,
        observed_data,
        color="red",
        label="Observed Data Points",
        zorder=5,
    )
    axes[0].plot(
        z_fine,
        initial_interp,
        label="Initial Guess Prediction (MAP)",
        color="black",
        linestyle="--",
        linewidth=1.5,
    )
    axes[0].scatter(
        z_normalized,
        initial_data,
        color="black",
        label="Initial Data Points (MAP)",
        zorder=5,
    )
    axes[0].fill_between(
        z_fine,
        initial_model_error_lower,
        initial_model_error_upper,
        color="purple",
        alpha=0.3,
        label="Initial Model Error Band",
    )
    axes[0].set_title("Initial Predictions vs Observed Data")
    axes[0].set_xlabel("Normalized Distance (z)")
    axes[0].set_ylabel("Ion Velocity")
    axes[0].legend()
    axes[0].grid()

    # Subplot 2: Posterior Prediction vs Observed Data
    axes[1].plot(
        z_fine,
        observed_interp,
        label="Observed Data (MultiLogBohm)",
        color="red",
        linestyle=":",
        linewidth=2,
    )
    axes[1].scatter(
        z_normalized,
        observed_data,
        color="red",
        label="Observed Data Points",
        zorder=5,
    )
    axes[1].plot(
        z_fine,
        posterior_interp,
        label="Posterior Prediction (Last Sample)",
        color="blue",
        linewidth=2,
    )
    axes[1].fill_between(
        z_fine,
        posterior_model_error_lower,
        posterior_model_error_upper,
        color="blue",
        alpha=0.3,
        label="Posterior Model Error Band",
    )
    axes[1].set_title("Posterior Predictions vs Observed Data")
    axes[1].set_xlabel("Normalized Distance (z)")
    axes[1].legend()
    axes[1].grid()

    # Adjust layout and save
    plt.tight_layout()
    plot_path = os.path.join(
        "..", "neldermead/mcmc-plots-1123", "comparison_corrected_error_bands.png"
    )
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}.")
    plt.show()

def plot_combined_corrected_error_bands(
    z_normalized, observed_data, initial_data, posterior_prediction
):
    """
    Create a single combined plot for initial and posterior predictions with corrected error bands,
    using point-wise differences and interpolation for smooth visual comparison.
    """
    # Interpolation for smooth curves
    z_fine = np.linspace(z_normalized[0], z_normalized[-1], 500)
    observed_interp = interp1d(z_normalized, observed_data, kind="cubic")(z_fine)
    initial_interp = interp1d(z_normalized, initial_data, kind="cubic")(z_fine)
    posterior_interp = interp1d(z_normalized, posterior_prediction, kind="cubic")(z_fine)

    # Compute error bands
    initial_model_error_upper = np.maximum(initial_interp, observed_interp)
    initial_model_error_lower = np.minimum(initial_interp, observed_interp)

    posterior_model_error_upper = np.maximum(posterior_interp, observed_interp)
    posterior_model_error_lower = np.minimum(posterior_interp, observed_interp)

    plt.figure(figsize=(12, 8))

    # Observed data
    plt.plot(
        z_fine,
        observed_interp,
        label="Observed Data (MultiLogBohm)",
        color="red",
        linestyle=":",
        linewidth=2,
    )
    plt.scatter(
        z_normalized,
        observed_data,
        color="red",
        label="Observed Data Points",
        zorder=5,
    )

    # Initial prediction with error bands
    plt.plot(
        z_fine,
        initial_interp,
        label="Initial Guess Prediction (MAP)",
        color="black",
        linestyle="--",
        linewidth=1.5,
    )
    plt.scatter(
        z_normalized,
        initial_data,
        color="black",
        label="Initial Data Points (MAP)",
        zorder=5,
    )
    plt.fill_between(
        z_fine,
        initial_model_error_lower,
        initial_model_error_upper,
        color="purple",
        alpha=0.3,
        label="Initial Model Error Band",
    )

    # Posterior prediction with error bands
    plt.plot(
        z_fine,
        posterior_interp,
        label="Posterior Prediction (Last Sample)",
        color="blue",
        linewidth=2,
    )
    plt.fill_between(
        z_fine,
        posterior_model_error_lower,
        posterior_model_error_upper,
        color="blue",
        alpha=0.3,
        label="Posterior Model Error Band",
    )

    plt.xlabel("Normalized Distance (z)")
    plt.ylabel("Ion Velocity")
    plt.title("Combined Comparison of Predictions with Observed Data and Error Bands")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(
        "..", "neldermead/mcmc-plots-1123", "combined_comparison_corrected_error_bands.png"
    )
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}.")
    plt.show()


def main():
    """Main function to calculate and compare the posterior prediction using the last MCMC sample."""
    # File paths
    samples_path = os.path.join("..", "mcmc-results-11-23-24", "final_mcmc_samples_w_2.0_1.csv")
    initial_simulation_path = os.path.join("..", "mcmc-results-11-23-24", "mcmc_w_2.0_initial_mcmc.json")
    observed_data_path = os.path.join("..", "mcmc-results-11-23-24", "mcmc_w_2.0_observed_data_map.json")

    # Load MCMC samples
    try:
        samples = pd.read_csv(samples_path, header=None)
        samples.columns = ["log_v1", "log_alpha"]
        samples["v1"] = 10 ** samples["log_v1"]
        samples["alpha"] = 10 ** samples["log_alpha"]
        samples["v2"] = samples["v1"] * samples["alpha"]
    except FileNotFoundError:
        print(f"Samples file not found at {samples_path}. Please ensure the path is correct.")
        return

    # Load initial simulation data
    try:
        with open(initial_simulation_path, "r") as f:
            initial_simulation_data = json.load(f)
            initial_data = np.array(initial_simulation_data["ion_velocity"][0])  # Extract ion_velocity
    except FileNotFoundError:
        print(f"Initial simulation file not found at {initial_simulation_path}.")
        return

    # Load observed data
    try:
        with open(observed_data_path, "r") as f:
            observed_data_map = json.load(f)
            observed_data = np.array(observed_data_map["ion_velocity"][0])  # Extract ion_velocity
    except FileNotFoundError:
        print(f"Observed data file not found at {observed_data_path}.")
        return

    # Ensure configuration is complete
    config_spt_100 = config_multilogbohm.copy()
    config_spt_100["anom_model"] = "TwoZoneBohm"

    # Debugging: Print configuration
    print("Simulation configuration:", config_spt_100)

    # Generate posterior prediction using the last sample and save the result
    posterior_filename = os.path.join("..", "mcmc-results-11-23-24", "posterior_prediction_last_sample.json")
    posterior_prediction, z_normalized = generate_posterior_prediction_last_sample(
        samples, config_spt_100, posterior_filename, subsample_step=10
    )
    posterior_model_error = np.abs(posterior_prediction - observed_data)
    initial_model_error = np.abs(initial_data - observed_data)

    # # Visualization
    # if posterior_prediction is not None:
    #     plot_comparison_with_error_and_truth(
    #         z_normalized,
    #         posterior_prediction,
    #         observed_data,
    #         initial_data
    #     )
    plot_comparison_with_subplot(
    z_normalized=z_normalized,
    observed_data=observed_data,
    initial_data=initial_data,
    posterior_prediction=posterior_prediction
)
    plot_combined_corrected_error_bands(
    z_normalized=z_normalized,
    observed_data=observed_data,
    initial_data=initial_data,
    posterior_prediction=posterior_prediction
)

    plot_comparison_with_corrected_error(
    z_normalized=z_normalized,
    observed_data=observed_data,
    initial_data=initial_data,
    posterior_prediction=posterior_prediction
)

    plot_comparison_with_error_and_truth(z_normalized, posterior_prediction, observed_data, initial_data)

if __name__ == "__main__":
    main()

# def main():
#     """Main function to calculate and compare the posterior prediction using the last MCMC sample."""
#     # File paths
#     results_dir = os.path.join("..", "results-mcmc")
#     samples_path = os.path.join(results_dir, "final_mcmc_samples_w_2.0.csv")
#     initial_simulation_path = os.path.join(results_dir, "mcmc_w_2.0_initial_mcmc.json")
#     truth_data_path = os.path.join(results_dir, "mcmc_w_2.0_observed_data_map.json")

#     # Load MCMC samples
#     samples = pd.read_csv(samples_path, header=None)
#     samples.columns = ["log_v1", "log_alpha"]
#     samples["v1"] = 10 ** samples["log_v1"]
#     samples["alpha"] = 10 ** samples["log_alpha"]
#     samples["v2"] = samples["v1"] * samples["alpha"]

#     # Load initial simulation data
#     with open(initial_simulation_path, "r") as f:
#         initial_simulation_data = json.load(f)

#     # Load ground truth data
#     with open(truth_data_path, "r") as f:
#         truth_data = json.load(f)

#     # Extract observed data
#     z_normalized = np.array(truth_data["z_normalized"])
#     observed_data = np.mean(truth_data["ion_velocity"], axis=0)

#     # Extract initial predictions and error
#     initial_predictions = np.mean(initial_simulation_data["ion_velocity"], axis=0)
#     initial_model_error = np.abs(initial_predictions - observed_data)

#     # Ensure configuration is complete
#     config_spt_100 = config_multilogbohm.copy()
#     config_spt_100["anom_model"] = "TwoZoneBohm"

#     # Generate posterior prediction using the last sample
#     posterior_filename = os.path.join(results_dir, "posterior_prediction_last_sample.json")
#     posterior_prediction, _ = generate_posterior_prediction_last_sample(samples, config_spt_100, posterior_filename)

#     # Calculate posterior model error
#     posterior_model_error = np.abs(observed_data - posterior_prediction)

#     # Visualization
#     plot_comparison_with_error_and_truth(
#         z_normalized, 
#         initial_predictions, 
#         posterior_prediction, 
#         observed_data, 
#         initial_model_error, 
#         posterior_model_error
#     )

