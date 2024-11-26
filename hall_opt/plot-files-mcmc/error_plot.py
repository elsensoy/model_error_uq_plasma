import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neldermead.map_nelder_mead import hallthruster_jl_wrapper, config_multilogbohm, save_results_to_json

def subsample_data(data, step=10):
    """Subsample the data by taking every nth element."""
    if isinstance(data, list):
        return data[::step]  # Every nth element from the list
    return data

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
        posterior_prediction = np.mean(ion_velocity, axis=0)  # Mean over space/time

        # Save the posterior prediction
        save_results_to_json(
            {
                "z_normalized": simulation_result["z_normalized"],
                "posterior_prediction": posterior_prediction.tolist()
            },
            save_filename,
            save_every_n_grid_points=subsample_step
        )

        print(f"Posterior prediction saved to {save_filename}.")
        return posterior_prediction, simulation_result["z_normalized"]

    except Exception as e:
        print(f"Error during TwoZoneBohm simulation for v1={v1}, v2={v2}: {e}")
        return None, None

def plot_comparison_with_error_and_truth(z_normalized, initial_predictions, posterior_prediction, observed_data, initial_guess_data):
    """Generate a comparison plot including initial, posterior, and observed data."""
    # Ensure all arrays match the size of z_normalized
    def adjust_to_match(array, z_norm, label):
        if len(array) != len(z_norm):
            print(f"Adjusting {label} to match z_normalized dimensions...")
            return np.interp(
                z_norm,  # New x-values
                np.linspace(z_norm[0], z_norm[-1], len(array)),  # Original x-values
                array  # Original y-values
            )
        return array

    initial_predictions = adjust_to_match(initial_predictions, z_normalized, "initial_predictions")
    observed_data = adjust_to_match(observed_data, z_normalized, "observed_data")
    initial_guess_data = adjust_to_match(initial_guess_data, z_normalized, "initial_guess_data")

    # Calculate model errors
    posterior_model_error = np.abs(observed_data - posterior_prediction)
    initial_model_error = np.abs(observed_data - initial_predictions)

    plt.figure(figsize=(12, 8))

    # Plot observed data
    plt.plot(z_normalized, observed_data, label="Observed Data (MultiLogBohm)", color="red", linestyle=":")

    # Plot initial guess data
    plt.plot(z_normalized, initial_guess_data, label="Initial Guess Prediction (MAP)", color="purple", linestyle="--")

    # Plot posterior prediction from the last sample
    plt.plot(z_normalized, posterior_prediction, label="Posterior Prediction (Last Sample)", color="blue")

    # Add model error bands
    plt.fill_between(
        z_normalized,
        posterior_prediction - posterior_model_error,
        posterior_prediction + posterior_model_error,
        color="blue",
        alpha=0.3,
        label="Posterior Model Error Band"
    )

    plt.fill_between(
        z_normalized,
        initial_predictions - initial_model_error,
        initial_predictions + initial_model_error,
        color="purple",
        alpha=0.3,
        label="Initial Model Error Band"
    )

    plt.xlabel("Normalized Distance (z)")
    plt.ylabel("Ion Velocity")
    plt.title("Comparison of Predictions with Observed Data and Model Error Bands")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join("..", "results-mcmc", "comparison_with_err_and_truth.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}.")
    plt.show()

def main():
    """Main function to calculate and compare the posterior prediction using the last MCMC sample."""
    # File paths
    RESULTS_DIR = os.path.join("..", "results-mcmc")
    samples_path = os.path.join(RESULTS_DIR, "final_mcmc_samples_w_2.0.csv")
    initial_simulation_path = os.path.join(RESULTS_DIR, "mcmc_w_2.0_initial_mcmc.json")
    observed_data_path = os.path.join(RESULTS_DIR, "mcmc_w_2.0_observed_data_map.json")

    # Load MCMC samples
    samples = pd.read_csv(samples_path, header=None)
    samples.columns = ["log_v1", "log_alpha"]
    samples["v1"] = 10 ** samples["log_v1"]
    samples["alpha"] = 10 ** samples["log_alpha"]
    samples["v2"] = samples["v1"] * samples["alpha"]

    # Load initial simulation data
    with open(initial_simulation_path, "r") as f:
        initial_simulation_data = json.load(f)

    # Load observed data
    with open(observed_data_path, "r") as f:
        truth_data = json.load(f)

    # Extract and subsample data for plotting
    z_normalized = subsample_data(initial_simulation_data["z_normalized"], step=10)
    observed_data = subsample_data(np.mean(np.array(truth_data["ion_velocity"]), axis=0), step=10)
    initial_predictions = subsample_data(np.mean(initial_simulation_data["ion_velocity"], axis=0), step=10)

    # Load initial guess data (pre-MCMC target data)
    initial_guess_path = os.path.join(RESULTS_DIR, "mcmc_w_2.0_initial_mcmc.json")
    with open(initial_guess_path, "r") as f:
        initial_guess_data = np.mean(np.array(json.load(f)["ion_velocity"]), axis=0)

    initial_guess_data = subsample_data(initial_guess_data, step=10)

    # Generate posterior prediction for the last sample
    posterior_filename = os.path.join(RESULTS_DIR, "posterior_prediction_last_sample.json")
    posterior_prediction, z_normalized = generate_posterior_prediction_last_sample(
        samples, config_multilogbohm, posterior_filename, subsample_step=10
    )

    # Visualization
    if posterior_prediction is not None:
        plot_comparison_with_error_and_truth(
            z_normalized,
            initial_predictions,
            posterior_prediction,
            observed_data,
            initial_guess_data
        )

if __name__ == "__main__":
    main()
