import juliacall
import tempfile
import json
import os
import time
import sys
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mapOpt.map_ink import (
    run_multilogbohm_simulation, 
    hallthruster_jl_wrapper,
    compute_neg_log_posterior,
    save_results_to_json,
	config_multilogbohm
)

# Initialize Julia environment
jl = juliacall.Main
jl.seval("using HallThruster")
jl.seval("using JSON3")
julia_file_path = os.path.join("..", "extract_metrics.jl")
jl.include(julia_file_path)


# Save the best initial guesses to a JSON file
def save_initial_guesses(ion_velocity_weight, v1, v2):
    """
    Save the best initial guesses for v1 and v2 for each ion_velocity_weight to a JSON file.
    """
    filename = os.path.join("..", "results-L-BFGS-B", "test_initial_guesses.json")

    # Load existing data or initialize an empty dictionary
    try:
        with open(filename, 'r') as f:
            initial_guess_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        initial_guess_data = {}

    # Update the dictionary with the new initial guesses for the current weight
    initial_guess_data[str(ion_velocity_weight)] = {
        'v1': v1,
        'v2': v2
    }

    # Save the updated dictionary back to the JSON file
    with open(filename, 'w') as f:
        json.dump(initial_guess_data, f, indent=4)
    
    print(f"Initial guesses for ion_velocity_weight {ion_velocity_weight} saved to {filename}.")

def test_initial_guesses():
    # Define the list of ion_velocity_weights you want to test
    ion_velocity_weights = [0.1, 1.0, 2.0, 3.0, 5.0, 10.0]

    # Run the test for each ion velocity weight
    for ion_velocity_weight in ion_velocity_weights:
        print(f"\nTesting initial guesses for ion_velocity_weight = {ion_velocity_weight}...\n")

        # Step 1: Run the MultiLogBohm simulation to generate the ground truth (using your function)
        ground_truth_data = run_multilogbohm_simulation(config_multilogbohm, ion_velocity_weight)

        # Save the ground truth data to a file
        save_results_to_json(ground_truth_data, f'test_w_{ion_velocity_weight}_observed_data_map.json')

        # Step 2: Set up the TwoZoneBohm configuration and test different initial guesses
        config_spt_100 = config_multilogbohm.copy()
        config_spt_100['anom_model'] = 'TwoZoneBohm'
        initial_guesses = [[-2, 0.5], [-3, 0.6], [-1, 0.7]]
        best_v1, best_v2 = None, None
        lowest_loss = None

        for idx, initial_guess in enumerate(initial_guesses):
            v1_initial = np.exp(initial_guess[0])
            alpha_initial = np.exp(initial_guess[1])
            v2_initial = alpha_initial * v1_initial

            print(f"Running simulation for initial guess {idx + 1}: v1 = {v1_initial}, v2 = {v2_initial}...")

            # Run the simulation for the current initial guess
            initial_result = hallthruster_jl_wrapper(v1_initial, v2_initial, config_spt_100, use_time_averaged=True)

            # Compute the loss (negative log-posterior)
            current_loss = compute_neg_log_posterior([np.log(v1_initial), np.log(alpha_initial)], ground_truth_data, config_spt_100, ion_velocity_weight)
            print(f"Initial guess {idx + 1}: v1 = {v1_initial}, v2 = {v2_initial}, loss = {current_loss:.6f}")

            # Track the best guess
            if lowest_loss is None or current_loss < lowest_loss:
                lowest_loss = current_loss
                best_v1, best_v2 = v1_initial, v2_initial

        # Print the best initial guess
        print(f"Best initial guess for ion_velocity_weight {ion_velocity_weight}: v1 = {best_v1}, v2 = {best_v2}, loss = {lowest_loss:.6f}")

        # Save the best initial guess to a JSON file
        save_initial_guesses(ion_velocity_weight, best_v1, best_v2)

if __name__ == "__main__":
    test_initial_guesses()

