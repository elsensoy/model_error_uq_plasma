import juliacall
import tempfile
import json
import os
import time
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

# Initialize Julia environment
jl = juliacall.Main
jl.seval("using HallThruster")
jl.seval("using JSON3")
jl.include("extract_metrics.jl")

# -----------------------------
# 1. MultiLogBohm Configuration (Ground Truth)
# -----------------------------
config_multilogbohm = {
    "channel_length": 0.025,
    "inner_radius": 0.0345,
    "outer_radius": 0.05,
    "magnetic_field_file": "bfield_spt100.csv",
    "magnetically_shielded": False,
    "propellant": "Xenon",
    "wall_material": "BNSiO2",
    "anode_potential": 300,
    "cathode_potential": 0.0,
    "anode_mass_flow_rate": 1e-5,
    "anom_model": "MultiLogBohm",
    "anom_model_coeffs": [
        0.02,
        0.024,
        0.028,
        0.033,
        0.04,
        0.004,
        0.004,
        0.05
    ],
    "cathode_location_m": 0.05,
    "max_charge": 1,
    "num_cells": 100,
    "dt_s": 1e-6,
    "duration_s": 1e-3,
    "num_save": 100,
    "flux_function": "rusanov",
    "limiter": "van_leer",
    "reconstruct": True,
    "ion_wall_losses": False,
    "electron_ion_collisions": True,
    "sheath_loss_coefficient": 1.0,
    "ion_temp_K": 1000.0,
    "neutral_temp_K": 500.0,
    "neutral_velocity_m_s": 150.0,
    "cathode_electron_temp_eV": 3.0,
    "inner_outer_transition_length_m": 0.01,
    "background_pressure_Torr": 0.0,
    "background_temperature_K": 100.0
}
# -----------------------------
# 2. TwoZoneBohm Configuration (for MLE)
# -----------------------------
config_spt_100 = config_multilogbohm.copy()
config_spt_100['anom_model'] = 'TwoZoneBohm'
config_spt_100['anom_model_coeffs'] = [-2, 0.5]  # Initial guesses for v1 and v2

# -----------------------------
# 3. Helper Functions for Running Simulations and Extracting Results
# -----------------------------
def run_simulation(config):
    """Run the HallThruster simulation using the provided configuration and track the time."""
    print("Starting simulation...")
    
    start_time = time.time()
    
    # Write config to a temporary file to pass it to the Julia code
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as temp_file:
        json.dump(config, temp_file)
        config_file_path = temp_file.name

    try:
        print("Running simulation...")
        result = jl.HallThruster.run_simulation(config_file_path, is_path=True, verbose=True)
        os.unlink(config_file_path)  # Delete the temp file after running the simulation
        
        end_time = time.time()
        print(f"Simulation completed in {end_time - start_time:.2f} seconds.")
        return result
    except juliacall.JuliaError as e:
        raise RuntimeError(f"Julia simulation failed: {e}")

def hallthruster_jl_wrapper(v1, v2, config, use_time_averaged=True, save_every_n_time_steps=10, save_every_n_grid_points=10):
    """Run the HallThruster simulation and extract metrics with subsampling logic."""
    print(f"Running simulation with v1: {v1}, v2: {v2}")
    config_copy = config.copy()
    config_copy["anom_model_coeffs"] = [v1, v2]
    
    # Run the simulation
    result = run_simulation(config_copy)

    # Apply time-averaging and subsampling for spatial metrics (always time-averaged)
    try:
        print("Extracting metrics (time-averaged for spatial metrics)...")
        extracted_data_json = extract_and_subsample_metrics(result, config_copy['num_save'], save_every_n_time_steps, save_every_n_grid_points, use_time_averaged=use_time_averaged)
    except Exception as e:
        print(f"Error in data extraction: {e}")
        return None

    return json.loads(extracted_data_json)


def extract_and_subsample_metrics(solution, n_save, save_every_n_time_steps, save_every_n_grid_points, use_time_averaged):
    """
    Extract metrics from the solution object, always apply time-averaging for spatial metrics,
    and convert the data into JSON-compatible formats.
    """
    # Always time-average spatial metrics
    result_data = jl.SolutionMetrics.extract_time_averaged_metrics(
        solution, n_save, save_every_n_grid_points=save_every_n_grid_points, save_every_n_time_steps=save_every_n_time_steps
    )
    
    # Subsample non-spatial metrics (thrust, discharge_current) with or without time-averaging
    if not use_time_averaged:
        result_data = jl.SolutionMetrics.extract_performance_metrics(
            solution, save_every_n_grid_points=save_every_n_grid_points, save_every_n_time_steps=save_every_n_time_steps
        )
    
    # Convert the result to JSON-compatible Python types
    result_data = json.loads(result_data)

    # Convert non-serializable types to lists or basic types
    def convert_to_list(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return [convert_to_list(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_list(value) for key, value in obj.items()}
        else:
            return obj

    result_data = convert_to_list(result_data)
    
    return json.dumps(result_data)


def run_multilogbohm_simulation(config, use_time_averaged=True, save_every_n_time_steps=10, save_every_n_grid_points=10):
    """
    Run the MultiLogBohm simulation and save the ground truth with optional time-averaging for non-spatial metrics.
    """
    print("Running MultiLogBohm simulation...")  
    result = run_simulation(config)

    # Initialize a dictionary to store ground truth data
    ground_truth_data = {}

    # Extract and subsample both non-spatial and spatial metrics
    extracted_data_json = extract_and_subsample_metrics(result, config['num_save'], save_every_n_time_steps, save_every_n_grid_points, use_time_averaged)
    
    # Parse the results and update the ground truth data
    ground_truth_data.update(json.loads(extracted_data_json))
    
    # Save the results to a JSON file
    print("Saving the ground truth data to 'nm_multilogbohm_ground_truth.json'...")
    save_results_to_json(ground_truth_data, 'nm_multilogbohm_ground_truth.json')
    
    print("MultiLogBohm simulation and data extraction complete.")
    return ground_truth_data


# Utility to save results to JSON
def save_results_to_json(result_dict, filename):
    """Save the results as a JSON file."""
    with open(filename, 'w') as json_file:
        json.dump(result_dict, json_file, indent=4)
        
# Utility to save results to JSON
# def save_results_to_json(result_dict, filename):
#     """Save the results as a JSON file."""
#     with open(filename, 'w') as json_file:
#         json.dump(result_dict, json_file, indent=4)

# def extract_solution_data_julia(solution):
#     """Extract solution data from Julia's simulation output."""
#     extracted_data_json = jl.SolutionMetrics.extract_performance_metrics(solution)
#     return json.loads(extracted_data_json)


def load_json_data(filename):
    """Load data from a JSON file and handle potential errors."""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        print(f"Data loaded successfully from {filename}")
        return data
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {filename}: {e}")
        return None
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return None


# -----------------------------
# 4. Prior and likelihood
# -----------------------------

def prior_logpdf(v1_log, v2_log):
    prior1 = norm.logpdf(v1_log, loc=np.log(1/160), scale=np.sqrt(2))
    prior2 = norm.logpdf(v2_log, loc=np.log(1/16), scale=np.sqrt(2))
    return prior1 + prior2

def log_likelihood(simulated_data, observed_data, sigma=0.08, ion_velocity_weight=3.0):
    """Compute the log-likelihood of the observed data given the simulated data."""
    log_likelihood_value = 0

    # Check the keys in the simulated and observed data
    print("Keys in simulated_data:", simulated_data.keys())
    print("Keys in observed_data:", observed_data.keys())

    # Thrust and discharge current are 1D arrays
    for key in ['thrust', 'discharge_current']:
        if key in observed_data and key in simulated_data:
            simulated_metric = np.array(simulated_data[key])
            observed_metric = np.array(observed_data[key])
            residual = simulated_metric - observed_metric
            log_likelihood_value += -0.5 * np.sum((residual / sigma) ** 2)
        else:
            print(f"Warning: Key '{key}' not found in data.")

    # Ion velocity is 2D (space and time averaged), so apply a lower weight
    if "ion_velocity" in observed_data and "ion_velocity" in simulated_data:
        simulated_ion_velocity = np.array(simulated_data["ion_velocity"])
        observed_ion_velocity = np.array(observed_data["ion_velocity"])
        residual = simulated_ion_velocity - observed_ion_velocity
        log_likelihood_value += -0.5 * np.sum((residual / (sigma / ion_velocity_weight)) ** 2)
    else:
        print(f"Warning: Ion velocity data not found in simulation or observed data.")
    
    return log_likelihood_value

def compute_neg_log_posterior(v_log, observed_data, config, sigma=0.08):
    """Compute the negative log-posterior for MAP optimization."""
    v1_log, alpha_log = v_log

	    # Manually enforce bounds for v1 and alpha
    if v1_log < -5 or v1_log > 0 or alpha_log < 0 or alpha_log > 3:
        return 1e6  # A large penalty value outside the bounds

    v1 = np.exp(v1_log)  # nu_1
    alpha = np.exp(alpha_log)  # scaling factor
    v2 = alpha * v1  # nu_2 is guaranteed to be greater than nu_1

    # Run the simulation and get the time-averaged results
    simulated_result = hallthruster_jl_wrapper(v1, v2, config)

    # Compute the log-likelihood
    log_likelihood_value = log_likelihood(simulated_result, observed_data, sigma, ion_velocity_weight =3.0)

    # Compute the log-prior for v1 and alpha
    log_prior_value = prior_logpdf(v1_log, alpha_log)

    # Combine likelihood and prior
    log_posterior_value = log_likelihood_value + log_prior_value

    # Return the negative log-posterior for optimization
    return -log_posterior_value

# -----------------------------
# 6. Run MLE or MAP
# -----------------------------
def run_map(observed_data, config, ion_velocity_weight=3.0, maxiter=100, callback=None, initial_guess=None):
# Added penalty function for bound enforcement in Nelder-Mead
bounds_penalty = lambda v_log: 1e6 if not (-5 <= v_log[0] <= 0 and 0 <= v_log[1] <= 3) else 0

# Objective function with penalty
result = minimize(
    lambda v_log: compute_neg_log_posterior(v_log, observed_data, config, ion_velocity_weight) + bounds_penalty(v_log),
    initial_guess,
    method='Nelder-Mead',
    callback=callback, 
    options={'maxiter': maxiter, 'fatol': 1e-6, 'xatol': 1e-6}
)

    if result.success:
        v1_log_opt, alpha_log_opt = result.x
        v1_opt = np.exp(v1_log_opt)
        alpha_opt = np.exp(alpha_log_opt)
        v2_opt = alpha_opt * v1_opt
        print(f"MAP estimates: v1 = {v1_opt:.6f}, v2 = {v2_opt:.6f}")
        return v1_opt, v2_opt, result.fun  # Return the function value (negative log-posterior)
    else:
        print(f"MAP optimization failed: {result.message}")
        return None, None, None


def run_map_and_optimization(ground_truth_data, config, ion_velocity_weight=3.0, maxiter=100, use_time_averaged=True, save_every_n_time_steps=10, save_every_n_grid_points=10):
    """
    Run MAP optimization for multiple initial guesses, track the progress, and choose the best result.
    """
	
    print("Running MAP optimization...")

    # List of initial guesses for the optimization process
    initial_guesses = [[-2, 0.5], [-3, 1.0], [-1, 2.0]]  

    # Counter to track the iteration number
    iteration_counter = [0]
    

    results = []

    def iteration_callback(v_log):
        """Inner callback to handle iteration-specific logging."""
        callback(v_log, iteration_counter, config, use_time_averaged, save_every_n_time_steps, save_every_n_grid_points)

    # Loop through all initial guesses
    for idx, initial_guess in enumerate(initial_guesses):
        print(f"Running optimization with initial guess {idx+1}: v1_log = {initial_guess[0]}, alpha_log = {initial_guess[1]}")

        # Run the MAP optimization for this initial guess
        v1_opt, v2_opt, neg_log_post = run_map(
            ground_truth_data, 
            config, 
            ion_velocity_weight, 
            maxiter, 
            callback=iteration_callback, 
            initial_guess=initial_guess
        )

        # Log the result for this initial guess
        if v1_opt is not None and v2_opt is not None:
            print(f"Initial guess {idx+1}: v1 = {v1_opt:.6f}, v2 = {v2_opt:.6f}, negative log-posterior = {neg_log_post:.6f}")
            results.append((v1_opt, v2_opt, neg_log_post))
        else:
            print(f"Initial guess {idx+1} optimization failed.")

    # Print all  v1, v2 values
    print("\nAll optimization results:")
    for idx, (v1_opt, v2_opt, neg_log_post) in enumerate(results):
        print(f"Guess {idx+1}: v1 = {v1_opt:.6f}, v2 = {v2_opt:.6f}, negative log-posterior = {neg_log_post:.6f}")

    # Find the best result (minimizing the negative log-posterior)
    if results:
        best_v1_opt, best_v2_opt, best_neg_log_post = min(results, key=lambda x: x[2])
        print(f"\nBest result: v1 = {best_v1_opt:.6f}, v2 = {best_v2_opt:.6f}, negative log-posterior = {best_neg_log_post:.6f}")

        # Run the final simulation with the best values
        print(f"Running TwoZoneBohm simulation with optimized values v1: {best_v1_opt}, v2: {best_v2_opt}...")
        optimized_result = hallthruster_jl_wrapper(
            best_v1_opt, best_v2_opt, 
            config, 
            use_time_averaged=use_time_averaged, 
            save_every_n_time_steps=save_every_n_time_steps, 
            save_every_n_grid_points=save_every_n_grid_points
        )
        save_results_to_json(optimized_result, 'nm_optimized_twozonebohm_result.json')
        print("Optimized results saved.")
        return best_v1_opt, best_v2_opt
    else:
        raise RuntimeError("MAP optimization failed for all initial guesses.")

# -----------------------------
# Saving Iteration Results
# -----------------------------

def callback(v_log, iteration_counter, config, use_time_averaged, save_every_n_time_steps, save_every_n_grid_points):
    """Callback function to log v1, v2, and metrics during optimization."""
    iteration_counter[0] += 1

    # Compute v1 and v2 for the current iteration
    v1_iter = np.exp(v_log[0])
    alpha_iter = np.exp(v_log[1])
    v2_iter = alpha_iter * v1_iter
    print(f"Running TwoZoneBohm for iteration {iteration_counter[0]} with v1: {v1_iter:.4f}, v2: {v2_iter:.4f}...")

    # Run the simulation to get the metrics for this iteration
    iteration_metrics = hallthruster_jl_wrapper(
        v1_iter, v2_iter, config, 
        use_time_averaged=use_time_averaged, 
        save_every_n_time_steps=save_every_n_time_steps, 
        save_every_n_grid_points=save_every_n_grid_points
    )

    # Save the iteration result along with the metrics
    save_map_iteration(v_log, iteration_counter[0], filename="nm_map_iteration_results.json")
    save_iteration_metrics(iteration_metrics, v_log, iteration_counter[0], filename="nm_iteration_metrics.json")

def save_map_iteration(v_log, iteration, filename="nm_map_iteration_results.json"):
    """Save v1, v2 values to map_iteration_results.json for each iteration."""
    v1 = np.exp(v_log[0])
    alpha = np.exp(v_log[1])
    v2 = alpha * v1
    iteration_result = {"iteration": iteration, "v1": v1, "v2": v2}

    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []

    data.append(iteration_result)
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def save_iteration_metrics(metrics, v_log, iteration, filename="nm_iteration_metrics.json"):
    """Save the simulation metrics along with v1, v2 for each iteration."""
    #  v1 and v2 from v_log
    v1 = np.exp(v_log[0])
    alpha = np.exp(v_log[1])
    v2 = alpha * v1

    #  the iteration metrics, including v1, v2, and the actual metrics
    iteration_metrics = {
        "iteration": iteration,
        "v1": v1,  
        "v2": v2,  
        "metrics": metrics  # Add the metrics for this iteration
    }

    # Load existing data, or initialize an empty list if the file doesn't exist or is empty
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []

    # Check for duplicates (e.g., duplicate iterations) and overwrite if necessary
    for i, entry in enumerate(data):
        if entry["iteration"] == iteration:
            # Overwrite the existing entry for this iteration
            data[i] = iteration_metrics
            break
    else:
        # Append new entry if no match found
        data.append(iteration_metrics)

    # Write the updated data back to the file
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)



def main():
    start_time = time.time()
    ion_velocity_weight = 3.0

    print("Generating ground truth data...")
    ground_truth_data = run_multilogbohm_simulation(
        config_multilogbohm, 
        use_time_averaged=True, 
        save_every_n_grid_points=10  # Pass the subsampling value 
    )
    save_results_to_json(ground_truth_data, 'nm_observed_data.json')
    print("Ground truth data saved.")


    print("Running TwoZoneBohm simulation with the best initial guess...")
    best_initial_guess = [-2, 0.5]  
    best_v1_initial = np.exp(best_initial_guess[0])
    best_alpha_initial = np.exp(best_initial_guess[1])
    best_v2_initial = best_alpha_initial * best_v1_initial

    # Run the TwoZoneBohm simulation with the best initial guess
    initial_guess_result = hallthruster_jl_wrapper(
        best_v1_initial, best_v2_initial, 
        config_spt_100, 
        use_time_averaged=True, 
        save_every_n_time_steps=10, 
        save_every_n_grid_points=10
    )
    save_results_to_json(initial_guess_result, 'nm_initial_guess_twozonebohm_result.json')
    print(f"Initial guess simulation results saved with v1: {best_v1_initial}, v2: {best_v2_initial}.")

    # Step 3: Run MAP optimization with multiple initial guesses
    print("Running MAP optimization with multiple initial guesses...")

    best_v1_opt, best_v2_opt = run_map_and_optimization(
        ground_truth_data, 
        config_spt_100, 
        ion_velocity_weight=3.0, 
        maxiter=100, 
        use_time_averaged=True, 
        save_every_n_time_steps=10, 
        save_every_n_grid_points=10
    )

    if best_v1_opt is not None and best_v2_opt is not None:
        # Step 4: Run the final TwoZoneBohm simulation with the best parameters from optimization
        print(f"Running TwoZoneBohm simulation with optimized values v1: {best_v1_opt}, v2: {best_v2_opt}...")
        optimized_result = hallthruster_jl_wrapper(
            best_v1_opt, best_v2_opt, 
            config_spt_100, 
            use_time_averaged=True, 
            save_every_n_time_steps=10, 
            save_every_n_grid_points=10
        )
        save_results_to_json(optimized_result, 'nm_twozonebohm_result.json')
        print("Optimized results saved.")
    else:
        print("MAP optimization failed for all initial guesses.")

    # End the timer,calculate the duration
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"MAP took {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    main()
