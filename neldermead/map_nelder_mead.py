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
julia_file_path = os.path.join("..", "extract_metrics.jl")
jl.include(julia_file_path)

# -----------------------------
# 1. MultiLogBohm Configuration (Ground Truth)
# -----------------------------
config_multilogbohm = {
    "channel_length": 0.025,
    "inner_radius": 0.0345,
    "outer_radius": 0.05,
    "magnetic_field_file": os.path.join("..", "bfield_spt100.csv"),
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
# 2. TwoZoneBohm Configuration 
# -----------------------------
config_spt_100 = config_multilogbohm.copy()
config_spt_100['anom_model'] = 'TwoZoneBohm'
# -----------------------------
# 3. Helper Functions for Running Simulations and Extracting Results
# -----------------------------
def subsample_data(data, step=10):
    """subsample the data by taking every nth element."""
    if isinstance(data, list):
        return data[::step]  # every nth element from the list
    return data 

def save_results_to_json(result_dict, filename, save_every_n_grid_points=10, subsample_for_saving=True):
    """
    save the results as a JSON file, ensure the directory exists.
    subsample only when saving and keep original data untouched for processing.
    """
    spatial_keys = ['ion_velocity', 'z_normalized']
    
    # create a copy to avoid modifying the original result_dict in memory
    result_dict_copy = result_dict.copy()

    for key in spatial_keys:
        if key in result_dict_copy:
            print(f"Original {key} data shape: {np.array(result_dict_copy[key]).shape}")
            
            # Subsample only for saving, if enabled
            if subsample_for_saving:
                if key == 'z_normalized' and len(result_dict_copy[key]) <= save_every_n_grid_points:
                    print(f"{key} already subsampled. Skipping subsampling.")
                else:
                    if key == 'z_normalized':
                        result_dict_copy[key] = subsample_data(result_dict_copy[key], save_every_n_grid_points)
                    else:
                        result_dict_copy[key] = [subsample_data(sublist, save_every_n_grid_points) for sublist in result_dict_copy[key]]
            
            print(f"Subsampled {key} data shape for saving: {np.array(result_dict_copy[key]).shape}")

    # Ensure the results directory exists
    results_dir = os.path.join("..", "results-Nelder-Mead")
    os.makedirs(results_dir, exist_ok=True)  # Create directory if it doesn't exist

    # Define the full path for the JSON file
    result_file_path = os.path.join(results_dir, filename)

    # Save the subsampled results to the file
    try:
        with open(result_file_path, 'w') as json_file:
            json.dump(result_dict_copy, json_file, indent=4)
        print(f"Results successfully saved to {result_file_path}")
    except Exception as e:
        print(f"Failed to save the results: {e}")

def run_simulation(config):
    """Run the HallThruster simulation using the provided configuration and track the time."""
    print("Starting simulation...")
    
    # Start the timer
    start_time = time.time()
    
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as temp_file:
        json.dump(config, temp_file)
        config_file_path = temp_file.name

    try:
        print("Running simulation...")
        result = jl.HallThruster.run_simulation(config_file_path, is_path=True, verbose=True)
        
        # Delete the temporary file after the simulation
        os.unlink(config_file_path)
        
        # End the timer and calculate the duration
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Simulation completed in {elapsed_time:.2f} seconds.")

        return result
    except juliacall.JuliaError as e:
        print(f"Julia simulation failed with error: {e}. Continuing to the next iteration.")
        os.unlink(config_file_path)  # Clean up the temporary file even if there's an error
        return None  # Return None to indicate failure
    except Exception as e:
        print(f"Unexpected error during simulation: {e}. Continuing to the next iteration.")
        os.unlink(config_file_path)  # Clean up the temporary file for any other exception
        return None  # Return None to indicate failure

def hallthruster_jl_wrapper(v1, v2, config, use_time_averaged=True, save_every_n_grid_points=None):
    """Run the HallThruster simulation with parameters v1 and v2, and extract metrics."""
    print(f"Running simulation with v1: {v1}, v2: {v2}")
    config_copy = config.copy()
    config_copy["anom_model_coeffs"] = [v1, v2]
    
    # Run the simulation
    result = run_simulation(config_copy)
    if result is None:
        print("Simulation failed. Skipping to the next iteration.")
    else:
        print("Simulation succeeded.")
    # Process the result

    # time-averaging if chosen
    if use_time_averaged:
        print("Applying time-averaging to the simulation results...")
        try:
            extracted_data_json = jl.SolutionMetrics.extract_time_averaged_metrics(
                result, int(0.4 * config_copy['num_save']), save_every_n_grid_points=1  # Full data, no subsampling
            )
        except Exception as e:
            print(f"Time-averaging failed: {e}")
            print("Falling back to non-time-averaged metrics...")
            extracted_data_json = jl.SolutionMetrics.extract_performance_metrics(
                result, save_every_n_grid_points=1  # Full data, no subsampling
            )
    else:
        print("Extracting non-time-averaged metrics...")
        extracted_data_json = jl.SolutionMetrics.extract_performance_metrics(
            result, save_every_n_grid_points=1  # Full data, no subsampling
        )
    extracted_data = json.loads(extracted_data_json)
    # No subsampling here, only when saving
    return extracted_data


def extract_solution_data_julia(solution):
    extracted_data_json = jl.SolutionMetrics.extract_performance_metrics(solution)
    return json.loads(extracted_data_json)


def load_json_data(filename):
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

def run_multilogbohm_simulation(config, ion_velocity_weight, use_time_averaged=True, save_every_n_grid_points=10):
    print("Running MultiLogBohm simulation...")  
    # Step 1: Run the simulation (this runs a non-time-averaged simulation)
    result = run_simulation(config)
    if result is None:
        print("Simulation failed. Skipping to the next iteration.")
    else:
        print("Simulation succeeded.")
    


    # Initialize a dictionary to store ground truth data (both spatial and non-spatial metrics)
    ground_truth_data = {}

    # Step 2: extract non-spatial scalars thrust and discharge current
    print("Extracting final thrust and discharge current from simulation results...")
    ground_truth_data['thrust'] = jl.HallThruster.thrust(result)[-1]  # Final thrust value
    ground_truth_data['discharge_current'] = jl.HallThruster.discharge_current(result)[-1]  # Final discharge current value

    # Step 3: then spatial metrics (e.g., ion velocity)
    if use_time_averaged:
        print("Applying time-averaging to spatial metrics starting from the last 60 % of the simulation...")
        try:
            spatial_data = jl.SolutionMetrics.extract_time_averaged_metrics(
                result, int(0.4 * config['num_save']), save_every_n_grid_points=save_every_n_grid_points
            )
            print("Time-averaged spatial metrics successfully extracted.")
        except Exception as e:
            print(f"Error during time-averaging: {e}")
            return None
    else:
        print("Extracting non-time-averaged spatial metrics...")
        spatial_data = jl.SolutionMetrics.extract_performance_metrics(result)

    # Step 4: Convert the spatial data to a dictionary (without subsampling here)
    ground_truth_data.update(json.loads(spatial_data))  # Convert from JSON to dict
    
    print("MultiLogBohm simulation and data extraction complete.")
    
    
    return ground_truth_data

# -----------------------------
# 4. Prior and likelihood
# -----------------------------
#old version changed in greatlakesuser
def prior_logpdf(v1_log, alpha_log):
    prior1 = norm.logpdf(v1_log, loc=np.log10(1/160), scale=np.sqrt(2))
    prior2 = norm.logpdf(alpha_log, loc=np.log10(1/16), scale=np.sqrt(2))
    return prior1 + prior2


def log_likelihood(simulated_data, observed_data, sigma=0.08, ion_velocity_weight=2.0):
    """Compute the log-likelihood of the observed data given the simulated data."""
    log_likelihood_value = 0

    # DEBUG:Check the keys in the simulated and observed data
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

    # ion velocity is 2D (space and time averaged), so we apply a lower weight
    if "ion_velocity" in observed_data and "ion_velocity" in simulated_data:
        simulated_ion_velocity = np.array(simulated_data["ion_velocity"])
        observed_ion_velocity = np.array(observed_data["ion_velocity"])

        # DEBUG: Print the shapes of both arrays
        print(f"Shape of simulated_ion_velocity: {simulated_ion_velocity.shape}")
        print(f"Shape of observed_ion_velocity: {observed_ion_velocity.shape}")

       
        if simulated_ion_velocity.shape == observed_ion_velocity.shape:
            residual = simulated_ion_velocity - observed_ion_velocity
            log_likelihood_value += -0.5 * np.sum((residual / (sigma / ion_velocity_weight)) ** 2)
        else:
            print("Shapes are not compatible for subtraction.")
    else:
        print(f"Warning: Ion velocity data not found in simulation or observed data.")

    return log_likelihood_value

def compute_neg_log_posterior(v_log, observed_data, config, ion_velocity_weight, sigma=0.08):
    #Compute the negative log-posterior for MAP optimization.
    v1_log, alpha_log = v_log
    v1 = np.exp(v1_log)  # nu_1
    alpha = np.exp(alpha_log)  # scaling factor
    v2 = alpha * v1  # nu_2 is guaranteed to be greater than nu_1

    # Run the simulation and get the time-averaged results (no subsampling here)
    simulated_result = hallthruster_jl_wrapper(v1, v2, config, use_time_averaged=True, save_every_n_grid_points=None)

    # Use full data for log-likelihood computation
    log_likelihood_value = log_likelihood(simulated_result, observed_data, sigma=sigma, ion_velocity_weight=ion_velocity_weight)

    # Compute the log-prior for v1 and alpha
    log_prior_value = prior_logpdf(v1_log, alpha_log)

    # Return the negative log-posterior
    return -(log_likelihood_value + log_prior_value)

# -----------------------------
# 6. Run MAP
# -----------------------------
def callback(v_log, iteration_counter, config, use_time_averaged, save_every_n_grid_points, observed_data, ion_velocity_weight, loss_history):
    """Callback function to log v1, v2, metrics, and loss during optimization."""
    iteration_counter[0] += 1

    # Compute v1 and v2 for the current iteration
    v1_iter = np.exp(v_log[0])
    alpha_iter = np.exp(v_log[1])
    v2_iter = alpha_iter * v1_iter
    print(f"Running TwoZoneBohm for iteration {iteration_counter[0]} with v1: {v1_iter:.4f}, v2: {v2_iter:.4f}...")

    # Run the simulation to get the full metrics for this iteration (no subsampling)
    iteration_metrics = hallthruster_jl_wrapper(
        v1_iter, v2_iter, config, 
        use_time_averaged=use_time_averaged, 
        save_every_n_grid_points=None  # No subsampling in the optimization process
    )
    current_loss = compute_neg_log_posterior(v_log, observed_data, config, ion_velocity_weight)
    loss_history.append(current_loss)
    print(f"Iteration {iteration_counter[0]} loss: {current_loss}")

    # Subsample only when saving to JSON for visualization (ex. plotting)
    save_results_to_json(iteration_metrics, f'nm_w_{ion_velocity_weight}_iteration_metrics.json', save_every_n_grid_points, subsample_for_saving=True)

    # Save the iteration result (v1 and v2)
    save_map_iteration(v_log, iteration_counter[0], filename=f"nm_w_{ion_velocity_weight}_map_iteration_results.json")

def run_map_single_initial_guess(observed_data, config, ion_velocity_weight=2.0, save_every_n_grid_points=None):

    initial_guess = [-2, 0.5]

    best_result = None
    best_v1, best_v2 = None, None
    best_metrics = None  

    iteration_counter = [0]
    loss_history = []

    print(f"Running MAP optimization with a single initial guess: {initial_guess}")

    def iteration_callback(v_log):
        callback(v_log, iteration_counter, config, True, save_every_n_grid_points, observed_data, ion_velocity_weight, loss_history)

    # Penalty function for bounds
    def bounds_penalty(v_log):
        penalty = 0
        if not (-5 <= v_log[0] <= 0):
            penalty += (v_log[0] - max(-5, min(v_log[0], 0))) ** 2
        if not (0 <= v_log[1] <= 3):
            penalty += (v_log[1] - max(0, min(v_log[1], 3))) ** 2
        return penalty

    # Optimization call with adjusted options
    result = minimize(
        lambda v_log: compute_neg_log_posterior(v_log, observed_data, config, ion_velocity_weight) + bounds_penalty(v_log),
        initial_guess,
        method='Nelder-Mead',
        callback=iteration_callback,
        options={'maxfev': 50000, 'fatol': 1e-3, 'xatol': 1e-3}  # Adjusted options for more evaluations
    )

    if result.success:
        v1_log_opt, alpha_log_opt = result.x
        v1_opt = np.exp(v1_log_opt)
        alpha_opt = np.exp(alpha_log_opt)
        v2_opt = alpha_opt * v1_opt
        loss = result.fun

        print(f"MAP estimates: v1 = {v1_opt:.6f}, v2 = {v2_opt:.6f}, loss = {loss:.6f}")

        best_result = result
        best_v1, best_v2 = v1_opt, v2_opt

        best_metrics = hallthruster_jl_wrapper(
            v1_opt, v2_opt, config,
            use_time_averaged=True,
            save_every_n_grid_points=None
        )

    else:
        print(f"MAP optimization failed: {result.message}")

    weight_str = str(ion_velocity_weight).replace('.', '_')
    loss_filename = f'nm_loss_values_w_{weight_str}.json'
    with open(loss_filename, 'w') as f:
        json.dump(loss_history, f, indent=4)
    print(f"Loss values saved to {loss_filename}")

    if best_result:
        best_initial_guess = {
            "v1": best_v1,
            "v2": best_v2
        }
        best_initial_guess_filename = f'nm_best_initial_guess_w_{weight_str}.json'
        with open(best_initial_guess_filename, 'w') as f:
            json.dump(best_initial_guess, f, indent=4)
        print(f"Best initial guess saved to {best_initial_guess_filename}")

        best_result_filename = f'nm_best_result_w_{weight_str}.json'
        save_results_to_json(best_metrics, best_result_filename, save_every_n_grid_points)
        print(f"Best metrics saved to {best_result_filename}")

        return best_v1, best_v2

    else:
        print("MAP optimization failed.")
        return None, None

# -----------------------------
# Saving Iteration Results
# -----------------------------
def save_map_iteration(v_log, iteration, filename):
    """Save v1, v2 values to map_iteration_results.json for each iteration."""
    v1 = np.exp(v_log[0])
    alpha = np.exp(v_log[1])
    v2 = alpha * v1
    iteration_result = {"iteration": iteration, "v1": v1, "v2": v2}

    # Load existing data, or initialize an empty list
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []

    # Add the new result and save to file
    data.append(iteration_result)
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)




def save_iteration_metrics(metrics, v_log, iteration, filename):
    """Save the simulation metrics along with v1, v2 for each iteration."""
    # Calculate v1 and v2 from v_log
    v1 = np.exp(v_log[0])
    alpha = np.exp(v_log[1])
    v2 = alpha * v1

    # Prepare the iteration metrics, including v1, v2, and the actual metrics
    iteration_metrics = {
        "iteration": iteration,
        "v1": v1,  # add v1 to the result
        "v2": v2,  # add v2 to the result
        "metrics": metrics  # add the metrics for this iteration
    }

    # load existing data, or initialize an empty list if the file doesn't exist or is empty
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []

    # check for duplicates (if duplicate iterations present this occurs especially in the first couple of iterations i am not sure what caused it but this snippet might not be needed now) 
    for i, entry in enumerate(data):
        if entry["iteration"] == iteration:
            # Overwrite the existing entry for this iteration
            data[i] = iteration_metrics
            break
    else:
        # Append new entry if no match found
        data.append(iteration_metrics)

    # write the updated data back to the file
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
# -----------------------------
# -----------------------------

def main():
    start_time = time.time()

    # List of ion_velocity_weights
    #ion_velocity_weights = [0.1, 1.0, 2.0, 3.0, 5.0, 10.0, 1e-10]
    ion_velocity_weights = [2.0]


    for ion_velocity_weight in ion_velocity_weights:
        print(f"\nRunning simulations with ion_velocity_weight = {ion_velocity_weight}...\n")

        ground_truth_data = run_multilogbohm_simulation(config_multilogbohm, ion_velocity_weight)
        save_results_to_json(ground_truth_data, f'nm_w_{ion_velocity_weight}_observed_data_map.json', save_every_n_grid_points=10)
		
        print(f"Ground truth data for ion_velocity_weight {ion_velocity_weight} saved.")

        # Step 2:the best initial guess by running the simulation for each initial guess and selecting the best one
        print("Running simulations with multiple initial guesses...")
        initial_guesses = [[-2, 0.5], [-3, 0.6], [-1, 0.7]]
        best_initial_result = None
        best_v1, best_v2 = None, None
        lowest_loss = None

        for idx, initial_guess in enumerate(initial_guesses):
            v1_initial = np.exp(initial_guess[0])
            alpha_initial = np.exp(initial_guess[1])
            v2_initial = alpha_initial * v1_initial

            print(f"Running simulation for initial guess {idx + 1}: v1 = {v1_initial}, v2 = {v2_initial}...")
            
            initial_result = hallthruster_jl_wrapper(
                v1_initial, v2_initial, 
                config_spt_100, 
                use_time_averaged=True, 
                save_every_n_grid_points=10
            )

            # we compute the loss (negative log-posterior) for the initial guess
            current_loss = compute_neg_log_posterior([np.log(v1_initial), np.log(alpha_initial)], ground_truth_data, config_spt_100, ion_velocity_weight)
            print(f"Initial guess {idx + 1}: v1 = {v1_initial}, v2 = {v2_initial}, loss = {current_loss:.6f}")

            if lowest_loss is None or current_loss < lowest_loss:
                lowest_loss = current_loss
                best_initial_result = initial_result
                best_v1, best_v2 = v1_initial, v2_initial

        # Step 3: saving the results of the best initial guess
        print(f"Best initial guess found: v1 = {best_v1}, v2 = {best_v2}, with the lowest loss = {lowest_loss:.6f}")
        save_results_to_json(best_initial_result, f'nm_w_{ion_velocity_weight}_best_initial_guess_result.json', save_every_n_grid_points=10)
        print(f"Best initial guess results for ion_velocity_weight {ion_velocity_weight} saved.")

        # Step 4: map optimization with a single initial guess (starting from the best initial guess)
        print("Running MAP optimization with multiple initial guesses...")
        v1_opt, v2_opt = run_map_single_initial_guess(
            ground_truth_data, 
            config_spt_100, 
            ion_velocity_weight, 
            save_every_n_grid_points=10  # Pass subsampling value for saving results
        )
        

        if v1_opt is not None and v2_opt is not None:
            # Step 5: Run the final TwoZoneBohm simulation with optimized parameters
            print(f"Running TwoZoneBohm simulation with optimized values v1: {v1_opt}, v2: {v2_opt} for ion_velocity_weight {ion_velocity_weight}...")
            optimized_result = hallthruster_jl_wrapper(
                v1_opt, v2_opt, 
                config_spt_100, 
                use_time_averaged=True, 
                save_every_n_grid_points=10
            )
            # save optimized results with a unique filename
            save_results_to_json(optimized_result, f'nm_w_{ion_velocity_weight}_optimized_twozonebohm_result.json', save_every_n_grid_points=10)
            print(f"Optimized results for ion_velocity_weight {ion_velocity_weight} saved.")
        else:
            print(f"MAP optimization failed for ion_velocity_weight {ion_velocity_weight}.")

    # End the timer and calculate the duration
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nAll simulations completed. Total time: {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    main()