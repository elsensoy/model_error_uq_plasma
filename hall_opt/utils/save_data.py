import os
import json
import numpy as np

# -----------------------------
# Utility Functions
# -----------------------------

def load_config(config_path):
    """Load the JSON configuration file."""
    try:
        with open(config_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None

def subsample_data(data, step=10):
    """Subsample spatial or temporal data."""
    if isinstance(data, list):
        if isinstance(data[0], list):  # 2D data (e.g., ion velocity)
            return [row[::step] for row in data]  # Subsample columns
        return data[::step]  # Subsample 1D list
    return data  # Return unchanged if not a list

# -----------------------------
# Saving Results
# -----------------------------

def save_results_to_json(result_dict, filename, results_dir="results", save_every_n_grid_points=10, subsample_for_saving=True):

    # Filter required keys
    required_keys = ['thrust', 'discharge_current', 'ion_velocity', 'z_normalized']
    result_dict_copy = {key: result_dict[key] for key in required_keys if key in result_dict}

    # Subsample data if required
    if subsample_for_saving:
        for key in ['z_normalized', 'ion_velocity']:
            if key in result_dict_copy and result_dict_copy[key] is not None:
                result_dict_copy[key] = subsample_data(result_dict_copy[key], save_every_n_grid_points)

    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)

    # Save results to JSON
    result_file_path = os.path.join(results_dir, filename)
    with open(result_file_path, 'w') as json_file:
        json.dump(result_dict_copy, json_file, indent=4)
    print(f"Results successfully saved to {result_file_path}")


def load_json_data(filename):
    """Load JSON data from a file."""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        print(f"Data loaded successfully from {filename}")
        return data
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None

def save_metadata(metadata, filename="mcmc_metadata.json", directory="mcmc/results"):

    os.makedirs(directory, exist_ok=True)  
    filepath = os.path.join(directory, filename)
    
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to {filepath}")

   
def save_parameters_linear(iteration, v1, alpha, results_dir, filename="parameters_linear.json"):
    filepath = os.path.join(results_dir, filename)
    data = {"iteration": iteration, "v1": v1, "alpha": alpha}

    # Initialize log as an empty list if the file is empty or doesn't exist
    if os.path.exists(filepath):
        if os.path.getsize(filepath) > 0:  # Check if the file is non-empty
            with open(filepath, 'r') as file:
                try:
                    log = json.load(file)
                except json.JSONDecodeError:
                    print(f"Warning: {filename} is corrupted or empty. Initializing new log.")
                    log = []
        else:
            print(f"Warning: {filename} is empty. Initializing new log.")
            log = []
    else:
        log = []

    log.append(data)

    with open(filepath, 'w') as file:
        json.dump(log, file, indent=4)
    print(f"Iteration {iteration}: Saved linear parameters to {filename}")


def save_parameters_log(iteration, v1_log, alpha_log, results_dir, filename="parameters_log.json"):
    filepath = os.path.join(results_dir, filename)
    data = {"iteration": iteration, "v1_log": v1_log, "alpha_log": alpha_log}

    if os.path.exists(filepath):
        with open(filepath, 'r') as file:
            log = json.load(file)
    else:
        log = []

    log.append(data)

    with open(filepath, 'w') as file:
        json.dump(log, file, indent=4)
    print(f"Iteration {iteration}: Saved log-space parameters to {filename}")

def save_failing_samples_to_file(file_path):
    """
    Save the failing samples to a JSON file, ensuring the file is created even if there are no samples.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if failing_samples:
        with open(file_path, "w") as f:
            json.dump(failing_samples, f, indent=4)
        print(f"Failing samples saved to {file_path}")
    else:
        # Write an empty JSON array if no samples
        with open(file_path, "w") as f:
            json.dump([], f, indent=4)
        print(f"No failing samples to save. Empty file created at {file_path}.")

