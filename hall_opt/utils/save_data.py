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

   