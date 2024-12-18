import os
import pickle
import json
import numpy as np

# -----------------------------
# Saving Results
# -----------------------------

def load_config(config_path):
    """Load the JSON configuration file."""
    with open(config_path, 'r') as file:
        return json.load(file)

def subsample_data(data, step=10):
    if isinstance(data, list):
        if isinstance(data[0], list):  # 2D data (e.g., ion velocity)
            return [row[::step] for row in data]  # Subsample spatially (columns)
        return data[::step]  # Subsample spatially for 1D list
    return data  # Return as-is if not a list

def update_twozonebohm_config(config, v1, v2):                
    config_copy = config.copy()  # Ensure the original config is not mutated
    config_copy["anom_model"] = {"type": "TwoZoneBohm", "c1": v1, "c2": v2}
    return config_copy


def save_results_to_json(result_dict, filename, save_every_n_grid_points=10, subsample_for_saving=True):
 
    required_keys = ['thrust', 'discharge_current', 'ion_velocity', 'z_normalized']
    result_dict_copy = {key: result_dict[key] for key in required_keys if key in result_dict}

    if subsample_for_saving:
        for key in ['z_normalized', 'ion_velocity']:
            if key in result_dict_copy and result_dict_copy[key] is not None:
                result_dict_copy[key] = subsample_data(result_dict_copy[key], save_every_n_grid_points)

    # Save results
    results_dir = "map_/results-map"
    os.makedirs(results_dir, exist_ok=True)
    result_file_path = os.path.join(results_dir, filename)

    with open(result_file_path, 'w') as json_file:
        json.dump(result_dict_copy, json_file, indent=4)
    print(f"Results successfully saved to {result_file_path}")

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


def save_parameters(iteration, v1, v2, filename="parameter_log.json"):
    """Save optimization parameters to a JSON file."""
    data = {"iteration": iteration, "v1": v1, "v2": v2}
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            log = json.load(file)
    else:
        log = []

    log.append(data)

    with open(filename, 'w') as file:
        json.dump(log, file, indent=4)
    print(f"Iteration {iteration}: Saved v1 = {v1:.4f}, v2 = {v2:.4f} to {filename}")
