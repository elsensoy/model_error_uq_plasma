import os
import pickle
import json
import numpy as np


def get_next_results_dir(base_dir="results", base_name="mcmc-results"):
  
    #Generate the next results directory with an enumerated suffix.
    # 'mcmc-results-1', 'mcmc-results-2', etc.
    #     Example structure:
    # results/
    # ├── mcmc-results-1/
    # ├── mcmc-results-2/
    # 
    base_dir = os.path.abspath(base_dir)  # Convert to an absolute path
    os.makedirs(base_dir, exist_ok=True)  # Ensure the base_dir exists
    i = 1
    while True:
        dir_name = os.path.join(base_dir, f"{base_name}-{i}")
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)  # Create the directory
            print(f"Created results directory: {dir_name}")
            return dir_name
        i += 1



# Load optimized parameters from JSON
def load_optimized_params(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data.get("v1"), data.get("v2")

def load_json_data(filepath):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {filepath}")
        return None

def get_next_filename(base_filename, directory, extension=".csv"):
    """
    Generate the next available filename incremented in the directory.
    """
    i = 1
    full_path = os.path.join(directory, f"{base_filename}_{i}{extension}")
    while os.path.exists(full_path):
        i += 1
        full_path = os.path.join(directory, f"{base_filename}_{i}{extension}")
    return full_path


def save_metadata(metadata, filename="mcmc_metadata.json", directory="results"):

    os.makedirs(directory, exist_ok=True)  # Ensure the directory exists
    filepath = os.path.join(directory, filename)
    
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to {filepath}")

def subsample_data(data, step=10):
    """Subsample the data by taking every nth element."""
    if isinstance(data, list):
        return data[::step]  # Every nth element from the list
    return data 

def save_results_to_json(result_dict, filename="mcmc_results.json", directory="results", save_every_n_grid_points=10, subsample_for_saving=True):
    """
    Save the results as a JSON file, ensuring the directory exists.
    Subsample only when saving and keep original data untouched for processing.
    """
    spatial_keys = ['ion_velocity', 'z_normalized']
    result_dict_copy = result_dict.copy()  # Avoid modifying the original result_dict

    for key in spatial_keys:
        if key in result_dict_copy:
            print(f"Original {key} data shape: {np.array(result_dict_copy[key]).shape}")
            
            if subsample_for_saving:
                if key == 'z_normalized' and len(result_dict_copy[key]) <= save_every_n_grid_points:
                    print(f"{key} already subsampled. Skipping subsampling.")
                else:
                    if key == 'z_normalized':
                        result_dict_copy[key] = subsample_data(result_dict_copy[key], save_every_n_grid_points)
                    else:
                        result_dict_copy[key] = [subsample_data(sublist, save_every_n_grid_points) for sublist in result_dict_copy[key]]
            
            print(f"Subsampled {key} data shape for saving: {np.array(result_dict_copy[key]).shape}")

    # Define the full path for the JSON file
    result_file_path = os.path.join(directory, filename)

    try:
        with open(result_file_path, 'w') as json_file:
            json.dump(result_dict_copy, json_file, indent=4)
        print(f"Results successfully saved to {result_file_path}")
    except Exception as e:
        print(f"Failed to save the results: {e}")

def load_mcmc_config(json_path):
    
    #Load MCMC configuration from a JSON file.
 
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Configuration file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        config = json.load(f)
    
    # Convert values as needed
    config["initial_cov"] = np.array(config["initial_cov"])# Convert covariance matrix to numpy array
    config['results_dir'] = os.path.abspath(config['results_dir'])  # Ensure absolute path for results directory
    return config