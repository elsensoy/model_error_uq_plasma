# MCMC results directory path
results_dir = os.path.join("..", "mcmc-results-11-23-24")
# Path to results directory
RESULTS_NELDERMEAD = os.path.join("..", "results-Nelder-Mead")

initial_guess_path = os.path.join(RESULTS_NELDERMEAD, "best_initial_guess_w_2_0.json")

# -----------------------------
# 1.TwoZoneBohm Configuration
# -----------------------------
config_spt_100 = config_multilogbohm.copy()
config_spt_100['anom_model'] = 'TwoZoneBohm'

# -----------------------------
# 2. Helper Functions for Saving/Loading Results
# -----------------------------
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

def get_next_filename(base_filename, directory=results_dir, extension=".csv"):
    """
    Generate the next available filename with an incremented suffix in the specified directory.
    """
    i = 1
    # Ensure the file path includes the directory and extension
    full_path = os.path.join(directory, f"{base_filename}_{i}{extension}")
    while os.path.exists(full_path):
        i += 1
        full_path = os.path.join(directory, f"{base_filename}_{i}{extension}")
    return full_path

def save_metadata(metadata, filename="mcmc_metadata.json", directory=results_dir):
    """Save metadata to a JSON file with automatic filename incrementing."""
    # Use get_next_filename to determine a unique path in the directory
    full_path = get_next_filename(filename.split('.')[0], directory=directory, extension=".json")
    
    with open(full_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to {full_path}")


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
    results_dir = os.path.join("..", "mcmc-results-11-23-24")
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

def create_specific_config(config):
    specific_config = {
        "Thruster Model": "SPT-100 Hall Thruster",
        "Spatial Resolution and Grid": {
            "num_cells": config.get("num_cells", 100),
            "channel_length": config.get("channel_length", 0.025)
        },
        "Simulation Duration": {
            "duration_s": config.get("duration_s", 1e-3),
            "num_save": config.get("num_save", 100)
        },
        "Magnetic Field and Boundary Conditions": {
            "magnetic_field_file": config.get("magnetic_field_file", "bfield_spt100.csv"),
            "anode_potential": config.get("anode_potential", 300),
            "cathode_potential": config.get("cathode_potential", 0)
        },
        "Additional Model Configurations": {
            "Propellant and Wall Material": {
                "propellant": config.get("propellant", "Xenon"),
                "wall_material": config.get("wall_material", "BNSiO2")
            },
            "Ion and Neutral Temperatures": {
                "ion_temp_K": config.get("ion_temp_K", 1000),
                "neutral_temp_K": config.get("neutral_temp_K", 500),
                "neutral_velocity_m_s": config.get("neutral_velocity_m_s", 150)
            },
            "Anomalous Transport Coefficients": {
                "description": "Anomalous coefficients c(z) adjust the anomalous collision frequency.",
                "equation": r"\nu_{AN}(z) = c(z) \cdot \omega_{ce}(z)"
            }
        }
    }
    return specific_config

# ------