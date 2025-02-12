import os
import json
import numpy as np
from ..config.dict import Settings

# -----------------------------
# Utility Functions
# -----------------------------

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

def save_results_to_json(
    settings: Settings, 
    result_dict: dict, 
    filename: str, 
    results_dir=str,
    save_every_n_grid_points=int, 
    subsample_for_saving=True
):
    # Determine correct results directory
    if settings.ground_truth.gen_data:
        results_dir = settings.ground_truth.output_file
    elif settings.general.run_map:
        results_dir = settings.map.base_dir  # Uses `map-results-N/`
    elif settings.general.run_mcmc:
        results_dir = settings.mcmc.base_dir  # Uses `mcmc-results-N/`
    else:
        raise ValueError("ERROR: Neither MAP nor MCMC is enabled. Cannot save metrics.")

    # make sure directory exists
    # os.makedirs(results_dir, exist_ok=True)

    # Filter required keys
    required_keys = ['thrust', 'discharge_current', 'ion_velocity', 'z_normalized']
    result_dict_copy = {key: result_dict[key] for key in required_keys if key in result_dict}

    #  ion_velocity contains only the first entry
    if "ion_velocity" in result_dict_copy and isinstance(result_dict_copy["ion_velocity"], list):
        if len(result_dict_copy["ion_velocity"]) > 0 and isinstance(result_dict_copy["ion_velocity"][0], list):
            result_dict_copy["ion_velocity"] = result_dict_copy["ion_velocity"][0]  # Pick first set

    # Subsample data 
     # Subsample data 
    if subsample_for_saving:
        for key in ['z_normalized', 'ion_velocity']:
            if key in result_dict_copy and result_dict_copy[key] is not None:
                result_dict_copy[key] = subsample_data(result_dict_copy[key], save_every_n_grid_points)

    # Save results to JSON
    result_file_path = os.path.join(results_dir, filename)
    with open(result_file_path, 'w') as json_file:
        json.dump(result_dict_copy, json_file, indent=4)
    
    print(f"Results successfully saved to {result_file_path}")


def save_metadata(settings: Settings, metadata: dict, filename="metadata.json"):
    """
    Saves metadata in the appropriate MAP or MCMC directory.
    """
    if settings.general.run_map:
        directory = settings.map.base_dir  # Uses `map-results-N/`
    elif settings.general.run_mcmc:
        directory = settings.mcmc.base_dir  # Uses `mcmc-results-N/`
    else:
        raise ValueError("ERROR: Neither MAP nor MCMC is enabled. Cannot save metadata.")

    # make sure directory exists
    # os.makedirs(directory, exist_ok=True)

    # Save metadata
    filepath = os.path.join(directory, filename)
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"Metadata saved to {filepath}")
