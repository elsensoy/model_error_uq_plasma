import os
import json
import numpy as np
from ..config.dict import Settings
from pathlib import Path
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
    if settings.gen_data:
        results_dir = settings.ground_truth.results_dir
        print("DEBUG ground truth ion_velocity:", result_dict.get("ui"))

    elif settings.run_map:
        results_dir = settings.map.base_dir  # Uses `map-results-N/`
    elif settings.run_mcmc:
        results_dir = settings.mcmc.base_dir  # Uses `mcmc-results-N/`
    else:
        raise ValueError("ERROR: Neither MAP nor MCMC is enabled. Cannot save metrics.")



    # Filter required keys
    required_keys = ['thrust', 'discharge_current', 'ui', 'z_normalized']
    result_dict_copy = {key: result_dict[key] for key in required_keys if key in result_dict}

    #  ion_velocity contains only the first entry
    if "ui" in result_dict_copy and isinstance(result_dict_copy["ui"], list):
        if len(result_dict_copy["ui"]) > 0 and isinstance(result_dict_copy["ui"][0], list):
            result_dict_copy["ui"] = result_dict_copy["ui"][0]  # Pick first set

    # Subsample data 
     # Subsample data 
    if subsample_for_saving:
        for key in ['z_normalized', 'ui']:
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
    if settings.run_map:
        directory = settings.map.base_dir  # Uses `map-results-N/`
    elif settings.run_mcmc:
        directory = settings.mcmc.base_dir  # Uses `mcmc-results-N/`
    else:
        raise ValueError("ERROR: Neither MAP nor MCMC is enabled. Cannot save metadata.")

    # make sure directory exists
    # os.makedirs(directory, parents=True,exist_ok=True)

    # Save metadata
    filepath = os.path.join(directory, filename)
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"Metadata saved to {filepath}")

# In main.py (after resolve_all_paths, and after flags like run_map/gen_data are available):

def create_used_directories(settings):
    dirs = set()

    base_dir = Path(settings.output_dir)
    dirs.add(base_dir)

    if settings.gen_data:
        dirs.add(base_dir / "ground_truth")
        dirs.add(base_dir / "postprocess")

    if settings.run_map:
        dirs.add(base_dir / "map")

    if settings.run_mcmc:
        dirs.add(base_dir / "mcmc")

    if settings.plotting:
        dirs.add(base_dir / "plots")

    # Create only the directories we now know we need
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    print("[INFO] Created directories:")
    for d in dirs:
        print(f"  - {d.resolve()}")
