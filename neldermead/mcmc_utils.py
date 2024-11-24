import os
import pickle
import json
import numpy as np


def save_checkpoint(checkpoint_path, iteration, all_samples, acceptance_status, sampler_state):
    """Save the current MCMC state to a checkpoint file."""
    checkpoint_data = {
        "iteration": iteration,
        "all_samples": all_samples,
        "acceptance_status": acceptance_status,
        "sampler_state": sampler_state,
    }
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    print(f"Checkpoint saved at iteration {iteration} to {checkpoint_path}.")


def load_checkpoint(checkpoint_path):
    """Load MCMC state from a checkpoint file."""
    with open(checkpoint_path, 'rb') as f:
        checkpoint_data = pickle.load(f)
    print(f"Checkpoint loaded from {checkpoint_path}.")
    return checkpoint_data


def save_metadata(metadata, filename="mcmc_metadata.json", directory="."):
    """Save metadata to a JSON file."""
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, filename), 'w') as f:
        json.dump(metadata, f, indent=4)


def get_next_filename(base_filename, directory=".", extension=".csv"):
    """Generate the next available filename with an incremented suffix."""
    i = 1
    full_path = os.path.join(directory, f"{base_filename}_{i}{extension}")
    while os.path.exists(full_path):
        i += 1
        full_path = os.path.join(directory, f"{base_filename}_{i}{extension}")
    return full_path
