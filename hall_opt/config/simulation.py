import os
import sys
import json
import numpy as np
from pathlib import Path
from hall_opt.utils.save_data import load_json_data, save_failing_samples_to_file
from hall_opt.config.settings_loader import Settings

# HallThruster Path Setup
hallthruster_path = "/home/elidasensoy/.julia/packages/HallThruster/tHQQa/python"
if hallthruster_path not in sys.path:
    sys.path.append(hallthruster_path)

import hallthruster as het

# -----------------------------
# Helper Functions
# -----------------------------

def run_simulation_with_config(config, simulation, postprocess, config_type="MultiLogBohm", iteration=None, v1=None, v2=None, failing_samples=None):
    """
    Run the simulation with the given configuration and handle cases where the simulation fails.
    Logs failures to `failing_samples` if provided.
    """
    failing_samples = failing_samples or []  # Initialize if None
    config_copy = config.copy()  # Avoid mutating the original config
    input_data = {"config": config_copy, "simulation": simulation, "postprocess": postprocess}

    print(f"Running simulation with {config_type} configuration...")

    try:
        # Run the simulation
        solution = het.run_simulation(input_data)

        # Check if the simulation failed
        retcode = solution["output"].get("retcode", "unknown")  # Default to "unknown" if retcode is missing
        if retcode != "success":
            print(f"Simulation failed with retcode: {retcode}")
            failing_samples.append({
                "iteration": iteration,
                "v1": v1,
                "v2": v2,
                "config_type": config_type,
                "retcode": retcode,
                "reason": "Simulation failure",
                "config": config_copy,  # Save the failing config for debugging
            })
            return None  # Indicate failure

        # Validate simulation output
        metrics = solution["output"].get("average", {})
        if not metrics or any(not np.isfinite(value) for value in metrics.values() if isinstance(value, (float, int))):
            print("Metrics are missing or invalid. Logging failure.")
            failing_samples.append({
                "iteration": iteration,
                "v1": v1,
                "v2": v2,
                "config_type": config_type,
                "reason": "Invalid metrics",
                "config": config_copy,
            })
            return None  # Indicate failure

        # Return the valid solution
        return solution

    except KeyError as e:
        print(f"KeyError during simulation: {e}. Logging failure.")
        failing_samples.append({
            "iteration": iteration,
            "v1": v1,
            "v2": v2,
            "config_type": config_type,
            "reason": f"KeyError: {str(e)}",
        })
        return None

    except Exception as e:
        print(f"Unexpected error during simulation: {e}. Logging failure.")
        failing_samples.append({
            "iteration": iteration,
            "v1": v1,
            "v2": v2,
            "config_type": config_type,
            "reason": f"Unexpected error: {str(e)}",
        })
        return None


def update_twozonebohm_config(config, v1, v2):
    """
    Update the TwoZoneBohm configuration with new v1 and v2 values.
    """
    config_copy = config.copy()  # Avoid mutating the original config
    config_copy["anom_model"] = {"type": "TwoZoneBohm", "c1": v1, "c2": v2}
    return config_copy


def load_config_from_settings(settings: Settings):
    """
    Load the simulation configuration from a `Settings` object.
    """
    config_multilogbohm = settings.config_multilogbohm
    config_spt_100 = settings.config_spt_100
    simulation = settings.simulation
    postprocess = settings.postprocess
    return config_multilogbohm, config_spt_100, simulation, postprocess
