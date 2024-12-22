import sys
import json
import math
import os
import time
import numpy as np
from scipy.stats import norm
from utils.save_data import load_json_data, subsample_data, save_results_to_json

# Add HallThruster Python API to the system path
sys.path.append("/path/to/HallThruster/python") 
import hallthruster as het

# -----------------------------
# 1. MultiLogBohm Configuration (Ground Truth)
# -----------------------------
config_multilogbohm = {
    "thruster": {
        "name": "SPT-100",
        "geometry": {
            "channel_length": 0.025,
            "inner_radius": 0.0345,
            "outer_radius": 0.05,
        },
        "magnetic_field": {
            "file": os.path.join("config/bfield_spt100.csv"),
        }
    },
    "discharge_voltage": 300.0,
    "anode_mass_flow_rate": 1e-5,
    "domain": (0.0, 0.08),
    "anom_model": {
        "type": "MultiLogBohm",
        "zs": [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
        "cs": [0.02, 0.024, 0.028, 0.033, 0.04, 0.004, 0.004, 0.05]
    },
    "ncharge": 3,
}

config_spt_100 = {
    "thruster": {
        "name": "SPT-100",
        "geometry": {
            "channel_length": 0.025,
            "inner_radius": 0.0345,
            "outer_radius": 0.05,
        },
        "magnetic_field": {
            "file": "config/bfield_spt100.csv"
        }
    },
    "discharge_voltage": 300.0,
    "anode_mass_flow_rate": 5e-6,
    "domain": (0.0, 0.08),
    "anom_model": {
        "type": "TwoZoneBohm",
        "c1": -2.0,
        "c2": 0.5,
    },
    "ncharge": 3
}

simulation = {
    "dt": 1e-6,
    "adaptive": True,
    "grid": {
        "type": "EvenGrid",
        "num_cells": 100,
    },
    "num_save": 100,
    "duration": 1e-3,
}

postprocess = {
    "output_file": "map_/results-map/output_twozonebohm.json",
    "save_time_resolved": False,
    "average_start_time": 0.4 * 1e-3
}


# -----------------------------
# Helper Functions
# -----------------------------


# todo : check if the failure aligns with the expected behaviour after -neginf.
def run_simulation_with_config(config, simulation, postprocess, config_type="MultiLogBohm"):
    """
    Run the simulation and handle cases where the simulation fails,
    including `retcode: error` or `failure`.
    """
    config_copy = config.copy() 
    input_data = {"config": config_copy, "simulation": simulation, "postprocess": postprocess}

    print(f"Running simulation with {config_type} configuration...")
    try:
        # Run the simulation
        solution = het.run_simulation(input_data)

        # Check if the simulation failed
        retcode = solution["output"].get("retcode")
        if retcode != "success":
            print(f"Simulation failed with retcode: {retcode}")
            if retcode in {"failure", "error"}:
                print("Simulation detected NaN, Inf, or some other error state. Penalizing with -np.inf.")
            return -np.inf  # Penalize failure or error cases

        # Return the valid solution
        return solution

    except KeyError as e:
        print(f"KeyError during simulation: {e}. Penalizing with -np.inf.")
        return -np.inf
    except Exception as e:
        print(f"Unexpected error during simulation: {e}. Penalizing with -np.inf.")
        return -np.inf  # Penalize unexpected errors

#tested
def update_twozonebohm_config(config, v1, v2):

    config_copy = config.copy()
    config_copy["anom_model"] = {"type": "TwoZoneBohm", "c1": v1, "c2": v2}
    return config_copy
