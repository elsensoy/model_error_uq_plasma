import sys
import json
import math
import os
import time
import numpy as np
from scipy.stats import norm
from utils.save_data import load_json_data, subsample_data, save_results_to_json

# Add HallThruster Python API to the system path
sys.path.append("/path/to/HallThruster/python")  # Replace with the correct path
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
            "file": os.path.join("assets/bfield_spt100.csv"),
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
            "file": "assets/bfield_spt100.csv"
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
def run_simulation_with_config(config, simulation, postprocess, config_type="MultiLogBohm"):
 
    config_copy = config.copy()  # Ensure the original config is not mutated
    input_data = {"config": config_copy, "simulation": simulation, "postprocess": postprocess}

    print(f"Running simulation with {config_type} configuration...")
    try:
        solution = het.run_simulation(input_data)
        if solution["output"]["retcode"] != "success":
            print(f"Simulation failed with retcode: {solution['output']['retcode']}")
            return None
        return solution
    except Exception as e:
        print(f"Error during simulation with {config_type}: {e}")
        return None


def update_twozonebohm_config(config, v1, v2):

    config_copy = config.copy()  # Ensure the original config is not mutated
    config_copy["anom_model"] = {"type": "TwoZoneBohm", "c1": v1, "c2": v2}
    return config_copy
