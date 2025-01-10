import sys
import json
import math
import os
import time
import numpy as np
from scipy.stats import norm

# Add HallThruster Python API to the system path
sys.path.append("/root/.julia/packages/HallThruster/J4Grt/python")  # hallthruster python path 
import hallthruster as het

input_data = {
    "config": {
        "thruster": {
            "name": "SPT-100",
            "geometry": {
                "channel_length": 0.025,
                "inner_radius": 0.0345,
                "outer_radius": 0.05,
            },
            "magnetic_field": {
                "file": "bfield_spt100.csv",
            },
        },
        "discharge_voltage": 300.0,
        "anode_mass_flow_rate": 5e-6,
        "domain": (0.0, 0.08),
        "anom_model": {
            "type": "TwoZoneBohm",
            "c1": 0.01,
            "c2": 0.1,
        },
        "ncharge": 3,
    },
    "simulation": {
        "grid": {"type": "EvenGrid", "num_cells": 100},
        "dt": 1e-9,
        "duration": 1e-3,
        "num_save": 100,
        "adaptive": True,
    },
    "postprocess": {
        "output_file": "output.json",
        "save_time_resolved": False,
        "average_start_time": 0.5 * 1e-3,
    },
}

solution = het.run_simulation(input_data)
print(solution.keys())  # Check keys in the output
 
 
output_data = solution['output']
print("Keys in 'output':", output_data.keys())

if 'average' in output_data:
    print("Keys in 'average':", output_data['average'].keys())
else:
    print("'average' key is missing in the output.")

if 'average' in output_data:
    averaged_data = output_data['average']
    print("Thrust:", averaged_data.get('thrust'))
    print("Discharge Current:", averaged_data.get('discharge_current'))
    print("Ion Velocity (ui):", averaged_data.get('ui'))
    print("Z (z_normalized):", averaged_data.get('z'))
else:
    print("Averaged data is not available in the output.")
