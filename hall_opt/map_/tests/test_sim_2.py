import sys
sys.path.append("/root/.julia/packages/HallThruster/J4Grt/python")  # hallthruster python path 
import hallthruster as het

print("Running a test simulation...")

config = {
    "thruster": {
        "name": "SPT-100",
        "geometry": {
            "channel_length": 0.025,
            "inner_radius": 0.0345,
            "outer_radius": 0.05,
        },
        "magnetic_field": {
            "file": "bfield_spt100.csv"
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
    "dt": 5e-9,
    "adaptive": True,
    "grid": {"type": "EvenGrid", "num_cells": 100},
    "num_save": 100,
    "duration": 1e-3,
}

postprocess = {
    "output_file": "output.json",
    "save_time_resolved": False,
    "average_start_time": 5e-4,
}

input_data = {"config": config, "simulation": simulation, "postprocess": postprocess}

solution = het.run_simulation(input_data)
print("Simulation completed successfully!")
print("Output keys:", solution.keys())
