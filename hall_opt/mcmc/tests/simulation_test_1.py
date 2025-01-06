import json
import math
import os
import sys
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt

# HallThruster Path Setup
hallthruster_path = "/root/.julia/packages/HallThruster/J4Grt/python"
if hallthruster_path not in sys.path:
    sys.path.append(hallthruster_path)

import hallthruster as het
from config.simulation import simulation, config_spt_100, postprocess, config_multilogbohm, update_twozonebohm_config, run_simulation_with_config
from utils.save_data import load_json_data, subsample_data, save_results_to_json
from utils.iter_methods import load_optimized_params, get_next_filename, get_next_results_dir,load_mcmc_config

# 1. example parameters
v1_log = np.log10(1 / 160)  # Example parameter (log-space)
alpha_log = 0.5             

# 2. Convert to linear-space
v1 = 10**v1_log
alpha = 10**alpha_log
v2 = alpha * v1

# 3. Update the simulation configuration
updated_config = update_twozonebohm_config(config_spt_100, v1, v2)

# 4. Run the simulation
simulation_result = run_simulation_with_config(
    updated_config, simulation, postprocess, config_type="TwoZoneBohm"
)

# 5. Evaluate the simulation result
if simulation_result:
    print("Simulation ran successfully.")
    metrics = simulation_result["output"]["average"]
    print("Extracted Metrics:", json.dumps(metrics, indent=4))
else:
    print("Simulation failed. Check logs for details.")
