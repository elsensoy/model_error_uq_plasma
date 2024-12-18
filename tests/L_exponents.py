import json
import os
import numpy as np
import nolds
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
#output : Largest Lyapunov Exponent (extended): 0.11238903298501549~ weak, chaotic
# Load truth data
#SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True (sklearn fix- deprecated package)
truth_data_path = os.path.join("..", "results/mcmc_observed_data_map.json")
with open(truth_data_path, "r") as f:
    truth_data = json.load(f)

# Extract ion_velocity
ion_velocity_series = np.array(truth_data["ion_velocity"][0], dtype=float)

# Debugging
print(f"Ion velocity (truth data): {ion_velocity_series}")
print(f"Data type: {ion_velocity_series.dtype}")
print(f"Contains NaN: {np.isnan(ion_velocity_series).any()}")
print(f"Contains Inf: {np.isinf(ion_velocity_series).any()}")

# Extend data using interpolation
x_original = np.arange(len(ion_velocity_series))
x_extended = np.linspace(0, len(ion_velocity_series) - 1, 50)  # Extend to 50 points
interpolator = interp1d(x_original, ion_velocity_series, kind='linear')
ion_velocity_extended = interpolator(x_extended)

# Recompute Lyapunov exponent
lyap_exp = nolds.lyap_r(ion_velocity_extended, emb_dim=5, min_tsep=1)  # Reduced emb_dim
print(f"Largest Lyapunov Exponent (extended): {lyap_exp}")

