import os
import pandas as pd
import arviz as az
import numpy as np


file_path = os.path.join("..", "mcmc-results-20", "final_samples_1.csv")

# Load MCMC samples
mcmc_samples = pd.read_csv(file_path, header=None)
mcmc_samples.columns = ["log_v1", "log_alpha"]  # Rename columns
mcmc_samples["v1"] = 10 ** mcmc_samples["log_v1"]
mcmc_samples["alpha"] = 10 ** mcmc_samples["log_alpha"]
mcmc_samples["v2"] = mcmc_samples["v1"] * mcmc_samples["alpha"]

# Extract chains
v1_chain = mcmc_samples["v1"].values
v2_chain = mcmc_samples["v2"].values

# Compute ESS using Arviz
ess_v1 = az.ess(v1_chain)
ess_v2 = az.ess(v2_chain)

print(f"Effective Sample Size (v1): {ess_v1}")
print(f"Effective Sample Size (v2): {ess_v2}")
