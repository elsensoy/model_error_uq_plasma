import os
import numpy as np
from common_setup import load_data

# Base directory for results
base_results_dir = os.path.join("..", "results", "mcmc-results-1")

data = load_data(base_results_dir)

# Define number of chains
num_chains = 4  # Ensure this divides the number of iterations evenly
num_iterations = data.shape[0]
assert num_iterations % num_chains == 0, "Number of rows must be divisible by num_chains."

# Reshape data into chains: (num_chains, num_samples_per_chain, num_parameters)
chains = data.reshape(num_chains, -1, data.shape[1])

# Compute Gelman-Rubin statistic
def gelman_rubin(chains):
    """
    Computes the Gelman-Rubin statistic for multiple MCMC chains.
    """
    m, n = chains.shape[0], chains.shape[1]  # Number of chains and samples per chain

    chain_means = np.mean(chains, axis=1)  # Mean of each chain
    overall_mean = np.mean(chain_means)    

    B = n * np.var(chain_means, ddof=1)   # Between-chain variance
    W = np.mean(np.var(chains, axis=1, ddof=1))  # Within-chain variance

    V_hat = ((n - 1) / n) * W + (1 / n) * B  # Marginal posterior variance
    R_hat = np.sqrt(V_hat / W)  # Gelman-Rubin statistic

    return R_hat

# Compute R_hat for each parameter
R_hat_values = []
for param_index in range(chains.shape[2]):
    param_chains = chains[:, :, param_index]  # Extract chains for one parameter
    R_hat = gelman_rubin(param_chains)
    R_hat_values.append(R_hat)
    print(f"Parameter {param_index + 1}: Gelman-Rubin statistic (R_hat) = {R_hat}")

# Analyze results
print("R_hat values for all parameters:", R_hat_values)
non_converged_params = [i + 1 for i, R in enumerate(R_hat_values) if R > 1.1]
if non_converged_params:
    print(f"Warning: Parameters {non_converged_params} did not converge.")
else:
    print("All parameters have converged.")
