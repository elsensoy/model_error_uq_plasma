import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from MCMCIterators.samplers import DelayedRejectionAdaptiveMetropolis
from utils.statistics import prior_logpdf
from utils.save_data import load_json_data, subsample_data, save_results_to_json
from utils.iter_methods import load_optimized_params, get_next_filename, get_next_results_dir,load_mcmc_config

# Define the Posterior (Prior-Only)
# -----------------------------
def log_posterior_prior_only(v_log):
    v1_log, alpha_log = v_log
    return prior_logpdf(v1_log, alpha_log)

# -----------------------------
# MCMC Sampling
# -----------------------------
def run_prior_only_mcmc(initial_sample, initial_cov, iterations, save_interval=10, results_dir="mcmc_results_prior_only"):
    os.makedirs(results_dir, exist_ok=True)
    samples_file = os.path.join(results_dir, "prior_only_samples.csv")

    sampler = DelayedRejectionAdaptiveMetropolis(
        log_posterior_prior_only, np.array(initial_sample), initial_cov,
        adapt_start=10, eps=1e-6, sd=2.4**2 / len(initial_sample), interval=10, level_scale=1e-1
    )

    all_samples = []
    for iteration in range(iterations):
        try:
            proposed_sample, _, accepted = next(sampler)
            all_samples.append(proposed_sample)
            if (iteration + 1) % save_interval == 0:
                print(f"Iteration {iteration + 1}: Saved intermediate samples to {samples_file}")
                np.savetxt(samples_file, np.array(all_samples), delimiter=',')

        except Exception as e:
            print(f"Error at iteration {iteration + 1}: {e}")
            break

    np.savetxt(samples_file, np.array(all_samples), delimiter=',')
    print(f"Final samples saved to {samples_file}")
    return np.array(all_samples)

# -----------------------------
# Run the Test
# -----------------------------
if __name__ == "__main__":
    #map results manually logged
    v1_opt = 0.02000351122413226
    v2_opt = 0.02007329109585801 
    # Load MCMC configuration
    json_config_path = "config/mcmc_config.json"  
    mcmc_config = load_mcmc_config(json_config_path)

    # Extract parameters from configuration
    initial_guess_path = mcmc_config["initial_guess_path"]
    ion_velocity_weight = mcmc_config["ion_velocity_weight"]
    iterations = mcmc_config["iterations"]
    initial_cov = mcmc_config["initial_cov"]
 
    initial_sample = [v1_opt, v2_opt]
    print("Running MCMC with MAP-derived initial guess...")
    samples = run_prior_only_mcmc(initial_sample, initial_cov, iterations)

    # Plot the results
    plt.hist(samples[:, 0], bins=50, alpha=0.7, label="v1_log")
    plt.hist(samples[:, 1], bins=50, alpha=0.7, label="alpha_log")
    plt.legend()
    plt.title("Samples from the Prior with MAP Initial Guess")
    plt.xlabel("Log10 Parameters")
    plt.ylabel("Frequency")
    plt.show()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'prior_only.png'))
