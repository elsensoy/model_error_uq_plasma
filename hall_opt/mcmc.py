import os
import json
import sys
import numpy as np
import logging
from scipy.stats import norm
from datetime import datetime
from MCMCIterators.samplers import DelayedRejectionAdaptiveMetropolis 
from utils.map_nelder_mead import hallthruster_jl_wrapper, log_likelihood, config_multilogbohm, config_spt_100, run_simulation, run_multilogbohm_simulation
from utils.mcmc_utils import load_json_data, load_mcmc_config, load_optimized_params, get_next_filename, save_metadata, subsample_data, save_results_to_json, get_next_results_dir

results_dir = get_next_results_dir(base_dir="..", base_name="results/mcmc-results")

# -----------------------------
# 2. Prior and Posterior
# -----------------------------
def prior_logpdf(v1_log, alpha_log):
    # Gaussian prior on log10(c1)
    prior1 = norm.logpdf(v1_log, loc=np.log10(1/160), scale=np.sqrt(2))
    
    # Uniform prior on log10(alpha) in [0, 2]
    if alpha_log <= 0 or alpha_log > 2:
        print(f"Invalid prior: log10(alpha)={alpha_log} is out of range [0, 2].")
        return -np.inf  # Reject invalid samples
    
    prior2 = 0  # log(1) for a uniform distribution in valid range
    
    return prior1 + prior2


def log_posterior(v_log, observed_data, config, ion_velocity_weight=2.0):
    v1_log, alpha_log = v_log
    v1 = 10 ** v1_log
    alpha = 10 ** alpha_log
    v2 = alpha * v1
    
    print(f"DEBUG: v1_log = {v1_log}, alpha_log = {alpha_log}, v1 = {v1}, v2 = {v2}")

    # Enforce physical constraint: v2 >= v1
    if v2 < v1:
        print(f"Rejecting invalid proposal due to invalid initial parameters: v1 = {v1}, v2 = {v2}")
        return -np.inf  # Strong penalty for invalid posterior

    try:
        # Run the simulation
        simulated_data = hallthruster_jl_wrapper(v1, v2, config)
        
        # Handle simulation failure
        if simulated_data is None:
            print(f"Simulation failed for v1: {v1}, v2: {v2}. Setting log-likelihood to 1.")
            return -np.inf
        else:
            # Compute the log-prior
            log_prior_value = prior_logpdf(v1_log, alpha_log)
            # Compute log-likelihood normally if the simulation succeeds
            log_likelihood_value = log_likelihood(simulated_data, observed_data, ion_velocity_weight=ion_velocity_weight)

        # Compute the posterior as the sum of prior and likelihood
        log_posterior_value = log_prior_value + log_likelihood_value
        print(f"log_prior = {log_prior_value}, log_likelihood = {log_likelihood_value}, log_posterior = {log_posterior_value}")
        return log_posterior_value

    except Exception as e:
        print(f"Error in log-posterior computation: {e}")
        return -np.inf  # Treat as invalid posterior

def setup_logger():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler("mcmc.log")  # File output
        ]
    )
def mcmc_inference(logpdf, initial_sample, initial_cov, iterations, save_interval=10, results_dir=results_dir):
    """
    Run MCMC sampling using DRAM, run simulations for each sample, and save the simulator output.
    """
    # Initialize directories and logging
    setup_logger()
    logging.debug(f"Initial covariance matrix:\n{initial_cov}")
    initial_sample = np.array(initial_sample)
    #initial_cov = np.array([[0.8, 0], [0, 0.08]]) #covariance scaled down 
    # Initialize DRAM sampler
    sampler = DelayedRejectionAdaptiveMetropolis(
        logpdf, initial_sample, initial_cov, adapt_start=10, eps=1e-6,
        sd=2.4**2 / len(initial_sample), interval=10, level_scale=1e-1
    )

     # Prepare directories and file paths
    metrics_dir = os.path.join(results_dir, "iteration_metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    final_samples_file = get_next_filename("final_samples", directory=results_dir, extension=".csv")

    # Initialize storage for samples
    all_samples = []

    # Run MCMC sampling
    for iteration in range(iterations):
        try:
            # Get the next sample from the sampler
            proposed_sample, log_posterior, accepted = next(sampler)
            logging.debug(f"Iteration {iteration + 1}: Sample={proposed_sample}, Log Posterior={log_posterior}, Accepted={accepted}")

            # Store the sample
            all_samples.append(proposed_sample)

            # Transform MCMC sample to simulation inputs
            v1_log, alpha_log = proposed_sample
            v1, alpha = 10 ** v1_log, 10 ** alpha_log
            v2 = alpha * v1
            logging.debug(f"Transformed parameters for simulation: v1={v1}, v2={v2}")

            # Run the simulation
            simulation_result = hallthruster_jl_wrapper(v1, v2, config_spt_100)
            if simulation_result is not None:
                # Save simulator output (iteration metrics) as a JSON file
                metrics_filename = os.path.join(metrics_dir, f"iteration_{iteration + 1}_metrics.json")
                save_results_to_json(simulation_result, metrics_filename)
                logging.info(f"Iteration metrics saved for iteration {iteration + 1} to {metrics_filename}")
            else:
                logging.warning(f"Simulation failed for iteration {iteration + 1}. No metrics saved.")

        except Exception as e:
            logging.error(f"Error during iteration {iteration + 1}: {e}")
            raise  # Stop execution on error instead of continuing

    # Save final results
    np.savetxt(final_samples_file, np.array(all_samples), delimiter=',')
    # Return all samples and the acceptance rate
    return np.array(all_samples), sampler.accept_ratio()


def run_mcmc_with_optimized_params(json_path, observed_data, config, ion_velocity_weight, iterations, initial_cov, results_dir=results_dir):
    # Load optimized parameters as the initial guess
    v1_opt, v2_opt = load_optimized_params(json_path)
    if v1_opt is None or v2_opt is None:
        raise ValueError("Failed to load initial guess parameters.")

    # Convert v1 and alpha to log10 space
    v_log_initial = [np.log10(v1_opt), np.log10(v2_opt / v1_opt)]
    print(f"Initial log parameters: {v_log_initial}")

    # Run MCMC sampling
    samples, acceptance_rate = mcmc_inference(
        lambda v_log: log_posterior(v_log, observed_data, config),
        v_log_initial,
        initial_cov=initial_cov,
        iterations=iterations,
        save_interval=10,
        results_dir=results_dir
    )

    # Save final samples and metadata for analysis in results_dir
    final_samples_file = get_next_filename("final_mcmc_samples", directory=results_dir, extension=".csv")

    np.savetxt(final_samples_file, samples, delimiter=',')
    print(f"Final MCMC samples saved to {final_samples_file}")

    metadata = {
        "timestamp": datetime.now().isoformat(),  # Added timestamp in ISO format
        "initial_guess": {"v1": v1_opt, "v2": v2_opt},
        "initial_cov": initial_cov.tolist(),
        "v_log_initial": v_log_initial,
        "iterations": iterations,
        "acceptance_rate": acceptance_rate
    }
    save_metadata(metadata, filename=os.path.join(results_dir, f"mcmc_metadata.json"))

def main():
    # MCMC configuration
    # results_dir = get_next_results_dir(base_dir="..", base_name="mcmc-results")
    json_config_path = "mcmc_config.json"  
    mcmc_config = load_mcmc_config(json_config_path)

    # parameters from configuration
    # results_dir = mcmc_config["results_dir"]
    initial_guess_path = mcmc_config["initial_guess_path"]
    ion_velocity_weight = mcmc_config["ion_velocity_weight"]
    iterations = mcmc_config["iterations"]
    initial_cov = mcmc_config["initial_cov"]

    # results directory
    os.makedirs(results_dir, exist_ok=True)
    ground_truth_data = run_multilogbohm_simulation(config_multilogbohm, ion_velocity_weight)
    save_results_to_json(ground_truth_data, f'mcmc_observed_data_map.json',directory=results_dir,save_every_n_grid_points=10)
		
    v1_opt, v2_opt = load_optimized_params(initial_guess_path)
    if v1_opt is None or v2_opt is None:
        print("Failed to load initial guess parameters.")
        return

    # Run the simulation with the initial guess parameters
    print(f"Running initial simulation with v1: {v1_opt}, v2: {v2_opt}")
    initial_simulation_result = hallthruster_jl_wrapper(v1_opt, v2_opt, config_spt_100, use_time_averaged=True, save_every_n_grid_points=10)
    save_results_to_json(initial_simulation_result, f'mcmc_pre_mcmc_initial.json', directory=results_dir, save_every_n_grid_points=10)
    print(f"Initial simulation result saved")

    
    # Run MCMC sampling
    try:
        print("MCMC sampling...")
        run_mcmc_with_optimized_params(
            json_path=initial_guess_path,
            observed_data=ground_truth_data,
            config=config_spt_100,
            ion_velocity_weight=ion_velocity_weight,
            iterations=iterations,
            initial_cov=initial_cov
        )
        print("MCMC sampling completed successfully.")
    except Exception as e:
        print(f"Error during MCMC sampling: {e}")
        return

if __name__ == "__main__":
    main()

