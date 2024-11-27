import os
import json
import numpy as np
import logging
from scipy.stats import norm
from MCMCIterators.samplers import DelayedRejectionAdaptiveMetropolis
from hall_opt.map_nelder_mead import hallthruster_jl_wrapper, config_multilogbohm,  run_simulation, run_multilogbohm_simulation
from hall_opt.mcmc_utils import load_json_data, load_optimized_params, get_next_filename, save_metadata, subsample_data, save_results_to_json, create_specific_config

# MCMC results directory path
results_dir = os.path.join("..", "mcmc-results-11-25-24")
# Path to results directory
RESULTS_NELDERMEAD = os.path.join("..", "results-Nelder-Mead")

initial_guess_path = os.path.join(RESULTS_NELDERMEAD, "best_initial_guess_w_2_0.json")

# -----------------------------
# 1.TwoZoneBohm Configuration
# -----------------------------
config_spt_100 = config_multilogbohm.copy()
config_spt_100['anom_model'] = 'TwoZoneBohm'

# -----------------------------
# 2. Prior and likelihood
# -----------------------------

# Define log-prior function
def prior_logpdf(v1_log, alpha_log):
    # Gaussian prior on log10(c1)
    prior1 = norm.logpdf(v1_log, loc=np.log10(1/160), scale=np.sqrt(2))
    
    # Uniform prior on log10(alpha) in [0, 2]
    # if alpha_log <= 0 or alpha_log > 2:
    #     print(f"Invalid prior: log10(alpha)={alpha_log} is out of range [0, 2].")
    #     return -np.inf  # Reject invalid samples
    
    prior2 = 0  # log(1) for a uniform distribution in valid range
    
    return prior1 + prior2




def log_likelihood(simulated_data, observed_data, sigma=0.08, ion_velocity_weight=2.0):
    """Compute the log-likelihood of the observed data given the simulated data."""
    log_likelihood_value = 0

    # DEBUG:Check the keys in the simulated and observed data
    # print("Keys in simulated_data:", simulated_data.keys())
    # print("Keys in observed_data:", observed_data.keys())

    # Thrust and discharge current are 1D arrays
    for key in ['thrust', 'discharge_current']:
        if key in observed_data and key in simulated_data:
            simulated_metric = np.array(simulated_data[key])
            observed_metric = np.array(observed_data[key])
            residual = simulated_metric - observed_metric
            log_likelihood_value += -0.5 * np.sum((residual / sigma) ** 2)
        else:
            print(f"Warning: Key '{key}' not found in data.")

    # ion velocity is 2D (space and time averaged), so we apply a lower weight
    if "ion_velocity" in observed_data and "ion_velocity" in simulated_data:
        simulated_ion_velocity = np.array(simulated_data["ion_velocity"])
        observed_ion_velocity = np.array(observed_data["ion_velocity"])

        # DEBUG: Print the shapes of both arrays
        # print(f"Shape of simulated_ion_velocity: {simulated_ion_velocity.shape}")
        # print(f"Shape of observed_ion_velocity: {observed_ion_velocity.shape}")
        if simulated_ion_velocity.shape == observed_ion_velocity.shape:
            residual = simulated_ion_velocity - observed_ion_velocity
            log_likelihood_value += -0.5 * np.sum((residual / (sigma / ion_velocity_weight)) ** 2)
        else:
            print("Shapes are not compatible for subtraction.")
    else:
        print(f"Warning: Ion velocity data not found in simulation or observed data.")

    return log_likelihood_value


def log_posterior(v_log, observed_data, config, ion_velocity_weight=2.0):
    v1_log, alpha_log = v_log
    v1 = 10 ** v1_log
    alpha = 10 ** alpha_log
    v2 = alpha * v1
    
    print(f"DEBUG: v1_log = {v1_log}, alpha_log = {alpha_log}, v1 = {v1}, v2 = {v2}")

    # Enforce physical constraint: v2 >= v1
    # if v2 < v1:
    #     print(f"Rejecting invalid proposal due to invalid initial parameters: v1 = {v1}, v2 = {v2}")
    #     #Invalid posterior
    #     return -1e6  # Invalid posterior penalize dram


    try:
        # Run the simulation
        simulated_data = hallthruster_jl_wrapper(v1, v2, config)
        
        # Handle simulation failure by setting likelihood to 1
        if simulated_data is None:
            print(f"Simulation failed for v1: {v1}, v2: {v2}. Setting log-likelihood to 1.")
            log_likelihood_value = 0  # Neutral log-likelihood
        else:
            # Compute log-likelihood normally if the simulation succeeds
            log_likelihood_value = log_likelihood(simulated_data, observed_data, ion_velocity_weight=ion_velocity_weight)

        # Compute the log-prior
        log_prior_value = prior_logpdf(v1_log, alpha_log)

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

def mcmc_inference(logpdf, initial_sample, iterations=200, save_interval=10, base_path="mcmc-results-11-25-24"):
    setup_logger()
    initial_sample = np.array(initial_sample)
    initial_cov = np.array([[0.5, 0], [0, 0.01]])
    logging.debug(f"Initial covariance matrix:\n{initial_cov}")

    sampler = DelayedRejectionAdaptiveMetropolis(
        logpdf, initial_sample, initial_cov, adapt_start=10, eps=1e-6,
        sd=2.4**2 / len(initial_sample), interval=10, level_scale=1e-1
    )

    all_samples = []
    acceptance_status = []
    delayed_rejections = 0
    dram_acceptances = 0
    initial_acceptances = 0
    rejections = 0
    invalid_proposals = 0  # Counter for invalid proposals
    total_attempted = 0

    results_dir = os.path.join("..", base_path)
    os.makedirs(results_dir, exist_ok=True)

    checkpoint_file = get_next_filename(f"{base_path}_checkpoint", results_dir, ".csv")
    final_samples_file = get_next_filename(f"{base_path}_final_samples", results_dir, ".csv")
    final_status_file = get_next_filename(f"{base_path}_final_status", results_dir, ".txt")

    logging.info(f"Checkpoint file: {checkpoint_file}")
    logging.info(f"Final samples file: {final_samples_file}")
    logging.info(f"Final status file: {final_status_file}")

    for iteration in range(iterations):
        try:
            total_attempted += 1
            logging.debug(f"Starting iteration {iteration + 1}/{iterations}. Total attempted: {total_attempted}")

            # Generate the next proposal
            result = next(sampler)
            sample = result[0]
            delayed_rejection = False

            # Handle valid proposals
            accepted = result[1]
            if not accepted:
                delayed_rejection = True
                delayed_rejections += 1
                logging.debug("First proposal rejected. DRAM considering second proposal.")
                result = next(sampler)
                sample = result[0]
                accepted = result[1]

            if accepted:
                if delayed_rejection:
                    logging.info("DRAM accepted a previously rejected proposal.")
                    dram_acceptances += 1
                else:
                    logging.info("Proposal accepted initially.")
                    initial_acceptances += 1
                acceptance_status.append('T')
            else:
                logging.info("Proposal rejected after DRAM.")
                rejections += 1
                acceptance_status.append('F')

            all_samples.append(sample)

            # Calculate and log the current acceptance rate
            acceptance_rate = (initial_acceptances + dram_acceptances) / total_attempted
            logging.info(
                f"Iteration {iteration + 1}: "
                f"Acceptance Rate = {acceptance_rate:.4f} | "
                f"Initial Acceptances = {initial_acceptances} | "
                f"DRAM Acceptances = {dram_acceptances} | "
                f"Rejections = {rejections} | "
                f"Delayed Rejections = {delayed_rejections}"
            )

            if (iteration + 1) % save_interval == 0:
                with open(checkpoint_file, 'a') as f:
                    f.write(f"{iteration + 1},{sample[0]},{sample[1]}\n")
                logging.info(f"Checkpoint saved at iteration {iteration + 1}. Acceptance rate = {acceptance_rate:.4f}")

        except Exception as e:
            logging.error(f"Error during iteration {iteration + 1}: {e}")
            acceptance_status.append('Error')
            all_samples.append(None)
            continue

    # Final acceptance rate
    acceptance_rate = (initial_acceptances + dram_acceptances) / total_attempted if total_attempted > 0 else 0
    logging.info(f"Final acceptance rate: {acceptance_rate:.4f}")
    logging.info(f"Initial acceptances: {initial_acceptances}")
    logging.info(f"DRAM acceptances: {dram_acceptances}")
    logging.info(f"Total rejections: {rejections}")
    logging.info(f"Invalid proposals: {invalid_proposals}")
    logging.info(f"Delayed rejections: {delayed_rejections}")
    logging.info(f"Final samples saved to: {final_samples_file}")
    logging.info(f"Final acceptance status saved to: {final_status_file}")

    np.savetxt(final_samples_file, np.array(all_samples, dtype=object), delimiter=',')
    with open(final_status_file, 'w') as status_f:
        for idx, status in enumerate(acceptance_status, start=1):
            status_f.write(f"{idx}: {status}\n")

    return np.array(all_samples, dtype=object), acceptance_rate, initial_cov


def run_mcmc_with_optimized_params(json_path, observed_data, config, ion_velocity_weight=2.0, iterations=200):
    # Load optimized parameters as the initial guess
    v1_opt, v2_opt = load_optimized_params(json_path)
    
    # Verify that parameters are loaded
    if v1_opt is None or v2_opt is None:
        print(f"Error: Optimized parameters could not be loaded. v1_opt: {v1_opt}, v2_opt: {v2_opt}")
        return  

    try:
        # Convert v1 and alpha to log10 space
        v_log_initial = [np.log10(v1_opt), np.log10(v2_opt / v1_opt)]
        print("Initial log parameters:", v_log_initial)  # Debugging: Verify initial log values
    except Exception as e:
        print(f"Error calculating v_log_initial: {e}")
        return
    
    # Run MCMC sampling
    print("Running MCMC sampling based on loaded optimized parameters...")
    base_path = f"mcmc_samples_1"
    
    samples, acceptance_rate, initial_cov = mcmc_inference(
        lambda v_log: log_posterior(v_log, observed_data, config, ion_velocity_weight=ion_velocity_weight),
        v_log_initial,
        iterations=iterations,
        save_interval=10,
        base_path=base_path
    )
    
    print(f"MCMC sampling complete with acceptance rate: {acceptance_rate:.2f}")
    
    # Save final samples and metadata for analysis in results_dir
    final_samples_file = get_next_filename(f"final_mcmc_samples_2_w_{ion_velocity_weight}")
    np.savetxt(final_samples_file, samples, delimiter=',')
    print(f"Final MCMC samples saved to {final_samples_file}")

    metadata = {
        "initial_guess": {"v1": v1_opt, "v2": v2_opt},
        "initial_cov": initial_cov.tolist(),
        "v_log_initial": v_log_initial,
        "iterations": iterations,
        "acceptance_rate": acceptance_rate,
        "ion_velocity_weight": ion_velocity_weight,
        "saved_file": base_path,
        "final_samples_file": final_samples_file,
        "model": "TwoZoneBohm",
        "config": create_specific_config(config)
    }
    
    save_metadata(metadata, filename=os.path.join(results_dir, f"mcmc_metadata_w_{ion_velocity_weight}.json"))
# Main function
def main():
    # Load the initial guess parameters from the JSON file

    ion_velocity_weight = [2.0]
    for ion_velocity_weight in ion_velocity_weight:
        # ground_truth_data = run_multilogbohm_simulation(config_multilogbohm, ion_velocity_weight)
        # save_results_to_json(ground_truth_data, f'mcmc_w_{ion_velocity_weight}_observed_data_map.json', save_every_n_grid_points=10)
		
    v1_opt, v2_opt = load_optimized_params(initial_guess_path)
    if v1_opt is None or v2_opt is None:
        print("Failed to load initial guess parameters.")
        return

    # Run the simulation with the initial guess parameters
    print(f"Running initial simulation with v1: {v1_opt}, v2: {v2_opt}")
    initial_simulation_result = hallthruster_jl_wrapper(v1_opt, v2_opt, config_spt_100, use_time_averaged=True, save_every_n_grid_points=10)
    save_results_to_json(initial_simulation_result, f'mcmc_w_{ion_velocity_weight}_initial_mcmc.json', save_every_n_grid_points=10)
    print(f"Initial simulation result saved")

    # Load the observed data for MCMC (or use the simulation result directly)
    #observed_data = ground_truth_data  # Using the result directly
    observed_data = load_json_data(os.path.join(RESULTS_DIR, mcmc_w_2.0_observed_data_map.json))

    # Run MCMC with the initial guess as the starting point
    print("Starting MCMC sampling...")
    run_mcmc_with_optimized_params(
        json_path=initial_guess_path,
        observed_data=observed_data,
        config=config_spt_100,
        ion_velocity_weight=2.0,
        iterations=200
    )

if __name__ == "__main__":
    main()








