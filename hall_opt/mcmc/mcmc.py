import os
import json
import sys
import numpy as np
import logging
from scipy.stats import norm
from datetime import datetime
from MCMCIterators.samplers import DelayedRejectionAdaptiveMetropolis 
from map_.simulation import simulation, config_spt_100, postprocess, config_multilogbohm, update_twozonebohm_config, run_simulation_with_config
from utils.save_data import load_json_data, subsample_data, save_results_to_jso
from iter_methods import load_optimized_params, get_next_filename, save_metadata, get_next_results_dir
from statistics import log_likelihood, prior_logpdf, log_posterior

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)  # Ensures hall_opt is at the top of the search path

# Add HallThruster Python API to sys.path
hallthruster_path = "/root/.julia/packages/HallThruster/J4Grt/python"
if hallthruster_path not in sys.path:
    sys.path.append(hallthruster_path)

print("Updated sys.path:", sys.path)

import hallthruster as het

results_dir = get_next_results_dir(base_dir="..", base_name="results/mcmc-results")

def setup_logger():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler("mcmc.log")  # File output
        ]
    )
def mcmc_inference(logpdf, initial_sample, initial_cov, iterations, save_interval=10, results_dir="results"):

    os.makedirs(results_dir, exist_ok=True)
    metrics_dir = os.path.join(results_dir, "iteration_metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    final_samples_file = os.path.join(results_dir, "final_samples.csv")

    # Initialize sampler
    sampler = DelayedRejectionAdaptiveMetropolis(
        logpdf, np.array(initial_sample), initial_cov, adapt_start=10, eps=1e-6,
        sd=2.4**2 / len(initial_sample), interval=10, level_scale=1e-1
    )

    all_samples = []

    # MCMC Iterations
    for iteration in range(iterations):
        try:
            # Get the next sample
            proposed_sample, log_posterior, accepted = next(sampler)
            v1_log, alpha_log = proposed_sample
            v1, alpha = 10 ** v1_log, 10 ** alpha_log
            v2 = alpha * v1

            print(f"Iteration {iteration + 1}: v1={v1:.4f}, v2={v2:.4f}, Log Posterior={log_posterior:.4f}, Accepted={accepted}")
            all_samples.append(proposed_sample)

            # Run simulation
            updated_config = update_twozonebohm_config(config_spt_100, v1, v2)
            simulation_result = run_simulation_with_config(updated_config, simulation, postprocess, "TwoZoneBohm")

            # Save simulation output
            if simulation_result:
                metrics_filename = os.path.join(metrics_dir, f"iteration_{iteration + 1}_metrics.json")
                save_results_to_json(simulation_result, metrics_filename)
                print(f"Metrics saved for iteration {iteration + 1} to {metrics_filename}")

            # Save samples at intervals
            if (iteration + 1) % save_interval == 0:
                np.savetxt(final_samples_file, np.array(all_samples), delimiter=',')
                print(f"Saved samples to {final_samples_file} (iteration {iteration + 1})")

        except Exception as e:
            print(f"Error at iteration {iteration + 1}: {e}")
            break

    # Save final samples
    np.savetxt(final_samples_file, np.array(all_samples), delimiter=',')
    print(f"Final samples saved to {final_samples_file}")
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
        "timestamp": datetime.now().isoformat(), 
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

