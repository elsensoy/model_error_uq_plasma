import os
import json
import sys
import numpy as np
import logging
from scipy.stats import norm
from datetime import datetime
from MCMCIterators.samplers import DelayedRejectionAdaptiveMetropolis 

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)  # Ensures hall_opt is at the top of the search path

# Add HallThruster Python API to sys.path
hallthruster_path = "/root/.julia/packages/HallThruster/J4Grt/python"
if hallthruster_path not in sys.path:
    sys.path.append(hallthruster_path)

print("Updated sys.path:", sys.path)

import hallthruster as het

from config.simulation import simulation, config_spt_100, postprocess, config_multilogbohm, update_twozonebohm_config, run_simulation_with_config
from utils.save_data import load_json_data, subsample_data, save_results_to_json, save_failing_samples_to_file
from utils.iter_methods import load_optimized_params, get_next_filename, save_metadata, get_next_results_dir,load_mcmc_config
from utils.statistics import log_likelihood, prior_logpdf, log_posterior

results_dir = "mcmc/results"

def setup_logger():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler("mcmc.log")  # File output
        ]
    )

failing_samples = []  # Ensure this is globally defined

def mcmc_inference(logpdf, initial_sample, initial_cov, iterations, save_interval=10, results_dir=results_dir):
    """
    Run MCMC inference and save results in both log-space and linear-space.
    """
    # Create a new results directory for this MCMC run
    run_dir = get_next_results_dir(base_dir=results_dir, base_name="mcmc-results")
    metrics_dir = os.path.join(run_dir, "iteration_metrics")
    os.makedirs(metrics_dir, exist_ok=True)  # Directory for iteration metrics

    final_samples_log_file = os.path.join(run_dir, "final_samples_log.csv")
    final_samples_linear_file = os.path.join(run_dir, "final_samples_linear.csv") 
    failing_samples_file = os.path.join(run_dir, "failing_samples.json")  

    # Initialize sampler
    sampler = DelayedRejectionAdaptiveMetropolis(
        logpdf, np.array(initial_sample), initial_cov, adapt_start=10, eps=1e-6,
        sd=2.4**2 / len(initial_sample), interval=1, level_scale=1e-1
    )

    all_samples = []  # Log-space samples
    all_samples_linear = []  # Linear-space samples
    failing_samples = []  # Track failing samples

    # MCMC Iterations
    for iteration in range(iterations):
        try:
            # Get the next sample
            proposed_sample, log_posterior, accepted = next(sampler)
            v1_log, alpha_log = proposed_sample

            # Transform to linear-space for simulation 
            v1, alpha = 10 ** v1_log, 10 ** alpha_log
            v2 = alpha * v1

            print(f"Iteration {iteration + 1}: v1={v1:.4f}, v2={v2:.4f}, Log Posterior={log_posterior:.4f}, Accepted={accepted}")
            
            # Append log-space and linear-space values
            all_samples.append(proposed_sample)
            all_samples_linear.append([v1, alpha])

            # Run simulation for current iteration
            updated_config = update_twozonebohm_config(config_spt_100, v1, v2)
            simulation_result = run_simulation_with_config(
                updated_config, simulation, postprocess, "TwoZoneBohm",
                iteration=iteration + 1, v1=v1, v2=v2
            )

            if simulation_result:
                # Extract metrics similar to observed and initial data
                averaged_metrics = simulation_result["output"]["average"]
                iteration_metrics = {
                    "thrust": averaged_metrics["thrust"],
                    "discharge_current": averaged_metrics["discharge_current"],
                    "ion_velocity": averaged_metrics["ui"][0],
                    "z_normalized": averaged_metrics["z"]
                }
                # Save iteration metrics
                metrics_filename = os.path.join(metrics_dir, f"iteration_{iteration + 1}_metrics.json")
                save_results_to_json(iteration_metrics, metrics_filename, results_dir=metrics_dir)
                print(f"Metrics saved for iteration {iteration + 1} to {metrics_filename}")
            else:
                # Log failing samples
                print(f"Simulation failed at iteration {iteration + 1} for v1={v1:.4f}, v2={v2:.4f}.")
                failing_samples.append({
                    "iteration": iteration + 1,
                    "v1_log": v1_log,
                    "alpha_log": alpha_log,
                    "v1": v1,
                    "v2": v2,
                    "reason": "Simulation failure"
                })

            # Save samples and failing samples at intervals
            if (iteration + 1) % save_interval == 0:
                np.savetxt(final_samples_log_file, np.array(all_samples), delimiter=',')
                np.savetxt(final_samples_linear_file, np.array(all_samples_linear), delimiter=',')
                print(f"Saved samples to {final_samples_log_file} and {final_samples_linear_file} (iteration {iteration + 1})")

                # Save failing samples
                with open(failing_samples_file, "w") as f:
                    json.dump(failing_samples, f, indent=4)
                print(f"Saved failing samples to {failing_samples_file} (iteration {iteration + 1})")

        except Exception as e:
            # Handle any unexpected errors in sampling
            print(f"Error at iteration {iteration + 1}: {e}")
            break

    # Save final samples
    np.savetxt(final_samples_log_file, np.array(all_samples), delimiter=',')
    np.savetxt(final_samples_linear_file, np.array(all_samples_linear), delimiter=',')
    print(f"Final samples saved to {final_samples_log_file} (log-space) and {final_samples_linear_file} (linear-space)")

    # Save any remaining failing samples
    with open(failing_samples_file, "w") as f:
        json.dump(failing_samples, f, indent=4)
    print(f"Final failing samples saved to {failing_samples_file}")

    return np.array(all_samples), np.array(all_samples_linear), sampler.accept_ratio()

def run_mcmc_with_optimized_params(json_path, observed_data, config, ion_velocity_weight, iterations, initial_cov, results_dir="mcmc/results"):
    """
    Run MCMC with optimized parameters and save results to a dynamic results directory.
    """
    # Load optimized parameters as the initial guess
    v1_opt, alpha_opt = load_optimized_params(json_path)
    if v1_opt is None or alpha_opt is None:
        raise ValueError("Failed to load initial guess parameters.")
    initial_sample = [v1_opt, alpha_opt]

    # Create a new results directory for this MCMC run
    run_results_dir = get_next_results_dir(base_dir=results_dir, base_name="mcmc-results")

    # Run MCMC sampling
    samples, samples_linear, acceptance_rate = mcmc_inference(
        lambda v_log: log_posterior(v_log, observed_data, config, simulation, postprocess),
        initial_sample,
        initial_cov=initial_cov,
        iterations=iterations,
        save_interval=10,
        results_dir=run_results_dir
    )

    # Save metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "initial_cov": initial_cov.tolist(),
        "initial sample": initial_sample,
        "iterations": iterations,
        "acceptance_rate": acceptance_rate
    }
    save_metadata(metadata, filename="mcmc_metadata.json", directory=run_results_dir)

def main():
    # Load MCMC configuration
    json_config_path = "config/mcmc_config.json"  
    mcmc_config = load_mcmc_config(json_config_path)

    # Extract parameters from configuration
    initial_guess_path = mcmc_config["initial_guess_path"]
    ion_velocity_weight = mcmc_config["ion_velocity_weight"]
    iterations = mcmc_config["iterations"]
    initial_cov = mcmc_config["initial_cov"]

    # Create results directory
    os.makedirs(results_dir, exist_ok=True)

    print("Running ground truth simulation (MultiLogBohm)...")
    ground_truth_postprocess = postprocess.copy()
    ground_truth_postprocess["output_file"] = "/mnt/c/Users/elsensoy/model_error_uq_plasma/hall_opt/map_/results-map/ground_truth.json"

    ground_truth_solution = run_simulation_with_config(
        config_multilogbohm, simulation, ground_truth_postprocess, config_type="MultiLogBohm"
    )

    if not ground_truth_solution:
        print("Ground truth simulation failed. Exiting.")
        return

    # Step 2: Extract Observed Data
    averaged_metrics = ground_truth_solution["output"]["average"]
    observed_data = {
        "thrust": averaged_metrics["thrust"],
        "discharge_current": averaged_metrics["discharge_current"],
        "ion_velocity": averaged_metrics["ui"][0], 
        "z_normalized": averaged_metrics["z"]
    }
    save_results_to_json(observed_data, f'mcmc_observed_data_map.json', results_dir = results_dir)
    print("Observed data extracted and saved.")

    # Step 3: Load Initial Guess Parameters
    v1_opt, alpha_opt = load_optimized_params(initial_guess_path)
    if v1_opt is None or alpha_opt is None:
        print("Failed to load initial guess parameters. Exiting.")
        return
    
    v2_opt = v1_opt * alpha_opt
    # Step 4: Run Initial Simulation for Validation
    print(f"Validating initial simulation with v1: {v1_opt}, v2: {v2_opt}")
    initial_config = update_twozonebohm_config(config_spt_100, v1_opt, v2_opt)
    initial_simulation_result = run_simulation_with_config(
        initial_config, simulation, postprocess, config_type="TwoZoneBohm"
    )

    if not initial_simulation_result:
        print("Initial simulation failed. Exiting.")
        return
    # Step 2: Extract initial Data
    averaged_metrics = initial_simulation_result["output"]["average"]
    initial_data = {
        "thrust": averaged_metrics["thrust"],
        "discharge_current": averaged_metrics["discharge_current"],
        "ion_velocity": averaged_metrics["ui"][0], 
        "z_normalized": averaged_metrics["z"]
    }
    save_results_to_json(initial_data, f'mcmc_pre_mcmc_initial.json', results_dir = results_dir)
    print("Initial data extracted and saved.")

    # Step 5: MCMC Sampling
    try:
        print("Starting MCMC sampling...")
        run_mcmc_with_optimized_params(
            json_path=initial_guess_path,
            observed_data=observed_data,
            config=config_spt_100,
            ion_velocity_weight=ion_velocity_weight,
            iterations=iterations,
            initial_cov=initial_cov,
            results_dir=results_dir
        )
        print("MCMC sampling completed successfully.")

    except Exception as e:
        print(f"Error during MCMC sampling: {e}")
        return

if __name__ == "__main__":
    main()