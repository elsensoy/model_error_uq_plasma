import os
import json
import sys
import numpy as np
from MCMCIterators.samplers import DelayedRejectionAdaptiveMetropolis
from hall_opt.config.loader import Settings, load_yml_settings, extract_anom_model
from hall_opt.config.run_model import run_simulation_with_config
from hall_opt.utils.save_data import save_results_to_json
from hall_opt.utils.iter_methods import get_next_results_dir
from hall_opt.utils.statistics import log_posterior

# HallThruster Path Setup
hallthruster_path = "/home/elida/.julia/packages/HallThruster/tHQQa/python"
if hallthruster_path not in sys.path:
    sys.path.append(hallthruster_path)

import hallthruster as het


def mcmc_inference(
    logpdf,
    initial_sample,
    initial_cov,
    iterations,
    save_interval,
    checkpoint_interval,
    results_dir,
    save_metadata_flag,
):
    """
    Perform MCMC inference, saving results at intervals and creating checkpoints.
    """
    # Create directories for results and iteration metrics
    run_dir = get_next_results_dir(base_dir=results_dir, base_name="mcmc-results")
    metrics_dir = os.path.join(run_dir, "iteration_metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    # Define file paths for saving results
    final_samples_log_file = os.path.join(run_dir, "final_samples_log.csv")
    final_samples_linear_file = os.path.join(run_dir, "final_samples_linear.csv")
    checkpoint_file = os.path.join(run_dir, "checkpoint.json")

    # Initialize MCMC sampler
    sampler = DelayedRejectionAdaptiveMetropolis(
        logpdf,
        np.array(initial_sample),
        initial_cov,
        adapt_start=10,
        eps=1e-6,
        sd=2.4**2 / len(initial_sample),
        interval=1,
        level_scale=1e-1,
    )

    # Store samples and metadata
    all_samples = []
    all_samples_linear = []
    metadata = {
        "iterations": iterations,
        "save_interval": save_interval,
        "checkpoint_interval": checkpoint_interval,
        "initial_cov": initial_cov.tolist(),
        "initial_sample": initial_sample,
        "acceptance_rate": None,
    }

    print(f"Starting MCMC inference with {iterations} iterations...")
    print(f"Results will be saved to: {run_dir}")

    # Run MCMC iterations
    for iteration in range(iterations):
        try:
            proposed_sample, log_posterior_val, accepted = next(sampler)
            c1_log, alpha_log = proposed_sample
            c1, alpha = np.exp(c1_log), np.exp(alpha_log)
            c2 = c1 * alpha

            print(
                f"Iteration {iteration + 1}: c1={c1:.4f}, c2={c2:.4f}, "
                f"Log Posterior={log_posterior_val:.4f}, Accepted={accepted}"
            )

            all_samples.append(proposed_sample)
            all_samples_linear.append([c1, c2])

            # Save metrics for this iteration
            iteration_metrics = {
                "iteration": iteration + 1,
                "c1": c1,
                "alpha": alpha,
                "c2": c2,
                "log_posterior": log_posterior_val,
                "accepted": accepted,
            }
            metrics_filename = os.path.join(metrics_dir, f"iteration_{iteration + 1}_metrics.json")
            save_results_to_json(iteration_metrics, metrics_filename)

            # Save checkpoints at specified intervals
            if (iteration + 1) % checkpoint_interval == 0:
                checkpoint_data = {
                    "iteration": iteration + 1,
                    "all_samples": all_samples,
                    "all_samples_linear": all_samples_linear,
                }
                save_results_to_json(checkpoint_data, checkpoint_file)

            # Save results at regular intervals
            if (iteration + 1) % save_interval == 0:
                np.savetxt(final_samples_log_file, np.array(all_samples), delimiter=",")
                np.savetxt(final_samples_linear_file, np.array(all_samples_linear), delimiter=",")

        except Exception as e:
            print(f"Error during MCMC iteration {iteration + 1}: {e}")
            break

    # Save final results
    np.savetxt(final_samples_log_file, np.array(all_samples), delimiter=",")
    np.savetxt(final_samples_linear_file, np.array(all_samples_linear), delimiter=",")
    metadata["acceptance_rate"] = sampler.accept_ratio()

    if save_metadata_flag:
        save_results_to_json(metadata, os.path.join(run_dir, "mcmc_metadata.json"))

    print(f"Final samples saved to {final_samples_log_file} (log-space) and {final_samples_linear_file} (linear-space)")
    print(f"Acceptance rate: {sampler.accept_ratio():.2%}")

    return all_samples, all_samples_linear, sampler.accept_ratio()


def run_mcmc_with_final_map_params(
    final_map_params,
    observed_data,
    config,
    settings,
    simulation,
    ion_velocity_weight,
    iterations,
    initial_cov,
    results_dir="mcmc/results",
):
    """
    Run MCMC with optimized parameters loaded from YAML settings.
    """
    # Load optimized parameters
    c1_opt, alpha_opt = load_final_map_params(final_map_params_path)
    if c1_opt is None or alpha_opt is None:
        raise ValueError("Failed to load initial guess parameters.")

    print(f"Loaded initial MAP parameters: c1_opt={c1_opt}, alpha_opt={alpha_opt}")

    initial_sample = [np.log10(c1_opt), np.log10(alpha_opt)]

    # Extract MCMC parameters from YAML settings
    mcmc_params = settings.mcmc_params
    save_interval = mcmc_params["save_interval"]
    checkpoint_interval = mcmc_params["checkpoint_interval"]
    save_metadata_flag = mcmc_params["save_metadata"]

    print(f"Starting MCMC with {iterations} iterations...")
    print(f"Results will be saved to: {results_dir}")

    # Perform MCMC inference
    all_samples, all_samples_linear, acceptance_rate = mcmc_inference(
        lambda c_log: log_posterior(c_log, observed_data, settings=settings),
        initial_sample,
        initial_cov=initial_cov,
        iterations=iterations,
        save_interval=save_interval,
        checkpoint_interval=checkpoint_interval,
        results_dir=results_dir,
        save_metadata_flag=save_metadata_flag,
    )

    print(f"MCMC completed. Acceptance rate: {acceptance_rate:.2%}")
    return all_samples, all_samples_linear, acceptance_rate
