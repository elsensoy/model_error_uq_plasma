import os
import json
import sys
import numpy as np
from MCMCIterators.samplers import DelayedRejectionAdaptiveMetropolis
from hall_opt.utils.save_data import save_results_to_json, save_metadata
from hall_opt.utils.iter_methods import get_next_results_dir
from hall_opt.utils.statistics import log_posterior


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
    Perform MCMC inference, saving final results and periodic checkpoints.
    """

    # Create a new directory for this MCMC run (mcmc-results-1, mcmc-results-2, ...)
    run_dir = get_next_results_dir(results_dir, base_name="mcmc-results")
    os.makedirs(run_dir, exist_ok=True)  # Ensure directory exists

    print(f"Using results directory: {run_dir}")

    # Define file paths for saving final results
    final_samples_log_file = os.path.join(run_dir, "final_samples_log.csv")
    final_samples_linear_file = os.path.join(run_dir, "final_samples_linear.csv")
    checkpoint_file = os.path.join(run_dir, "checkpoint.json")

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

    all_samples = []
    all_samples_linear = []

    print(f"Starting MCMC inference with {iterations} iterations...")

    for iteration in range(iterations):
        try:
            proposed_sample, log_posterior_val, accepted = next(sampler)
            c1_log, alpha_log = proposed_sample
            c1, alpha = np.exp(c1_log), np.exp(alpha_log)
            c2 = c1 * alpha

            print(f"Iteration {iteration + 1}: c1={c1:.4f}, c2={c2:.4f}, Accepted={accepted}")

            all_samples.append(proposed_sample)
            all_samples_linear.append([c1, c2])

            # Save results incrementally at regular intervals
            if (iteration + 1) % save_interval == 0:
                np.savetxt(final_samples_log_file, np.array(all_samples), delimiter=",", fmt="%.6f")
                np.savetxt(final_samples_linear_file, np.array(all_samples_linear), delimiter=",", fmt="%.6f")
                print(f"Saved samples at iteration {iteration + 1}")

            # Save checkpoint periodically
            if (iteration + 1) % checkpoint_interval == 0:
                checkpoint_data = {
                    "iteration": iteration + 1,
                    "all_samples": all_samples,
                    "all_samples_linear": all_samples_linear,
                }
                save_results_to_json(checkpoint_data, checkpoint_file)
                print(f"Checkpoint saved at iteration {iteration + 1}")

        except Exception as e:
            print(f"Error during MCMC iteration {iteration + 1}: {e}")
            break


    # Save final results
    np.savetxt(final_samples_log_file, np.array(all_samples), delimiter=",")
    np.savetxt(final_samples_linear_file, np.array(all_samples_linear), delimiter=",")

    # Save metadata
    acceptance_rate = sampler.accept_ratio()
    metadata = {
        "iterations": iterations,
        "acceptance_rate": acceptance_rate,
        "initial_sample": initial_sample.tolist(),
        "initial_cov": initial_cov.tolist(),
        "final_results_dir": run_dir,
    }

    if save_metadata_flag:
        save_metadata(metadata, filename="mcmc_metadata.json", directory=run_dir)
        print(f"Metadata saved to {run_dir}/mcmc_metadata.json")

    print(f"Final samples saved to {run_dir}")
    print(f"Acceptance rate: {acceptance_rate:.2%}")

    return all_samples, all_samples_linear, acceptance_rate


def run_mcmc_with_final_map_params(
    final_map_params,
    observed_data,
    config,
    settings,
    simulation,
    ion_velocity_weight,
    iterations,
    initial_cov
):
    """
    Run MCMC with optimized parameters loaded from YAML settings.
    """

    results_dir = settings.general_settings["results_dir"]
    print(f"Using results directory from settings: {results_dir}")

    # Load final MAP parameters from JSON
    final_map_params_path = settings.optimization_params["map_params"]["final_map_params"]

    if not os.path.exists(final_map_params_path):
        raise FileNotFoundError(f"MAP results file not found at {final_map_params_path}")

    with open(final_map_params_path, "r") as f:
        params = json.load(f)

    if not isinstance(params, list) or len(params) != 2:
        raise ValueError("Expected a list with two values [c1_log, alpha_log]")

    initial_sample = np.array(params, dtype=np.float64)

    all_samples, all_samples_linear, acceptance_rate = mcmc_inference(
        lambda c_log: log_posterior(np.array(c_log, dtype=np.float64), observed_data, settings),
        initial_sample,
        initial_cov=np.array(initial_cov, dtype=np.float64),
        iterations=iterations,
        save_interval=settings.optimization_params["mcmc_params"]["save_interval"],
        checkpoint_interval=settings.optimization_params["mcmc_params"]["checkpoint_interval"],
        results_dir=results_dir,  # Use settings-based results directory
        save_metadata_flag=settings.optimization_params["mcmc_params"]["save_metadata"],
    )

    print(f"MCMC completed. Acceptance rate: {acceptance_rate:.2%}")
    return all_samples, all_samples_linear, acceptance_rate
