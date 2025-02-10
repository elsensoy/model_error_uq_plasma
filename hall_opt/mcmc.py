import os
import json
import sys
from pathlib import Path
import numpy as np
from typing import Dict, Any
from hall_opt.config.verifier import Settings
from MCMCIterators.samplers import DelayedRejectionAdaptiveMetropolis
from hall_opt.utils.save_data import save_results_to_json, save_metadata
from hall_opt.utils.iter_methods import get_next_filename, get_next_results_dir
from hall_opt.utils.statistics import log_posterior

def mcmc_inference(
    logpdf,
    initial_sample,
    initial_cov,
    iterations,
    save_interval,
    checkpoint_interval,
    settings,
):

    #  base directory is used
    results_dir = settings.mcmc.base_dir  #   set from `mcmc-results-N/`
    os.makedirs(results_dir, exist_ok=True)

    print(f"Using MCMC results directory: {results_dir}")

    #  `iter_metrics/` exists inside `mcmc-results-N/`
    iter_metrics_dir = os.path.join(results_dir, "iter_metrics")
    os.makedirs(iter_metrics_dir, exist_ok=True)

    # Fix incorrect path for final files 
    final_samples_log_file = os.path.join(results_dir, "final_samples_log.csv")
    final_samples_linear_file = os.path.join(results_dir, "final_samples_linear.csv")
    checkpoint_file = os.path.join(results_dir, "checkpoint.json")
    metadata_file = os.path.join(results_dir, "mcmc_metadata.json")

    print(f"MCMC iterations will be saved in: {iter_metrics_dir}")

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

    all_samples = []
    all_samples_linear = []

    print(f" Starting MCMC inference with {iterations} iterations...")

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

            #  iteration results inside `mcmc-results-N/iter_metrics/`
            iter_filename = get_next_filename("metrics", iter_metrics_dir, extension=".json")
            iter_file_path = os.path.join(iter_metrics_dir, os.path.basename(iter_filename))

            iteration_data = {
                "iteration": iteration + 1,
                "c1_log": c1_log,
                "alpha_log": alpha_log,
                "c1": c1,
                "c2": c2,
                "accepted": accepted
            }

            with open(iter_file_path, "w") as f:
                json.dump(iteration_data, f, indent=4)

            print(f"Iteration metrics saved to {iter_file_path}")

            #  final samples incrementally inside `mcmc-results-N/`
            if (iteration + 1) % save_interval == 0:
                np.savetxt(final_samples_log_file, np.array(all_samples), delimiter=",", fmt="%.6f")
                np.savetxt(final_samples_linear_file, np.array(all_samples_linear), delimiter=",", fmt="%.6f")
                print(f"Saved samples at iteration {iteration + 1}")

            #  checkpoint periodically inside `mcmc-results-N/`
            if (iteration + 1) % checkpoint_interval == 0:
                checkpoint_data = {
                    "iteration": iteration + 1,
                    "all_samples": all_samples,
                    "all_samples_linear": all_samples_linear,
                }
                save_results_to_json(checkpoint_data, checkpoint_file)
                print(f"Checkpoint saved at iteration {iteration + 1}")

        except Exception as e:
            print(f" Error during MCMC iteration {iteration + 1}: {e}")
            break

    #  final results inside `mcmc-results-N/`
    np.savetxt(final_samples_log_file, np.array(all_samples), delimiter=",")
    np.savetxt(final_samples_linear_file, np.array(all_samples_linear), delimiter=",")

    # Save metadata inside `mcmc-results-N/`
    acceptance_rate = sampler.accept_ratio()
    metadata = {
        "iterations": iterations,
        "acceptance_rate": acceptance_rate,
        "initial_sample": initial_sample.tolist(),
        "initial_cov": initial_cov.tolist(),
        "final_results_dir": results_dir,
    }

    if settings.mcmc.save_metadata:
        save_metadata(metadata, filename="mcmc_metadata.json", directory=results_dir)
        print(f"Metadata saved to {metadata_file}")

    print(f"Final samples saved to {results_dir}")
    print(f"Acceptance rate: {acceptance_rate:.2%}")

    return all_samples, all_samples_linear, acceptance_rate

def run_mcmc_with_final_map_params(observed_data, settings):
    """Run MCMC using optimized MAP parameters and save results inside `hall_opt/results/mcmc/mcmc-results-N/`."""

    #   `mcmc-results-N/` is created inside `hall_opt/results/mcmc/`
    settings.mcmc.mcmc_results_dir = os.path.join(settings.general.results_dir, "mcmc")
    settings.mcmc.base_dir = get_next_results_dir(settings.mcmc.mcmc_results_dir, "mcmc-results")

    print(f" Using base directory for this MCMC run: {settings.mcmc.base_dir}")

    # check dir
    Path(settings.mcmc.base_dir).mkdir(parents=True, exist_ok=True)

    #  Load final MAP parameters from JSON
    final_map_params_path = Path(settings.map.base_dir) / "final_map_params.json"

    if not final_map_params_path.exists():
        raise FileNotFoundError(f" MAP results file not found at {final_map_params_path}")

    with open(final_map_params_path, "r") as f:
        params = json.load(f)

    if not isinstance(params, list) or len(params) != 2:
        raise ValueError(" Expected a list with two values [c1_log, alpha_log]")

    initial_sample = np.array(params, dtype=np.float64)

    #  Start MCMC inference
    all_samples, all_samples_linear, acceptance_rate = mcmc_inference(
        lambda c_log: log_posterior(np.array(c_log, dtype=np.float64), observed_data, settings),
        initial_sample,
        initial_cov=np.array(settings.mcmc.initial_cov, dtype=np.float64),
        iterations=settings.mcmc.save_interval,
        save_interval=settings.mcmc.save_interval,
        checkpoint_interval=settings.mcmc.checkpoint_interval,
        settings=settings,  #  Pass settings object so paths are used dynamically
    )

    print(f" MCMC completed. Acceptance rate: {acceptance_rate:.2%}")
    return all_samples, all_samples_linear, acceptance_rate
