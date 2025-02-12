import os
import json
import sys
from pathlib import Path
import numpy as np
from typing import Dict, Any
from hall_opt.config.dict import Settings
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
    settings
):
    """Performs MCMC inference, saving results in structured directories."""

    # Set up result directories
    mcmc_settings = settings.mcmc
    results_dir = Path(settings.mcmc.results_dir)
    mcmc_base_dir = Path(get_next_results_dir(results_dir, "mcmc-results"))
    settings.mcmc.base_dir = mcmc_base_dir  # Update settings with resolved directory

    final_samples_log_file = os.path.join(settings.mcmc.base_dir, "final_samples_log.csv")
    checkpoint_file = os.path.join(settings.mcmc.base_dir, "checkpoint.json")

    print(f"MCMC base directory: {settings.mcmc.base_dir}")

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
    checkpoint_samples = []  # Stores every 10th sample

    print(f"Starting MCMC inference with {settings.general.iterations} iterations...")

    for iteration in range(iterations):
        try:
            # Generate a new sample
            proposed_sample, log_posterior_val, accepted = next(sampler)
            c1_log, alpha_log = proposed_sample
            c1, alpha = np.exp(c1_log), np.exp(alpha_log)
            c2 = c1 * alpha

            print(f"Iteration {iteration + 1}: c1={c1:.4f}, c2={c2:.4f}, Accepted={accepted}")

            # Store only MCMC parameters
            all_samples.append(proposed_sample)

            # Save only MCMC parameters incrementally
            if (iteration + 1) % save_interval == 0:
                np.savetxt(final_samples_log_file, np.array(all_samples), delimiter=",", fmt="%.6f")

            # Save checkpoint every `checkpoint_interval` iterations (only MCMC parameters)
            if (iteration + 1) % checkpoint_interval == 0:
                checkpoint_samples.append(list(proposed_sample))  # Store every 10th sample
                checkpoint_data = {
                    "iteration": iteration + 1,
                    "checkpoint_samples": checkpoint_samples
                }

                with open(checkpoint_file, "w") as f:
                    json.dump(checkpoint_data, f, indent=4)

                print(f"Checkpoint saved at iteration {iteration + 1}")

        except Exception as e:
            print(f" Error during MCMC iteration {iteration + 1}: {e}")
            break

    print(f"Final samples saved to {settings.mcmc.base_dir}")
    return all_samples

def run_mcmc_with_final_map_params(observed_data: Dict[str, Any],
    settings: Settings):
    """
    Run MCMC with optimized parameters loaded from YAML settings.
    """

    config_file = "hall_opt/config/settings.yaml"  

    # Ensure `mcmc-results-N/` is determined BEFORE running MCMC
  
    final_map_params_path = os.path.join(settings.map.final_map_params_file)
    
    try:
        with open(final_map_params_path, "r") as f:
            final_map_params = json.load(f)  # This is likely a dictionary
    except FileNotFoundError:
        print(f" ERROR: Final MAP parameters file missing: {final_map_params_path}")
        return None

    print(f"DEBUG: Loaded MAP parameters: {final_map_params}")

    #  **Extract only c1_log and alpha_log into a list**
    try:
        params = [final_map_params["c1_log"], final_map_params["alpha_log"]]
    except KeyError:
        raise ValueError(" ERROR: Expected `c1_log` and `alpha_log` in final_map_params.json")
    print(f"DEBUG: Extracted parameter list for MCMC: {params}")


    all_samples = mcmc_inference(
        lambda c_log: log_posterior(np.array(c_log, dtype=np.float64), observed_data, settings, config_file),
        initial_sample=np.array(params, dtype=np.float64),
        initial_cov=np.array(settings.mcmc.initial_cov, dtype=np.float64),
        iterations=settings.general.iterations,
        save_interval=settings.mcmc.save_interval,
        checkpoint_interval=settings.mcmc.checkpoint_interval,
        settings=settings
    )
    return all_samples


