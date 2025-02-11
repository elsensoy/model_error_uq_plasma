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

# def mcmc_inference(
#     logpdf,
#     initial_sample,
#     initial_cov,
#     iterations,
#     save_interval,
#     checkpoint_interval,
#     settings
#     # save_metadata
# ):
#     """Performs MCMC inference, saving results in structured directories using YAML attributes."""

#     # Ensure `mcmc-results-N/iter_metrics/` exist
#     mcmc_settings = settings.mcmc
#     results_dir = Path(settings.mcmc.results_dir)
#     mcmc_base_dir = Path(get_next_results_dir(results_dir, "mcmc-results"))
#     # mcmc_base_dir.mkdir(parents=True, exist_ok=True)
#     settings.mcmc.base_dir = mcmc_base_dir  # Update settings with resolved directory

    
#     print(f" MCMC base directory: {settings.mcmc.base_dir}")
    
#     # Ensure the correct mcmc results directory (e.g., mcmc-results-1/, map-results-2/)
#     # settings.mcmc.base_dir = get_next_results_dir(results_dir, "mcmc-results")
#     # print(f"Using MCMC base directory: {mcmc_base_dir}")
    
#     # Define final output file paths dynamically from settings
#     iter_metrics_dir = os.path.join(mcmc_base_dir/"iter_metrics")
#     # os.makedirs(iter_metrics_dir, exist_ok=True)
#     final_samples_log_file = os.path.join(settings.mcmc.base_dir, "final_samples_log.csv")
#     final_samples_linear_file = os.path.join(settings.mcmc.base_dir, "final_samples_linear.csv")
#     checkpoint_file = os.path.join(settings.mcmc.base_dir, "checkpoint.json")
#     # metadata_file = os.path.join(settings.mcmc.base_dir, "mcmc_metadata.json")

#     print(f"MCMC output will be saved in: {results_dir}")

#     # Initialize MCMC sampler
#     sampler = DelayedRejectionAdaptiveMetropolis(
#         logpdf,
#         np.array(initial_sample),
#         initial_cov,
#         adapt_start=10,
#         eps=1e-6,
#         sd=2.4**2 / len(initial_sample),
#         interval=1,
#         level_scale=1e-1,
#     )

#     all_samples = []
#     all_samples_linear = []

#     print(f"Starting MCMC inference with {iterations} iterations...")



#     for iteration in range(iterations):
#         try:
#             proposed_sample, log_posterior_val, accepted = next(sampler)
#             c1_log, alpha_log = proposed_sample
#             c1, alpha = np.exp(c1_log), np.exp(alpha_log)
#             c2 = c1 * alpha

#             print(
#                 f"Iteration {iteration + 1}: c1={c1:.4f}, c2={c2:.4f}, "
#                 f"Log Posterior={log_posterior_val:.4f}, Accepted={accepted}"
#             )

#             # all_samples.append(proposed_sample)
#             # all_samples_linear.append([c1, c2])

#             iteration_data = {
#                 "iteration": iteration + 1,
#                 "c1_log": c1_log,
#                 "alpha_log": alpha_log,
#                 "c1": c1,
#                 "c2": c2,
#                 "accepted": accepted
#             }


#             print(f"Iteration {iteration + 1}: c1={c1:.4f}, c2={c2:.4f}, Accepted={accepted}")

#             all_samples.append(proposed_sample)
#             all_samples_linear.append([c1, c2])

#             # Generate next filename inside `mcmc-results-N/iter_metrics/`
#             iter_filename = get_next_filename("metrics", iter_metrics_dir, extension=".json")
#             iter_file_path = os.path.join(iter_metrics_dir, iter_filename)

#             with open(iter_file_path, "w") as f:
#                 json.dump(iteration_data, f, indent=4)

#             print(f"Iteration metrics saved to {iter_file_path}")

#             # Save final samples incrementally
#             if (iteration + 1) % save_interval == 0:
#                 np.savetxt(final_samples_log_file, np.array(all_samples), delimiter=",", fmt="%.6f")
#                 np.savetxt(final_samples_linear_file, np.array(all_samples_linear), delimiter=",", fmt="%.6f")
#                 print(f"Saved samples at iteration {iteration + 1}")


#             # Save checkpoint periodically
#             if (iteration + 1) % checkpoint_interval == 0:
#                 # Store only every 10th sample in the checkpoint
#                 if "checkpoint_samples" not in locals():
#                     checkpoint_samples = []  # Initialize list on first use

#                 checkpoint_samples.append(list(all_samples[-1]))  # Append only every checkpoint_interval-th sample

#                 checkpoint_data = {
#                     "iteration": iteration + 1,
#                     "checkpoint_samples": checkpoint_samples,  # Store only sampled checkpoints
#                 }

#                 with open(checkpoint_file, "w") as f:
#                     json.dump(checkpoint_data, f, indent=4)
#                 print(f"Checkpoint saved at iteration {iteration + 1}")
                
#         except Exception as e:
#             print(f" Error during MCMC iteration {iteration + 1}: {e}")
#             break

#     # Save final results
#     np.savetxt(final_samples_log_file, np.array(all_samples), delimiter=",")
#     np.savetxt(final_samples_linear_file, np.array(all_samples_linear), delimiter=",")

#     # Save metadata
#     acceptance_rate = sampler.accept_ratio()
#     # metadata = {
#     #     "iterations": iterations,
#     #     "acceptance_rate": acceptance_rate,
#     #     "initial_sample": initial_sample.tolist(),
#     #     "initial_cov": initial_cov.tolist(),
#     #     "final_results_dir": settings.mcmc.results_dir,
#     # }

#     # if settings.mcmc.save_metadata:
#     #     save_metadata(metadata, filename="mcmc_metadata.json", directory=settings.mcmc.base_dir)
#     #     print(f"Metadata saved to {metadata_file}")

#     print(f"Final samples saved to {settings.mcmc.base_dir}")
#     print(f"Acceptance rate: {acceptance_rate:.2%}")

#     return all_samples, all_samples_linear, acceptance_rate

def mcmc_inference(
    logpdf,
    initial_sample,
    initial_cov,
    iterations,
    save_interval,
    checkpoint_interval,
    settings
):
    """Performs MCMC inference, saving results in structured directories using YAML attributes."""

    # Ensure `mcmc-results-N/iter_metrics/` exist
    mcmc_settings = settings.mcmc
    results_dir = Path(settings.mcmc.results_dir)
    mcmc_base_dir = Path(get_next_results_dir(results_dir, "mcmc-results"))
    settings.mcmc.base_dir = mcmc_base_dir  # Update settings with resolved directory

    print(f" MCMC base directory: {settings.mcmc.base_dir}")

    # Define final output file paths dynamically from settings
    iter_metrics_dir = os.path.join(mcmc_base_dir, "iter_metrics")
    final_samples_log_file = os.path.join(settings.mcmc.base_dir, "final_samples_log.csv")
    final_samples_linear_file = os.path.join(settings.mcmc.base_dir, "final_samples_linear.csv")
    checkpoint_file = os.path.join(settings.mcmc.base_dir, "checkpoint.json")

    print(f"MCMC output will be saved in: {results_dir}")

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
    checkpoint_samples = []  # Initialize before the loop to persist across iterations

    print(f"Starting MCMC inference with {iterations} iterations...")

    for iteration in range(iterations):
        try:
            # Generate a new sample
            proposed_sample, log_posterior_val, accepted = next(sampler)
            c1_log, alpha_log = proposed_sample
            c1, alpha = np.exp(c1_log), np.exp(alpha_log)
            c2 = c1 * alpha

            print(
                f"Iteration {iteration + 1}: c1={c1:.4f}, c2={c2:.4f}, "
                f"Log Posterior={log_posterior_val:.4f}, Accepted={accepted}"
            )

            print(f"DEBUG: Before appending - all_samples length: {len(all_samples)}")
            all_samples.append(proposed_sample)
            all_samples_linear.append([c1, c2])
            print(f"DEBUG: After appending - all_samples length: {len(all_samples)}")

            # Append samples correctly (no duplicates)
            all_samples.append(proposed_sample)
            all_samples_linear.append([c1, c2])

            # Correctly structured iteration data
            iteration_data = {
                "iteration": iteration + 1,
                "c1_log": c1_log,
                "alpha_log": alpha_log,
                "c1": c1,
                "c2": c2,
                "accepted": accepted
            }

            # Save iteration metrics (only MCMC parameters, no extra simulation metrics)
            iter_filename = get_next_filename("metrics", iter_metrics_dir, extension=".json")
            iter_file_path = os.path.join(iter_metrics_dir, iter_filename)

            with open(iter_file_path, "w") as f:
                json.dump(iteration_data, f, indent=4)

            print(f"Iteration metrics saved to {iter_file_path}")

            # Save final samples incrementally
            if (iteration + 1) % save_interval == 0:
                np.savetxt(final_samples_log_file, np.array(all_samples), delimiter=",", fmt="%.6f")
                np.savetxt(final_samples_linear_file, np.array(all_samples_linear), delimiter=",", fmt="%.6f")
                print(f"Saved samples at iteration {iteration + 1}")

            # Save checkpoint periodically
            if (iteration + 1) % checkpoint_interval == 0:
                checkpoint_samples.append(list(proposed_sample))  # Store only every 10th sample

                checkpoint_data = {
                    "iteration": iteration + 1,
                    "checkpoint_samples": checkpoint_samples  # Store only sampled checkpoints
                }

                with open(checkpoint_file, "w") as f:
                    json.dump(checkpoint_data, f, indent=4)

                print(f"Checkpoint saved at iteration {iteration + 1}")

        except Exception as e:
            print(f" Error during MCMC iteration {iteration + 1}: {e}")
            break

    # Save final results at the end of all iterations
    np.savetxt(final_samples_log_file, np.array(all_samples), delimiter=",")
    np.savetxt(final_samples_linear_file, np.array(all_samples_linear), delimiter=",")

    # Save metadata (optional)
    acceptance_rate = sampler.accept_ratio()
    print(f"Final samples saved to {settings.mcmc.base_dir}")
    print(f"Acceptance rate: {acceptance_rate:.2%}")

    return all_samples, all_samples_linear, acceptance_rate


def run_mcmc_with_final_map_params(observed_data: Dict[str, Any],
    settings: Settings):
    """
    Run MCMC with optimized parameters loaded from YAML settings.
    """
 
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


    initial_sample = np.array(params, dtype=np.float64)

    all_samples, all_samples_linear, acceptance_rate = mcmc_inference(
        lambda c_log: log_posterior(np.array(c_log, dtype=np.float64), observed_data, settings),
        initial_sample,
        initial_cov=np.array(settings.mcmc.initial_cov, dtype=np.float64),
        iterations=settings.mcmc.save_interval,
        save_interval=settings.mcmc.save_interval,
        checkpoint_interval=settings.mcmc.checkpoint_interval,
        settings=settings
    )


    print(f"MCMC completed. Acceptance rate: {acceptance_rate:.2%}")
    return all_samples, all_samples_linear, acceptance_rate


