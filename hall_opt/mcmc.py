import os
import json
import numpy as np
import logging
from datetime import datetime
from MCMCIterators.samplers import DelayedRejectionAdaptiveMetropolis
from hall_opt.config.loader import Settings, extract_anom_model
from hall_opt.config.run_model import run_simulation_with_config
from utils.save_data import save_results_to_json, save_metadata
from utils.iter_methods import load_optimized_params, get_next_results_dir
from utils.statistics import log_posterior

# Logger setup
def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger("MCMC")

logger = setup_logger()

def mcmc_inference(
    logpdf,
    initial_sample,
    initial_cov,
    iterations,
    save_interval=10,
    checkpoint_interval=10,
    results_dir="mcmc/results",
    save_metadata_flag=True
):
    """
    Perform MCMC inference, saving results at intervals and creating checkpoints.
    """
    run_dir = get_next_results_dir(base_dir=results_dir, base_name="mcmc-results")
    metrics_dir = os.path.join(run_dir, "iteration_metrics")
    os.makedirs(metrics_dir, exist_ok=True)

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
    failing_samples = []
    metadata = {
        "iterations": iterations,
        "save_interval": save_interval,
        "checkpoint_interval": checkpoint_interval,
        "initial_cov": initial_cov.tolist(),
        "initial_sample": initial_sample,
        "acceptance_rate": None,
    }

    for iteration in range(iterations):
        try:
            proposed_sample, log_posterior_val, accepted = next(sampler)
            c1_log, alpha_log = proposed_sample
            c1, alpha = np.exp(c1_log), np.exp(alpha_log)
            c2 = c1 * alpha

            logger.info(
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
            logger.info(f"Saved metrics for iteration {iteration + 1} to {metrics_filename}")

            # Save checkpoints at specified intervals
            if (iteration + 1) % checkpoint_interval == 0:
                checkpoint_data = {
                    "iteration": iteration + 1,
                    "all_samples": all_samples,
                    "all_samples_linear": all_samples_linear,
                }
                save_results_to_json(checkpoint_data, checkpoint_file)
                logger.info(f"Saved checkpoint at iteration {iteration + 1} to {checkpoint_file}")

            # Save results at regular intervals
            if (iteration + 1) % save_interval == 0:
                np.savetxt(final_samples_log_file, np.array(all_samples), delimiter=",")
                np.savetxt(final_samples_linear_file, np.array(all_samples_linear), delimiter=",")
                logger.info(f"Saved samples to {final_samples_log_file} and {final_samples_linear_file}")

        except Exception as e:
            logger.error(f"Error during MCMC iteration {iteration + 1}: {e}")
            break

    # Save final results
    np.savetxt(final_samples_log_file, np.array(all_samples), delimiter=",")
    np.savetxt(final_samples_linear_file, np.array(all_samples_linear), delimiter=",")
    metadata["acceptance_rate"] = sampler.accept_ratio()

    if save_metadata_flag:
        save_results_to_json(metadata, os.path.join(run_dir, "mcmc_metadata.json"))
        logger.info(f"Metadata saved to {os.path.join(run_dir, 'mcmc_metadata.json')}")

    logger.info(f"Final samples saved to {final_samples_log_file} (log-space) and {final_samples_linear_file} (linear-space)")

    return all_samples, all_samples_linear, sampler.accept_ratio()

def run_mcmc_with_optimized_params(
    map_initial_guess_path,
    observed_data,
    config,
    settings,
    simulation,
    ion_velocity_weight,
    iterations,
    initial_cov,
    results_dir="mcmc/results"
):
    """
    Run MCMC with optimized parameters.
    """
    # Load optimized parameters
    c1_opt, alpha_opt = load_optimized_params(map_initial_guess_path)
    if c1_opt is None or alpha_opt is None:
        raise ValueError("Failed to load initial guess parameters.")

    initial_sample = [np.log10(c1_opt), np.log10(alpha_opt)]

    # Extract MCMC parameters from settings
    mcmc_params = settings.mcmc_params
    save_interval = mcmc_params.get("save_interval", 10)
    checkpoint_interval = mcmc_params.get("checkpoint_interval", 10)
    save_metadata_flag = mcmc_params.get("save_metadata", True)

    # Perform MCMC inference
    all_samples, all_samples_linear, acceptance_rate = mcmc_inference(
        lambda c_log: log_posterior(c_log, observed_data, settings=settings),
        initial_sample,
        initial_cov=initial_cov,
        iterations=iterations,
        save_interval=save_interval,
        checkpoint_interval=checkpoint_interval,
        results_dir=results_dir,
        save_metadata_flag=save_metadata_flag
    )

    return all_samples, all_samples_linear, acceptance_rate
