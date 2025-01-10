import os
import json
import numpy as np
import logging
from datetime import datetime
from MCMCIterators.samplers import DelayedRejectionAdaptiveMetropolis
from config.simulation import (
    simulation, 
    config_spt_100, 
    postprocess, 
    config_multilogbohm, 
    update_twozonebohm_config, 
    run_simulation_with_config
)
from utils.save_data import save_results_to_json, save_metadata
from utils.iter_methods import load_optimized_params, get_next_results_dir
from utils.statistics import log_posterior

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
    results_dir="mcmc/results"
):
    """
    Perform MCMC inference, saving results at intervals.
    """
    run_dir = get_next_results_dir(base_dir=results_dir, base_name="mcmc-results")
    metrics_dir = os.path.join(run_dir, "iteration_metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    final_samples_log_file = os.path.join(run_dir, "final_samples_log.csv")
    final_samples_linear_file = os.path.join(run_dir, "final_samples_linear.csv")

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

    for iteration in range(iterations):
        try:
            proposed_sample, log_posterior_val, accepted = next(sampler)
            v1_log, alpha_log = proposed_sample
            v1, alpha = 10 ** v1_log, 10 ** alpha_log
            v2 = alpha * v1

            logger.info(
                f"Iteration {iteration + 1}: v1={v1:.4f}, v2={v2:.4f}, Log Posterior={log_posterior_val:.4f}, Accepted={accepted}"
            )

            all_samples.append(proposed_sample)
            all_samples_linear.append([v1, alpha])

            updated_config = update_twozonebohm_config(config_spt_100, v1, v2)
            simulation_result = run_simulation_with_config(
                updated_config, simulation, postprocess, "TwoZoneBohm"
            )

            if simulation_result:
                averaged_metrics = simulation_result["output"]["average"]
                iteration_metrics = {
                    "thrust": averaged_metrics["thrust"],
                    "discharge_current": averaged_metrics["discharge_current"],
                    "ion_velocity": averaged_metrics["ui"][0],
                    "z_normalized": averaged_metrics["z"],
                }
                metrics_filename = os.path.join(metrics_dir, f"iteration_{iteration + 1}_metrics.json")
                save_results_to_json(iteration_metrics, metrics_filename)
                logger.info(f"Saved metrics for iteration {iteration + 1} to {metrics_filename}")
            else:
                logger.warning(f"Simulation failed at iteration {iteration + 1}.")
                failing_samples.append({
                    "iteration": iteration + 1,
                    "v1_log": v1_log,
                    "alpha_log": alpha_log,
                    "v1": v1,
                    "v2": v2,
                })

            if (iteration + 1) % save_interval == 0:
                np.savetxt(final_samples_log_file, np.array(all_samples), delimiter=",")
                np.savetxt(final_samples_linear_file, np.array(all_samples_linear), delimiter=",")
                logger.info(f"Saved samples to {final_samples_log_file} and {final_samples_linear_file}")

        except Exception as e:
            logger.error(f"Error during MCMC iteration {iteration + 1}: {e}")
            break

    np.savetxt(final_samples_log_file, np.array(all_samples), delimiter=",")
    np.savetxt(final_samples_linear_file, np.array(all_samples_linear), delimiter=",")
    logger.info(f"Final samples saved to {final_samples_log_file} (log-space) and {final_samples_linear_file} (linear-space)")

    return all_samples, all_samples_linear, sampler.accept_ratio()


def run_mcmc_with_optimized_params(
    initial_guess_path,
    observed_data,
    config,
    ion_velocity_weight,
    iterations,
    initial_cov,
    results_dir="mcmc/results"
):
    """
    Run MCMC with optimized parameters.
    """
    v1_opt, alpha_opt = load_optimized_params(initial_guess_path)
    if v1_opt is None or alpha_opt is None:
        raise ValueError("Failed to load initial guess parameters.")

    initial_sample = [np.log10(v1_opt), np.log10(alpha_opt)]

    all_samples, all_samples_linear, acceptance_rate = mcmc_inference(
        lambda v_log: log_posterior(v_log, observed_data, config),
        initial_sample,
        initial_cov=initial_cov,
        iterations=iterations,
        results_dir=results_dir
    )

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "initial_cov": initial_cov.tolist(),
        "initial_sample": initial_sample,
        "iterations": iterations,
        "acceptance_rate": acceptance_rate,
    }
    save_metadata(metadata, filename="mcmc_metadata.json", directory=results_dir)

    return all_samples, all_samples_linear, acceptance_rate
