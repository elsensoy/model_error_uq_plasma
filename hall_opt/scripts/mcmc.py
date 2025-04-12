import os
import json
import sys
from pathlib import Path
import numpy as np
from typing import Dict, Any, Optional 
from hall_opt.config.dict import Settings
from MCMCIterators.samplers import DelayedRejectionAdaptiveMetropolis
from hall_opt.utils.iter_methods import get_next_filename, get_next_results_dir
from hall_opt.posterior.statistics import log_posterior
from hall_opt.utils.parse import find_file_anywhere
from datetime import datetime
from hall_opt.utils.save_data import save_metadata
from hall_opt.config.evaluate.checkpoint import restore_mcmc_checkpoint

def mcmc_inference(
    logpdf,
    initial_sample,
    initial_cov,
    iterations,
    save_interval,
    checkpoint_interval,
    settings,
    burn_in,
    observed_data 
):
    """Performs MCMC inference, saving results in structured directories."""

    # Set up result directories
    mcmc_base_dir = Path(settings.output_dir) / "mcmc"
    mcmc_base_dir.mkdir(parents=True, exist_ok=True)
    settings.mcmc.base_dir = get_next_results_dir(str(mcmc_base_dir), "mcmc-results")
    Path(settings.mcmc.base_dir).mkdir(parents=True, exist_ok=True)
    final_samples_log_file = os.path.join(settings.mcmc.base_dir, "final_samples_log.csv")
    final_samples_mcmc_file = os.path.join(settings.mcmc.base_dir, "final_samples.csv")
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
    checkpoint_samples = []   
    burn_in_samples = [] 
    checkpoint_index = 1

    print(f"Starting MCMC inference with {settings.mcmc.max_iter} iterations with burn in {settings.mcmc.burn_in}...")

    for iteration in range(iterations):
        try:
            # Generate a new sample
            proposed_sample, log_posterior_val, accepted = next(sampler)
            c1_log, alpha_log = proposed_sample
            c1, alpha = np.exp(c1_log), np.exp(alpha_log)
            c2 = c1 * alpha

            print(f"Iteration {iteration + 1}: c1={c1:.4f}, c2={c2:.4f}, Accepted={accepted}")
            print(f"Log Posterior Value {log_posterior_val}")
            if iteration < burn_in:
                burn_in_samples.append(proposed_sample)
                continue  # Skip saving/logging for burn-in

            # Store only post-burn-in MCMC parameters
            all_samples.append(proposed_sample)

            # Save MCMC parameters incrementally
            if (iteration + 1) % save_interval == 0:
                np.savetxt(final_samples_log_file, np.array(all_samples), delimiter=",", fmt="%.6f")
                np.savetxt(final_samples_mcmc_file, np.array(all_samples), delimiter=",", fmt="%.6f")             

            # Save checkpoint every `checkpoint_interval` iterations
            if (iteration + 1) % checkpoint_interval == 0:
                # Load existing checkpoint data
                if os.path.exists(checkpoint_file):
                    with open(checkpoint_file, "r") as f:
                        all_checkpoints = json.load(f)
                else:
                    all_checkpoints = {}

                # Add new checkpoint entry
                all_checkpoints[str(checkpoint_index)] = {
                    "iteration": iteration + 1,
                    "sample": list(proposed_sample)
                }

                # Save updated checkpoints
                with open(checkpoint_file, "w") as f:
                    json.dump(all_checkpoints, f, indent=4)

                print(f"Checkpoint #{checkpoint_index} saved at iteration {iteration + 1}")
                checkpoint_index += 1


            # Prepare metadata dictionary
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "initial_parameters": list(initial_sample),
                "final_sample": list(all_samples[-1]) if all_samples else None,
                "final_log_posterior": float(log_posterior(np.array(all_samples[-1]), observed_data, settings)) if all_samples else None,
                "used_covariance": initial_cov.tolist(),
                "total_iterations": iterations,
                "burn_in": burn_in,
                "samples_saved": len(all_samples),
                "checkpoint_iterations": list(range(checkpoint_interval, iterations + 1, checkpoint_interval)),
                "acceptance_rate": sampler.accept_ratio()  # Using built-in tracking
            }

            # Save metadata
            save_metadata(settings, metadata, filename="mcmc_metadata.json")

        except Exception as e:
            print(f" Error during MCMC iteration {iteration + 1}: {e}")
            break
    # Final save in case loop ended before hitting save_interval
    if all_samples:
        print(f"[INFO] Saving final samples (n={len(all_samples)}) to log file...")
        np.savetxt(final_samples_log_file, np.array(all_samples), delimiter=",", fmt="%.6f")


    print(f"Final samples saved to {settings.mcmc.base_dir}")
    return all_samples




def run_mcmc_with_final_map_params(observed_data: Dict[str, Any], settings: Settings):
    """
    Run MCMC using the most recently modified MAP result file found dynamically.
    Includes checkpoint restore support.
    """
    # Overwrite default MCMC values with user-provided input if available

    if settings.mcmc_settings:
        print("[DEBUG] Found user-defined mcmc_settings in YAML. Applying overrides...")
        user = settings.mcmc_settings

        if user.max_iter is not None:
            print(f"[DEBUG] Overriding max_iter: {settings.mcmc.max_iter} -> {user.max_iter}")
            settings.mcmc.max_iter = user.max_iter

        if user.burn_in is not None:
            print(f"[DEBUG] Overriding burn_in: {settings.mcmc.burn_in} -> {user.burn_in}")
            settings.mcmc.burn_in = user.burn_in

        if user.initial_cov is not None:
            print(f"[DEBUG] Overriding initial_cov:\n{settings.mcmc.initial_cov} -> {user.initial_cov}")
            settings.mcmc.initial_cov = user.initial_cov

        if user.save_interval is not None:
            print(f"[DEBUG] Overriding save_interval: {settings.mcmc.save_interval} -> {user.save_interval}")
            settings.mcmc.save_interval = user.save_interval

        if user.checkpoint_interval is not None:
            print(f"[DEBUG] Overriding checkpoint_interval: {settings.mcmc.checkpoint_interval} -> {user.checkpoint_interval}")
            settings.mcmc.checkpoint_interval = user.checkpoint_interval

    else:
        print("[DEBUG] No user-defined mcmc_settings found in YAML. Using all defaults.")

    # Ask the user
    print("Enter checkpoint iteration to resume from (0 to start fresh):")
    try:
        resume_iter = int(input().strip())
    except ValueError:
        print("[ERROR] Invalid input. Please enter an integer.")
        return None

    # If resuming, try to restore last sample from checkpoint
    if resume_iter > 0:
        checkpoint_path = Path(settings.output_dir) / "mcmc" / settings.mcmc.base_dir.name / "checkpoint.json"
        restored_sample = restore_mcmc_checkpoint(checkpoint_path, resume_iter)  # <-- pass checkpoint number

        if restored_sample is None:
            print("[ERROR] Could not restore from checkpoint. Exiting.")
            return None

        x_log = restored_sample
        burn_in = 0  # Skip burn-in when resuming

    else:
        # Use dynamic search instead of relying on initial_data
        map_filename = "optimization_result.json"
        found_path = find_file_anywhere(
            filename=map_filename,
            start_dir=settings.output_dir,
            max_depth_up=1,
            exclude_dirs=["venv", ".venv", "__pycache__"]
        )

        if not found_path:
            print(f"[ERROR] Could not locate '{map_filename}'. MCMC cannot proceed.")
            return None

        print(f"[INFO] Found MAP optimization result at: {found_path}")

        # Load the parameters
        try:
            with open(found_path, "r") as f:
                result_data = json.load(f)

            x_log = result_data.get("x_log")
            if not isinstance(x_log, list) or len(x_log) != 2:
                print(f"[ERROR] 'x_log' missing or invalid in {map_filename}")
                return None

            print(f"[DEBUG] Using extracted x_log for MCMC: c1_log = {x_log[0]}, alpha_log = {x_log[1]}")

        except Exception as e:
            print(f"[ERROR] Failed to read MAP file '{found_path}': {e}")
            return None

        burn_in = settings.mcmc.burn_in  # Apply burn-in when starting fresh


    # Run the MCMC
    return mcmc_inference(
        lambda c_log: log_posterior(np.array(c_log, dtype=np.float64), observed_data, settings),
        initial_sample=np.array(x_log, dtype=np.float64),
        initial_cov=np.array(settings.mcmc.initial_cov, dtype=np.float64),
        iterations=settings.mcmc.max_iter,
        save_interval=settings.mcmc.save_interval,
        checkpoint_interval=settings.mcmc.checkpoint_interval,
        settings=settings,
        burn_in=burn_in,
        observed_data=observed_data
    )
