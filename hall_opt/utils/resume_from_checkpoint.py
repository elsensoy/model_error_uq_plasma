import pickle
import os
import numpy as np
from MCMCIterators.samplers import DelayedRejectionAdaptiveMetropolis
from neldermead.mcmc_utils import save_checkpoint, load_checkpoint, save_metadata
from neldermead.map_nelder_mead import create_config  # Import  config function


def resume_from_checkpoint(
    logpdf,
    checkpoint_path,
    additional_iterations,
    save_interval=10,
    base_path="mcmc-results",
    metadata_filename="resumed_mcmc_metadata.json",
    config=None
):
    """
    Resume ( draft ) MCMC from a saved checkpoint and save results, metadata, and final sampling.

    Parameters:
        logpdf: Callable
            Log-posterior function.
        checkpoint_path: str
            Path to the saved checkpoint file.
        additional_iterations: int
            Number of additional iterations to run.
        save_interval: int
            Interval for saving checkpoints.
        base_path: str
            Directory for saving results.
        metadata_filename: str
            Name of the metadata file to save.
        config: dict
            Configuration dictionary for creating metadata.

    Returns:
        np.ndarray, float: The samples and final acceptance rate.
    """
    # Load the checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    start_iteration = checkpoint["iteration"]
    all_samples = checkpoint["all_samples"]
    acceptance_status = checkpoint["acceptance_status"]
    sampler_state = checkpoint["sampler_state"]

    # Restore the sampler state
    sampler = DelayedRejectionAdaptiveMetropolis(
        logpdf, sampler_state["current_sample"], sampler_state["covariance"],
        adapt_start=sampler_state["adapt_start"], eps=sampler_state["eps"],
        sd=sampler_state["sd"], interval=sampler_state["interval"],
        level_scale=sampler_state["level_scale"]
    )
    sampler.set_state(sampler_state)

    # Prepare for saving results
    results_dir = os.path.join(base_path)
    os.makedirs(results_dir, exist_ok=True)
    checkpoint_file = os.path.join(results_dir, "resume_checkpoint.csv")
    final_samples_file = os.path.join(results_dir, "final_samples.csv")
    final_status_file = os.path.join(results_dir, "final_status.txt")

    # Initialize the checkpoint file
    if start_iteration == 0:
        with open(checkpoint_file, "w") as f:
            f.write("iteration,lambda1,lambda2,accepted\n")  # Add header

    # Resume MCMC sampling
    acceptances = sum(acceptance_status)
    for i in range(start_iteration, start_iteration + additional_iterations):
        try:
            result = next(sampler)
            sample = result[0]
            accepted = result[1]
            all_samples.append(sample)
            acceptance_status.append(accepted)
            if accepted:
                acceptances += 1

            # Save checkpoints at intervals
            if (i + 1) % save_interval == 0:
                with open(checkpoint_file, "a") as f:
                    f.write(f"{i + 1},{sample[0]},{sample[1]},{'T' if accepted else 'F'}\n")
                save_checkpoint(
                    checkpoint_path, i + 1, all_samples, acceptance_status, sampler.get_state()
                )
                print(f"Checkpoint saved at iteration {i + 1}.")

        except Exception as e:
            print(f"Error at iteration {i + 1}: {e}")
            break

    # Final save
    acceptance_rate = acceptances / len(all_samples)
    np.savetxt(final_samples_file, np.array(all_samples), delimiter=",")
    with open(final_status_file, "w") as status_f:
        for idx, status in enumerate(acceptance_status, start=1):
            status_f.write(f"{idx}: {'T' if status else 'F'}\n")

    print(f"Final samples saved to: {final_samples_file}")
    print(f"Final acceptance status saved to: {final_status_file}")

    # Save metadata
    specific_config = create_specific_config(config) if config else {}
    metadata = {
        "start_iteration": start_iteration,
        "additional_iterations": additional_iterations,
        "total_iterations": start_iteration + additional_iterations,
        "final_acceptance_rate": acceptance_rate,
        "samples_file": final_samples_file,
        "status_file": final_status_file,
        "checkpoint_file": checkpoint_file,
        "config": specific_config
    }
    save_metadata(metadata, filename=os.path.join(results_dir, metadata_filename))

    print(f"Metadata saved to: {metadata_filename}")
    return np.array(all_samples), acceptance_rate
