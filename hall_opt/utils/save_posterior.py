import os
import json
import numpy as np
import pathlib
from scipy.stats import norm
from hall_opt.config.run_model import run_model
from hall_opt.utils.iter_methods import get_next_filename,get_next_results_dir
from hall_opt.config.verifier import Settings
from hall_opt.utils.save_data import save_results_to_json


# -----------------------------
# 4. Save Posterior Using `save_results.json
# -----------------------------
def save_posterior(
    settings: Settings,
    c1_log: float,
    alpha_log: float,
    log_posterior_value: float,
):
    """Appends log sampling  `map_sampling.json` or `mcmc_sampling.json`
       inside the  `map-{N}/iter_metrics/` directory.
    """

    base_dir = settings.map.base_dir if settings.general.run_map else settings.mcmc.base_dir

    #   `iter_metrics/` directory exists inside `map-{N}/`
    iter_metrics_dir = os.path.join(base_dir, "iter_metrics")
    os.makedirs(iter_metrics_dir, exist_ok=True)

    #  fixed filename inside `iter_metrics/`
    filename = os.path.join(iter_metrics_dir, "map_sampling.json" if settings.general.run_map else "mcmc_sampling.json")

    if os.path.exists(filename):
        with open(filename, "r") as json_file:
            try:
                posterior_list = json.load(json_file)
                if not isinstance(posterior_list, list):  # If it's not a list, reset
                    posterior_list = []
            except json.JSONDecodeError:
                posterior_list = []
    else:
        posterior_list = []

    #  Append new posterior data
    posterior_list.append({
        "c1_log": c1_log,
        "alpha_log": alpha_log,
        "log_posterior": log_posterior_value
    })

    #  Save back to file
    with open(filename, "w") as json_file:
        json.dump(posterior_list, json_file, indent=4)

    print(f" Posterior value appended to {filename}")

# -----------------------------
# 5. Save Extracted Metrics Using `save_results_to_json`
# -----------------------------

def save_metrics(settings, extracted_metrics, output_dir=None, base_name="metrics", use_json_dump=False):
    """Saves extracted simulation metrics to file."""

    if use_json_dump:
        # Use ground truth output file
        output_file = settings.postprocess.output_file["MultiLogBohm"]
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, "w") as json_file:
            json.dump(extracted_metrics, json_file, indent=4)
        print(f"Ground truth metrics saved to {output_file}")

    else:
        # Define directory based on the method being used (MAP or MCMC)
        if settings.general.run_map:
            base_dir = settings.map.results_dir
        elif settings.general.run_mcmc:
            base_dir = settings.mcmc.results_dir
        else:
            raise ValueError("ERROR: Neither MAP nor MCMC is enabled. Cannot save metrics.")

        metrics_dir = os.path.join(base_dir, "iter_metrics")
        os.makedirs(metrics_dir, exist_ok=True)

        # Ensure filename is always set before using it
        filename = f"{base_name}.json"  # Assign default filename

        # Save results
        save_results_to_json(
            result_dict=extracted_metrics,  # Fix: Ensure this argument is passed
            filename=os.path.basename(filename),  # Fix: Ensure filename is always defined
            results_dir=metrics_dir,
        )

        print(f"Metrics saved to {metrics_dir}/{filename}")