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

def save_metrics(
    settings: Settings,
    extracted_metrics: dict,
    output_dir: None, 
    base_name: str = "metrics",
    use_json_dump: bool = False,
    save_every_n_grid_points: int = 10 
):
    #Saves extracted simulation metrics inside `map-results-N/iter_metrics/` or `mcmc-results-N/iter_metrics/`.

    if use_json_dump:
        # save to ground_truth output file
        output_file = settings.postprocess.output_file["MultiLogBohm"]
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as json_file:
            json.dump(extracted_metrics, json_file, indent=4)
        print(f" Ground truth metrics saved to {output_file}")
    
    else:
        #  Use the `map-results-N/` or `mcmc-results-N/` directory created before iterations start
        if settings.general.run_map:
            base_dir = settings.map.base_dir  # Uses pre-defined map-results-N/
        elif settings.general.run_mcmc:
            base_dir = settings.mcmc.base_dir  # Uses pre-defined mcmc-results-N/
        else:
            raise ValueError("ERROR: Neither MAP nor MCMC is enabled. Cannot save metrics.")

        #  Define `iter_metrics/` inside `map-results-N/` or `mcmc-results-N/`
        metrics_dir = os.path.join(base_dir, "iter_metrics")

        #   `map-results-N/iter_metrics/` or `mcmc-results-N/iter_metrics/` exists
        os.makedirs(metrics_dir, exist_ok=True)

        #  Generate next available `metrics_X.json` filename inside `iter_metrics/`
        filename = get_next_filename(base_name, metrics_dir, extension=".json")

    save_results_to_json(
        save_results_to_json(
            result_dict=extracted_metrics,
            filename=os.path.basename(filename),
            results_dir=metrics_dir,
            save_every_n_grid_points=save_every_n_grid_points,  # Apply subsampling
            subsample_for_saving=True  # Enable subsampling
        )
        )

    print(f"Metrics saved to {os.path.join(metrics_dir, filename)}")