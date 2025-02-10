import os
import json
import numpy as np
import pathlib
from scipy.stats import norm
from hall_opt.utils.iter_methods import get_next_filename
from hall_opt.config.dict import Settings
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
    settings,
    extracted_metrics: dict,
    output_dir: str = None,
    base_name: str = "metrics",
    use_json_dump: bool = False,
    save_every_n_grid_points: int = 10,  # Subsampling frequency
    subsample_for_saving: bool = True
):
    """
    Saves extracted simulation metrics correctly inside:
      - `map-results-N/iter_metrics/metrics_X.json`
      - `mcmc-results-N/iter_metrics/metrics_X.json`
    
    Uses `save_results_to_json()` for JSON formatting and optional subsampling.

    Args:
        settings (Settings): Parsed YAML settings.
        extracted_metrics (dict): Simulation results (metrics).
        output_dir (str, optional): Override directory.
        base_name (str, optional): Filename prefix.
        use_json_dump (bool, optional): Save ground truth metrics separately.
        save_every_n_grid_points (int, optional): Grid subsampling rate.
        subsample_for_saving (bool, optional): Enable subsampling before saving.
    """

    print(" DEBUG: Entering `save_metrics()`...")

    # Prevent Saving Empty Metrics
    if not extracted_metrics:
        print("ERROR: No valid metrics to save. Skipping save operation.")
        return

    #  Determine correct save directory
    if use_json_dump:
        # Ground truth results go to predefined postprocess output
        output_file = settings.postprocess.output_file["MultiLogBohm"]
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as json_file:
            json.dump(extracted_metrics, json_file, indent=4)
        print(f"Ground truth metrics saved to {output_file}")
        return

    # if settings.ground_truth.gen_data:
    #     output_file = settings.ground_truth.output_file["output_ground_truth"]
    #  # iterative results directory (MAP/MCMC)
    if settings.general.run_map:
        base_dir = settings.map.base_dir  # Uses `map-results-N/`
    elif settings.general.run_mcmc:
        base_dir = settings.mcmc.base_dir  # Uses `mcmc-results-N/`
    else:
        raise ValueError("ERROR: Neither MAP nor MCMC is enabled. Cannot save metrics.")

    #  `iter_metrics/` directory inside `map-results-N/` or `mcmc-results-N/`
    metrics_dir = os.path.join(base_dir, "iter_metrics")
    os.makedirs(metrics_dir, exist_ok=True)  # Ensure it exists

    print(f"DEBUG: Metrics will be saved in: {metrics_dir}")

    #  Generate unique filename (e.g., `metrics_1.json`, `metrics_2.json`)
    metrics_file_path = get_next_filename(base_name, metrics_dir)

    print(f" DEBUG: Saving metrics to {metrics_file_path}")

    #  Save using `save_results_to_json()`
    try:
        save_results_to_json(
            settings=settings,
            result_dict=extracted_metrics,
            filename=metrics_file_path,
            results_dir=metrics_dir,
            save_every_n_grid_points=save_every_n_grid_points,
            subsample_for_saving=subsample_for_saving
        )
        print(f"SUCCESS: Metrics saved to {metrics_file_path}")
    except Exception as e:
        print(f"ERROR: Failed to save metrics to {metrics_file_path}: {e}")