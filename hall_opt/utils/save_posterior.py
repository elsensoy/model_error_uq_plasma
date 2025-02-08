import os
import json
import numpy as np
import pathlib
from scipy.stats import norm
from .iter_methods import get_next_filename
from ..config.dict import Settings
from .save_data import save_results_to_json


# -----------------------------
# 4. Save metrics Using `save_results.json
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

    print(" DEBUG: Entering `save_metrics()`...")

    # Prevent Saving Empty Metrics
    if not extracted_metrics:
        print("ERROR: No valid metrics to save. Skipping save operation.")
        return
    if settings.general.run_map:
        base_dir = settings.map.base_dir  # Uses `map-results-N/`
    elif settings.general.run_mcmc:
        base_dir = settings.mcmc.base_dir  # Uses `mcmc-results-N/`
    else:
        raise ValueError("ERROR: Neither MAP nor MCMC is enabled. Cannot save metrics.")

    #  `iter_metrics/` directory inside `map-results-N/` or `mcmc-results-N/`
    metrics_dir = os.path.join(base_dir, "iter_metrics")
    os.makedirs(metrics_dir, exist_ok=True)  # make sure it exists

    #   unique filename (e.g., `metrics_1.json`, `metrics_2.json`)
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
