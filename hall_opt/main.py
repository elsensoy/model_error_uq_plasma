# Import necessary modules
import sys
import os
import json
import yaml
import numpy as np
import argparse
import logging
from pathlib import Path
from hall_opt.config.load_settings import extract_anom_model
from hall_opt.config.run_model import run_model
from hall_opt.map import run_map_workflow
from hall_opt.mcmc import run_mcmc_with_final_map_paramss

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run MAP and MCMC estimation workflow.")
    parser.add_argument("--settings", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()


    # Resolve base directory paths
    base_results_dir = Path(settings["general_settings"]["results_dir"])

    # Create directories if they don't exist
    base_results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results directory set to: {base_results_dir}")

    observed_data = None

    # Step 1: Generate ground truth data
    if settings["general_settings"]["gen_data"]:
        print("Generating ground truth data using MultiLogBohm...")
        try:
            multilogbohm_config = extract_anom_model(settings, model_type="MultiLogBohm")
            ground_truth_solution = run_model(
                config=multilogbohm_config,
                settings=settings,
                simulation=settings["simulation"],      # Pass general simulation settings
                postprocess=settings["postprocess"],    # Pass postprocessing settings
                model_type="MultiLogBohm",
            )

            if ground_truth_solution:
                averaged_metrics = ground_truth_solution["output"]["average"]
                observed_data = {
                    "thrust": averaged_metrics["thrust"],
                    "discharge_current": averaged_metrics["discharge_current"],
                    "ion_velocity": averaged_metrics["ui"][0],
                    "z_normalized": averaged_metrics["z"],
                }
                print("Ground truth data generated and extracted.")
            else:
                print("Error: Ground truth simulation failed.")
                sys.exit(1)
        except Exception as e:
            print(f"Error during ground truth generation: {e}")
            sys.exit(1)

    # Step 2: Run MAP estimation
    if settings["general_settings"]["run_map"]:
        print("Running MAP estimation using TwoZoneBohm...")
        map_results_dir = base_results_dir / "map_results"
        map_results_dir.mkdir(parents=True, exist_ok=True)

        try:
            final_map_params_path = map_results_dir / settings["optimization_params"]["map_params"]["final_map_params_file"]

            c1_opt, alpha_opt = run_map_workflow(
                observed_data=observed_data,
                settings=settings,
                simulation=settings["simulation"],
                results_dir=str(map_results_dir),
            )

            if c1_opt is not None and alpha_opt is not None:
                print(f"MAP optimization completed: c1={c1_opt}, alpha={alpha_opt}")
            else:
                print("Error: MAP optimization failed.")
        except Exception as e:
            print(f"Error during MAP estimation: {e}")
            sys.exit(1)

    # Step 3: Run MCMC sampling
    if settings["general_settings"]["run_mcmc"]:
        print("Running MCMC sampling using TwoZoneBohm...")
        mcmc_results_dir = base_results_dir / "mcmc_results"
        mcmc_results_dir.mkdir(parents=True, exist_ok=True)

        observed_data["ion_velocity"] = np.array(observed_data["ion_velocity"], dtype=np.float64)

        try:
            run_mcmc_with_final_map_params(
                final_map_params= final_map_params_path,
                observed_data=observed_data,
                config=extract_anom_model(settings, model_type="TwoZoneBohm"),
                simulation=settings["simulation"],
                settings=settings,
                ion_velocity_weight=settings["general_settings"]["ion_velocity_weight"],
                iterations=settings["general_settings"]["iterations"],
                initial_cov=settings["optimization_params"]["mcmc_params"]["initial_cov"],
            )
            print("MCMC sampling completed successfully.")
        except Exception as e:
            print(f"Error during MCMC sampling: {e}")
            sys.exit(1)

    # Step 4: Generate plots (if enabled)
    if settings["general_settings"]["plotting"]:
        print("Generating plots...")
        plots_dir = base_results_dir / settings["plotting"]["plots_subdir"]
        plots_dir.mkdir(parents=True, exist_ok=True)
        print(f"Plots will be saved in: {plots_dir}")

if __name__ == "__main__":
    main()
