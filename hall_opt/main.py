# Import necessary modules
import sys
import os
import yaml
import numpy as np
import logging
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
# Import verifier and run functions
from config.verifier import verify_all_yaml, extract_anom_model, GeneralSettings
from config.run_model import run_model
from map import run_map_workflow
from mcmc import run_mcmc_with_final_map_params

def main():


    # Validate all YAML files before execution
    settings = verify_all_yaml()
    if settings is None:
        print("ERROR: One or more YAML configuration files are invalid. Exiting...")
        sys.exit(1)

    # Resolve base directory paths
    base_results_dir = Path(settings.results_dir)
    base_results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results directory set to: {base_results_dir}")

    observed_data = None

    # Step 1: Generate ground truth data
    if settings.gen_data:
        print("Generating ground truth data using MultiLogBohm...")
        try:
            multilogbohm_config = extract_anom_model(settings, model_type="MultiLogBohm")
            ground_truth_solution = run_model(
                config=multilogbohm_config,
                settings=settings,
                simulation=settings.simulation,      # Pass general simulation settings
                postprocess=settings.postprocess,    # Pass postprocessing settings
                model_type="MultiLogBohm",
            )

            if ground_truth_solution:
                averaged_metrics = ground_truth_solution["output"]["average"]
                observed_data = {
                    "thrust": averaged_metrics.get("thrust", 0),
                    "discharge_current": averaged_metrics.get("discharge_current", 0),
                    "ion_velocity": averaged_metrics.get("ui", [0])[0],
                    "z_normalized": averaged_metrics.get("z", 0),
                }
                print("Ground truth data generated and extracted.")
            else:
                print("ERROR: Ground truth simulation failed.")
                sys.exit(1)
        except Exception as e:
            print(f"ERROR during ground truth generation: {e}")
            sys.exit(1)

    # Step 2: Run MAP estimation
    if settings.run_map:
        print("Running MAP estimation using TwoZoneBohm...")
        map_results_dir = base_results_dir / "map_results"
        map_results_dir.mkdir(parents=True, exist_ok=True)

        try:
            final_map_params_path = map_results_dir / settings.map_params.final_map_params_file

            c1_opt, alpha_opt = run_map_workflow(
                observed_data=observed_data,
                settings=settings,
                simulation=settings.simulation,
                results_dir=str(map_results_dir),
            )

            if c1_opt is not None and alpha_opt is not None:
                print(f"MAP optimization completed: c1={c1_opt}, alpha={alpha_opt}")
            else:
                print("ERROR: MAP optimization failed.")
        except Exception as e:
            print(f"ERROR during MAP estimation: {e}")
            sys.exit(1)

    # Step 3: Run MCMC sampling
    if settings.run_mcmc:
        print("Running MCMC sampling using TwoZoneBohm...")
        mcmc_results_dir = base_results_dir / "mcmc_results"
        mcmc_results_dir.mkdir(parents=True, exist_ok=True)

        observed_data["ion_velocity"] = np.array(observed_data["ion_velocity"], dtype=np.float64)

        try:
            run_mcmc_with_final_map_params(
                final_map_params=final_map_params_path,
                observed_data=observed_data,
                config=extract_anom_model(settings, model_type="TwoZoneBohm"),
                simulation=settings.simulation,
                settings=settings,
                ion_velocity_weight=settings.ion_velocity_weight,
                iterations=settings.iterations,
                initial_cov=settings.mcmc_params.initial_cov,
            )
            print("MCMC sampling completed successfully.")
        except Exception as e:
            print(f"ERROR during MCMC sampling: {e}")
            sys.exit(1)

    # Step 4: Generate plots (if enabled)
    if settings.plotting:
        print("Generating plots...")
        plots_dir = base_results_dir / settings.plotting.plots_subdir
        plots_dir.mkdir(parents=True, exist_ok=True)
        print(f"Plots will be saved in: {plots_dir}")

if __name__ == "__main__":
    main()
