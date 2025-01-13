import sys
import os
import argparse
from pathlib import Path
from hall_opt.config.loader import Settings, load_yml_settings,extract_anom_model
from hall_opt.config.run_model import run_simulation_with_config
from map import run_map_workflow
from mcmc import run_mcmc_with_optimized_params

# Add HallThruster to the Python path
hallthruster_path = "/home/elidasensoy/.julia/packages/HallThruster/tHQQa/python"
if hallthruster_path not in sys.path:
    sys.path.append(hallthruster_path)

import hallthruster as het
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run MAP and MCMC estimation workflow.")
    parser.add_argument("--settings", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    # Load settings
    print("Loading settings...")
    settings_path = Path(args.settings)
    yml_dict = load_yml_settings(settings_path)
    settings = Settings(**yml_dict)

    # Ensure results directory exists
    results_dir = Path(settings.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results directory set to: {results_dir}")

    observed_data = None
    twozonebohm_config = extract_anom_model(settings, model_type="TwoZoneBohm")

    # Step 1: Generate ground truth data
    if settings.gen_data:
        print("Generating ground truth data using MultiLogBohm...")
        multilogbohm_config = extract_anom_model(settings, model_type="MultiLogBohm")
        ground_truth_solution = run_simulation_with_config(
            config=multilogbohm_config,
            simulation=settings.simulation,
            postprocess=settings.postprocess,
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
            return

    # Step 2: Run MAP estimation
    if settings.run_map:
        print("Running MAP estimation using TwoZoneBohm...")
        map_results_dir = results_dir / "map_results"
        map_results_dir.mkdir(parents=True, exist_ok=True)

        c1_opt, alpha_opt = run_map_workflow(
            observed_data=observed_data,
            settings=settings,
            simulation=settings.simulation,
            results_dir=str(map_results_dir),
        )

        if c1_opt is not None and alpha_opt is not None:
            print(f"MAP optimization completed: c1={c1_opt}, alpha={alpha_opt}")
        else:
            print("Error: MAP optimization failed.")

    # Step 3: Run MCMC sampling
    if settings.run_mcmc:
        print("Running MCMC sampling using TwoZoneBohm...")
        mcmc_results_dir = results_dir / "mcmc_results"
        mcmc_results_dir.mkdir(parents=True, exist_ok=True)

        try:
            run_mcmc_with_optimized_params(
                map_initial_guess_path=settings.optimized_param,  # Path to optimized params
                observed_data=observed_data,  # Observed data from ground truth
                config=twozonebohm_config,  # Configuration for TwoZoneBohm
                simulation=settings.simulation,  # Simulation parameters
                settings=settings,
                ion_velocity_weight=settings.ion_velocity_weight,  # Ion velocity weight
                iterations=settings.iterations,  # Number of MCMC iterations
                initial_cov=settings.mcmc_initial_cov,  # Initial covariance matrix
                results_dir=str(mcmc_results_dir),  # Directory for MCMC results
            )
            print("MCMC sampling completed successfully.")
        except Exception as e:
            print(f"Error during MCMC sampling: {e}")

    # Step 4: Generate plots 
    # if settings.plotting:
    #     print("Generating plots...")
    #     plots_dir = results_dir / "plots"
    #     generate_all_plots(input_dir=str(results_dir), output_dir=str(plots_dir))
    #     print(f"All plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
