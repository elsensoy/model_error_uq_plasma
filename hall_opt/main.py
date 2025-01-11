import sys
import os
import argparse
from pathlib import Path
from hall_opt.config.settings_loader import Settings, load_yml_settings
from hall_opt.config.simulation import run_simulation_with_config
from map import run_map_workflow
from mcmc import run_mcmc_with_optimized_params
# from hall_opt.plotting import generate_all_plots

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
hallthruster_path = "/home/elidasensoy/.julia/packages/HallThruster/tHQQa/python"
if hallthruster_path not in sys.path:
    sys.path.append(hallthruster_path)

import hallthruster as het

# Main workflow
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run MCMC and MAP estimation workflow.")
    parser.add_argument("--settings", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    # Load YAML configuration
    print("Loading settings...")
    settings_path = Path(args.settings)
    yml_dict = load_yml_settings(settings_path)
    settings = Settings(**yml_dict)

    # Extract configurations from settings
    config_multilogbohm = settings.config_multilogbohm
    config_spt_100 = settings.config_spt_100
    simulation = settings.simulation
    postprocess = settings.postprocess

    # Ensure data directory exists
    results_dir = Path(settings.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Data directory set to: {results_dir}")

    observed_data = None

    # Step 1: Generate ground truth data
    if settings.gen_data:
        print("Generating ground truth data using MultiLogBohm...")
        ground_truth_postprocess = postprocess.copy()
        ground_truth_postprocess["output_file"] = str(results_dir / "ground_truth.json")

        ground_truth_solution = run_simulation_with_config(
            config_multilogbohm, simulation, ground_truth_postprocess, config_type="MultiLogBohm"
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

        v1_opt, alpha_opt = run_map_workflow(
            observed_data=observed_data,
            settings=settings,
            config_spt_100=config_spt_100,
            simulation=simulation,
            postprocess=postprocess,
            results_dir=str(map_results_dir),
            final_params_file="final_parameters.json"
        )

        if v1_opt is not None and alpha_opt is not None:
            print(f"MAP optimization completed: v1={v1_opt}, alpha={alpha_opt}")
        else:
            print("Error: MAP optimization failed.")

    # Step 3: Run MCMC sampling
    if settings.run_mcmc:
        print("Running MCMC sampling using TwoZoneBohm...")
        mcmc_results_dir = settings.results_dir / "mcmc_results"
        mcmc_results_dir.mkdir(parents=True, exist_ok=True)

        try:
            run_mcmc_with_optimized_params(
                json_path=settings.optimized_param,
                observed_data=observed_data,
                config=config_spt_100,
                ion_velocity_weight=settings.ion_velocity_weight,
                iterations=settings.iterations,
                initial_cov=settings.initial_cov,
                results_dir=str(mcmc_results_dir),
            )
            print("MCMC sampling completed successfully.")
        except Exception as e:
            print(f"Error during MCMC sampling: {e}")

    # Step 4: Generate plots
    # Uncomment if generate_all_plots function is ready to use
    # if settings.plotting:
    #     print("Generating plots...")
    #     plots_dir = results_dir / "plots"
    #     generate_all_plots(input_dir=str(results_dir), output_dir=str(plots_dir))
    #     print(f"All plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
