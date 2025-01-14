import sys
import os
import argparse
import logging
from pathlib import Path
from hall_opt.config.loader import Settings, load_yml_settings,extract_anom_model
from hall_opt.config.run_model import run_simulation_with_config
from map import run_map_workflow
from mcmc import run_mcmc_with_final_map_params


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run MAP and MCMC estimation workflow.")
    parser.add_argument("--settings", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    # Load settings
    settings_path = Path(args.settings)
    if not settings_path.exists():
        print(f"Error: Settings file not found at {settings_path}")
        return

    print("Loading settings...")
    yml_dict = load_yml_settings(settings_path)
    settings = Settings(**yml_dict)

    # Ensure results directory exists
    results_dir = Path(settings.general_settings["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results directory set to: {results_dir}")

    observed_data = None

    # Step 1: Generate ground truth data
    if settings.general_settings["gen_data"]:
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
    if settings.general_settings["run_map"]:
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
    if settings.general_settings["run_mcmc"]:
        print("Running MCMC sampling using TwoZoneBohm...")
        mcmc_results_dir = results_dir / "mcmc_results"
        mcmc_results_dir.mkdir(parents=True, exist_ok=True)

        try:
            run_mcmc_with_optimized_params(
                map_initial_guess_path=settings.optimization_params["map_params"]["final_map_params"],
                observed_data=observed_data,
                config=extract_anom_model(settings, model_type="TwoZoneBohm"),
                simulation=settings.simulation,
                settings=settings,
                ion_velocity_weight=settings.general_settings["ion_velocity_weight"],
                iterations=settings.general_settings["iterations"],
                initial_cov=settings.optimization_params["mcmc_params"]["initial_cov"],
                results_dir=str(mcmc_results_dir),
            )
            print("MCMC sampling completed successfully.")
        except Exception as e:
            print(f"Error during MCMC sampling: {e}")

    # Step 4: Generate plots (if enabled)
    if settings.general_settings["plotting"]:
        print("Generating plots...")
        plots_dir = results_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        # Assuming generate_all_plots is implemented
        # generate_all_plots(input_dir=str(results_dir), output_dir=str(plots_dir))
        print(f"All plots saved to: {plots_dir}")



    # Step 4: Generate plots 
    # if settings.plotting:
    #     print("Generating plots...")
    #     plots_dir = results_dir / "plots"
    #     generate_all_plots(input_dir=str(results_dir), output_dir=str(plots_dir))
    #     print(f"All plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()