import sys
import numpy as np
from pathlib import Path
from hall_opt.config.verifier import verify_all_yaml
from hall_opt.config.run_model import run_model
# from hall_opt.utils.save_data import save_results_to_json  
#TODO: postprocess validation needed
#TODO: add mkdir logic. ensure the results_dir is fully integrated.
#TODO: Config debug needed. see err_msg.txt. could be a name conflict with het
def main():
    # Validate YAML files before execution
    settings = verify_all_yaml()
    if settings is None:
        print("ERROR: One or more YAML configuration files are invalid. Exiting...")
        sys.exit(1)

    # Extract validated sections
    general_settings = settings["general"]
    config_settings = settings["config_settings"]
    ground_truth = settings["ground_truth"]

    # Resolve base directory paths
    base_results_dir = Path(general_settings.results_dir).resolve()
    base_results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results directory set to: {base_results_dir}")

    observed_data = None

    # Step 1: Generate ground truth data
    if ground_truth.gen_data:
        print("Generating ground truth data using MultiLogBohm...")

        try:
            # Run simulation
            ground_truth_solution = run_model(
                config_settings=config_settings,  
                simulation=config_settings,  
                postprocess=config_settings.postprocess,  
                model_type="MultiLogBohm",
            )

            if ground_truth_solution:
                # Extract results
                averaged_metrics = ground_truth_solution["output"]["average"]
                observed_data = {
                    "thrust": averaged_metrics.get("thrust", 0),
                    "discharge_current": averaged_metrics.get("discharge_current", 0),
                    "ion_velocity": averaged_metrics.get("ui", [0])[0],
                    "z_normalized": averaged_metrics.get("z", 0),
                }

                print("\n Ground truth data successfully generated!\n")
                print(f"Thrust: {observed_data['thrust']} N")
                print(f"Discharge Current: {observed_data['discharge_current']} A")
                print(f"Ion Velocity: {observed_data['ion_velocity']} m/s")
                print(f"Z-Normalized: {observed_data['z_normalized']}\n")

                # #  Save results
                # save_results_to_json(
                #     result_dict=observed_data,
                #     filename="ground_truth_results.json",
                #     results_dir=str(base_results_dir),
                #     save_every_n_grid_points=10,
                #     subsample_for_saving=True
                # )

            else:
                print(" ERROR: Ground truth simulation failed.")
                sys.exit(1)

        except Exception as e:
            print(f" ERROR during ground truth generation: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
