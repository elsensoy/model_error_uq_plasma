import sys
import argparse
import json
import subprocess
from pathlib import Path
import os
import yaml

# print("HallThruster imported successfully!")
HALL_OPT_DIR = os.path.dirname(os.path.abspath(__file__))  
print(f"[DEBUG] hall_opt directory: {HALL_OPT_DIR}")

#TODO: Replace python path. This should resolve path finding issue if run.py fails finding het.
#( find yours by running `
#``julia
# using HallThruster; 
# HallThruster.PYTHON_PATH
# ```)
hallthruster_path = "/home/elida/.julia/packages/HallThruster/cq07j/python"
if hallthruster_path not in sys.path:
    sys.path.append(hallthruster_path)

import hallthruster as het
print("HallThruster imported successfully from main!")


from hall_opt.config.verifier import verify_all_yaml, load_yaml  
from hall_opt.scripts.map import run_map_workflow
from hall_opt.scripts.mcmc import run_mcmc_with_final_map_params
from hall_opt.scripts.gen_data import get_ground_truth_data
from hall_opt.plotting.posterior_plots import generate_plots
from hall_opt.plotting.simplex_plot import visualize_final_simplex
from hall_opt.utils.parse import get_yaml_path, parse_arguments
from hall_opt.utils.save_data import create_used_directories, save_results_to_json
from hall_opt.utils.data_loader import find_latest_results_dir

def main():
    try:
             
        args = parse_arguments()
        yaml_path = get_yaml_path(args.method_yaml)  # resolves path inside config/ if needed

        print(f"[DEBUG] Using YAML file: {yaml_path}")

        # -----------------------------
        # Step 2: Load and Validate YAML
        # -----------------------------
        yaml_data = load_yaml(yaml_path)  # Load YAML as dict
        print(f"[DEBUG] Loaded YAML data from: {yaml_path}")

        settings = verify_all_yaml(yaml_data, source_path=yaml_path)  # validate and resolve

        if settings is None:
            print("[ERROR] Configuration verification failed. Exiting...")
            sys.exit(1)

        print(f"[INFO] YAML file registered as: {settings.general.config_file}")
        print("[INFO] Configuration successfully loaded and verified!")

    # -----------------------------
    #  Step 3:Override YAML Settings with Loaded Config
    # -----------------------------
        valid_flags = ["run_map", "run_mcmc", "gen_data", "plotting"]

        print("\n[DEBUG] Checking overrides from YAML file...")
        for flag in valid_flags:
            if flag in yaml_data:  # If the flag exists in the YAML file
                setattr(settings.general, flag, bool(yaml_data[flag]))
                print(f"Overriding: settings.{flag} = {getattr(settings.general, flag)}")

        # -----------------------------
        #  Debugging: Print Final Execution Flags
        # -----------------------------
        print("DEBUG: Final execution flags:")
        for flag in valid_flags:
            print(f"  {flag}: {getattr(settings.general, flag)}")

        # -----------------------------

        #  Step 4: Create Results Directory
        # -----------------------------

        create_used_directories(settings)

        # -----------------------------
        #  Step 5: Generate or Load Ground Truth Data
        # -----------------------------
        observed_data = None
        observed_data, metrics = get_ground_truth_data(settings)

        if observed_data is None:
            print("[FATAL] Cannot proceed without ground truth.")
            sys.exit(1)

        #  metrics only if they were generated (not when loading from CSV)
        if metrics:
            save_results_to_json(
                settings=settings,
                result_dict=metrics,
                filename="ground_truth_metrics.json",
                results_dir=str(Path(settings.output_dir) / "ground_truth"),
                save_every_n_grid_points=5,
                subsample_for_saving=True,
            )
        # -----------------------------
        #  Step 6: Run MAP Estimation (If Enabled)
        # -----------------------------
        if settings.run_map:
            print(f"DEBUG: observed_data: {type(observed_data)}, {observed_data is None}")
            if observed_data is None:
                print("ERROR: observed_data is missing, cannot run MAP estimation!")
            else:
                try:
                    print("DEBUG: Calling run_map_workflow()...")
                    optimized_params = run_map_workflow(observed_data, settings)
                    if optimized_params:
                        print(f"Final MAP parameters saved")
                    else:
                        print("ERROR: MAP optimization failed.")

                except Exception as e:
                    print(f"ERROR during MAP estimation: {e}")
                    sys.exit(1)
        else:
            print("DEBUG: MAP Estimation is disabled.")

        # -----------------------------
        #  Step 7: Run MCMC Sampling (If Enabled)
        # -----------------------------
        if settings.run_mcmc:
            print("Running MCMC sampling...")
            print(f"Using base directory for this MCMC run: {settings.mcmc.base_dir}")

            # Run MCMC Sampling
            mcmc_params = run_mcmc_with_final_map_params(observed_data, settings)
            if mcmc_params is None:
                print("ERROR: MCMC sampling failed. Exiting.")
                sys.exit(1)

            print("MCMC sampling completed!")

        # -----------------------------
        #  Step 8: Generate Plots (If Enabled)
        # -----------------------------
        if settings.general.plotting:
            print("Generating plots...")
            generate_plots(settings)

            # --- Call the final simplex visualization (Finding Latest Dir) ---
            print("Attempting to generate final simplex plot by finding latest MAP results...")

            map_parent_dir = Path(settings.output_dir) / "map"
            map_base_name = "map-results" # The base name used by get_next_results_dir

            # Find the latest MAP results directory
            latest_map_dir = find_latest_results_dir(str(map_parent_dir), map_base_name)

            if latest_map_dir:
                # Construct the path to the specific file within the latest directory
                optimization_result_file = latest_map_dir / "optimization_result.json"

                # Check if the optimization result file exists in that latest directory
                if optimization_result_file.is_file():
                    try:
                        print(f"Found results file in latest MAP directory: {optimization_result_file}")
                        # Call the simplex visualization function
                        visualize_final_simplex(str(optimization_result_file))
                        print("Final simplex plot displayed/saved.")
                    except FileNotFoundError:
                        print(f"[WARNING] File {optimization_result_file} not found despite initial check. Skipping.")
                    except Exception as e:
                        print(f"[WARNING] An error occurred during final simplex plot generation: {e}")

            else:
                print("[INFO] Plotting is disabled in settings (settings.general.plotting=False).")


    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt detected in main.py. Exiting.")
        sys.exit(0)  # Exit on Ctrl+C

if __name__ == "__main__":
    main()