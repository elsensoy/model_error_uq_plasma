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
from hall_opt.plotting.common_setup import get_common_paths
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
                save_every_n_grid_points=10,
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
                        final_map_params_path = Path(settings.map.output_dir) / "final_map_params.json"
                        with open(final_map_params_path, "w") as f:
                            json.dump(optimized_params, f, indent=4)
                        print(f"Final MAP parameters saved to {final_map_params_path}")
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


        # --- Generate General Plots (if enabled) ---
        # This function should internally handle prompting the user (MAP vs MCMC)
        # and use get_common_paths to find the data for the selected type.
        if settings.general.plotting:

            # --- Generate Final Simplex Plot (Specific to MAP) ---
            print("\n[INFO] Attempting to generate final simplex plot for MAP results...")

            from hall_opt.utils.parse import find_file_anywhere  # adjust path as needed

            # Use a flexible file search starting from output_dir
            map_file = find_file_anywhere(
                filename="optimization_result.json",
                start_dir=settings.output_dir,  # Or '.' or 'map_results' depending on your folder structure
                max_depth_up=1,
                exclude_dirs=["venv", ".venv", "__pycache__"]
            )

            if map_file and map_file.is_file():
                try:
                    print(f"[INFO] Found optimization result file: {map_file}")
                    visualize_final_simplex(str(map_file))
                    print("[INFO] Final simplex plot generation completed.")
                except Exception as e:
                    print(f"[ERROR] An error occurred during final simplex plot generation: {e}")
            else:
                print("[INFO] Could not find 'optimization_result.json' for MAP plotting. Skipping simplex plot.")

            print("\n[INFO] General plotting enabled. Starting generate_plots...")
            generate_plots(settings) # Assumes this function does its own path finding internally
            print("[INFO] generate_plots finished.")

            # --- Generate Final Simplex Plot (Specific to MAP) ---
            # Since this plot is specific to MAP results, we explicitly get the MAP paths here.
            # This avoids modifying generate_plots if it handles MCMC too,
            # and ensures we target the correct file even if generate_plots ran for MCMC.
            print("\n[INFO] Attempting to generate final simplex plot for MAP results...")

            # Get paths specifically for MAP analysis using the centralized function
            # Pass "map" directly as analysis_type
            map_paths = get_common_paths(settings, "map")

            # Retrieve the latest MAP directory path from the dictionary
            latest_map_dir = map_paths.get("latest_results_dir") # Use .get() for safety

            if latest_map_dir:
                # Construct the path to the specific file within the found directory
                optimization_result_file = latest_map_dir / "optimization_result.json"

                print(f"[DEBUG] Checking for MAP optimization file: {optimization_result_file}")

                # Check if the specific optimization result file exists
                if optimization_result_file.is_file():
                    try:
                        print(f"[INFO] Found optimization results file: {optimization_result_file}")
                        # Call the simplex visualization function with the correct path
                        visualize_final_simplex(str(optimization_result_file)) # Pass path as string if needed
                        print("[INFO] Final simplex plot generation attempted.")
                    except Exception as e:
                        print(f"[ERROR] An error occurred during final simplex plot generation: {e}")
                else:
                    print(f"[INFO] Optimization result file not found in the latest MAP directory ({latest_map_dir}). Skipping simplex plot.")
            else:
                # This message comes from get_common_paths/find_latest_results_dir if no dir was found
                print("[INFO] Could not find the latest MAP results directory.")

        else:
            print("\n[INFO] Plotting is disabled in settings (settings.general.plotting=False).")

        print("\n[INFO] Main script finished.")

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt detected in main.py. Exiting gracefully.")
        sys.exit(0)  # Exit on Ctrl+C
    except AttributeError as e:
         print(f"\n[ERROR] Missing setting attribute in main execution: {e}")
         sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred in main: {e}")
        sys.exit(1)

