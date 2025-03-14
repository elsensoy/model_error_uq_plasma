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

#TODO: Replace python path.
#( find yours by running `
#``julia
# using HallThruster; 
# HallThruster.PYTHON_PATH
# ```)
# hallthruster_path = "/home/elida/.julia/packages/HallThruster/cq07j/python"
# if hallthruster_path not in sys.path:
#     sys.path.append(hallthruster_path)

# import hallthruster as het
# print("HallThruster imported successfully from main!")


from hall_opt.config.verifier import verify_all_yaml  
from hall_opt.scripts.map import run_map_workflow
from hall_opt.scripts.mcmc import run_mcmc_with_final_map_params
from hall_opt.scripts.gen_data import generate_ground_truth
from hall_opt.plotting.posterior_plots import generate_plots
from hall_opt.utils.data_loader import load_data
from hall_opt.utils.resolve_paths import resolve_yaml_paths
from hall_opt.utils.parse import load_yaml, get_yaml_path, parse_arguments

def main():
    try:
        print("Starting main.py execution...")
        args = parse_arguments()
        yaml_path = get_yaml_path(args.method_yaml)

        print(f"[DEBUG] Using YAML file: {yaml_path}")  # Debugging message

        # Load the YAML file
        yaml_data = load_yaml(yaml_path)

        # Debug: Print the loaded YAML data
        print(f"[DEBUG] Loaded YAML data: {yaml_data}")

        # -----------------------------
        #  Step 2: Validate and Load settings.yaml
        # -----------------------------
        settings = verify_all_yaml()  # Load "settings.yaml"

        if settings is None:
            print("ERROR: Failed to load settings. Exiting...")
            sys.exit(1)

        resolve_yaml_paths(settings)

        # -----------------------------
        #  Step 3: Override YAML Settings with Loaded Config
        # -----------------------------
        valid_flags = ["run_map", "run_mcmc", "gen_data", "plotting"]

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
        #  Step 3: Create Results Directory
        # -----------------------------
        base_results_dir = Path(settings.results_dir)

        base_results_dir.mkdir(parents=True, exist_ok=True)

        print(f" Results directory set to: {base_results_dir}")
        # -----------------------------
        #  Step 4: Generate or Load Ground Truth Data
            # -----------------------------
        observed_data = None
        ground_truth_file = Path(settings.postprocess.output_file["MultiLogBohm"]).resolve()

        if settings.general.gen_data:
            print("DEBUG: `gen_data=True` -> Running ground truth generation...")
            observed_data = generate_ground_truth(settings)
        else:
            observed_data = load_data(settings, "ground_truth")
        if not ground_truth_file.exists():
                print(f"ERROR: Ground truth file '{ground_truth_file}' not found.")
                print(f"DEBUG: `gen_data=False` -> Trying to load ground truth data from {ground_truth_file}")
        #  Stop execution if no ground truth is found
        if observed_data is None:
            print("ERROR: Ground truth data is required but missing. Exiting.")

        # -----------------------------
        #  Step 5: Run MAP Estimation (If Enabled)
        # -----------------------------
        if settings.general.run_map:
            print(f"DEBUG: observed_data: {type(observed_data)}, {observed_data is None}")
            if observed_data is None:
                print("ERROR: observed_data is missing, cannot run MAP estimation!")
            else:
                try:
                    print("DEBUG: Calling run_map_workflow()...")
                    optimized_params = run_map_workflow(observed_data, settings, settings.general.config_file)

                    if optimized_params:
                        final_map_params_path = Path(settings.map.output) / "final_map_params.json"
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
        #  Step 6: Run MCMC Sampling (If Enabled)
        # -----------------------------
        if settings.general.run_mcmc:
            print("Running MCMC sampling...")
            print(f"Using base directory for this MCMC run: {settings.mcmc.base_dir}")

            # Run MCMC Sampling
            mcmc_params = run_mcmc_with_final_map_params(observed_data, settings)
            if mcmc_params is None:
                print("ERROR: MCMC sampling failed. Exiting.")
                sys.exit(1)

            print("MCMC sampling completed!")

        # -----------------------------
        #  Step 7: Generate Plots (If Enabled)
        # -----------------------------
        if settings.general.plotting:
            print("Generating plots...")
            generate_plots(settings)

        print("All processes completed successfully!")

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt detected in main.py. Exiting cleanly.")
        sys.exit(0)  # Exit cleanly on Ctrl+C

if __name__ == "__main__":
    main()