import sys
import argparse
import json
from pathlib import Path
from config.verifier import verify_all_yaml  
from scripts.map import run_map_workflow
from scripts.mcmc import run_mcmc_with_final_map_params
from scripts.gen_data import generate_ground_truth
from plotting.posterior_plots import generate_plots
from utils.data_loader import load_data
from utils.resolve_paths import resolve_yaml_paths


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run different components of the project.")
    
    parser.add_argument("--mcmc", action="store_true", help="Run MCMC sampling (overrides YAML)")
    parser.add_argument("--map", action="store_true", help="Run MAP estimation (overrides YAML)")
    parser.add_argument("--gen_data", action="store_true", help="Generate ground truth data (overrides YAML)")
    parser.add_argument("--plotting", action="store_true", help="Generate plots")
    return parser.parse_args()


def main():
    # Parse command-line arguments
    args = parse_arguments()

    # -----------------------------
    #  Step 1: Validate and Load YAML
    # -----------------------------
    settings = verify_all_yaml()

    if settings is None:
        print("ERROR: Failed to load settings. Exiting...")
        sys.exit(1)

    resolve_yaml_paths(settings)

    # -----------------------------
    #  Step 2: Override YAML Settings with Command-Line Args
    # -----------------------------
    if args.map:
        settings.general.run_map = True
    if args.mcmc:
        settings.general.run_mcmc = True
    if args.gen_data:
        settings.ground_truth.gen_data = True
    if args.plotting:
        settings.general.plotting =True

    # -----------------------------
    #  Step 3: Create Results Directory
    # -----------------------------
    base_results_dir = settings.general.results_dir

    # base_results_dir.mkdir(parents=True, exist_ok=True)
   
    print(f" Results directory set to: {base_results_dir}")
     # -----------------------------
    #  Step 4: Generate or Load Ground Truth Data
    # -----------------------------
    observed_data = None
    ground_truth_file = Path(settings.postprocess.output_file["MultiLogBohm"]).resolve()

    if settings.ground_truth.gen_data:
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
# Step 5: Run MAP estimation if enabled
    if settings.general.run_map:
        print("Running MAP estimation using TwoZoneBohm...")

        print(f"Using base directory for this MAP run: {settings.map.base_dir}")
        config_file = settings.general.config_file
        try:
            # Run MAP workflow
            optimized_params = run_map_workflow(observed_data, settings, config_file)

            if optimized_params:
                # Save final MAP parameters inside `map-results-N/`
                final_map_params_path = Path(settings.map.base_dir) / "final_map_params.json"
                with open(final_map_params_path, "w") as f:
                    json.dump(optimized_params, f, indent=4)
                print(f"Final MAP parameters saved to {final_map_params_path}")

            else:
                print("ERROR: MAP optimization failed.")

        except Exception as e:
            print(f"ERROR during MAP estimation: {e}")
            sys.exit(1)


    # -----------------------------
    #  Step 6: Run MCMC Sampling (If Enabled)
    # -----------------------------
    if settings.general.run_mcmc:
        print(" Running MCMC sampling...")
    
        print(f"Using base directory for this MCMC run: {settings.mcmc.base_dir}")

        # Run MCMC Sampling
        mcmc_params = run_mcmc_with_final_map_params(observed_data, settings)
        if mcmc_params is None:
            print(" ERROR: MCMC sampling failed. Exiting.")
            sys.exit(1)

        print(" MCMC sampling completed!")

    # -----------------------------
    #  Step 7: Generate Plots (If Enabled)
    # -----------------------------
    if settings.general.plotting:
        print(" Generating plots...")
        generate_plots(settings)
    print(" All processes completed successfully!")


if __name__ == "__main__":
    main()