import sys
import numpy as np
import os
import yaml
from pathlib import Path
from hall_opt.config.verifier import verify_all_yaml
from hall_opt.config.run_model import run_model
from pathlib import Path
from hall_opt.config.verifier import verify_all_yaml
from hall_opt.config.run_model import run_model
from hall_opt.debug.plot_ground_truth import plot_ion_velocity, load_results

def main():
    #  Step 1: Validate and Load YAML
    settings = verify_all_yaml()
    if settings is None:
        print(" ERROR: Failed to load settings. Exiting...")
        sys.exit(1)
    print(f"DEBUG: settings type = {type(settings)}")

    #  Step 2: Extract Required Configurations
    general_settings = settings.general
    config_settings = settings.config_settings 
    ground_truth = settings.ground_truth
    postprocess = settings.postprocess
    simulation = settings.simulation

    print(f"DEBUG: config_settings type = {type(config_settings)}")
    print(f"DEBUG: simulation type = {type(simulation)}")
    print(f"DEBUG: postprocess type = {type(postprocess)}")

    #  Step 3: Results Directory if It Doesn't Exist
    base_results_dir = Path(settings.postprocess.output_file["Multilogbohm"]).resolve()
    base_results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results directory set to: {base_results_dir}")
    
    #  Step 4: Generate Ground Truth Data (Only If Enabled)
    if ground_truth.gen_data:
        print("\nGenerating ground truth data using MultiLogBohm...")

        try:
            #  Step 5: Run Simulation
            ground_truth_solution = run_model(
                config_settings=config_settings,
                settings=settings,
                simulation=settings.simulation,      # Pass required simulation settings
                postprocess=settings.postprocess,    # Pass postprocessing settings
                model_type="MultiLogBohm",
            )

            #  Step 6: Extract & Display Results
            if ground_truth_solution:
                averaged_metrics = ground_truth_solution["output"]["average"]
                observed_data = {
                    "thrust": averaged_metrics.get("thrust", 0),
                    "discharge_current": averaged_metrics.get("discharge_current", 0),
                    "ion_velocity": averaged_metrics.get("ui", [0])[0],
                    "z_normalized": averaged_metrics.get("z", 0),
                }

                print("\n Ground truth data successfully generated!\n")
                print(f" Thrust: {observed_data['thrust']} N")
                print(f" Discharge Current: {observed_data['discharge_current']} A")
                print(f" Ion Velocity: {observed_data['ion_velocity']} m/s")
                print(f" Z-Normalized: {observed_data['z_normalized']}\n")
        
        except Exception as e:
            print(f" ERROR during ground truth generation: {e}")

        else:
            print(" ERROR: Ground truth simulation failed.")

