import sys
import numpy as np
from pathlib import Path
from pydantic import BaseModel  
from hall_opt.config.verifier import verify_all_yaml, extract_anom_model
from hall_opt.config.run_model import run_model

def main():
    # Validate all YAML files before execution
    settings = verify_all_yaml()
    if settings is None:
        print("ERROR: One or more YAML configuration files are invalid. Exiting...")
        sys.exit(1)

    # ✅ Extract sections correctly
    general_settings = settings["general"] 
    config_settings = settings["config"] 
    ground_truth = settings["ground_truth"]

    # Resolve base directory paths
    base_results_dir = Path(general_settings.results_dir).resolve()
    base_results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results directory set to: {base_results_dir}")

    # Step 1: Generate ground truth data
    if ground_truth.gen_data:
        print("Generating ground truth data using MultiLogBohm...")
        try:
           # multilogbohm_config = extract_anom_model(settings, model_type="MultiLogBohm")
            ground_truth_solution = run_model(
                config_settings=config_settings,  
                simulation=config_settings,  
                postprocess=config_settings.postprocess,  
                model_type="MultiLogBohm",
            )

            if ground_truth_solution:
                # Ensure `ground_truth_solution["output"]["average"]` is a dictionary
                averaged_metrics = ground_truth_solution["output"]["average"]
                if isinstance(averaged_metrics, BaseModel):
                    averaged_metrics = averaged_metrics.model_dump()  # Convert to dict if needed

                # Use `.get()` now that it's a dictionary
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

            else:
                print("ERROR: Ground truth simulation failed.")
                sys.exit(1)

        except Exception as e:
            print(f" ERROR during ground truth generation: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
