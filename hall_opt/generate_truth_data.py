import sys
import numpy as np
from pathlib import Path
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
# Import verifier and run functions
from config.verifier import verify_all_yaml, extract_anom_model
from config.run_model import run_model

def main():
    # Validate all YAML files before execution
    yaml_dir = Path(__file__).resolve().parent / "config"
    print(f"Using YAML directory: {yaml_dir}")
    settings = verify_all_yaml()
    if settings is None:
        print("ERROR: One or more YAML configuration files are invalid. Exiting...")
        sys.exit(1)
    

    # Resolve base directory paths
    base_results_dir = Path(settings.results_dir)
    base_results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results directory set to: {base_results_dir}")

    # Step 1: Generate ground truth data
    if settings.gen_data:
        print("Generating ground truth data using MultiLogBohm...")
        try:
            multilogbohm_config = extract_anom_model(settings, model_type="MultiLogBohm")
            ground_truth_solution = run_model(
                config=multilogbohm_config,
                settings=settings,
                simulation=settings.simulation,
                postprocess=settings.postprocess,
                model_type="MultiLogBohm",
            )

            if ground_truth_solution:
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

            else:
                print("ERROR: Ground truth simulation failed.")
                sys.exit(1)

        except Exception as e:
            print(f" ERROR during ground truth generation: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
