import os
import json
from hall_opt.config.dict import Settings
from hall_opt.config.run_model import run_model
from hall_opt.utils.save_posterior import save_metrics

def generate_ground_truth(settings: Settings):
    ground_truth = settings.ground_truth
    output_file = ground_truth.output_file  

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if ground_truth.gen_data:
        print("\nGenerating ground truth data using MultiLogBohm...")

        try:
            # Run the simulation
            ground_truth_solution = run_model(
                config_settings=settings.config_settings,
                settings=settings,
                simulation=settings.simulation,      
                postprocess=settings.postprocess,    
                model_type="MultiLogBohm",
            )

            if not ground_truth_solution:
                print("ERROR: Ground truth simulation failed.")
                return None
            
            # Extract Necessary Metrics (Same Structure as Metrics Files)
            metrics = ground_truth_solution.get("output", {}).get("average", {})
            if not metrics:
                print("ERROR: Invalid or missing metrics in ground truth simulation output.")
                return None

            extracted_metrics = {
                "thrust": metrics.get("thrust", [0]),
                "time": [settings.simulation.duration],
                "discharge_current": metrics.get("discharge_current", [0]),
                "z_normalized": metrics.get("z", []),
                "ion_velocity": [metrics.get("ui", [0])], 
            }

            # # Save Extracted Metrics Using `save_metrics`
            save_metrics(settings, extracted_metrics, output_dir=settings.ground_truth.results_dir, use_json_dump=True)

            print(f"Ground truth data successfully saved to {output_file}")
            return ground_truth_solution

        except Exception as e:
            print(f"ERROR during ground truth generation: {e}")
            return None