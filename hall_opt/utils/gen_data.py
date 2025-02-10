import os
import json
from hall_opt.config.verifier import Settings
from hall_opt.config.run_model import run_model
from hall_opt.utils.save_posterior import save_metrics

def generate_ground_truth(settings: Settings):
    """Generate and save ground truth data if gen_data is True, otherwise load fallback."""
    
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

            print(f"Ground truth data successfully saved to {output_file}")
            return ground_truth_solution

        except Exception as e:
            print(f"ERROR during ground truth generation: {e}")
            return None
