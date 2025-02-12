import os
import json
from pathlib import Path
from hall_opt.config.verifier import Settings
from hall_opt.config.run_model import run_model


def generate_ground_truth(settings: Settings):
    """Generate and save ground truth data if gen_data is True, otherwise load fallback."""
    
    ground_truth = settings.ground_truth
    # output_file = Path(settings.ground_truth.output_file["MultiLogBohm"]).resolve()

    # os.makedirs(os.path.dirname(output_file), exist_ok=True)

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

            print(f"Ground truth data successfully saved")
            return ground_truth_solution

        except Exception as e:
            print(f"ERROR during ground truth generation: {e}")
            return None
