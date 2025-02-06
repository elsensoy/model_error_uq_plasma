import os
import json
from hall_opt.config.verifier import Settings
from hall_opt.config.run_model import run_model

def generate_ground_truth(settings: Settings):
    """Generate and save ground truth data if gen_data is True."""
    
    ground_truth = settings.ground_truth
    output_file = ground_truth.output_file  

    # Ensure output directory exists
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
            
            # Save the FULL raw simulation output
            with open(output_file, "w") as file:
                json.dump(ground_truth_solution, file, indent=4)
            
            print(f"Ground truth data successfully saved to {output_file}")
            return ground_truth_solution

        except Exception as e:
            print(f"ERROR during ground truth generation: {e}")
            return None
    else:
        print("Ground truth generation is disabled in settings.")

    return None
