import os
import json
from pathlib import Path
from hall_opt.config.verifier import Settings
from hall_opt.config.run_model import run_model
def generate_ground_truth(settings: Settings):
    output_file = Path(settings.postprocess.output_file["MultiLogBohm"]).resolve()
    print(f"DEBUG: Expected output file: {output_file}")

    if settings.ground_truth.gen_data:
        print("DEBUG: Running ground truth generation...")
        observed_data = run_model(settings, settings.config_settings.dict(), 
                                  settings.simulation.dict(), settings.postprocess.dict(), 
                                  model_type="MultiLogBohm")

        if observed_data is None:
            print("ERROR: Ground truth simulation failed. No data generated.")
            return None

        print(f"DEBUG: Ground truth data before saving: {observed_data}")  # print data before saving

        # Ensure directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save the ground truth file
        with open(output_file, "w") as f:
            json.dump(observed_data, f, indent=4)

        print(f"Ground truth successfully saved to {output_file}")
    
    return output_file if output_file.exists() else None
