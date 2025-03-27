import os
import json
import sys
import hallthruster as het
from pathlib import Path
from typing import Optional, Dict, Any
from ..utils.data_loader import extract_anom_model
from ..config.dict import Settings
from pydantic import BaseModel

def run_model(
    settings: Settings,
    config_settings: Dict[str, Any],
    simulation: Dict[str, Any],
    postprocess: Dict[str, Any],
    model_type: str,
) -> Optional[Dict[str, Any]]:
    
    if not config_settings or "anom_model" not in config_settings:
        try:
            config_settings = extract_anom_model(settings, model_type)
        except ValueError as e:
            print(f"Configuration error: {e}")
            return None
    
    # Print updated `c1` and `c2` before running the simulation
    print("DEBUG: Before extract_anom_model, config_settings =", config_settings)

    if "anom_model" in config_settings and "TwoZoneBohm" in config_settings["anom_model"]:
        print(f"DEBUG: Using c1={config_settings['anom_model']['TwoZoneBohm']['c1']}, "
              f"c2={config_settings['anom_model']['TwoZoneBohm']['c2']} in the simulation")

    # Convert postprocess to dictionary
    postprocess_dict = settings.postprocess.model_dump()

    # Handle model-specific output files
    output_file = (
        postprocess_dict["output_file"].get(model_type)
        if isinstance(postprocess_dict.get("output_file"), dict)
        else postprocess_dict.get("output_file", f"{settings.general.results_dir}/postprocess/output.json")
    )
    
    postprocess_dict["output_file"] = output_file

    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Ensured directory exists for output file: {output_path.parent}")

    print(f"Running simulation with {model_type} configuration...")


    # Convert `simulation` to a dictionary
    simulation_dict = simulation.model_dump() if isinstance(simulation, BaseModel) else simulation

    input_data = {
        "config": config_settings,  # Pass updated config
        "simulation": simulation_dict,
        "postprocess": postprocess_dict,
    }
    print("DEBUG: Checking required file paths before running simulation...")

# Check magnetic field file
    # magnetic_field_path = os.path.abspath(config_settings["thruster"]["magnetic_field"]["file"])
    # print(f"DEBUG: Magnetic field file path: {magnetic_field_path}")
    # if not os.path.exists(magnetic_field_path):
    #     print(f"ERROR: Magnetic field file '{magnetic_field_path}' is missing!")
    #     return None

    # Check working directory
    print("DEBUG: Current working directory:", os.getcwd())


    try:
        #uncommented this section to see the input model on cl with pretty printing
        json_input = json.dumps(input_data, indent=4)    
        print(f"DEBUG: JSON configuration sent to HallThruster:\n{json_input}")
        
        solution = het.run_simulation(input_data)
        if solution is None:
            print("ERROR: Simulation failed. Returning `None`.")
            sys.exit(1)  # Stop execution due to critical failure
        
        return solution  


    except KeyError as e:
        print(f" ERROR: Missing key in JSON: {e}")
        return None
    except Exception as e:
        print(f" ERROR during simulation: {e}")
        return None