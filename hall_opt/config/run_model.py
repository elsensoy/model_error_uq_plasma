import os
import json
import hallthruster as het
from typing import Optional, Dict, Any
from hall_opt.utils.data_loader import extract_anom_model
from hall_opt.config.dict import Settings
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

    print(f"Running simulation with {model_type} configuration...")

    # Convert `simulation` to a dictionary
    simulation_dict = simulation.model_dump() if isinstance(simulation, BaseModel) else simulation

    input_data = {
        "config": config_settings,  # Pass updated config
        "simulation": simulation_dict,
        "postprocess": postprocess_dict,
    }

    try:
        # Run the simulation
        solution = het.run_simulation(input_data)
        return solution  

    except Exception as e:
        print(f"ERROR during simulation: {e}")
        return None
