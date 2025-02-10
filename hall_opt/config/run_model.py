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
    
    print(f"DEBUG: Inside run_model()")
    print(f"DEBUG: config_settings type = {type(config_settings)}")
    print(f"DEBUG: simulation type = {type(simulation)}")
    print(f"DEBUG: postprocess type = {type(postprocess)}")

    # Extract and update the simulation configuration for the specified model type
    try:
        config_settings = extract_anom_model(settings, model_type)
    except ValueError as e:
        print(f"Configuration error: {e}")
        return None

    # Convert postprocess to dictionary
    postprocess_dict = settings.postprocess.model_dump()

    # Handle model-specific output files
    output_file = (
        postprocess_dict["output_file"].get(model_type)
        if isinstance(postprocess_dict.get("output_file"), dict)
        else postprocess_dict.get("output_file", f"{settings.general.results_dir}/postprocess/output.json")
    )
    
    #  the output file is updated
    postprocess_dict["output_file"] = output_file

    print(f"Running simulation with {model_type} configuration...")

    # Convert `simulation` to a dictionary
    simulation_dict = simulation.model_dump() if isinstance(simulation, BaseModel) else simulation

    input_data = {
        "config": config_settings,
        "simulation": simulation_dict,
        "postprocess": postprocess_dict,
    }

    try:
        # Run the simulation
        solution = het.run_simulation(input_data)
        # print(f"DEBUG: Raw simulation output = {solution}")

        return solution  # Return simulation output in-memory

    except Exception as e:
        print(f"ERROR during simulation: {e}")
        return None