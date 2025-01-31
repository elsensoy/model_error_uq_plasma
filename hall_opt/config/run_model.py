import hallthruster as het
import json
import numpy as np
import os
from typing import Optional, List, Dict, Any
from hall_opt.config.verifier import extract_anom_model
#TODO: output file dir needs to be added to pydantic

def run_model(
    config_settings: Dict[str, Any],
    simulation: Dict[str, Any],
    postprocess: Dict[str, Any],
    model_type: str
) -> Optional[Dict[str, Any]]:

    # Extract and update the simulation configuration for the specified model type
    try:
        config_dict = extract_anom_model(config_settings, model_type)
    except ValueError as e:
        print(f" Configuration error: {e}")
        return None

    # Convert Pydantic models to dictionaries if needed
    if hasattr(config_settings, "model_dump"):
        config_settings = config_settings.model_dump()
    
    if hasattr(simulation, "model_dump"):
        simulation = simulation.model_dump()
    
    if hasattr(postprocess, "model_dump"):
        postprocess = postprocess.model_dump()

    # Ensure postprocessing settings exist
    postprocess_dict = postprocess.copy()

    # Handle model-specific output files safely
    output_file = postprocess_dict.get("output_file", {}).get(model_type)

    if not output_file:
        print(f" Warning: No output file defined for model type '{model_type}'. Using default './hall_opt/results/output.json'.")
        output_file = "./hall_opt/results/output.json"

    # Update the postprocessing settings with the selected output file
    postprocess_dict["output_file"] = output_file

    # Debugging output
    print(f"Running simulation with {model_type} configuration...")
    print(f" Output file for postprocessing: {output_file}")

    # Prepare input data for simulation
    input_data = {
        "config": config_dict,  
        "simulation": simulation,  
        "postprocess": postprocess_dict,  
    }

    # Ensure directory exists before running the simulation
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    try:
        # Run the simulation
        solution = het.run_simulation(input_data)

        # Validate simulation metrics
        metrics = solution["output"].get("average", {})
        if not metrics or any(not np.isfinite(value) for value in metrics.values() if isinstance(value, (float, int))):
            print("Warning: Invalid or missing metrics in simulation output.")
            return None

        return solution

    except Exception as e:
        print(f"ERROR during simulation: {e}")
        return None
