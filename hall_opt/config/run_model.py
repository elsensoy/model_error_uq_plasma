import hallthruster as het
import json
import numpy as np
import os
from typing import Optional, Dict, Any
from hall_opt.config.dict import Settings
from hall_opt.utils.data_loader import extract_anom_model
from pydantic import BaseModel
from hall_opt.utils.save_data import save_results_to_json

def run_model(
    settings: Settings,
    config_settings: Dict[str, Any],  
    simulation: Dict[str, Any],
    postprocess: Dict[str, Any],
    model_type: str,
) -> Optional[Dict[str, Any]]:
    
    """Runs the Hall Thruster simulation, extracts metrics, and validates output."""
    
    # Debug: Ensure we are working with Pydantic objects or dictionaries
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

    # Convert postprocess to dictionary before accessing keys
    postprocess_dict = settings.postprocess.model_dump()

    # Handle model-specific output files
    output_file = (
        postprocess_dict["output_file"].get(model_type) 
        if isinstance(postprocess_dict.get("output_file"), dict) 
        else postprocess_dict.get("output_file", "hall_opt/results/output.json")
    )
    
    # Ensure the output file is updated
    postprocess_dict["output_file"] = output_file

    # Debugging output
    print(f"Running simulation with {model_type} configuration...")
    print(f"Output file for postprocessing: {postprocess_dict['output_file']}")

#   `simulation` is converted to a dictionary
    simulation_dict = simulation.model_dump() if isinstance(simulation, BaseModel) else simulation

    input_data = {
        "config": config_settings,  # Anomalous model-specific configuration
        "simulation": simulation_dict,  #  Use dictionary version
        "postprocess": postprocess_dict,  # Postprocessing settings
    }

    # Ensure Output Directory Exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Debug: Print Input Data Before Running
    # print("DEBUG: input_data = ", json.dumps(input_data, indent=4))

    try:
        # Run the simulation 
        solution = het.run_simulation(input_data)
        # print(f"DEBUG: Raw simulation output = {solution}")


        # Extract & Validate Simulation Metrics
        if not solution:
            print("ERROR: Simulation returned an invalid response.")
            return None

        metrics = solution["output"].get("average", {})
        
        # Extract Specific Metrics
        extracted_metrics = {
            "thrust": metrics.get("thrust", 0),
            "discharge_current": metrics.get("discharge_current", 0),
            "ion_velocity": metrics.get("ui", [0])[0],  # First value if ui is a list
            "z_normalized": metrics.get("z", 0),
        }

        # Debugging Output for Extracted Metrics
        print("\nExtracted Simulation Metrics:")
        print(f"Thrust: {extracted_metrics['thrust']} N")
        print(f"Discharge Current: {extracted_metrics['discharge_current']} A")
        print(f"Ion Velocity: {extracted_metrics['ion_velocity']} m/s")
        print(f"Z-Normalized: {extracted_metrics['z_normalized']}")

        save_results_to_json(
            extracted_metrics, 
            filename=os.path.basename(output_file),
            results_dir=os.path.dirname(output_file),
            save_every_n_grid_points=10, 
            subsample_for_saving=True
        )

        return extracted_metrics  #  only the structured extracted metrics

    except Exception as e:
        print(f"ERROR during simulation: {e}")
        return None
