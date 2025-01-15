import sys
import json
import numpy as np
from typing import Optional, List, Dict, Any
from hall_opt.config.loader import Settings, extract_anom_model

# HallThruster Path Setup
hallthruster_path = "/home/elida/.julia/packages/HallThruster/tHQQa/python"
if hallthruster_path not in sys.path:
    sys.path.append(hallthruster_path)

import hallthruster as het


def run_simulation_with_config(
    settings: Settings,
    config: Dict[str, Any],
    simulation: Dict[str, Any],
    postprocess: Dict[str, Any],
    model_type: str,
    failing_samples: Optional[list] = None
) -> Optional[Dict[str, Any]]:
    """
    Run a simulation with the given configuration.
    
    Args:
        settings (Settings): The loaded settings object.
        config (Dict[str, Any]): Simulation-specific configuration.
        simulation (Dict[str, Any]): Simulation parameters.
        postprocess (Dict[str, Any]): Postprocessing configuration.
        model_type (str): The type of anomalous transport model (e.g., "TwoZoneBohm", "MultiLogBohm").
        failing_samples (list, optional): A list to store details of failed simulations.
    
    Returns:
        Optional[Dict[str, Any]]: The simulation results, or None if the simulation fails.
    """
    failing_samples = failing_samples or []

    # Extract and update the simulation configuration for the specified model type
    try:
        config = extract_anom_model(settings, model_type)
    except ValueError as e:
        print(f"Configuration error: {e}")
        return None

    # Extract and validate the output file for the current model type
# Prepare postprocessing settings
    postprocess = settings.postprocess.copy()

    # Handle model-specific output files
    if isinstance(postprocess.get("output_file"), dict):
        
        output_file = postprocess["output_file"].get(model_type)
        if output_file is None:
            print(f"Warning: No output file defined for model type '{model_type}'. Using default './results/output.json'.")
            output_file = "./results/output.json"
    else:
        # Use the output_file as a string, or fallback to default
        output_file = postprocess.get("output_file", "hall_opt/results/output.json")

    # Update the postprocessing settings with the selected output file
    postprocess["output_file"] = output_file

    # Debugging output
    print(f"Running simulation with {model_type} configuration...")
    print(f"Output file for postprocessing: {postprocess['output_file']}")


    input_data = {
        "config": config,  # Anomalous model-specific configuration
        "simulation": simulation,    # General simulation parameters
        "postprocess": postprocess,  # Postprocessing settings
    }

    print(f"Running simulation with {model_type} configuration...")

    try:
        # Run the simulation 
        solution = het.run_simulation(input_data)

        # Check simulation success
        retcode = solution["output"].get("retcode", "unknown")
        if retcode != "success":
            print(f"Simulation failed with retcode: {retcode}")
            failing_samples.append({
                "model_type": model_type,
                "retcode": retcode,
                "reason": "Simulation failure",
                "config": config,
            })
            return None

        # Validate simulation metrics
        metrics = solution["output"].get("average", {})
        if not metrics or any(not np.isfinite(value) for value in metrics.values() if isinstance(value, (float, int))):
            print("Invalid or missing metrics in simulation output.")
            failing_samples.append({
                "model_type": model_type,
                "reason": "Invalid metrics",
                "config": config,
            })
            return None

        return solution

    except Exception as e:
        print(f"Error during simulation: {e}")
        failing_samples.append({
            "model_type": model_type,
            "reason": f"Unexpected error: {str(e)}",
            "config": config,
        })
        return None
