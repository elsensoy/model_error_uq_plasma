import sys
import json
import numpy as np
from typing import Dict, Any, Optional
from hall_opt.config.loader import Settings

# HallThruster Path Setup
hallthruster_path = "/home/elidasensoy/.julia/packages/HallThruster/tHQQa/python"
if hallthruster_path not in sys.path:
    sys.path.append(hallthruster_path)

import hallthruster as het

# -----------------------------
#  Run a simulation with the given configuration and handle failures.
# -----------------------------
from typing import Dict, Any, Optional
import hallthruster as het
import numpy as np

def run_simulation_with_config(
    config: Dict[str, Any],
    simulation: Dict[str, Any],
    postprocess: Dict[str, Any],
    model_type: str,
    failing_samples: Optional[list] = None
) -> Optional[Dict[str, Any]]:
    """
    Run a simulation with the given configuration.
    """
    failing_samples = failing_samples or []

    # Extract and validate the output file for the current model type
    if "output_file" in postprocess and isinstance(postprocess["output_file"], dict):
        output_file = postprocess["output_file"].get(model_type)
        if output_file is None:
            raise ValueError(f"No output file defined for model type '{model_type}' in postprocess.")
        postprocess["output_file"] = output_file
    elif not isinstance(postprocess["output_file"], str):
        raise ValueError("postprocess['output_file'] must be a string or a dictionary with model-specific keys.")

    # Prepare input for simulation
    input_data = {
        "config": config,
        "simulation": simulation,
        "postprocess": postprocess,
    }

    print(f"Running simulation with {model_type} configuration...")

    try:
        solution = het.run_simulation(input_data)
        return solution
    except Exception as e:
        print(f"Error during simulation: {e}")
        failing_samples.append({"reason": str(e), "config": config})
        return None
