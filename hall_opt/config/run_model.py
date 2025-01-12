import sys
import json
import numpy as np
from typing import Dict, Any, Optional
from hall_opt.config.settings_loader import Settings

# HallThruster Path Setup
hallthruster_path = "/home/elidasensoy/.julia/packages/HallThruster/tHQQa/python"
if hallthruster_path not in sys.path:
    sys.path.append(hallthruster_path)

import hallthruster as het

# -----------------------------
# Run Simulation based on Specified Model Configuration
# -----------------------------

def run_simulation_with_config(
    settings: Settings,
    model_type: str,
    iteration: Optional[int] = None,
    c1: Optional[float] = None,
    c2: Optional[float] = None,
    failing_samples: Optional[list] = None
) -> Optional[Dict[str, Any]]:

    failing_samples = failing_samples or []

    # Extract and update the simulation configuration for the specified model type
    try:
        simulation_config = extract_anom_model(settings, model_type)
    except ValueError as e:
        print(f"Configuration error: {e}")
        return None

    # Prepare postprocessing settings
    postprocess = settings.postprocess.copy()
    postprocess["output_file"] = simulation_config.get("output_file", "./results/output.json")

    # Prepare input for the simulation
    input_data = {
        "config": simulation_config,
        "simulation": settings.simulation,
        "postprocess": postprocess,
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
                "iteration": iteration,
                "c1": c1,
                "c2": c2,
                "retcode": retcode,
                "reason": "Simulation failure",
                "config": simulation_config,
            })
            return None

        # Validate simulation metrics
        metrics = solution["output"].get("average", {})
        if not metrics or any(not np.isfinite(value) for value in metrics.values() if isinstance(value, (float, int))):
            print("Invalid or missing metrics in simulation output.")
            failing_samples.append({
                "iteration": iteration,
                "c1": c1,
                "c2": c2,
                "reason": "Invalid metrics",
                "config": simulation_config,
            })
            return None

        return solution

    except Exception as e:
        print(f"Error during simulation: {e}")
        failing_samples.append({
            "iteration": iteration,
            "c1": c1,
            "c2": c2,
            "reason": f"Unexpected error: {str(e)}",
        })
        return None
