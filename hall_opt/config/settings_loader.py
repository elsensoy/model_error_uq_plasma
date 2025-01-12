import os
import sys
import yaml
from typing import Dict, Any
from pydantic import BaseModel, Field

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
hallthruster_path = "/home/elidasensoy/.julia/packages/HallThruster/tHQQa/python"
if hallthruster_path not in sys.path:
    sys.path.append(hallthruster_path)

import hallthruster as het
from pathlib import Path


class Settings(BaseModel):
    results_dir: str = Field(..., description="Directory to save all results.")
    gen_data: bool = Field(..., description="Flag to generate ground truth data.")
    run_map: bool = Field(False, description="Flag to run MAP estimation.")
    run_mcmc: bool = Field(False, description="Flag to run MCMC sampling.")
    ion_velocity_weight: float = Field(2.0, description="Weight for ion velocity in MAP and MCMC.")
    plotting: bool = Field(False, description="Enable or disable plotting.")
    initial_guess_path: str = Field(..., description="Path to initial guess parameters for MCMC.")
    optimized_param: str = Field(..., description="Path to optimized parameters for MCMC.")
    iterations: int = Field(1000, description="Number of MCMC iterations.")
    initial_cov: list = Field(..., description="Initial covariance matrix for MCMC.")
    simulation_config: Dict[str, Any] = Field(..., description="Unified simulation configuration.")
    simulation: Dict[str, Any] = Field(..., description="Simulation parameters.")
    postprocess: Dict[str, Any] = Field(..., description="Postprocessing parameters.")


def load_yml_settings(path: Path) -> Dict[str, Any]:
    """
    Load YAML configuration from a file.
    """
    try:
        with path.open("r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError as error:
        print("Error: YAML settings file not found.")
        raise
    except yaml.YAMLError as error:
        print("Error: Invalid YAML format.")
        raise


def extract_anom_model(settings: Settings, model_type: str) -> Dict[str, Any]:
    """
    Extract and update the anomalous transport model configuration based on the model type.
    """
    config = settings.simulation_config.copy()
    parameters = config["anom_model"]["parameters"]

    if model_type == "TwoZoneBohm":
        config["anom_model"] = {
            "type": model_type,
            "c1": parameters["c1"],
            "c2": parameters["c2"],
        }
        config["output_file"] = parameters.get("output_file", "./results/output_twozonebohm.json")
    elif model_type == "MultiLogBohm":
        config["anom_model"] = {
            "type": model_type,
            "zs": parameters["zs"],
            "cs": parameters["cs"],
        }
        config["output_file"] = parameters.get("output_file", "./results/output_multilogbohm.json")
    else:
        raise ValueError(f"Unsupported anomalous transport model type: {model_type}")

    return config
