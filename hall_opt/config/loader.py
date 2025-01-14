import os
import sys
import yaml
import logging
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ValidationError
from pathlib import Path
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
hallthruster_path = "/home/elidasensoy/.julia/packages/HallThruster/tHQQa/python"
if hallthruster_path not in sys.path:
    sys.path.append(hallthruster_path)

import hallthruster as het

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class Settings(BaseModel):
    # General settings
    general_settings: Dict[str, Any] = Field(
        ...,
        description="General settings such as results directory, flags, and iteration counts."
    )

    # Optimization parameters, containing both MAP and MCMC parameters
    optimization_params: Dict[str, Dict[str, Any]] = Field(
        ...,
        description="Optimization parameters including MAP and MCMC settings."
    )

    # Unified simulation configuration
    simulation_config: Dict[str, Any] = Field(
        ...,
        description="Configuration of the simulation, including thruster properties and anomalous models."
    )

    # Simulation parameters
    simulation: Dict[str, Any] = Field(
        ...,
        description="Simulation parameters like time step and grid resolution."
    )

    # Postprocessing settings
    postprocess: Dict[str, Any] = Field(
        ...,
        description="Postprocessing configuration, including output files and processing flags."
    )

    # Input parameters
    inputs: Dict[str, Any] = Field(
        ...,
        description="Input parameters for the simulation."
    )

    # Outputs configuration
    outputs: List[Dict[str, Any]] = Field(
        ...,
        description="List of output metrics, their descriptions, and domains."
    )

# try:

#     settings = Settings(**yml_dict)
# except ValidationError as e:
#     print("Validation Error:")
#     print(e.errors())  # Prints a list of validation errors
#     print(e.json())  # Prints errors in JSON format

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
    try:
        config = settings.simulation_config
        thruster_config = config.get("thruster", {})
        anom_model_config = thruster_config.get("anom_model", {})
        if model_type not in anom_model_config:
            raise KeyError(f"Anomalous model type '{model_type}' not found in configuration.")
        model_config = anom_model_config[model_type]

        # Merge common configuration with model-specific configuration
        base_config = {
            "thruster": thruster_config,

        }
        base_config["anom_model"] = {**model_config, "type": model_type}
        return base_config
    except KeyError as e:
        raise KeyError(f"Configuration error: {e}")
