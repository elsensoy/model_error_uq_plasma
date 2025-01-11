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

try:
    import pathlib
    print("Pathlib imported successfully.")
except ImportError as e:
    print(f"Pathlib import error: {e}")

# Configuration model using Pydantic 
class Settings(BaseModel):
    results_dir: str = Field(..., description="Directory to save all results.")
    gen_data: bool = Field(..., description="Flag to generate ground truth data.")
    run_map: bool = Field(False, description="Flag to run MAP estimation.")
    run_mcmc: bool = Field(False, description="Flag to run MCMC sampling.")
    ion_velocity_weight: float = Field(2.0, description="Weight for ion velocity in MAP and MCMC.")
    plotting: bool = Field(False, description="Enable or disable plotting.")
    initial_guess_path: str = Field(..., description="Path to initial guess parameters for MCMC.")
    iterations: int = Field(1000, description="Number of MCMC iterations.")
    initial_cov: list = Field(..., description="Initial covariance matrix for MCMC.")
   # Simulation-specific configurations
    config_multilogbohm: Dict[str, Any] = Field(
        ..., description="Configuration for the MultiLogBohm ground truth model."
    )
    config_spt_100: Dict[str, Any] = Field(
        ..., description="Configuration for the TwoZoneBohm model."
    )
    simulation: Dict[str, Any] = Field(
        ..., description="Simulation parameters."
    )
    postprocess: Dict[str, Any] = Field(
        ..., description="Postprocessing parameters."
    )

# Helper function to load YAML configuration
def load_yml_settings(path: pathlib.Path) -> Any:
    try:
        with path.open("r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError as error:
        print("Error: YAML settings file not found.")
        raise
    except yaml.YAMLError as error:
        print("Error: Invalid YAML format.")
        raise
