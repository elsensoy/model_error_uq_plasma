import os
import sys
import yaml
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from pathlib import Path
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
hallthruster_path = "/home/elidasensoy/.julia/packages/HallThruster/tHQQa/python"
if hallthruster_path not in sys.path:
    sys.path.append(hallthruster_path)

import hallthruster as het


class Settings(BaseModel):
    # General settings
    results_dir: str = Field(..., description="Directory to save all results.")
    gen_data: bool = Field(..., description="Flag to generate ground truth data.")
    run_map: bool = Field(False, description="Flag to run MAP estimation.")
    run_mcmc: bool = Field(False, description="Flag to run MCMC sampling.")
    initial_guess_path: str = Field(..., description="Path to initial guess parameters for MCMC.")
    ion_velocity_weight: float = Field(2.0, description="Weight for ion velocity in MAP and MCMC.")
    plotting: bool = Field(False, description="Enable or disable plotting.")
    iterations: int = Field(1000, description="Number of MCMC iterations.")

    # Simulation-related settings
    simulation_config: Dict[str, Any] = Field(..., description="Unified simulation configuration.")
    simulation: Dict[str, Any] = Field(..., description="Simulation parameters.")
    postprocess: Dict[str, Any] = Field(..., description="Postprocessing parameters.")

    # MAP-related settings
    map_method: str = Field("Nelder-Mead", description="Optimization method for MAP.")
    map_initial_guess_path: Optional[str] = Field(None, description="Path to initial guess parameters for MAP.")
    map_maxfev: int = Field(5000, description="Maximum number of function evaluations for MAP.")
    map_fatol: float = Field(1e-3, description="Function tolerance for MAP optimization.")
    map_xatol: float = Field(1e-3, description="Step size tolerance for MAP optimization.")
    optimized_param: str = Field(..., description="Path to optimized parameters for MCMC.")
    map_iteration_log_file: Optional[str] = Field(None, description="File to log MAP iterations.")
    
    # MCMC-related settings
    mcmc_save_interval: int = Field(10, description="Interval for saving MCMC samples.")
    mcmc_checkpoint_interval: int = Field(10, description="Interval for saving MCMC checkpoints.")
    mcmc_save_metadata: bool = Field(True, description="Flag to save MCMC metadata.")
    mcmc_results_dir: Optional[str] = Field(None, description="Directory for saving MCMC results.")
    mcmc_final_samples_file_log: Optional[str] = Field(None, description="File to save final MCMC samples in log-space.")
    mcmc_final_samples_file_linear: Optional[str] = Field(None, description="File to save final MCMC samples in linear space.")
    mcmc_checkpoint_file: Optional[str] = Field(None, description="File to save MCMC checkpoints.")
    mcmc_metadata_file: Optional[str] = Field(None, description="File to save MCMC metadata.")
    initial_cov: list = Field(..., description="Initial covariance matrix for MCMC.")

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
    parameters = config["anom_model"].get(model_type, {})

    if model_type == "TwoZoneBohm":
        config["anom_model"] = {
            "type": model_type,
            "c1": parameters["c1"],
            "c2": parameters["c2"],
        }
    elif model_type == "MultiLogBohm":
        config["anom_model"] = {
            "type": model_type,
            "zs": parameters["zs"],
            "cs": parameters["cs"],
        }
    else:
        raise ValueError(f"Unsupported anomalous transport model type: {model_type}")

    return config
