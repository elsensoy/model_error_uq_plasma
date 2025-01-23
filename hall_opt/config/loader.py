import os
import sys
import yaml
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ValidationError

# Ensure HallThruster Python path is included
hallthruster_path = "/home/elida/.julia/packages/HallThruster/tHQQa/python"
if hallthruster_path not in sys.path:
    sys.path.append(hallthruster_path)

import hallthruster as het


class GeometryConfig(BaseModel):
    channel_length: float
    inner_radius: float
    outer_radius: float

class MagneticFieldConfig(BaseModel):
    file: str

class AnomModelConfig(BaseModel):
    TwoZoneBohm: Dict[str, float]
    MultiLogBohm: Dict[str, List[float]]

class Config(BaseModel):
    name: str
    geometry: GeometryConfig
    magnetic_field: MagneticFieldConfig
    discharge_voltage: float
    anode_mass_flow_rate: float
    domain: List[float]
    propellant: str
    ion_wall_losses: bool
    solve_plume: bool
    apply_thrust_divergence_correction: bool
    neutral_ingestion_multiplier: float
    ncharge: int
    transition_length: float
    neutral_velocity: float
    anom_model: AnomModelConfig

class OptimizationParams(BaseModel):
    map_params: Dict[str, Any]
    mcmc_params: Dict[str, Any]  
    map_initial_guess_path: str = Field(..., description="Path to initial guess parameters for MCMC.")
    final_map_params: Dict[str, Any] = Field(..., description="MAP and MCMC optimization parameters.")

class Settings(BaseModel):
    general_settings: Dict[str, Any] = Field(..., description="General settings for the simulation.")
    config: Dict[str, Any] = Field(..., description="Configuration for the thruster simulation.")
    simulation: Dict[str, Any] = Field(..., description="Simulation parameters.")
    postprocess: Dict[str, Any] = Field(..., description="Postprocessing settings.")
    inputs: Dict[str, Any] = Field(..., description="Input values for the simulation.")
    outputs: List[Dict[str, Any]] = Field(..., description="Expected output metrics.")
    optimization_params: Dict[str, Any] = Field(..., description="MAP and MCMC optimization parameters.")

def load_yml_settings(path: Path) -> Settings:
    """
    Load the YAML file and validate it against the `Settings` Pydantic model.
    """
    try:
        with path.open("r") as file:
            yaml_data = yaml.safe_load(file)  # Load YAML into a dictionary
        # print("Loaded YAML data:", yaml_data)  # Debugging: Print raw YAML
        return Settings(**yaml_data)  # Validate and parse with Pydantic
    except FileNotFoundError:
        print(f"Error: YAML file not found at {path}")
        raise
    except ValidationError as e:
        print(f"Validation error: {e.json()}")
        print("Invalid YAML settings data:", yaml_data)  # Debugging
        raise
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML format. {str(e)}")
        raise


def extract_anom_model(settings: Settings, model_type: str) -> Dict[str, Any]:

    try:
        # Access the config from the settings
        config = settings.config

        # Debugging: Print the simulation configuration
        # print("Simulation Config:", config)

        # Access the anomalous transport model configuration
        anom_model_config = config["anom_model"]

        # Debugging: Print the requested model type and available models
        print("Model Type Requested:", model_type)
        print("Available Models in Config:", anom_model_config.keys())

        if model_type not in anom_model_config:
            raise KeyError(f"Anomalous model type '{model_type}' not found in configuration.")

        # Extract the specific model configuration
        model_config = anom_model_config[model_type]

        # Combine the simulation config with the specific model
        base_config = config.copy()  # Create a copy of the simulation config
        base_config["anom_model"] = {**model_config, "type": model_type}  # Update the model-specific part

        # # Debugging: Print the final extracted configuration
        # print(f"Extracted Model Config: {base_config}")

        return base_config

    except KeyError as e:
        print(f"Configuration error: {e}")
        raise
