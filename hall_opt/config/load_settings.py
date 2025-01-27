import yaml
from pydantic import BaseModel
from pathlib import Path
from typing import List
#set file paths to str for testing 

# Pydantic models for map.yaml and mcmc.yaml
class MapConfig(BaseModel):
    map_initial_guess_file: str
    iteration_log_file: str
    final_map_params_file: str
    method: str
    maxfev: int
    fatol: float
    xatol: float


class MCMCConfig(BaseModel):
    save_interval: int
    checkpoint_interval: int
    save_metadata: bool
    final_samples_file_log: str
    final_samples_file_linear: str
    checkpoint_file: str
    metadata_file: str
    initial_cov: List[List[float]]


# Load YAML file
def load_yaml(file_path: str):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


# Validate YAML file against Pydantic model
def load_and_validate_yaml(file_path: str, model: BaseModel):
    data = load_yaml(file_path)
    validated_data = model(**data)
    print(f"{file_path} is valid.")
    return validated_data

#TODO: implement generate settings workflow