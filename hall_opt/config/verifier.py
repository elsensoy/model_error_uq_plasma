import yaml
from pydantic import BaseModel, ValidationError, field_validator
from pathlib import Path
from typing import List, Dict, Any

class ThrusterConfig(BaseModel):
    name: str
    geometry: Dict[str, float]
    magnetic_field: Dict[str, str]

class Config(BaseModel):
    thruster: ThrusterConfig
    discharge_voltage: int
    anode_mass_flow_rate: float
    domain: List[float]
    ncharge: int
    anom_model: Dict[str, Any]
    postprocess: Dict[str, Any]

class GeneralSettings(BaseModel):  
    results_dir: str
    gen_data: bool
    run_map: bool
    run_mcmc: bool
    plotting: bool
    ion_velocity_weight: float
    iterations: int

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

class PlottingConfig(BaseModel):
    plots_subdir: str
    metrics_subdir: str
    enabled_plots: List[str]

# Load YAML with error handling
def load_yaml(file_path: str) -> dict:
    try:
        with open(file_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"ERROR: YAML file not found: {file_path}")
        return None
    except yaml.YAMLError as e:
        print(f"ERROR: YAML parsing error in {file_path}: {e}")
        return None

# Validate YAML using Pydantic
def load_and_validate_yaml(file_path: str, model: BaseModel):
    data = load_yaml(file_path)
    if data is None:
        return None

    # Extract "config" from config.yaml
    if file_path == "config.yaml" and "config" in data:
        data = data["config"]

    try:
        validated_data = model(**data)
        print(f"{file_path} is valid.")
        return validated_data
    except ValidationError as e:
        print(f"ERROR: Validation error in {file_path}:\n{e}")
        return None
    

	
def verify_all_yaml():
    print("\nVerifying all YAML configuration files...\n")

    # Ensure we correctly set the `config` directory
    yaml_dir = Path(__file__).resolve().parent  # This is `hall_opt/config/`

    # Explicitly define full paths for each YAML file
    settings_path = yaml_dir / "settings.yaml"
    config_path = yaml_dir / "config.yaml"
    map_path = yaml_dir / "map.yaml"
    mcmc_path = yaml_dir / "mcmc.yaml"
    plotting_path = yaml_dir / "plotting.yaml"

    # Validate each YAML file using the correct paths
    settings_data = load_and_validate_yaml(settings_path, GeneralSettings)
    config = load_and_validate_yaml(config_path, Config)
    map_config = load_and_validate_yaml(map_path, MapConfig)
    mcmc_config = load_and_validate_yaml(mcmc_path, MCMCConfig)
    plotting_config = load_and_validate_yaml(plotting_path, PlottingConfig)

    # Check if any YAML file failed validation
    if None in [settings_data, config, map_config, mcmc_config, plotting_config]:
        print("\nERROR: One or more YAML files failed validation. Exiting...\n")
        return None

    print("\n All YAML files are valid. Proceeding with execution...\n")

    return settings_data

# Workflow logic
def process_workflow(settings: GeneralSettings):
    results_dir = Path(settings.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if settings.gen_data:
        print("Generating data... (To be implemented)")

    if settings.run_map:
        print("Running MAP configuration...")
        map_config = load_and_validate_yaml("map.yaml", MapConfig)
        if map_config:
            print("MAP configuration validated.")

    if settings.run_mcmc:
        print("Running MCMC configuration...")
        mcmc_config = load_and_validate_yaml("mcmc.yaml", MCMCConfig)
        if mcmc_config:
            print("MCMC configuration validated.")

    if settings.plotting:
        print("Generating plots... (To be implemented)")

    print(f"Workflow completed with {settings.iterations} iterations.")

# Extract anomalous transport model
def extract_anom_model(config: Config, model_type: str) -> Dict[str, Any]:
    try:
        anom_model_config = config.anom_model

        if model_type not in anom_model_config:
            raise KeyError(f"Anomalous model type '{model_type}' not found in configuration.")

        model_config = anom_model_config[model_type]

        # Merge base config with the selected model
        base_config = config.model_dump()

        base_config["anom_model"] = {**model_config, "type": model_type}

        print(f"Extracted model config for {model_type}")
        return base_config

    except KeyError as e:
        print(f"ERROR: {e}")
        return None

# Main execution for testing 
# if __name__ == "__main__":
#     settings = verify_all_yaml()
#     if settings:
#         process_workflow(settings)
