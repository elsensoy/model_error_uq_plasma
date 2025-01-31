import yaml
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Any, Optional
from pathlib import Path
from typing_extensions import Annotated  # add annotation 

###TODO DONE: Defaults & Annotations
class ThrusterConfig(BaseModel):
    name: str = Field(..., description="Thruster name")
    geometry: Dict[str, float] = Field(..., description="Geometry dimensions")
    magnetic_field: Dict[str, str] = Field(..., description="Magnetic field file path")


class PostProcessConfig(BaseModel):
    output_file: Dict[str, str] = Field(
        ..., description="Mapping of model type to corresponding output file"
    )
    save_time_resolved: bool = Field(
        default=False, description="Flag to save time-resolved data"
    )
    average_start_time: float = Field(
        default=0.0004, description="Start time for averaging process"
    )

class Config(BaseModel):
    thruster: ThrusterConfig
    discharge_voltage: Annotated[int, Field(ge=0, description="Discharge voltage in V")]
    anode_mass_flow_rate: Annotated[float, Field(gt=0, description="Mass flow rate in kg/s")]
    domain: Annotated[List[float], Field(min_length=2, max_length=2)]
    ncharge: Annotated[int, Field(ge=1, description="Number of charge states")]
    anom_model: Dict[str, Any]
    postprocess: PostProcessConfig 
    


class GeneralSettings(BaseModel):
    results_dir: str = Field(default="../results", description="Base directory for results")
    run_map: bool = Field(default=False, description="Run MAP estimation")
    run_mcmc: bool = Field(default=False, description="Run MCMC sampling")
    plotting: bool = Field(default=False, description="Generate plots")
    ion_velocity_weight: float = Field(default=2.0, description="Weighting factor for ion velocity")
    iterations: Annotated[int, Field(gt=0, description="Number of iterations")]

    # Automatically convert `results_dir` to an absolute path
    def absolute_paths(self):
        self.results_dir = str(Path(self.results_dir).resolve())

class MapConfig(BaseModel):
    map_initial_guess_file: str = Field(..., description="Initial MAP guess file")
    iteration_log_file: str = Field(..., description="MAP iteration log file")
    final_map_params_file: str = Field(..., description="Final MAP parameter file")
    method: str = Field(default="Nelder-Mead", description="MAP optimization method")
    maxfev: int = Field(default=5000, ge=100, description="Maximum function evaluations")
    fatol: float = Field(default=0.003, gt=0, description="Function tolerance")
    xatol: float = Field(default=0.003, gt=0, description="Step size tolerance")


class MCMCConfig(BaseModel):
    save_interval: int = Field(default=10, ge=1, description="MCMC save interval")
    checkpoint_interval: int = Field(default=10, ge=1, description="Checkpoint interval")
    save_metadata: bool = Field(default=True, description="Save metadata flag")
    final_samples_file_log: str = Field(..., description="Final MCMC log file")
    final_samples_file_linear: str = Field(..., description="Final MCMC linear file")
    checkpoint_file: str = Field(..., description="MCMC checkpoint file")
    metadata_file: str = Field(..., description="MCMC metadata file")
    initial_cov: List[List[float]] = Field(..., description="Initial covariance matrix")


class PlottingConfig(BaseModel):
    plots_subdir: str = Field(default="plots-mcmc", description="Directory for plots")
    metrics_subdir: str = Field(default="iteration_metrics", description="Directory for metrics")
    enabled_plots: List[str] = Field(default=["autocorrelation", "trace", "posterior", "metric_plots"],
                                     description="List of enabled plots")


class GroundTruthConfig(BaseModel):
    gen_data: bool = Field(default=False, description="Enable ground truth data generation")

#TODO DONE: error handling
def load_yaml(file_path: str) -> Optional[dict]:
    try:
        with open(file_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"ERROR: YAML file not found: {file_path}")
        return None
    except yaml.YAMLError as e:
        print(f"ERROR: YAML parsing error in {file_path}: {e}")
        return None


def load_and_validate_yaml(data: dict, model: BaseModel, section: str):
    #Validate a specific section
    if section not in data:
        print(f"ERROR: Missing section '{section}' in settings.yaml")
        return None
    try:
        validated_data = model(**data[section])
        print(f"settings.yaml [{section}] is valid.")
        return validated_data
    except ValidationError as e:
        print(f"eRROR: Validation error in settings.yaml [{section}]:\n{e}")
        return None


def verify_all_yaml():
    print("\n Verifying settings.yaml configuration...\n")

    yaml_dir = Path(__file__).resolve().parent
    yaml_path = yaml_dir / "settings.yaml"

    # Load entire YAML file
    settings_data = load_yaml(yaml_path)
    if settings_data is None:
        return None

    #  Validate general settings (Always Required)
    general_settings = load_and_validate_yaml(settings_data, GeneralSettings, "general")
    if general_settings is None:
        print("\n ERROR: General settings validation failed. Exiting...\n")
        return None

    #  Convert `config_settings` into a Pydantic `Config` object
    try:
        config_settings = Config(**settings_data["config_settings"])
    except ValidationError as e:
        print(f"\n ERROR: Validation failed for config_settings:\n{e}")
        return None

    #  Conditionally validate sections based on general settings
    ground_truth = load_and_validate_yaml(settings_data, GroundTruthConfig, "ground_truth") if "ground_truth" in settings_data else None
    map_config = load_and_validate_yaml(settings_data, MapConfig, "map") if general_settings.run_map else None
    mcmc_config = load_and_validate_yaml(settings_data, MCMCConfig, "mcmc") if general_settings.run_mcmc else None
    plotting_config = load_and_validate_yaml(settings_data, PlottingConfig, "plots") if general_settings.plotting else None

    # Final validation check
    if None in [general_settings, config_settings] or (
        ground_truth and ground_truth.gen_data and ground_truth is None
    ) or (
        general_settings.run_map and map_config is None
    ) or (
        general_settings.run_mcmc and mcmc_config is None
    ) or (
        general_settings.plotting and plotting_config is None
    ):
        print("\n ERROR: One or more YAML sections failed validation. Exiting...\n")
        return None

    print("\n All required YAML sections are valid. Proceeding with execution...\n")

    return {
        "general": general_settings,
        "config_settings": config_settings,  
        "ground_truth": ground_truth,
        "map": map_config,
        "mcmc": mcmc_config,
        "plots": plotting_config,
    }


# Extract anomalous transport model(might delete later)
def extract_anom_model(config: Config, model_type: str) -> Dict[str, Any]:
    try:
        anom_model_config = config.anom_model

        if model_type not in anom_model_config:
            raise KeyError(f"Anomalous model type '{model_type}' not found in configuration.")

        model_config = anom_model_config[model_type]

        base_config = config.model_dump()

        base_config["anom_model"] = {**model_config, "type": model_type}

        print(f"Extracted model config for {model_type}")
        return base_config

    except KeyError as e:
        print(f"ERROR: {e}")
        return None
