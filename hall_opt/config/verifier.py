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

class Simulation(BaseModel):
    dt: float = Field(..., description="Time step size")
    adaptive: bool = Field(..., description="Whether to use adaptive time stepping")
    grid: Dict[str, Any] = Field(..., description="Grid configuration")
    num_save: int = Field(..., description="Number of save points")
    duration: float = Field(..., description="Simulation duration")

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
    plots_subdir: str = Field(default="ground_truth_plots", description="Directory for plots")
    metrics_subdir: str = Field(default="iteration_metrics", description="Directory for metrics")
    enabled_plots: List[str] = Field(default=["ground_truth_plots"],
                                     description="List of enabled plots")


class GroundTruthConfig(BaseModel):
    gen_data: bool = Field(default=False, description="Enable ground truth data generation")

class Settings(BaseModel):
    general: GeneralSettings
    config_settings: Config
    simulation: Simulation
    postprocess: PostProcessConfig
    ground_truth: Optional[GroundTruthConfig]
    map: Optional[MapConfig]
    mcmc: Optional[MCMCConfig]
    plots: Optional[PlottingConfig]
    
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
    try:
        settings = Settings(**yaml_data)
        print("\n YAML settings successfully loaded and validated!\n")
        return settings
    except ValidationError as e:
        print(f"\n ERROR: YAML validation failed:\n{e}")
        return None
    
def verify_all_yaml() -> Optional[Settings]:
    """Loads and validates the entire settings.yaml using the unified Settings class."""
    print("\n Verifying settings.yaml configuration...\n")

    yaml_path = Path(__file__).resolve().parent / "settings.yaml"

    # Load entire YAML file
    settings_data = load_yaml(yaml_path)
    
    if settings_data is None:
        print("\nERROR: Failed to load settings.yaml. Exiting...\n")
        return None

    # Validate Required Sections (General, Config, PostProcess)
    try:
        settings = Settings(**settings_data)
        print("\nsettings.yaml is valid. Proceeding with execution...\n")
    except ValidationError as e:
        print(f"\nERROR: Validation failed for settings.yaml:\n{e}")
        return None

   
    class_mapping = {
        "ground_truth": "GroundTruthConfig",
        "map": "MapConfig",
        "mcmc": "MCMCConfig",
        "plots": "PlottingConfig",
    }

    # Required Sections - Always Validate
    required_sections = {
        "general": settings.general,
        "config_settings": settings.config_settings,
        "postprocess": settings.postprocess,
        "simulation": settings.simulation
    }

    for section, data in required_sections.items():
        if data is None:
            print(f"\nERROR: Missing required section '{section}' in settings.yaml\n")
            return None
        print(f"settings.yaml [{section}] is valid.")

    # Optional Sections - Validate Only If Enabled
    optional_sections = {
        "ground_truth": settings.ground_truth.gen_data if settings.ground_truth else False,
        "map": settings.general.run_map,
        "mcmc": settings.general.run_mcmc,
        "plots": settings.general.plotting,
    }

    for section, flag in optional_sections.items():
        if flag:
            if section not in settings_data:
                print(f"\nERROR: Missing required section '{section}' in settings.yaml\n")
                return None  # Exit if a required section is missing

            try:
                section_class = class_mapping.get(section, f"{section.capitalize()}Config")
                setattr(settings, section, eval(section_class)(**settings_data[section]))
                print(f"settings.yaml [{section}] is valid.")
            except ValidationError as e:
                print(f"\nERROR: Validation failed for {section}:\n{e}")
                return None

    return settings  # Return the validated settings object

def extract_anom_model(settings: Settings, model_type: str) -> Dict[str, Any]:
    """Extracts the anomalous model configuration for a given model type."""
    try:
        anom_model_config = settings.config_settings.anom_model
        if model_type not in anom_model_config:
            raise KeyError(f" ERROR: Anomalous model type '{model_type}' not found.")

        base_config = settings.config_settings.dict()
        base_config["anom_model"] = {**anom_model_config[model_type], "type": model_type}

        print(f" Extracted model config for {model_type}")
        return base_config

    except KeyError as e:
        print(f" ERROR: {e}")
        return None
