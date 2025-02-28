import yaml
from pydantic import BaseModel, Field, model_validator
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import yaml
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

### DONE: Defaults & Annotations

class ThrusterConfig(BaseModel):
    name: str = Field(..., description="Thruster name")
### Thruster Configuration
    geometry: Dict[str, float] = Field(
        default={"channel_length": 0.025, "inner_radius": 0.0345, "outer_radius": 0.05},
        description="Thruster geometry dimensions",
    )
    magnetic_field: Dict[str, str] = Field(
        default={"file": "config/bfield_spt100.csv"}, description="Magnetic field file path"
    )

class TwoZoneBohmModel(BaseModel):
    c1: float = Field(0.00625, description="Coefficient 1 for TwoZoneBohm")
    c2: float = Field(0.0625, description="Coefficient 2 for TwoZoneBohm")

class MultiLogBohmModel(BaseModel):
    zs: List[float] = Field(
        default=[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07], 
        description="Z values for MultiLogBohm"
    )
    cs: List[float] = Field(
        default=[0.02, 0.024, 0.028, 0.033, 0.04, 0.004, 0.004, 0.05], 
        description="C values for MultiLogBohm"
    )

class Config(BaseModel):
    """Main configuration model with optional fields and anomalous model validation."""

    thruster: Optional["ThrusterConfig"] = None
    discharge_voltage: Optional[int] = Field(300, ge=0, description="Discharge voltage in V")
    anode_mass_flow_rate: Optional[float] = Field(5.0e-6, gt=0, description="Mass flow rate in kg/s")
    domain: Optional[List[float]] = Field(default=[0, 0.08], min_length=2, max_length=2)
    anom_model: Optional[Dict[str, Any]] = None

    @model_validator(mode="before")
    @classmethod
    def validate_anom_model(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Ensures `anom_model` contains only `TwoZoneBohm` or `MultiLogBohm` and applies defaults only if missing."""
        
        anom_model = values.get("anom_model")

        if anom_model is None:
            print("DEBUG: No `anom_model` found. Defaulting to `MultiLogBohm`.")
            values["anom_model"] = {"MultiLogBohm": MultiLogBohmModel()} 

        else:
            valid_models = {"TwoZoneBohm", "MultiLogBohm"}
            selected_models = set(anom_model.keys()).intersection(valid_models)

            if len(selected_models) > 1:
                raise ValueError("ERROR: Only one anomalous model (TwoZoneBohm or MultiLogBohm) can be set at a time.")

        return values




### General Settings
class GeneralSettings(BaseModel):
    results_dir: str = "results"
    config_file: str = "config/settings.yaml"
    run_map: bool = False
    run_mcmc: bool = False
    plotting: bool = False
    subsampled: bool = False
    ion_velocity_weight: float = 2.0
    iterations: int = 20


    def absolute_paths(self):
        """Convert results_dir to an absolute path"""
        self.results_dir = str(Path(self.results_dir).resolve())

class PostProcessConfig(BaseModel):
    output_file: Dict[str, str] = Field(
        default={
            "TwoZoneBohm": "results_test/postprocess/output_twozonebohm.json",
            "MultiLogBohm": "results_test/postprocess/output_multilogbohm.json"
        })
    save_time_resolved: bool = Field(
        default=False, description="Flag to save time-resolved data"
    )
    average_start_time: float = Field(
        default=0.0004, description="Start time for averaging process"
    )
    


### Ground Truth Configuration
class GroundTruthConfig(BaseModel):
    gen_data: bool = True
    results_dir: str = "results/ground_truth"
    output_file: str = "results/ground_truth/output_ground_truth.json"

    def absolute_paths(self, results_dir: str):
        base_dir = Path(results_dir) / "ground_truth"
        self.results_dir = str(base_dir)
        self.output_file = str(base_dir / "output_ground_truth.json")


### MCMC Configuration
class MCMCConfig(BaseModel):
    results_dir: str = "results/mcmc"
    base_dir: str = "results/mcmc"
    save_interval: int = 10
    checkpoint_interval: int = 10
    save_metadata: bool = True
    final_samples_file_log: str = "results/mcmc/final_samples_log.csv"
    final_samples_file_linear: str = "results/mcmc/final_samples_linear.csv"
    checkpoint_file: str = "results/mcmc/checkpoint.json"
    metadata_file: str = "results/mcmc/mcmc_metadata.json"
    initial_cov: List[List[float]] = [[0.1, 0.05], [0.05, 0.1]]

    def absolute_paths(self, results_dir: str):
        base_dir = Path(results_dir) / "mcmc"
        self.results_dir = str(base_dir)
        self.base_dir = str(base_dir)


### MAP Configuration
class MapConfig(BaseModel):
    results_dir: str = "results/map"
    base_dir: str = "results/map"
    map_initial_guess_file: str = "results/map/initial_guess.json"
    iteration_log_file: str = "results/map/map_sampling.json"
    final_map_params_file: str = "results/map/final_map_params.json"
    method: str = "Nelder-Mead"
    maxfev: int = 5000
    fatol: float = 0.003
    xatol: float = 0.003


### Plotting Configuration
class PlottingConfig(BaseModel):
    results_dir: str = "results/plots"
    plots_subdir: str = "results/plots"
    metrics_subdir: str = "results/plots/iteration_metrics"
    enabled_plots: List[str] = ["autocorrelation", "trace", "posterior", "pair"]


class Simulation(BaseModel):
    dt: float = 1e-6
    adaptive: bool = Field(default= True, description="Whether to use adaptive time stepping")
    grid: Dict[str, Any] = Field(default={"type": "EvenGrid", "num_cells": 100})
    num_save: int = Field(default=1000, description="Number of save points")
    duration: float = 0.001

        
###  Final Settings Model (Updated)
class Settings(BaseModel):
    results_dir: str = Field(default="results_test", description="Base directory for results")
    gen_data: bool = Field(default=True, description="Enable ground truth data generation")
    general: Optional[GeneralSettings] = Field(default_factory=GeneralSettings)
    config_settings: Config = Field(default_factory=Config)
    postprocess: Optional[PostProcessConfig] = Field(default_factory=PostProcessConfig)
    ground_truth: Optional[GroundTruthConfig] = Field(default_factory=GroundTruthConfig)
    mcmc: Optional[MCMCConfig] = Field(default_factory=MCMCConfig)
    map: Optional[MapConfig] = Field(default_factory=MapConfig)
    plots: Optional[PlottingConfig] = Field(default_factory=PlottingConfig)
    simulation: Optional[Simulation] = Field(default_factory=Simulation)

    def resolve_all_paths(self):
        """Resolve any dynamic paths and placeholders."""
        base_results_dir = Path(self.results_dir).resolve()
        self.results_dir = str(base_results_dir)
        self.general.results_dir = self.results_dir  # Ensure `general` inherits the correct path

        for section in [self.ground_truth, self.mcmc, self.map, self.plots]:
            if section and hasattr(section, "absolute_paths"):
                section.absolute_paths(base_results_dir)

        print("All paths resolved dynamically!")
