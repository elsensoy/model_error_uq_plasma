import yaml
from pydantic import BaseModel, Field, model_validator
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import yaml
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


### Thruster Configuration
class ThrusterConfig(BaseModel):
    geometry: Dict[str, float] = Field(
        default={"channel_length": 0.025, "inner_radius": 0.0345, "outer_radius": 0.05},
        description="Thruster geometry dimensions",
    )
    magnetic_field: Dict[str, str] = Field(
        default={"file": "config/bfield_spt100.csv"}, description="Magnetic field file path"
    )


# Define individual models
class TwoZoneBohmModel(BaseModel):
    c1: float = 0.00625
    c2: float = 0.0625

class MultiLogBohmModel(BaseModel):
    zs: List[float] = Field(default=[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
    cs: List[float] = Field(default=[0.02, 0.024, 0.028, 0.033, 0.04, 0.004, 0.004, 0.05])

#  Main Config (No `parameters` field)
class AnomModelConfig(BaseModel):
    type: str = "MultiLogBohm"  # Default model type
    # c1: float = None  # Only for TwoZoneBohm
    # c2: float = None
    zs: List[float] = None  # Only for MultiLogBohm
    cs: List[float] = None

    @model_validator(mode="before")
    @classmethod
    def set_model_defaults(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Fills in the correct model parameters based on the `type`."""
        model_type = values.get("type", "MultiLogBohm")  # Default if missing

        if model_type == "TwoZoneBohm":
            values.update(TwoZoneBohmModel().model_dump())  # Set TwoZoneBohm parameters
        elif model_type == "MultiLogBohm":
            values.update(MultiLogBohmModel().model_dump())  # Set MultiLogBohm parameters
        else:
            raise ValueError(f" Invalid anom_model type: {model_type}")

        return values


# Handle Defaults for All Rrquired Fields
class ConfigSettings(BaseModel):
    thruster: ThrusterConfig = Field(default_factory=ThrusterConfig)
    anom_model: AnomModelConfig = Field(default_factory=AnomModelConfig)  # Keep Validation
    discharge_voltage: int = Field(300, ge=0)
    anode_mass_flow_rate: float = Field(5.0e-6, gt=0)
    domain: List[float] = Field(default=[0, 0.08], min_length=2, max_length=2)


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


### PostProcess Configuration
class PostProcessConfig(BaseModel):
    output_file: Dict[str, str] = Field(
        default={
            "TwoZoneBohm": "results_test/postprocess/output_twozonebohm.json",
            "MultiLogBohm": "results_test/postprocess/output_multilogbohm.json"
        }
    )


### Ground Truth Configuration
class GroundTruthConfig(BaseModel):
    gen_data: bool = False
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


### Simulation Configuration
class Simulation(BaseModel):
    dt: float = 1e-6
    grid: Dict[str, Any] = Field(default={"type": "EvenGrid", "num_cells": 100})
    duration: float = 0.001

###  Final Settings Model (Updated)
class Settings(BaseModel):
    results_dir: str = Field(default="results_test", description="Base directory for results")
    gen_data: bool = Field(default=False, description="Enable ground truth data generation")
    run_map: bool = Field(default=False, description="Enable map")
    run_mcmc: bool = Field(default=False, description="Enable mcmc")
    general: GeneralSettings = Field(default_factory=GeneralSettings)
    config_settings: ConfigSettings = Field(default_factory=ConfigSettings)
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