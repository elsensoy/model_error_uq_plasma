import yaml
from pydantic import BaseModel, Field, model_validator
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import yaml
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ThrusterConfig(BaseModel):
    name: str = Field("SPT-100", description="Thruster name")

    geometry: Dict[str, float] = Field(
        default={"channel_length": 0.025, "inner_radius": 0.0345, "outer_radius": 0.05},
        description="Thruster geometry dimensions",
    )
    magnetic_field: Dict[str, str] = Field(
        default={"file": "hall_opt/config/bfield_spt100.csv"}, description="Magnetic field file path"
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

#defaulting to classtwozonebohm parameter values if config isnt provided
class Config(BaseModel):
    """Main configuration model with default settings from `dict.py` and optional user overrides."""
    thruster: Optional[ThrusterConfig] = None  # Defaulted later in validator
    discharge_voltage: Optional[int] = Field(300, ge=0, description="Discharge voltage in V")
    anode_mass_flow_rate: Optional[float] = Field(5.0e-6, gt=0, description="Mass flow rate in kg/s")
    domain: Optional[List[float]] = Field(default=[0, 0.08], min_length=2, max_length=2)
    anom_model: Optional[Dict[str, Any]] = None  # Allows user override

    @model_validator(mode="before")
    @classmethod
    def validate_config(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Ensures `anom_model` and `thruster` have default values unless overridden."""
        
        # Ensure anom_model defaults exist
        anom_model = values.get("anom_model", {})
        if "MultiLogBohm" not in anom_model:
            print("DEBUG: Adding default `MultiLogBohm` model...")
            anom_model["MultiLogBohm"] = MultiLogBohmModel().model_dump()

        if "TwoZoneBohm" not in anom_model:
            print("DEBUG: Adding default `TwoZoneBohm` model...")
            anom_model["TwoZoneBohm"] = TwoZoneBohmModel().model_dump()

        values["anom_model"] = anom_model

        #  `thruster` is never `None` (Fix for `deserialize` error)
        if values.get("thruster") is None:
            print("DEBUG: Adding default `ThrusterConfig`...")
            values["thruster"] = ThrusterConfig().model_dump()

        return values



### General Settings
class GeneralSettings(BaseModel):
    results_dir: str = "results"
    config_file: str = "config/settings.yaml"
    run_map: bool = False
    gen_data: bool = False
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
    gen_data: bool = Field(
        default=True, description="Flag to enable gen_data"
    )
    results_dir: str = "results/ground_truth"
    output_file: str = "results/ground_truth/output_ground_truth.json"

    def absolute_paths(self, results_dir: str):
        base_dir = Path(results_dir) / "ground_truth"
        self.results_dir = str(base_dir)
        self.output_file = str(base_dir / "output_ground_truth.json")


### MCMC Configuration
class MCMCConfig(BaseModel):
    output_dir: str = "results_test/mcmc"
    base_dir: str = "results_test/mcmc"
    burn_in: int = Field(default = 50, description= "Discarded Iterations to Minimize Initial Bias")
    save_interval: int = 10
    checkpoint_interval: int = 10
    save_metadata: bool = True
    reference_data: str = "results_test/map/final_map_params.json"
    final_samples_file_log: str = "results_test/mcmc/final_samples_log.csv"
    final_samples_file_linear: str = "results_test/mcmc/final_samples_linear.csv"
    checkpoint_file: str = "results_test/mcmc/checkpoint.json"
    metadata_file: str = "results_test/mcmc/mcmc_metadata.json"
    initial_cov: List[List[float]] = [[0.1, 0.05], [0.05, 0.1]]
    max_iter: int =  Field(default= 100, description="Maximum iteration count" )

    def absolute_paths(self, results_dir: str):
        base_dir = Path(results_dir) / "mcmc"
        self.results_dir = str(base_dir)
        self.base_dir = str(base_dir)


class MapConfig(BaseModel):
    """Configuration for MAP optimization settings."""
    
    output_dir: str = Field(default="results_test/map", description="Directory for MAP results")
    base_dir: str = Field(default_factory=lambda: "results_test/map", description="Directory for Iterations")
    initial_guess: List[float] = Field(default=[-0.2, 0.5], description="Initial guess parameters for TwoZoneBohm model") 
    iteration_log_file: str = Field(default_factory=lambda: "results_test/map/map_sampling.json", description="MAP iteration log file")
    final_map_params_file: str = Field(default_factory=lambda: "results_test/map/final_map_params.json", description="Final MAP parameter file")
    method: str = Field(default="Nelder-Mead", description="MAP optimization method")
    maxfev: int = Field(default=5000, ge=100, description="Maximum function evaluations")
    fatol: float = Field(default=0.003, gt=0, description="Function tolerance")
    xatol: float = Field(default=0.003, gt=0, description="Step size tolerance")
    max_iter: int =  Field(default= 100, description="Maximum iteration count" )

    @model_validator(mode="before")
    @classmethod
    def resolve_file_paths(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Dynamically resolve file paths based on `base_dir`."""
        base_dir = values.get("base_dir", "results_test/map")
        values["iteration_log_file"] = f"{base_dir}/map_sampling.json"
        values["final_map_params_file"] = f"{base_dir}/final_map_params.json"
        return values

### Plotting Configuration
class PlottingConfig(BaseModel):
    results_dir: str = "results_test/plots"
    plots_subdir: str = "results_test/plots"
    metrics_subdir: str = "results_test/plots/iteration_metrics"
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
    output_dir: str = Field(default="sub_dir", description="method result directory for testing == results_test/[method_dir]")
    gen_data: bool = Field(default=False, description="Enable ground truth data generation")
    run_map: bool =  Field(default=False, description="Enable MAP optimization")
    plotting: bool =  Field(default=False, description="Enable Plots Generation")
    run_mcmc: bool = Field(default=False, description="Enable MCMC Sampling")
    reference_data: str = Field(default="results_test/map/final_map_params.json", description="initial point for mcmc sampling")
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
        self.general.results_dir = self.results_dir  

        for section in [self.ground_truth, self.mcmc, self.map, self.plots]:
            if section and hasattr(section, "absolute_paths"):
                section.absolute_paths(base_results_dir)

        print("All paths resolved dynamically!")