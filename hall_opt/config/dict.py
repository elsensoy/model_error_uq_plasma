import yaml
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Any, Optional
from pathlib import Path
from hall_opt.utils.resolve_paths import resolve_yaml_paths
from typing_extensions import Annotated  # add annotation 

### DONE: Defaults & Annotations

class ThrusterConfig(BaseModel):
    name: str = Field(..., description="Thruster name")
    geometry: Dict[str, float] = Field(..., description="Geometry dimensions")
    magnetic_field: Dict[str, str] = Field(..., description="Magnetic field file path")

class PostProcessConfig(BaseModel):
    #output_file: Dict[str, str] = Field(default="hall_opt/results/output_multilogbohm", description="Mapping of model type to corresponding output file")
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
    results_dir: str = Field(default="hall_opt/results", description="Base directory for results")
    run_map: bool = Field(default=False, description="Run MAP estimation")
    run_mcmc: bool = Field(default=False, description="Run MCMC sampling")
    plotting: bool = Field(default=False, description="Generate plots")
    ion_velocity_weight: float = Field(default=2.0, description="Weighting factor for ion velocity")
    iterations: Annotated[int, Field(gt=0, description="Number of iterations")]

    # Automatically convert `results_dir` to an absolute path
    def absolute_paths(self):
        self.results_dir = str(Path(self.results_dir).resolve())

class MapConfig(BaseModel):
    results_dir: str = Field(default="", description="Directory for MAP results")  # 
    base_dir: str = Field(default="", description="Directory for Iterations")
    map_initial_guess_file: str = Field(..., description="Initial MAP guess file")
    iteration_log_file: str = Field(..., description="MAP iteration log file")
    final_map_params_file: str = Field(..., description="Final MAP parameter file")
    method: str = Field(default="Nelder-Mead", description="MAP optimization method")
    maxfev: int = Field(default=5000, ge=100, description="Maximum function evaluations")
    fatol: float = Field(default=0.003, gt=0, description="Function tolerance")
    xatol: float = Field(default=0.003, gt=0, description="Step size tolerance")

    def absolute_paths(self, results_dir: str):
        """Convert paths to absolute paths based on results_dir."""
        if not self.map_results_dir:
            self.map_results_dir = str(Path(results_dir) / "map")  
        base_dir = Path(self.map_results_dir).resolve()
        self.map_results_dir = str(base_dir)
        self.base_dir = str(base_dir)
        self.map_initial_guess_file = str(base_dir / self.map_initial_guess_file)
        self.iteration_log_file = str(base_dir / self.iteration_log_file)
        self.final_map_params_file = str(base_dir / self.final_map_params_file)

class MCMCConfig(BaseModel):
    mcmc_results_dir: str = Field(default="", description="Directory for MCMC results")  #  Add default
    base_dir: str = Field(default="", description="Directory for Iterations")
    save_interval: int = Field(default=10, ge=1, description="MCMC save interval")
    checkpoint_interval: int = Field(default=10, ge=1, description="Checkpoint interval")
    save_metadata: bool = Field(default=True, description="Save metadata flag")
    final_samples_file_log: str = Field(..., description="Final MCMC log file")
    final_samples_file_linear: str = Field(..., description="Final MCMC linear file")
    checkpoint_file: str = Field(..., description="MCMC checkpoint file")
    metadata_file: str = Field(..., description="MCMC metadata file")
    initial_cov: List[List[float]] = Field(..., description="Initial covariance matrix")

    def absolute_paths(self, results_dir: str):
        """Convert paths to absolute paths based on results_dir."""
        if not self.mcmc_results_dir:
            self.mcmc_results_dir = str(Path(results_dir) / "mcmc")  
        base_dir = Path(self.mcmc_results_dir).resolve()
        self.mcmc_results_dir = str(base_dir)
        self.base_dir = str(base_dir)
        self.final_samples_file_log = str(base_dir / self.final_samples_file_log)
        self.final_samples_file_linear = str(base_dir / self.final_samples_file_linear)
        self.checkpoint_file = str(base_dir / self.checkpoint_file)
        self.metadata_file = str(base_dir / self.metadata_file)

class PlottingConfig(BaseModel):
    plots_subdir: str = Field(default="ground_truth_plots", description="Directory for plots")
    metrics_subdir: str = Field(default="iteration_metrics", description="Directory for metrics")
    enabled_plots: List[str] = Field(default=["ground_truth_plots"],
                                     description="List of enabled plots")

    def absolute_paths(self, results_dir: str):
        """Convert plot paths to absolute paths based on results_dir."""
        self.plots_subdir = str(Path(results_dir) / self.plots_subdir)
        self.metrics_subdir = str(Path(results_dir) / self.metrics_subdir)

class GroundTruthConfig(BaseModel):
    gen_data: bool = Field(default=False, description="Enable ground truth data generation")
    results_dir: str = Field(default="", description="Directory for ground truth output")
    output_file: str = Field(default="", description="Ground truth output file")

    def absolute_paths(self, results_dir: str):
        """ paths are placed inside the results directory."""
        if not self.results_dir:
            self.results_dir = str(Path(results_dir) / "ground_truth")  #  Set default if missing

        base_dir = Path(self.results_dir).resolve()
        self.results_dir = str(base_dir)
        self.output_file = str(base_dir / self.output_file)
        
class Settings(BaseModel):
    general: GeneralSettings
    config_settings: Config
    simulation: Simulation
    postprocess: PostProcessConfig
    ground_truth: Optional[GroundTruthConfig]
    map: Optional[MapConfig]
    mcmc: Optional[MCMCConfig]
    plots: Optional[PlottingConfig]

    def resolve_all_paths(self):
        base_results_dir = Path(self.general.results_dir).resolve()
        self.general.results_dir = str(base_results_dir)

        print(f" Resolving paths based on base directory: {base_results_dir}")

        # Apply `absolute_paths()` dynamically
        for section_name, section in self.__dict__.items():
            if hasattr(section, "absolute_paths"):
                print(f"🔹 Resolving paths for section: {section_name}")
                section.absolute_paths(base_results_dir)

        print(" All paths resolved dynamically!")