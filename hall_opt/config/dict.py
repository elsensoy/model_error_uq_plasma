import yaml
from pydantic import BaseModel, Field, model_validator
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import yaml
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import re


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
    thruster: Optional[ThrusterConfig] = None  # Defaulted earlier
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

class PostProcessConfig(BaseModel):
    output_file: Dict[str, str] = Field(
        default={
            "TwoZoneBohm": "${postprocess.results_dir}/output_twozonebohm.json",
            "MultiLogBohm": "${postprocess.results_dir}/output_multilogbohm.json"
        }
    )
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
    results_dir: str = "${ground_truth.results_dir}"
    output_file: str = "${postprocess.results_dir}/output_multilogbohm.json"

### MCMC Configuration
class MCMCConfig(BaseModel):
    output_dir: str = "${settings.output_dir}"
    base_dir: str ="${settings.output_dir}/mcmc"
    burn_in: int = Field(default = 50, description= "Discarded Iterations to Minimize Initial Bias")
    save_interval: int = 10
    checkpoint_interval: int = 10
    save_metadata: bool = True
    initial_data: str = "${map.final_map_params_file}"
    final_samples_file_log: str = "${mcmc.results_dir}/final_samples_log.csv"
    final_samples_file_linear: str = "${mcmc.results_dir}/final_samples_linear.csv"
    checkpoint_file: str = "${mcmc.results_dir}/checkpoint.json"
    metadata_file: str = "${mcmc.results_dir}/mcmc_metadata.json"
    initial_cov: List[List[float]] = [[0.1, 0.05], [0.05, 0.1]]
    max_iter: int =  Field(default= 100, description="Maximum iteration count" )


class MapConfig(BaseModel):
    """Configuration for MAP optimization settings."""
    
    output_dir: str = Field(default="${settings.output_dir}", description="Directory for MAP results")
    base_dir: str = Field(default_factory=lambda: "${settings.output_dir}", description="Directory for Iterations")
    initial_guess: List[float] = Field(default=[-0.2, 0.5], description="Initial guess parameters for TwoZoneBohm model") 
    iteration_log_file: str = Field(default_factory=lambda: "${settings.output_dir}/map_sampling.json", description="MAP iteration log file")
    final_map_params_file: str = Field(default_factory=lambda: "${settings.output_dir}/final_map_params.json", description="Final MAP parameter file")
    method: str = Field(default="Nelder-Mead", description="MAP optimization method")
    maxfev: int = Field(default=5000, ge=100, description="Maximum function evaluations")
    fatol: float = Field(default=0.003, gt=0, description="Function tolerance")
    xatol: float = Field(default=0.003, gt=0, description="Step size tolerance")
    max_iter: int =  Field(default= 100, description="Maximum iteration count" )

class MapInputSettings(BaseModel):
    initial_cs: List[float]
    max_iter: Optional[int] = 100
    algorithm: Optional[str] = "slsqp"


class PlottingConfig(BaseModel):
    results_dir: str = "${plots.results_dir}"
    plots_subdir: str = "${plots.results_dir}"
    metrics_subdir: str = "${plots.results_dir}/iteration_metrics"
    enabled_plots: List[str] = ["autocorrelation", "trace", "posterior", "pair"]


class Simulation(BaseModel):
    dt: float = 1e-6
    adaptive: bool = Field(default= True, description="Whether to use adaptive time stepping")
    grid: Dict[str, Any] = Field(default={"type": "EvenGrid", "num_cells": 100})
    num_save: int = Field(default=1000, description="Number of save points")
    duration: float = 0.001

# ---------------------------------------
# Master Settings Model
# ---------------------------------------

class Settings(BaseModel):
    results_dir: str = "results_test"
    output_dir: str = "results_test"
    gen_data: bool = False
    run_map: bool = False
    plotting: bool = False
    run_mcmc: bool = False
    reference_data: Optional[str] = "${ground_truth.output_file}"
    map_settings: Optional[MapInputSettings] = None
    general: Optional[GeneralSettings] = Field(default_factory=GeneralSettings)
    config_settings: Config = Field(default_factory=Config)
    postprocess: Optional[PostProcessConfig] = Field(default_factory=PostProcessConfig)
    ground_truth: Optional[GroundTruthConfig] = Field(default_factory=GroundTruthConfig)
    mcmc: Optional[MCMCConfig] = Field(default_factory=MCMCConfig)
    map: Optional[MapConfig] = Field(default_factory=MapConfig)
    plots: Optional[PlottingConfig] = Field(default_factory=PlottingConfig)
    simulation: Optional[Simulation] = Field(default_factory=Simulation)

    def resolve_all_paths(self, config_file: Optional[str] = None):
        if not self.output_dir:
            print("[INFO] No 'output_dir' provided in YAML. Using default: 'results_test'")
            self.output_dir = "results_test"

        base = Path(self.output_dir).resolve()
        self.output_dir = str(base)
        self.results_dir = str(base)

        if self.general:
            self.general.config_file = str(config_file or "unknown.yaml")

        # Define path substitutions
        path_map = {
            "${settings.output_dir}": base,
            "${postprocess.results_dir}": base / "postprocess",
            "${map.results_dir}": base / "map",
            "${mcmc.results_dir}": base / "mcmc",
            "${plots.results_dir}": base / "plots",
             "${map.optimization_output}": base / "map" / "optimization_result.json", 
            "${map.final_map_params_file}": Path(self.map.final_map_params_file),
            "${ground_truth.results_dir}": base / "ground_truth",
            "${ground_truth.output_file}": self.ground_truth.output_file
        }

        if self.reference_data:
            self.reference_data, _ = resolve_placeholders(self.reference_data, path_map)

        # Collect used placeholders
        used_keys = set()

        for section in vars(self).values():
            if isinstance(section, BaseModel):
                for key, val in vars(section).items():
                    resolved_val, section_used = resolve_placeholders(val, path_map, used_keys)
                    setattr(section, key, resolved_val)
                    used_keys.update(section_used)

        print(f"[INFO] All paths resolved relative to: {base}")


# ---------------------------------------
# Helper Function for Placeholder Resolution
# ---------------------------------------
def resolve_placeholders(value, path_map, used_keys=None):
    import re
    if used_keys is None:
        used_keys = set()
    pattern = re.compile(r"\$\{(.+?)\}")

    def _resolve_str(s):
        matches = pattern.findall(s)
        for match in matches:
            key = f"${{{match}}}"
            if key not in path_map:
                raise ValueError(f"[PLACEHOLDER ERROR] Undefined placeholder '{key}'")
            s = s.replace(key, str(path_map[key]))
            used_keys.add(key)
        return s

    if isinstance(value, str):
        return _resolve_str(value), used_keys
    elif isinstance(value, list):
        result = []
        for v in value:
            r, used_keys = resolve_placeholders(v, path_map, used_keys)
            result.append(r)
        return result, used_keys
    elif isinstance(value, dict):
        result = {}
        for k, v in value.items():
            r, used_keys = resolve_placeholders(v, path_map, used_keys)
            result[k] = r
        return result, used_keys
    return value, used_keys
