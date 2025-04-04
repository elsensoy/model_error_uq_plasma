import yaml
import re
from pathlib import Path
from pydantic import ValidationError,BaseModel
from typing import List, Dict, Any, Optional, Union
from config.dict import Settings  # Import the updated Settings model\
def load_yaml(file_path: str) -> Optional[dict]:
    """Loads YAML configuration file safely."""
    try:
        with open(file_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"ERROR: YAML file not found: {file_path}")
        return None
    except yaml.YAMLError as e:
        print(f"ERROR: YAML parsing error in {file_path}: {e}")
        return None
    

def verify_all_yaml(yaml_data: dict, source_path: Optional[str] = None) -> Optional[Settings]:
    from hall_opt.config.dict import Settings  # update path as needed

    try:
        settings = Settings(**yaml_data)

        # Set the source file path in general.config_file
        if settings.general:
            settings.general.config_file = str(source_path or "unknown.yaml")

        # Resolve all placeholders based on current settings
        settings.resolve_all_paths(config_file=source_path)

        return settings

    except ValidationError as ve:
        print(f"[ERROR] YAML validation failed:\n{ve}")
        return None
    except ValueError as ve:
        print(f"[ERROR] Placeholder resolution failed:\n{ve}")
        return None
    


def get_valid_optimization_method(
    method: Optional[str],
    source_yaml: Optional[str] = None
) -> str:
    # supported_methods = {"Nelder-Mead", "BFGS", "Powell", "L-BFGS-B", "TNC", "SLSQP"}  # support future expansion. only neldermead is implemented so far
    supported_methods = {"Nelder-Mead"} 
    if method is None:
        print(f"[INFO] No method specified in YAML ({source_yaml or 'unknown.yaml'}). Using default: 'Nelder-Mead'")
        return "Nelder-Mead"

    method_upper = method.upper()
    if method_upper in (m.upper() for m in supported_methods):
        print(f"[INFO] Using optimization method '{method}' from {source_yaml or 'unknown.yaml'}")
        return method

    print(f"[WARNING] Unsupported method '{method}' in {source_yaml or 'unknown.yaml'}. Falling back to 'Nelder-Mead'")
    return "Nelder-Mead"


def extract_anom_model(settings: Settings, model_type: str) -> Dict[str, Any]:
    """Extracts the anomalous model configuration for a given model type."""
    try:
        anom_model_config = settings.config_settings.anom_model
        if model_type not in anom_model_config:
            raise KeyError(f" ERROR: Anomalous model type '{model_type}' not found.")

        base_config = settings.config_settings.model_dump()
        base_config["anom_model"] = {**anom_model_config[model_type], "type": model_type}

        return base_config

    except KeyError as e:
        print(f" ERROR: {e}")
        return 


