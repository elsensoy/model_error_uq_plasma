from pydantic import BaseModel, Field, ValidationError, validator
import yaml
from pathlib import Path
from hall_opt.config.loader import Settings, load_yml_settings,extract_anom_model

# Load YAML
yaml_path = Path("//home/elida/Public/users/elsensoy/model_error_uq_plasma/hall_opt/config/settings.yaml")
yaml_data = yaml.safe_load(yaml_path.read_text())

# Create Settings instance
settings = Settings(**yaml_data)

# Access general settings
print(settings.general_settings["results_dir"])

class Settings(BaseModel):
    results_dir: str



def load_and_validate_settings(yaml_path: Path):
    """
    Load YAML file and validate it with the Settings class.
    """
    yml_dict = load_yml_config(yaml_path)
    try:
        settings = Settings(**yml_dict)  # Validate with Pydantic
        return settings
    except ValidationError as e:
        print(f"Validation error:\n{e}")
        raise