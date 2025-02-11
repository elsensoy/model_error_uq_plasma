import yaml
from pathlib import Path
from pydantic import ValidationError
from typing import Optional
from hall_opt.utils.resolve_paths import resolve_yaml_paths
from hall_opt.config.dict import Settings  #  Import Settings class

def load_yaml(file_path: str) -> Optional[dict]:
    """Loads YAML configuration file safely."""
    try:
        with open(file_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f" ERROR: YAML file not found: {file_path}")
        return None
    except yaml.YAMLError as e:
        print(f" ERROR: YAML parsing error in {file_path}: {e}")
        return None
def verify_all_yaml() -> Optional[Settings]:
    """Verifies, validates, and resolves paths in `settings.yaml`."""
    print("\n Verifying settings.yaml configuration...\n")

    yaml_path = Path(__file__).resolve().parent / "settings.yaml"

    # Load YAML data
    settings_data = load_yaml(yaml_path)
    if settings_data is None:
        print("ERROR: Failed to load settings.yaml. Exiting...")
        return None

    # Convert raw YAML dict into a Pydantic `Settings` object first
    try:
        settings = Settings(**settings_data)
        print("\n settings.yaml is valid. Proceeding with execution...\n")
    except ValidationError as e:
        print(f"\nERROR: Validation failed for settings.yaml:\n{e}")
        return None

    # Debug: Print `general.results_dir` before resolving
    print(f"DEBUG: Loaded `general.results_dir`: {settings.general.results_dir}")

    print(f"\nDEBUG: before resolving:")
    print(f"  general.results_dir: {settings.general.results_dir}")
    print(f"  mcmc.results_dir: {settings.mcmc.results_dir}")
    print(f"  mcmc.base_dir: {settings.mcmc.base_dir}")

    # Resolve placeholders in YAML paths using Pydantic object
    settings = resolve_yaml_paths(settings)

    print(f"\nDEBUG: After resolving:")
    print(f"  general.results_dir: {settings.general.results_dir}")
    print(f"  mcmc.results_dir: {settings.mcmc.results_dir}")
    print(f"  mcmc.base_dir: {settings.mcmc.base_dir}")

    # Debug: Print `general.results_dir` after resolving
    print(f" Results directory resolved to: {settings.general.results_dir}")

    # Ensure directories exist
    Path(settings.general.results_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.ground_truth.results_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.mcmc.results_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.map.results_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.mcmc.base_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.map.base_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.plots.plots_subdir).mkdir(parents=True, exist_ok=True)

    return settings
