import yaml
from pathlib import Path
from pydantic import ValidationError
from typing import Optional
from config.dict import Settings  # Import the updated Settings model


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


def verify_all_yaml(yaml_data: dict) -> Optional[Settings]:
    """Verifies, validates, and resolves paths in `settings.yaml`."""

    print("\nVerifying configuration...\n")

    try:
        settings = Settings(**yaml_data)  # Validate & create settings object
        print("[INFO] Configuration successfully loaded and verified!")
    except ValidationError as e:
        print(f"\n[ERROR] Validation failed for YAML configuration:\n{e}")
        return None

    # Debug: Print `results_dir` before resolving
    print(f"DEBUG: Loaded `results_dir`: {settings.results_dir}")

    print(f"\nDEBUG: Before resolving:")
    print(f"  general.results_dir: {settings.general.results_dir}")
    print(f"  mcmc.results_dir: {settings.mcmc.output_dir}")
    print(f"  mcmc.base_dir: {settings.mcmc.base_dir}")

    # Ensure required directories exist
    Path(settings.general.results_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.ground_truth.results_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.mcmc.output_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.map.output_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.mcmc.base_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.map.base_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.plots.plots_subdir).mkdir(parents=True, exist_ok=True)

    return settings