import yaml
from pathlib import Path
from pydantic import ValidationError
from typing import Optional
from config.dict import Settings  # Import the updated Settings model
from hall_opt.utils.parse import load_yaml

def verify_all_yaml(yaml_path: str) -> Optional[Settings]:
    """Verifies, validates, and resolves paths in the provided YAML file."""
    
    yaml_file_path = Path(yaml_path).resolve()  # Resolve absolute path

    if not yaml_file_path.exists():
        print(f"[ERROR] YAML file '{yaml_path}' not found. Exiting...")
        return None

    print(f"\nVerifying configuration file: {yaml_file_path}\n")

    # Load YAML data
    settings_data = load_yaml(yaml_file_path)
    if settings_data is None:
        print("[ERROR] Failed to load YAML file. Exiting...")
        return None

    # Convert raw YAML dict into a Pydantic `Settings` object
    try:
        settings = Settings(**settings_data)
        print("\n YAML file is valid. Proceeding with execution...\n")
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
