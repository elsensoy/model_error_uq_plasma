from pathlib import Path
from pydantic import BaseModel

def resolve_yaml_paths(settings):
    """Recursively resolve placeholders in file paths using `results_dir`."""

    # Ensure results_dir is a Path object
    base_results_dir = Path(settings.results_dir).resolve()
    settings.results_dir = str(base_results_dir)

    print(f"DEBUG: Resolving paths with base directory: {base_results_dir}")

    # Define replacement map
    path_map = {
        "${general.results_dir}": base_results_dir,
        "${postprocess.results_dir}": base_results_dir / "postprocess",
        "${map.results_dir}": base_results_dir / "map",
        "${mcmc.results_dir}": base_results_dir / "mcmc",
        "${plots.results_dir}": base_results_dir / "plots",
        "${ground_truth.results_dir}": base_results_dir / "ground_truth",
    }

    def resolve(value):
        """Recursively replace placeholders in strings, dicts, and lists."""
        if isinstance(value, str):
            for placeholder, resolved_path in path_map.items():
                value = value.replace(placeholder, str(resolved_path))
            return value
        elif isinstance(value, list):
            return [resolve(item) for item in value]
        elif isinstance(value, dict):
            return {k: resolve(v) for k, v in value.items()}
        return value  # Return original value if not str, list, or dict

    # Recursively apply replacements to all sections dynamically
    for section_name, section in vars(settings).items():
        if isinstance(section, BaseModel):  # Ensure it is a Pydantic model
            for key, value in vars(section).items():
                setattr(section, key, resolve(value))

    print("All YAML paths resolved successfully!")
    return settings
