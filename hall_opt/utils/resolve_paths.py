from pathlib import Path
from pydantic import BaseModel
project_root = Path(__file__).resolve().parent
def resolve_yaml_paths(settings):
    """Recursively resolve file paths inside settings using `general.results_dir`."""

    if not hasattr(settings, "general") or not hasattr(settings.general, "results_dir"):
        raise KeyError("ERROR: `general.results_dir` missing in settings.yaml")

    results_dir = Path(settings.general.results_dir).resolve()
    print(f"DEBUG: Resolving paths with base directory: {results_dir}")

    def resolve(value):
        """Replace placeholders dynamically in string paths."""
        if isinstance(value, str):
            return (
                value.replace("${general.results_dir}", str(results_dir))
                     .replace("${ground_truth.results_dir}", str(results_dir / "postprocess"))
                     .replace("${postprocess.output_file}", str(results_dir / "postprocess"))
                     .replace("${map.results_dir}", str(results_dir / "map"))
                     .replace("${mcmc.results_dir}", str(results_dir / "mcmc"))
                     .replace("${plots.results_dir}", str(results_dir / "plots"))
            )
        return value

    # Apply replacements to all sections dynamically
    for section_name, section in vars(settings).items():
        if isinstance(section, BaseModel):  #  it should be a Pydantic model
            for key, value in vars(section).items():
                if isinstance(value, str):
                    setattr(section, key, resolve(value))
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        section.__dict__[key][sub_key] = resolve(sub_value)

    print(" All YAML paths resolved successfully!")
    return settings