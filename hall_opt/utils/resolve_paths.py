from pathlib import Path

def resolve_yaml_paths(settings_data: dict):
    """Replaces placeholders like '${general.results_dir}' dynamically in all YAML paths."""

    # Ensure `general.results_dir` exists
    if "general" not in settings_data or "results_dir" not in settings_data["general"]:
        raise KeyError("ERROR: `general.results_dir` missing in settings.yaml")

    results_dir = Path(settings_data["general"]["results_dir"]).resolve()

    def replace_var(value):
        """Replaces placeholders dynamically."""
        if isinstance(value, str):
            value = value.replace("${general.results_dir}", str(results_dir))
            value = value.replace("${ground_truth.results_dir}", str(results_dir / "postprocess"))
            value = value.replace("${postprocess.output_file}", str(results_dir / "postprocess"))
            value = value.replace("${map.results_dir}", str(results_dir / "map"))
            value = value.replace("${mcmc.results_dir}", str(results_dir / "mcmc"))
            value = value.replace("${plots.results_dir}", str(results_dir / "plots"))
        return value

    # Apply replacements to all sections
    for section_name, section_data in settings_data.items():
        if isinstance(section_data, dict):
            for key, value in section_data.items():
                settings_data[section_name][key] = replace_var(value)

    print("All YAML paths resolved successfully!")
    return settings_data
