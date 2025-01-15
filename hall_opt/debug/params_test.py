import yaml
from pathlib import Path
from hall_opt.config.loader import Settings, load_yml_settings,extract_anom_model

# Load YAML
yaml_path = Path("/home/elida/Public/users/elsensoy/model_error_uq_plasma/hall_opt/config/settings.yaml")
yaml_data = yaml.safe_load(yaml_path.read_text())

# Create Settings instance
settings = Settings(**yaml_data)

# Access general settings
print(settings.general_settings["results_dir"])

# Access optimization parameters
map_params = settings.optimization_params["map_params"]
print(map_params["method"])

mcmc_params = settings.optimization_params["mcmc_params"]
print(mcmc_params["save_interval"])

#command: python -m hall_opt.debug.params_test --settings hall_opt/config/settings.yaml