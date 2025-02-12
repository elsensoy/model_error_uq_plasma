from pathlib import Path
import yaml

config_file = Path("hall_opt/config/settings.yaml")

try:
    with open(config_file, 'r') as file:
        config_settings = yaml.safe_load(file)
        print("YAML File Loaded Successfully!")
        print(config_settings)
except Exception as e:
    print(f"Error loading YAML: {e}")
