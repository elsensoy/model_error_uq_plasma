import sys
import argparse
from pathlib import Path
import os
import yaml

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run different methods of the project.")
    parser.add_argument("method_yaml", type=str, help="Path to a YAML configuration file (in root or config/).")
    return parser.parse_args()

def get_yaml_path(method_yaml):
    """
    Determines the correct path for the YAML file.
    - If `method_yaml` exists in the root directory, use it.
    - Otherwise, look for it inside the `config/` directory.
    """
    yaml_file = Path(method_yaml)

    # Case 1: If the file exists in the root directory, use it directly
    if yaml_file.exists():
        return yaml_file

    # Case 2: Otherwise, assume it's inside `config/`
    config_yaml_file = Path("hall_opt/config") / yaml_file
    if config_yaml_file.exists():
        return config_yaml_file

    # If neither file exists, print an error and exit
    print(f"ERROR: '{method_yaml}' not found in the root directory or 'config/' folder.")
    sys.exit(1)

def load_yaml(file_path):
    """Load the YAML file and return its contents as a dictionary."""
    try:
        with open(file_path, "r") as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"ERROR: Failed to load YAML file {file_path}: {e}")
        sys.exit(1)
