
# import os
# import sys
# from pathlib import Path

# # -----------------------------
# # Automatically find the project root
# # -----------------------------
# current_file = Path(__file__).resolve()
# project_root = current_file.parent  # Moves up to `model_error_uq_plasma/`

# # Ensure project root is in sys.path
# if str(project_root) not in sys.path:
#     sys.path.insert(0, str(project_root))

# # Ensure the hall_opt directory is also in sys.path
# hall_opt_path = project_root / "hall_opt"
# if str(hall_opt_path) not in sys.path:
#     sys.path.insert(0, str(hall_opt_path))

# # Debugging: Print sys.path to verify paths
# print(f"Project root: {project_root}")
# print(f"sys.path: {sys.path}")

# # -----------------------------
# # Import and run the main script
# # -----------------------------
# try:
#     from hall_opt.main import main  # Now the import should work everywhere
#     if __name__ == "__main__":
#         main()
# except ModuleNotFoundError as e:
#     print(f"Import Error: {e}")
#     print("Make sure 'hall_opt' has an __init__.py file and is correctly structured.")

import os
import sys
import yaml
from pathlib import Path

# -----------------------------
# Detect Project Root (Automatically)
# -----------------------------
project_root = Path(__file__).resolve().parent  # Detects `model_error_uq_plasma`
hall_opt_path = project_root / "hall_opt"

# Ensure project root is in sys.path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Ensure hall_opt directory is in sys.path
if str(hall_opt_path) not in sys.path:
    sys.path.insert(0, str(hall_opt_path))

# Debugging Output
print(f"Resolved Project Root: {project_root}")
print(f"Resolved Hall Opt Path: {hall_opt_path}")

# -----------------------------
# Load YAML file & Resolve Paths
# -----------------------------
yaml_path = hall_opt_path / "config/settings.yaml"

with open(yaml_path, "r") as f:
    settings = yaml.safe_load(f)

# Function to resolve placeholders manually
def resolve_variables(data, base_path):
    """ Recursively resolve placeholders in YAML. """
    if isinstance(data, dict):
        return {k: resolve_variables(v, base_path) for k, v in data.items()}
    elif isinstance(data, list):
        return [resolve_variables(v, base_path) for v in data]
    elif isinstance(data, str):
        return str(base_path / data) if not data.startswith(("${", "/")) else data
    else:
        return data

# Resolve paths based on `project_root`
settings = resolve_variables(settings, project_root)

# Convert important paths to absolute paths
results_dir = Path(settings["general"]["results_dir"]).resolve()

# Debugging Output
print(f"Resolved Results Directory: {results_dir}")

# -----------------------------
# Import and run the main script
# -----------------------------
try:
    from hall_opt.main import main  # Now the import should work everywhere
    if __name__ == "__main__":
        main()
except ModuleNotFoundError as e:
    print(f"Import Error: {e}")
    print("Make sure 'hall_opt' has an __init__.py file and is correctly structured.")
