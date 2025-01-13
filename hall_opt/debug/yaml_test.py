from pathlib import Path
from hall_opt.config.loader import Settings, load_yml_settings,extract_anom_model
def test_yaml_loading(yaml_path):
    try:
        settings = load_yml_settings(Path(yaml_path))
        print("YAML file loaded successfully.")
        print(settings)  # Print the loaded settings to verify correctness
    except Exception as e:
        print(f"Error loading YAML file: {e}")

# Test the function
test_yaml_loading("/mnt/c/Users/elsensoy/model_error_uq_plasma/hall_opt/config/settings.yaml")

def test_yaml_conditions(yaml_path):
    settings_dict = load_yml_settings(Path(yaml_path))
    settings = Settings(**settings_dict)

    # Check flags and corresponding actions
    if settings.gen_data:
        print("Generating ground truth data is enabled.")
    else:
        print("Generating ground truth data is disabled.")

    if settings.run_map:
        print("MAP estimation is enabled.")
    else:
        print("MAP estimation is disabled.")

    if settings.run_mcmc:
        print("MCMC sampling is enabled.")
    else:
        print("MCMC sampling is disabled.")

# Test the function
test_yaml_conditions("/mnt/c/Users/elsensoy/model_error_uq_plasma/hall_opt/config/settings.yaml")

def test_workflow_with_yaml(yaml_path):
    settings_dict = load_yml_settings(Path(yaml_path))
    settings = Settings(**settings_dict)

    if settings.gen_data:
        print("Simulating ground truth data... (mock)")
        # Mock function call
        mock_ground_truth_generation()

    if settings.run_map:
        print("Running MAP estimation... (mock)")
        # Mock function call
        mock_map_estimation()

    if settings.run_mcmc:
        print("Running MCMC sampling... (mock)")
        # Mock function call
        mock_mcmc_sampling()

# Mock functions for testing
def mock_ground_truth_generation():
    print("Mock: Ground truth data generated.")

def mock_map_estimation():
    print("Mock: MAP estimation completed.")

def mock_mcmc_sampling():
    print("Mock: MCMC sampling completed.")

# Test the function
test_workflow_with_yaml("/mnt/c/Users/elsensoy/model_error_uq_plasma/hall_opt/config/settings.yaml")
