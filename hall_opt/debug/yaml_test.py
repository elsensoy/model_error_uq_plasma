from pathlib import Path
from hall_opt.config.loader import Settings, load_yml_settings


def setup_yaml(yaml_path):
    """
    Load YAML configuration and return the settings dictionary and Settings object.
    """
    try:
        settings_dict = load_yml_settings(Path(yaml_path))
        settings = Settings(**settings_dict)
        return settings_dict, settings
    except Exception as e:
        raise RuntimeError(f"Error loading YAML file '{yaml_path}': {e}")


def test_yaml_loading(yaml_path):
    """
    Test if the YAML file can be loaded successfully.
    """
    try:
        settings_dict, _ = setup_yaml(yaml_path)
        print("YAML file loaded successfully.")
        assert isinstance(settings_dict, dict), "YAML file should load as a dictionary."
    except Exception as e:
        print(f"Error loading YAML file: {e}")


def test_yaml_conditions(yaml_path):
    """
    Test specific flags and configurations in the YAML file.
    """
    _, settings = setup_yaml(yaml_path)

    assert hasattr(settings, "gen_data"), "Missing 'gen_data' flag in settings."
    if settings.gen_data:
        print("Generating ground truth data is enabled.")
        assert settings.gen_data is True, "'gen_data' should be enabled."
    else:
        print("Generating ground truth data is disabled.")

    assert hasattr(settings, "run_map"), "Missing 'run_map' flag in settings."
    if settings.run_map:
        print("MAP estimation is enabled.")
        assert settings.run_map is True, "'run_map' should be enabled."
    else:
        print("MAP estimation is disabled.")

    assert hasattr(settings, "run_mcmc"), "Missing 'run_mcmc' flag in settings."
    if settings.run_mcmc:
        print("MCMC sampling is enabled.")
        assert settings.run_mcmc is True, "'run_mcmc' should be enabled."
    else:
        print("MCMC sampling is disabled.")


def test_workflow_with_yaml(yaml_path):
    """
    Test the workflow based on flags in the YAML file.
    """
    _, settings = setup_yaml(yaml_path)

    # Mock workflow based on the flags
    if settings.gen_data:
        print("Simulating ground truth data... (mock)")
        mock_ground_truth_generation()

    if settings.run_map:
        print("Running MAP estimation... (mock)")
        mock_map_estimation()

    if settings.run_mcmc:
        print("Running MCMC sampling... (mock)")
        mock_mcmc_sampling()


def test_missing_or_invalid_fields(yaml_path):
    """
    Test how the loader handles missing or invalid fields in the YAML file.
    """
    try:
        settings_dict, settings = setup_yaml(yaml_path)

        # Check for a required field
        assert "results_dir" in settings_dict, "The 'results_dir' key is missing in settings.yaml."
        assert settings.results_dir != "", "'results_dir' should not be empty."

        # Check if any unexpected fields exist
        valid_keys = {"gen_data", "run_map", "run_mcmc", "ion_velocity_weight", "plotting", "iterations"}
        for key in settings_dict:
            if key not in valid_keys:
                print(f"Warning: Unexpected key '{key}' found in settings.yaml.")
    except AssertionError as e:
        print(f"Test failed: {e}")


# Mock functions for testing
def mock_ground_truth_generation():
    print("Mock: Ground truth data generated.")


def mock_map_estimation():
    print("Mock: MAP estimation completed.")


def mock_mcmc_sampling():
    print("Mock: MCMC sampling completed.")


# Main Test Execution
if __name__ == "__main__":
    yaml_path = "//home/elida/Public/users/elsensoy/model_error_uq_plasma/hall_opt/config/settings.yaml"

    print("\nRunning test_yaml_loading...")
    test_yaml_loading(yaml_path)

    print("\nRunning test_yaml_conditions...")
    test_yaml_conditions(yaml_path)

    print("\nRunning test_workflow_with_yaml...")
    test_workflow_with_yaml(yaml_path)

    print("\nRunning test_missing_or_invalid_fields...")
    test_missing_or_invalid_fields(yaml_path)
