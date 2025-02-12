import numpy as np

def extract_metrics(solution, observed_data, settings):
    """
    Extracts relevant simulation metrics from the solution output.
    
    Parameters:
    - solution (dict): The output dictionary containing simulation metrics.
    - observed_data (dict): Observed experimental data for comparison.
    - settings (Settings): Configuration settings.
    
    Returns:
    - simulated_data (dict): Extracted simulation metrics.
    - simulated_ion_velocity (np.ndarray): Extracted ion velocity.
    - ion_velocity_weight (float): Ion velocity weight from settings.
    - output_dir (Path): Directory where metrics should be saved.
    """
    # Extract all metrics
    metrics = solution.get("output", {}).get("average", {})
    if not metrics:
        print("ERROR: Invalid or missing metrics in simulation output.")
        return None, None, None, None

    simulated_data = {
        "thrust": metrics.get("thrust", [0]),  # Default to [0] to avoid errors
        "discharge_current": metrics.get("discharge_current", 0),
        "z_normalized": metrics.get("z", []),
        "ion_velocity": metrics.get("ui", []) 
    }

    # Extract simulated ion velocity (ensure correct format)
    simulated_ion_velocity = np.array(metrics.get("ui", [0])[0], dtype=np.float64) if "ui" in metrics and metrics["ui"] else np.array([0])

    # Retrieve ion velocity weight from settings
    ion_velocity_weight = settings.general.ion_velocity_weight

    # Determine output directory based on the run mode
    if settings.general.run_map:
        output_dir = settings.map.base_dir  # Use MAP results directory
        print("DEBUG: Saving metrics in MAP results directory.")
    elif settings.general.run_mcmc:
        output_dir = settings.mcmc.base_dir  # Use MCMC results directory
        print("DEBUG: Saving metrics in MCMC results directory.")
    else:
        print("DEBUG: No MAP or MCMC executed, using general results directory.")
        output_dir = settings.general.results_dir  # Default general directory

    return simulated_data, simulated_ion_velocity, ion_velocity_weight, output_dir
