import json
import os
import matplotlib.pyplot as plt
from map_sim import run_map_and_optimization  
from scipy.interpolate import interp1d  
import numpy as np 

# Helper function to load JSON data
def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

# thrust residuals over iterations
def plot_thrust_residuals(observed_data, iteration_metrics, save_dir="plots_residuals_ion_increased_to_3_3"):
    os.makedirs(save_dir, exist_ok=True)

    iterations = [entry['iteration'] for entry in iteration_metrics]
    thrust_residuals = [observed_data['thrust'][-1] - entry['metrics']['thrust'][-1] for entry in iteration_metrics]

    plt.figure(figsize=(10, 6))

    # Plot thrust residuals over iterations
    plt.plot(iterations, thrust_residuals, marker='o', linestyle='-', color='red', label='Thrust Residuals')

    plt.title('Thrust Residuals Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Thrust Residual (Observed - Simulated)')
    plt.grid(True)
    plt.legend()

    plt.savefig(os.path.join(save_dir, 'thrust_residuals_ion_increased_to_3_3.png'))
    plt.close()
    print(f"Thrust residuals plot saved in {save_dir}/thrust_residuals_ion_increased_to_3_3.png")



# Function to plot ion velocity residuals over iterations (observed - simulated)
def plot_ion_velocity_residuals(observed_data, iteration_metrics, save_dir="plots_residuals_ion_increased_to_3_3"):
    os.makedirs(save_dir, exist_ok=True)

    # Create z_grid based on observed data (assuming it's on a uniform grid)
    z_grid_observed = np.linspace(0, 1, len(observed_data['ion_velocity'][0]))

    plt.figure(figsize=(10, 6))

    # Plot residuals for each iteration
    for entry in iteration_metrics:
        simulated_ion_velocity = entry['metrics']['ion_velocity'][0]  # Simulated ion velocity for the iteration
        
        # Create a z_grid for the simulated data (it may have a different number of points)
        z_grid_simulated = np.linspace(0, 1, len(simulated_ion_velocity))

        # Interpolate the simulated ion velocity to match the observed data's grid
        interpolate_simulated = interp1d(z_grid_simulated, simulated_ion_velocity, kind='linear', fill_value="extrapolate")
        simulated_ion_velocity_interpolated = interpolate_simulated(z_grid_observed)

        # Calculate residuals (Observed - Simulated)
        residuals = np.array(observed_data['ion_velocity'][0]) - simulated_ion_velocity_interpolated
        
        # Plot the residuals for this iteration
        plt.plot(z_grid_observed, residuals, linestyle='--', label=f"Iter {entry['iteration']}: Residuals", linewidth=1)

    plt.title('Ion Velocity Residuals Over Iterations')
    plt.xlabel('Normalized z')
    plt.ylabel('Residual (Observed - Simulated)')
    plt.grid(True)
    plt.legend()

    plt.savefig(os.path.join(save_dir, 'ion_velocity_residuals_ion_increased_to_3_3.png'))
    plt.close()
    print(f"Ion velocity residuals plot saved in {save_dir}/ion_velocity_residuals_ion_increased_to_3_3.png")
# Function to plot potential residuals over iterations
def plot_potential_residuals(observed_data, iteration_metrics, save_dir="plots_residuals_ion_increased_to_3_3"):
    os.makedirs(save_dir, exist_ok=True)

    z_grid = np.linspace(0, 1, len(observed_data['potential'][0]))  # Assuming same grid for observed and simulated

    plt.figure(figsize=(10, 6))

    # Plot residuals for each iteration
    for entry in iteration_metrics:
        simulated_potential = entry['metrics']['potential'][0]  # Simulated potential for the iteration
        residuals = np.array(observed_data['potential'][0]) - np.array(simulated_potential)

        # Plot the residuals for this iteration
        plt.plot(z_grid, residuals, linestyle='--', label=f"Iter {entry['iteration']}: Residuals", linewidth=1)

    plt.title('Potential Residuals Over Iterations')
    plt.xlabel('Normalized z')
    plt.ylabel('Residual (Observed - Simulated)')
    plt.grid(True)
    plt.legend()

    plt.savefig(os.path.join(save_dir, 'potential_residuals_ion_increased_to_3_3.png'))
    plt.close()
    print(f"Potential residuals plot saved in {save_dir}/potential_residuals_ion_increased_to_3_3.png")






# Function to plot thrust over iterations (including observed and initial guess)
def plot_thrust_over_iterations(observed_data, initial_data, iteration_metrics, save_dir="plots_residuals_ion_increased_to_3_3"):
    os.makedirs(save_dir, exist_ok=True)

    iterations = [entry['iteration'] for entry in iteration_metrics]
    thrust_values = [entry['metrics']['thrust'][-1] for entry in iteration_metrics]  # Assuming 'thrust' is a list

    plt.figure(figsize=(10, 6))

    # Plot observed and initial guess thrust
    plt.plot([1], [observed_data['thrust'][-1]], marker='o', color='red', label='Observed', markersize=8)
    plt.plot([1], [initial_data['thrust'][-1]], marker='o', color='blue', label='Initial Guess', markersize=8)

    # Plot thrust for each iteration
    plt.plot(iterations, thrust_values, marker='o', linestyle='-', color='purple', label='Thrust (MAP Iterations)')

    plt.title('Thrust Over Iterations (Including Observed & Initial Guess)')
    plt.xlabel('Iteration')
    plt.ylabel('Thrust')
    plt.grid(True)
    plt.legend()

    plt.savefig(os.path.join(save_dir, 'thrust_over_iterations_with_observed_ion_increased_to_3_3.png'))
    plt.close()
    print(f"Thrust plot saved in {save_dir}/thrust_over_iterations_with_observed_ion_increased_to_3_3.png")

# Function to plot ion_velocity over iterations (including observed and initial guess)
def plot_ion_velocity_over_iterations(observed_data, initial_data, iteration_metrics, save_dir="plots_residuals_ion_increased_to_3_3"):
    os.makedirs(save_dir, exist_ok=True)

    # Dynamically generate the z_grid based on the length of ion_velocity
    z_grid_observed = np.linspace(0, 1, len(observed_data['ion_velocity'][0]))
    z_grid_initial = np.linspace(0, 1, len(initial_data['ion_velocity'][0]))

    plt.figure(figsize=(10, 6))

    # Plot observed and initial guess ion velocity with dynamically generated z_grid
    plt.plot(z_grid_observed, observed_data['ion_velocity'][0], linestyle='-', color='red', label='Observed', linewidth=2)
    plt.plot(z_grid_initial, initial_data['ion_velocity'][0], linestyle='-.', color='blue', label='Initial Guess', linewidth=2)

    # Plot each iteration's ion velocity data with dynamically generated z_grid
    for entry in iteration_metrics:
        iter_values = entry['metrics']['ion_velocity'][0]  # Assuming ion_velocity is a list of lists
        z_grid_iter = np.linspace(0, 1, len(iter_values))  # Generate a z_grid for the iteration
        plt.plot(z_grid_iter, iter_values, linestyle='--', label=f"Iter {entry['iteration']}: v1={entry['v1']:.4f}, v2={entry['v2']:.4f}")

    plt.title('Ion Velocity Over Iterations (Including Observed & Initial Guess)')
    plt.xlabel('Normalized z')
    plt.ylabel('Ion Velocity')
    plt.grid(True)
    plt.legend()

    plt.savefig(os.path.join(save_dir, 'ion_velocity_over_iterations_with_observed_ion_increased_to_3_3.png'))
    plt.close()
    print(f"Ion velocity plot saved in {save_dir}/ion_velocity_over_iterations_with_observed_100.png")


def plot_discharge_current_over_iterations(observed_data, initial_data, iteration_metrics, save_dir="plot_ion_increased_to_3_3_100"):
    os.makedirs(save_dir, exist_ok=True)

    iterations = [entry['iteration'] for entry in iteration_metrics]
    discharge_current_values = [entry['metrics']['discharge_current'][-1] for entry in iteration_metrics]  

    plt.figure(figsize=(10, 6))

    # Plot observed and initial guess discharge current
    plt.plot([1], [observed_data['discharge_current'][-1]], marker='o', color='red', label='Observed', markersize=8)
    plt.plot([1], [initial_data['discharge_current'][-1]], marker='o', color='blue', label='Initial Guess', markersize=8)

    # Plot discharge current for each iteration
    plt.plot(iterations, discharge_current_values, marker='o', linestyle='-', color='purple', label='Discharge Current (MAP Iterations)')

    plt.title('Discharge Current Over Iterations (Including Observed & Initial Guess)')
    plt.xlabel('Iteration')
    plt.ylabel('Discharge Current')
    plt.grid(True)
    plt.legend()

    plt.savefig(os.path.join(save_dir, 'discharge_current_over_iterations_with_observed_ion_increased_to_3_3.png'))
    plt.close()
    print(f"Discharge current plot saved in {save_dir}/discharge_current_over_iterations_with_observed_ion_increased_to_3_3.png")


def plot_discharge_current_residuals(observed_data, iteration_metrics, save_dir="plots_residuals_ion_increased_to_3_3"):
    os.makedirs(save_dir, exist_ok=True)

    iterations = [entry['iteration'] for entry in iteration_metrics]
    
    # Assuming the discharge current is a list and we need the final value (last element)
    discharge_current_residuals = [
        observed_data['discharge_current'][-1] - entry['metrics']['discharge_current'][-1] 
        for entry in iteration_metrics
    ]

    plt.figure(figsize=(10, 6))

    # Plot discharge current residuals over iterations
    plt.plot(iterations, discharge_current_residuals, marker='o', linestyle='-', color='red', label='Discharge Current Residuals')

    plt.title('Discharge Current Residuals Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Discharge Current Residual (Observed - Simulated)')
    plt.grid(True)
    plt.legend()

    plt.savefig(os.path.join(save_dir, 'discharge_current_residuals_ion_increased_to_3_3.png'))
    plt.close()
    print(f"Discharge current residuals plot saved in {save_dir}/discharge_current_residuals.png")


def plot_potential_over_iterations(observed_data, initial_data, iteration_metrics, z_grid, save_dir="plots_residuals_ion_increased_to_3_3"):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))

    # Extract potential data correctly
    print(f"Shape of z_grid: {len(z_grid)}")
    print(f"Shape of observed_data['potential']: {np.shape(observed_data['potential'])}")

    # Extract the inner list (102 values) from the nested structure
    observed_potential = observed_data['potential'][0]
    print(f"Extracted observed potential: Length = {len(observed_potential)}")

    # Check if z_grid matches the potential data length
    if len(observed_potential) != len(z_grid):
        print(f"z_grid and potential lengths differ. Adjusting z_grid to match the potential data.")
        # Adjust z_grid to match the length of the potential data (102 points)
        z_grid = np.linspace(0, 1, len(observed_potential))

    # Extract initial potential similarly
    initial_potential = initial_data['potential'][0]

    # Plot observed and initial guess potential
    plt.plot(z_grid, observed_potential, linestyle='-', color='red', label='Observed', linewidth=2)
    plt.plot(z_grid, initial_potential, linestyle='-', color='blue', label='Initial Guess', linewidth=2)

    # Plot each iteration's potential data
    for entry in iteration_metrics:
        iter_values = entry['metrics']['potential'][0]
        if len(iter_values) == 1:
            iter_values = np.repeat(iter_values[0], len(z_grid))  # Handle single value case
        plt.plot(z_grid, iter_values, linestyle='--', label=f"Iter {entry['iteration']}: v1={entry['v1']:.4f}, v2={entry['v2']:.4f}")

    plt.title('Potential Over Iterations (Including Observed & Initial Guess)')
    plt.xlabel('Normalized z')
    plt.ylabel('Potential')
    plt.grid(True)
    plt.legend()

    plt.savefig(os.path.join(save_dir, 'potential_over_iterations_with_observed_ion_increased_to_3_3.png'))
    plt.close()
    print(f"Potential plot saved in {save_dir}/potential_over_iterations_with_observed_ion_increased_to_3_3.png")
def plot_discharge_current_over_iterations(observed_data, initial_data, iteration_metrics, save_dir="plots_residuals_ion_increased_to_3_3"):
    os.makedirs(save_dir, exist_ok=True)

    iterations = [entry['iteration'] for entry in iteration_metrics]
    discharge_current_values = [entry['metrics']['discharge_current'][-1] for entry in iteration_metrics]  # Assuming 'discharge_current' is a list

    plt.figure(figsize=(10, 6))

    # Plot observed and initial guess discharge current
    plt.plot([1], [observed_data['discharge_current'][-1]], marker='o', color='red', label='Observed', markersize=8)
    plt.plot([1], [initial_data['discharge_current'][-1]], marker='o', color='blue', label='Initial Guess', markersize=8)

    # Plot discharge current for each iteration
    plt.plot(iterations, discharge_current_values, marker='o', linestyle='-', color='purple', label='Discharge Current (MAP Iterations)')

    plt.title('Discharge Current Over Iterations (Including Observed & Initial Guess)')
    plt.xlabel('Iteration')
    plt.ylabel('Discharge Current')
    plt.grid(True)
    plt.legend()

    plt.savefig(os.path.join(save_dir, 'discharge_current_over_iterations_with_observed_ion_increased_to_3_3.png'))
    plt.close()
    print(f"Discharge current plot saved in {save_dir}/discharge_current_over_iterations_with_observed_ion_increased_to_3_3.png")


def main():
    # Step 1: Load necessary data
    ground_truth_data = load_json('observed_data_map.json')
    
    # Load initial guess data
    initial_data = load_json('initial_guess_twozonebohm_result.json')

    # Load observed data
    observed_data = load_json('observed_data_map.json')

    # Load iteration metrics from file
    iteration_metrics = load_json('iteration_metrics.json')

    # Ensure save_dir is passed as a string
    save_dir = "plots_residuals_ion_increased_to_3_3"

    # Plot thrust over iterations with observed and initial guess
    plot_thrust_over_iterations(observed_data, initial_data, iteration_metrics, save_dir)

    # Plot ion velocity over iterations with observed and initial guess
    plot_ion_velocity_over_iterations(observed_data, initial_data, iteration_metrics, save_dir)
    plot_discharge_current_over_iterations(observed_data, initial_data, iteration_metrics, save_dir)

    # Plot residuals
    plot_ion_velocity_residuals(observed_data, iteration_metrics, save_dir)
    plot_thrust_residuals(observed_data, iteration_metrics, save_dir)
    plot_discharge_current_residuals(observed_data, iteration_metrics, save_dir)
        # Plot potential over iterations with observed and initial guess
    plot_potential_over_iterations(observed_data, initial_data, iteration_metrics, save_dir)
    plot_potential_residuals(observed_data, iteration_metrics, save_dir)



if __name__ == "__main__":
    main()