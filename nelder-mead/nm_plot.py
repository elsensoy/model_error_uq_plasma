import json
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np

# Helper function to load JSON data
def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

# thrust residuals over iterations
def plot_thrust_residuals(observed_data, iteration_metrics, save_dir="plots_residuals_nm"):
    os.makedirs(save_dir, exist_ok=True)

    iterations = [entry['iteration'] for entry in iteration_metrics]
    thrust_residuals = [observed_data['thrust'][-1] - entry['metrics']['thrust'][-1] for entry in iteration_metrics]

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, thrust_residuals, marker='o', linestyle='-', color='red', label='Thrust Residuals')

    plt.title('Thrust Residuals Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Thrust Residual (Observed - Simulated)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'thrust_residuals.png'))
    plt.close()
    print(f"Thrust residuals plot saved in {save_dir}/thrust_residuals.png")

# Ion velocity residuals over iterations
def plot_ion_velocity_residuals(observed_data, iteration_metrics, save_dir="plots_residuals_nm"):
    os.makedirs(save_dir, exist_ok=True)

    # Create a uniform z_grid based on the observed ion velocity
    z_grid_observed = np.linspace(0, 1, len(observed_data['ion_velocity'][0]))

    plt.figure(figsize=(10, 6))

    # Plot residuals for each iteration
    for entry in iteration_metrics:
        simulated_ion_velocity = entry['metrics']['ion_velocity'][0]  # Simulated ion velocity
        
        # Create a z_grid for the simulated data
        z_grid_simulated = np.linspace(0, 1, len(simulated_ion_velocity))

        # Interpolate the simulated ion velocity to match the observed grid
        interpolate_simulated = interp1d(z_grid_simulated, simulated_ion_velocity, kind='linear', fill_value="extrapolate")
        simulated_ion_velocity_interpolated = interpolate_simulated(z_grid_observed)

        # Calculate residuals
        residuals = np.array(observed_data['ion_velocity'][0]) - simulated_ion_velocity_interpolated
        
        # Plot the residuals for this iteration
        plt.plot(z_grid_observed, residuals, linestyle='--', label=f"Iter {entry['iteration']} Residuals")

    plt.title('Ion Velocity Residuals Over Iterations')
    plt.xlabel('Normalized z')
    plt.ylabel('Residual (Observed - Simulated)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'ion_velocity_residuals.png'))
    plt.close()
    print(f"Ion velocity residuals plot saved in {save_dir}/ion_velocity_residuals.png")

# Function to plot thrust over iterations (including observed and initial guess)
def plot_thrust_over_iterations(observed_data, initial_data, iteration_metrics, save_dir="plots_residuals_nm"):
    os.makedirs(save_dir, exist_ok=True)

    iterations = [entry['iteration'] for entry in iteration_metrics]
    thrust_values = [entry['metrics']['thrust'][-1] for entry in iteration_metrics]

    plt.figure(figsize=(10, 6))
    plt.plot([1], [observed_data['thrust'][-1]], marker='o', color='red', label='Observed', markersize=8)
    plt.plot([1], [initial_data['thrust'][-1]], marker='o', color='blue', label='Initial Guess', markersize=8)
    plt.plot(iterations, thrust_values, marker='o', linestyle='-', color='purple', label='Thrust (MAP Iterations)')

    plt.title('Thrust Over Iterations (Including Observed & Initial Guess)')
    plt.xlabel('Iteration')
    plt.ylabel('Thrust')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'thrust_over_iterations.png'))
    plt.close()
    print(f"Thrust plot saved in {save_dir}/thrust_over_iterations.png")

# Ion velocity over iterations (including observed and initial guess)
def plot_ion_velocity_over_iterations(observed_data, initial_data, iteration_metrics, save_dir="plots_residuals_nm"):
    os.makedirs(save_dir, exist_ok=True)

    z_grid_observed = np.linspace(0, 1, len(observed_data['ion_velocity'][0]))
    z_grid_initial = np.linspace(0, 1, len(initial_data['ion_velocity'][0]))

    plt.figure(figsize=(10, 6))
    plt.plot(z_grid_observed, observed_data['ion_velocity'][0], linestyle='-', color='red', label='Observed', linewidth=2)
    plt.plot(z_grid_initial, initial_data['ion_velocity'][0], linestyle='-.', color='blue', label='Initial Guess', linewidth=2)

    # Plot each iteration's ion velocity
    for entry in iteration_metrics:
        iter_values = entry['metrics']['ion_velocity'][0]
        z_grid_iter = np.linspace(0, 1, len(iter_values))  # Generate z_grid for this iteration
        plt.plot(z_grid_iter, iter_values, linestyle='--', label=f"Iter {entry['iteration']}: v1={entry['v1']:.4f}, v2={entry['v2']:.4f}")

    plt.title('Ion Velocity Over Iterations (Including Observed & Initial Guess)')
    plt.xlabel('Normalized z')
    plt.ylabel('Ion Velocity')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'ion_velocity_over_iterations.png'))
    plt.close()
    print(f"Ion velocity plot saved in {save_dir}/ion_velocity_over_iterations.png")

# Discharge current over iterations (including observed and initial guess)
def plot_discharge_current_over_iterations(observed_data, initial_data, iteration_metrics, save_dir="plots_residuals_nm"):
    os.makedirs(save_dir, exist_ok=True)

    iterations = [entry['iteration'] for entry in iteration_metrics]
    discharge_current_values = [entry['metrics']['discharge_current'][-1] for entry in iteration_metrics]

    plt.figure(figsize=(10, 6))
    plt.plot([1], [observed_data['discharge_current'][-1]], marker='o', color='red', label='Observed', markersize=8)
    plt.plot([1], [initial_data['discharge_current'][-1]], marker='o', color='blue', label='Initial Guess', markersize=8)
    plt.plot(iterations, discharge_current_values, marker='o', linestyle='-', color='purple', label='Discharge Current (MAP Iterations)')

    plt.title('Discharge Current Over Iterations (Including Observed & Initial Guess)')
    plt.xlabel('Iteration')
    plt.ylabel('Discharge Current')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'discharge_current_over_iterations.png'))
    plt.close()
    print(f"Discharge current plot saved in {save_dir}/discharge_current_over_iterations.png")
import json
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np

# Helper function to load JSON data
def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

# thrust residuals over iterations
def plot_thrust_residuals(observed_data, iteration_metrics, save_dir="plots_residuals_nm"):
    os.makedirs(save_dir, exist_ok=True)

    iterations = [entry['iteration'] for entry in iteration_metrics]
    thrust_residuals = [observed_data['thrust'][-1] - entry['metrics']['thrust'][-1] for entry in iteration_metrics]

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, thrust_residuals, marker='o', linestyle='-', color='red', label='Thrust Residuals')

    plt.title('Thrust Residuals Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Thrust Residual (Observed - Simulated)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'thrust_residuals.png'))
    plt.close()
    print(f"Thrust residuals plot saved in {save_dir}/thrust_residuals.png")

# Ion velocity residuals over iterations
def plot_ion_velocity_residuals(observed_data, iteration_metrics, save_dir="plots_residuals_nm"):
    os.makedirs(save_dir, exist_ok=True)

    # Create a uniform z_grid based on the observed ion velocity
    z_grid_observed = np.linspace(0, 1, len(observed_data['ion_velocity'][0]))

    plt.figure(figsize=(10, 6))

    # Plot residuals for each iteration
    for entry in iteration_metrics:
        simulated_ion_velocity = entry['metrics']['ion_velocity'][0]  # Simulated ion velocity
        
        # Create a z_grid for the simulated data
        z_grid_simulated = np.linspace(0, 1, len(simulated_ion_velocity))

        # Interpolate the simulated ion velocity to match the observed grid
        interpolate_simulated = interp1d(z_grid_simulated, simulated_ion_velocity, kind='linear', fill_value="extrapolate")
        simulated_ion_velocity_interpolated = interpolate_simulated(z_grid_observed)

        # Calculate residuals
        residuals = np.array(observed_data['ion_velocity'][0]) - simulated_ion_velocity_interpolated
        
        # Plot the residuals for this iteration
        plt.plot(z_grid_observed, residuals, linestyle='--', label=f"Iter {entry['iteration']} Residuals")

    plt.title('Ion Velocity Residuals Over Iterations')
    plt.xlabel('Normalized z')
    plt.ylabel('Residual (Observed - Simulated)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'ion_velocity_residuals.png'))
    plt.close()
    print(f"Ion velocity residuals plot saved in {save_dir}/ion_velocity_residuals.png")

# Function to plot thrust over iterations (including observed and initial guess)
def plot_thrust_over_iterations(observed_data, initial_data, iteration_metrics, save_dir="plots_residuals_nm"):
    os.makedirs(save_dir, exist_ok=True)

    iterations = [entry['iteration'] for entry in iteration_metrics]
    thrust_values = [entry['metrics']['thrust'][-1] for entry in iteration_metrics]

    plt.figure(figsize=(10, 6))
    plt.plot([1], [observed_data['thrust'][-1]], marker='o', color='red', label='Observed', markersize=8)
    plt.plot([1], [initial_data['thrust'][-1]], marker='o', color='blue', label='Initial Guess', markersize=8)
    plt.plot(iterations, thrust_values, marker='o', linestyle='-', color='purple', label='Thrust (MAP Iterations)')

    plt.title('Thrust Over Iterations (Including Observed & Initial Guess)')
    plt.xlabel('Iteration')
    plt.ylabel('Thrust')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'thrust_over_iterations.png'))
    plt.close()
    print(f"Thrust plot saved in {save_dir}/thrust_over_iterations.png")

# Ion velocity over iterations (including observed and initial guess)
def plot_ion_velocity_over_iterations(observed_data, initial_data, iteration_metrics, save_dir="plots_residuals_nm"):
    os.makedirs(save_dir, exist_ok=True)

    z_grid_observed = np.linspace(0, 1, len(observed_data['ion_velocity'][0]))
    z_grid_initial = np.linspace(0, 1, len(initial_data['ion_velocity'][0]))

    plt.figure(figsize=(10, 6))
    plt.plot(z_grid_observed, observed_data['ion_velocity'][0], linestyle='-', color='red', label='Observed', linewidth=2)
    plt.plot(z_grid_initial, initial_data['ion_velocity'][0], linestyle='-.', color='blue', label='Initial Guess', linewidth=2)

    # Plot each iteration's ion velocity
    for entry in iteration_metrics:
        iter_values = entry['metrics']['ion_velocity'][0]
        z_grid_iter = np.linspace(0, 1, len(iter_values))  # Generate z_grid for this iteration
        plt.plot(z_grid_iter, iter_values, linestyle='--', label=f"Iter {entry['iteration']}: v1={entry['v1']:.4f}, v2={entry['v2']:.4f}")

    plt.title('Ion Velocity Over Iterations (Including Observed & Initial Guess)')
    plt.xlabel('Normalized z')
    plt.ylabel('Ion Velocity')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'ion_velocity_over_iterations.png'))
    plt.close()
    print(f"Ion velocity plot saved in {save_dir}/ion_velocity_over_iterations.png")

# Discharge current over iterations (including observed and initial guess)
def plot_discharge_current_over_iterations(observed_data, initial_data, iteration_metrics, save_dir="plots_residuals_nm"):
    os.makedirs(save_dir, exist_ok=True)

    iterations = [entry['iteration'] for entry in iteration_metrics]
    discharge_current_values = [entry['metrics']['discharge_current'][-1] for entry in iteration_metrics]

    plt.figure(figsize=(10, 6))
    plt.plot([1], [observed_data['discharge_current'][-1]], marker='o', color='red', label='Observed', markersize=8)
    plt.plot([1], [initial_data['discharge_current'][-1]], marker='o', color='blue', label='Initial Guess', markersize=8)
    plt.plot(iterations, discharge_current_values, marker='o', linestyle='-', color='purple', label='Discharge Current (MAP Iterations)')

    plt.title('Discharge Current Over Iterations (Including Observed & Initial Guess)')
    plt.xlabel('Iteration')
    plt.ylabel('Discharge Current')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'discharge_current_over_iterations.png'))
    plt.close()
    print(f"Discharge current plot saved in {save_dir}/discharge_current_over_iterations.png")
def main():
    # Load observed, initial, and iteration metrics data
    ground_truth_data = load_json('nm_observed_data.json')  # Make sure this is the Nelder-Mead generated data
    initial_data = load_json('nm_initial_guess_twozonebohm_result.json')  # Make sure this is the initial guess from Nelder-Mead run
    iteration_metrics = load_json('nm_iteration_metrics.json')  # Ensure this is from Nelder-Mead iteration logging

    save_dir = "plots_residuals_nm"

    # Plot thrust over iterations
    plot_thrust_over_iterations(ground_truth_data, initial_data, iteration_metrics, save_dir)

    # Plot ion velocity over iterations
    plot_ion_velocity_over_iterations(ground_truth_data, initial_data, iteration_metrics, save_dir)
    
    # Plot discharge current over iterations
    plot_discharge_current_over_iterations(ground_truth_data, initial_data, iteration_metrics, save_dir)

    # Plot residuals
    plot_ion_velocity_residuals(ground_truth_data, iteration_metrics, save_dir)
    plot_thrust_residuals(ground_truth_data, iteration_metrics, save_dir)

if __name__ == "__main__":
    main()
