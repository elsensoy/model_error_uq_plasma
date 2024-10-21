import json
import os
import matplotlib.pyplot as plt
from map_ink import run_map_multiple_initial_guesses
from scipy.interpolate import interp1d  
import numpy as np 

# Helper function to load JSON data
def load_json(filename):
    file_path = os.path.join("..", "results-LBFGSB", filename)  # Load from the results directory
    with open(file_path, 'r') as f:
        return json.load(f)

# Plot thrust over iterations with residuals
def plot_thrust_over_iterations_with_residuals(observed_data, initial_data, iteration_metrics, save_dir="plots_residuals"):
    os.makedirs(save_dir, exist_ok=True)

    iterations = [entry['iteration'] for entry in iteration_metrics]
    thrust_values = [entry['metrics']['thrust'][-1] for entry in iteration_metrics]
    thrust_residuals = [observed_data['thrust'][-1] - entry['metrics']['thrust'][-1] for entry in iteration_metrics]

    plt.figure(figsize=(10, 6))
    plt.plot([1], [observed_data['thrust'][-1]], marker='o', color='red', label='Observed', markersize=8)
    plt.plot([1], [initial_data['thrust'][-1]], marker='o', color='blue', label='Initial Guess', markersize=8)
    plt.plot(iterations, thrust_values, marker='o', linestyle='-', color='purple', label='Thrust (MAP Iterations)')
    
    # Plot residuals (Observed - Simulated)
    plt.plot(iterations, thrust_residuals, marker='x', linestyle='--', color='orange', label='Thrust Residuals')

    plt.title('Thrust Over Iterations (Including Residuals)')
    plt.xlabel('Iteration')
    plt.ylabel('Thrust')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'thrust_over_iterations_with_residuals.png'))
    plt.close()
    print(f"Thrust plot with residuals saved in {save_dir}/thrust_over_iterations_with_residuals.png")


# Plot ion velocity over iterations with residuals
def plot_ion_velocity_over_iterations_with_residuals(observed_data, initial_data, iteration_metrics, save_dir="plots_residuals"):
    os.makedirs(save_dir, exist_ok=True)

    z_grid_observed = np.linspace(0, 1, len(observed_data['ion_velocity'][0]))
    z_grid_initial = np.linspace(0, 1, len(initial_data['ion_velocity'][0]))

    plt.figure(figsize=(10, 6))
    plt.plot(z_grid_observed, observed_data['ion_velocity'][0], linestyle='-', color='red', label='Observed', linewidth=2)
    plt.plot(z_grid_initial, initial_data['ion_velocity'][0], linestyle='-.', color='blue', label='Initial Guess', linewidth=2)

    # Plot each iteration's ion velocity and residuals
    for entry in iteration_metrics:
        iter_values = entry['metrics']['ion_velocity'][0]
        z_grid_iter = np.linspace(0, 1, len(iter_values))  # Generate z_grid for this iteration
        plt.plot(z_grid_iter, iter_values, linestyle='--', label=f"Iter {entry['iteration']}: v1={entry['v1']:.4f}, v2={entry['v2']:.4f}")

        # Calculate residuals and plot them
        interpolated_values = np.interp(z_grid_observed, z_grid_iter, iter_values)
        residuals = observed_data['ion_velocity'][0] - interpolated_values
        plt.plot(z_grid_observed, residuals, linestyle=':', label=f"Iter {entry['iteration']} Residuals")

    plt.title('Ion Velocity Over Iterations (Including Residuals)')
    plt.xlabel('Normalized z')
    plt.ylabel('Ion Velocity')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'ion_velocity_over_iterations_with_residuals.png'))
    plt.close()
    print(f"Ion velocity plot with residuals saved in {save_dir}/ion_velocity_over_iterations_with_residuals.png")

def main():
    # Load data for multiple ion_velocity_weights
    datasets = {
        '0.1': {
            'observed_data': load_json('w_0.1_observed_data_map.json'),
            'iteration_metrics': load_json('w_0.1_map_iteration_results.json')
        },
        '1.0': {
            'observed_data': load_json('w_1.0_observed_data_map.json'),
            'iteration_metrics': load_json('w_1.0_iteration_metrics.json')
        },
        '2.0': {
            'observed_data': load_json('w_2.0_observed_data_map.json'),
            'iteration_metrics': load_json('w_2.0_iteration_metrics.json')
        },
        '3.0': {
            'observed_data': load_json('w_3.0_observed_data_map.json'),
            'iteration_metrics': load_json('w_3.0_iteration_metrics.json')
        },
        '5.0': {
            'observed_data': load_json('w_5.0_observed_data_map.json'),
            'iteration_metrics': load_json('w_5.0_iteration_metrics.json')
        },
        '10.0': {
            'observed_data': load_json('w_10.0_observed_data_map.json'),
            'iteration_metrics': load_json('w_10.0_iteration_metrics.json')
        },
        '1e-10': {
            'observed_data': load_json('w_1e-10_observed_data_map.json'),
            'iteration_metrics': load_json('w_1e-10_iteration_metrics.json')
        }
    }

    # Iterate over all weights and plot residuals for each weight
    for weight, data in datasets.items():
        observed_data = data['observed_data']
        iteration_metrics = data['iteration_metrics']

        print(f"Plotting residuals for ion_velocity_weight = {weight}...")

        # Plot thrust residuals for each weight
        plot_thrust_residuals(observed_data, iteration_metrics, save_dir=f"plots_residuals_w_{weight}")
        
        # Plot ion velocity residuals for each weight
        plot_ion_velocity_residuals(observed_data, iteration_metrics, save_dir=f"plots_residuals_w_{weight}")

        # Plot discharge current residuals for each weight
        plot_discharge_current_residuals(observed_data, iteration_metrics, save_dir=f"plots_residuals_w_{weight}")

if __name__ == "__main__":
    main()
