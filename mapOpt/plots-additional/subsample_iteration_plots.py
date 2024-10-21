import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Helper function to load JSON data
def load_json(filename):
    file_path = os.path.join("..", "results-LBFGSB", filename)  # Load from the results directory
    with open(file_path, 'r') as f:
        return json.load(f)

def subsample_iteration_data(iteration_data, step=10):
    """Subsample iteration data by taking every nth entry."""
    # Ensure the data is a list before subsampling
    if isinstance(iteration_data, list):
        return iteration_data[::step]  # Take every nth element from the list
    else:
        raise ValueError("iteration_data is expected to be a list.")


# Plot thrust over iterations (including observed and initial guess)
def plot_thrust_over_iterations(observed_data, initial_data, iteration_metrics, save_dir="plots_residuals", subsample_step=10):
    os.makedirs(save_dir, exist_ok=True)

    # Subsample the iteration data (every nth iteration)
    iteration_metrics_subsampled = subsample_iteration_data(iteration_metrics, subsample_step)

    iterations = [entry['iteration'] for entry in iteration_metrics_subsampled]
    thrust_values = [entry['metrics']['thrust'][-1] for entry in iteration_metrics_subsampled]

    plt.figure(figsize=(10, 6))
    plt.plot([1], [observed_data['thrust'][-1]], marker='o', color='red', label='Observed', markersize=8)
    plt.plot([1], [initial_data['thrust'][-1]], marker='o', color='blue', label='Initial Guess', markersize=8)
    plt.plot(iterations, thrust_values, marker='o', linestyle='-', color='purple', label='Thrust (MAP Iterations)')

    plt.title('Thrust Over Iterations (Including Observed & Initial Guess)')
    plt.xlabel('Iteration')
    plt.ylabel('Thrust')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'thrust_over_iterations_subsampled.png'))
    plt.close()
    print(f"Thrust plot saved in {save_dir}/thrust_over_iterations_subsampled.png")

def plot_ion_velocity_over_iterations(observed_data, initial_data, iteration_results, iteration_metrics, save_dir="plots_residuals", subsample_step=10):
    os.makedirs(save_dir, exist_ok=True)

    z_grid_observed = np.linspace(0, 1, len(observed_data['ion_velocity'][0]))
    z_grid_initial = np.linspace(0, 1, len(initial_data['ion_velocity'][0]))

    plt.figure(figsize=(10, 6))
    plt.plot(z_grid_observed, observed_data['ion_velocity'][0], linestyle='-', color='red', label='Observed', linewidth=2)
    plt.plot(z_grid_initial, initial_data['ion_velocity'][0], linestyle='-.', color='blue', label='Initial Guess', linewidth=2)

    # Subsample the iteration data (every nth iteration)
    iteration_results_subsampled = subsample_iteration_data(iteration_results, subsample_step)
    iteration_metrics_subsampled = subsample_iteration_data(iteration_metrics, subsample_step)

    # Plot each iteration's ion velocity using subsampled metrics
    for i, entry in enumerate(iteration_results_subsampled):
        iter_values = iteration_metrics_subsampled[i]['ion_velocity'][0]
        z_grid_iter = np.linspace(0, 1, len(iter_values))  # Generate z_grid for this iteration
        plt.plot(z_grid_iter, iter_values, linestyle='--', label=f"Iter {entry['iteration']}: v1={entry['v1']:.4f}, v2={entry['v2']:.4f}")

    plt.title('Ion Velocity Over Iterations (Including Observed & Initial Guess)')
    plt.xlabel('Normalized z')
    plt.ylabel('Ion Velocity')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'ion_velocity_over_iterations_subsampled.png'))
    plt.close()
    print(f"Ion velocity plot saved in {save_dir}/ion_velocity_over_iterations_subsampled.png")

# Plot discharge current over iterations (including observed and initial guess)
def plot_discharge_current_over_iterations(observed_data, initial_data, iteration_metrics, save_dir="plots_residuals", subsample_step=10):
    os.makedirs(save_dir, exist_ok=True)

    # Subsample the iteration data (every nth iteration)
    iteration_metrics_subsampled = subsample_iteration_data(iteration_metrics, subsample_step)

    iterations = [entry['iteration'] for entry in iteration_metrics_subsampled]
    discharge_current_values = [entry['metrics']['discharge_current'][-1] for entry in iteration_metrics_subsampled]

    plt.figure(figsize=(10, 6))
    plt.plot([1], [observed_data['discharge_current'][-1]], marker='o', color='red', label='Observed', markersize=8)
    plt.plot([1], [initial_data['discharge_current'][-1]], marker='o', color='blue', label='Initial Guess', markersize=8)
    plt.plot(iterations, discharge_current_values, marker='o', linestyle='-', color='green', label='Discharge Current (MAP Iterations)')

    plt.title('Discharge Current Over Iterations (Including Observed & Initial Guess)')
    plt.xlabel('Iteration')
    plt.ylabel('Discharge Current')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'discharge_current_over_iterations_subsampled.png'))
    plt.close()
    print(f"Discharge current plot saved in {save_dir}/discharge_current_over_iterations_subsampled.png")

# Main function to load data and create plots
def main():
    # Load data for multiple ion_velocity_weights
    datasets = {
        '0.1': {
            'observed_data': load_json('w_0.1_observed_data_map.json'),
            'initial_data': load_json('w_0.1_best_initial_guess_result.json'),
            'iteration_metrics': load_json('w_0.1_iteration_metrics.json'),
            'iteration_results': load_json('w_0.1_map_iteration_results.json')  # Load iteration results
        },
        '1.0': {
            'observed_data': load_json('w_1.0_observed_data_map.json'),
            'initial_data': load_json('w_1.0_best_initial_guess_result.json'),
            'iteration_metrics': load_json('w_1.0_iteration_metrics.json'),
            'iteration_results': load_json('w_1.0_map_iteration_results.json')
        },
        '3.0': {
            'observed_data': load_json('w_3.0_observed_data_map.json'),
            'initial_data': load_json('w_3.0_best_initial_guess_result.json'),
            'iteration_metrics': load_json('w_3.0_iteration_metrics.json'),
            'iteration_results': load_json('w_3.0_map_iteration_results.json')  # Load iteration results
        },
        '2.0': {
            'observed_data': load_json('w_2.0_observed_data_map.json'),
            'initial_data': load_json('w_2.0_best_initial_guess_result.json'),
            'iteration_metrics': load_json('w_2.0_iteration_metrics.json'),
            'iteration_results': load_json('w_2.0_map_iteration_results.json')
        },
        '5.0': {
            'observed_data': load_json('w_5.0_observed_data_map.json'),
            'initial_data': load_json('w_5.0_best_initial_guess_result.json'),
            'iteration_metrics': load_json('w_5.0_iteration_metrics.json'),
            'iteration_results': load_json('w_5.0_map_iteration_results.json')  # Load iteration results
        },
        '10.0': {
            'observed_data': load_json('w_10.0_observed_data_map.json'),
            'initial_data': load_json('w_10.0_best_initial_guess_result.json'),
            'iteration_metrics': load_json('w_10.0_iteration_metrics.json'),
            'iteration_results': load_json('w_10.0_map_iteration_results.json')
        },
        '10.0': {
            'observed_data': load_json('w_10.0_observed_data_map.json'),
            'initial_data': load_json('w_10.0_best_initial_guess_result.json'),
            'iteration_metrics': load_json('w_10.0_iteration_metrics.json')
        },
        '1e-10': {
            'observed_data': load_json('w_1e-10_observed_data_map.json'),
            'initial_data': load_json('w_1e-10_best_initial_guess_result.json'),
            'iteration_metrics': load_json('w_1e-10_iteration_metrics.json'),
			'iteration_results': load_json('w_1e-10_map_iteration_results.json')
        }
    }

    # Iterate over all weights and plot for each weight
    for weight, data in datasets.items():
        observed_data = data['observed_data']
        initial_data = data['initial_data']
        iteration_metrics = data['iteration_metrics']
        iteration_results = data['iteration_results']

        print(f"Plotting for ion_velocity_weight = {weight}...")

        # Subsample the iteration results (every 10th iteration)
        subsampled_iterations = subsample_iteration_data(iteration_results, step=10)

        # Define save directory
        save_dir = f"plots_residuals_w_{weight}"
        # Plot ion velocity over iterations (with subsampling)
        plot_ion_velocity_over_iterations(observed_data, initial_data, subsampled_iterations, save_dir, subsample_step=10)

        # Plot thrust over iterations (with subsampling)
        plot_thrust_over_iterations(observed_data, initial_data, subsampled_iterations, save_dir, subsample_step=10)


        # Plot discharge current over iterations (with subsampling)
        plot_discharge_current_over_iterations(observed_data, initial_data, subsampled_iterations, save_dir, subsample_step=10)

if __name__ == "__main__":
    main()