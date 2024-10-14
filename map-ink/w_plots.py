import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Helper function to load JSON data
def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

# Function to plot ion velocity comparison for multiple ion velocity weights
def plot_ion_velocity_comparison(datasets, labels, save_dir="plots_comparison", offset=True):
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))  # Larger figure size for better spacing
    
    for idx, data in enumerate(datasets):
        # Load the various data types for each weight
        observed_data = data['observed_data']
        optimized_data = data['optimized_data']
        initial_guess_data = data['initial_guess_data']
        
        # Convert ion velocity lists to numpy arrays
        observed_ion_velocity = np.array(observed_data['ion_velocity'][0])
        optimized_ion_velocity = np.array(optimized_data['ion_velocity'][0])
        initial_guess_ion_velocity = np.array(initial_guess_data['ion_velocity'][0])
        
        # Define grids based on ion velocity arrays
        z_grid_observed = np.linspace(0, 1, len(observed_ion_velocity))
        z_grid_optimized = np.linspace(0, 1, len(optimized_ion_velocity))
        z_grid_initial = np.linspace(0, 1, len(initial_guess_ion_velocity))

        # Apply offset to each dataset for better visual distinction (optional)
        offset_value = 1000 * idx if offset else 0

        # Plot the observed data (ground truth) once for all comparisons
        if idx == 0:
            plt.plot(z_grid_observed, observed_ion_velocity + offset_value, 
                     linestyle='-', color='red', label='Observed (Ground Truth)', linewidth=3, alpha=0.7)

        # Plot the initial guess ion velocity for each dataset
        plt.plot(z_grid_initial, initial_guess_ion_velocity + offset_value, 
                 linestyle='--', label=f'{labels[idx]} Initial Guess', linewidth=1.0, marker='o', markersize=6)

        # Plot the optimized ion velocity for each dataset
        plt.plot(z_grid_optimized, optimized_ion_velocity + offset_value, 
                 linestyle='-.', label=f'{labels[idx]} Optimized', linewidth=3)

    plt.title('Ion Velocity Comparison (Observed, Initial Guess, Optimized)', fontsize=14)
    plt.xlabel('Normalized z', fontsize=12)
    plt.ylabel('Ion Velocity', fontsize=12)
    plt.grid(True, linestyle=':', linewidth=1)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()  # Adjust layout to prevent overlapping elements
    plt.savefig(os.path.join(save_dir, 'ion_velocity_comparison.png'))
    plt.close()
    print(f"Ion velocity comparison plot saved in {save_dir}/ion_velocity_comparison.png")


# Function to plot thrust comparison for different ion velocity weights
def plot_thrust_comparison(datasets, labels, save_dir="plots_comparison"):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    
    for idx, data in enumerate(datasets):
        observed_data = data['observed_data']
        optimized_data = data['optimized_data']
        initial_guess_data = data['initial_guess_data']

        # Extract thrust values
        observed_thrust = observed_data['thrust'][-1]  # Final observed thrust
        optimized_thrust = optimized_data['thrust'][-1]  # Final optimized thrust
        initial_guess_thrust = initial_guess_data['thrust'][-1]  # Initial guess thrust

        # Plot thrust for observed, initial guess, and optimized results
        plt.bar(idx - 0.2, observed_thrust, width=0.2, label=f'{labels[idx]} Observed', color='red', alpha=0.7)
        plt.bar(idx, initial_guess_thrust, width=0.2, label=f'{labels[idx]} Initial Guess', color='blue', alpha=0.7)
        plt.bar(idx + 0.2, optimized_thrust, width=0.2, label=f'{labels[idx]} Optimized', color='green', alpha=0.7)

    plt.title('Thrust Comparison (Observed, Initial Guess, Optimized)', fontsize=14)
    plt.xlabel('Ion Velocity Weight', fontsize=12)
    plt.ylabel('Thrust', fontsize=12)
    plt.xticks(range(len(labels)), labels, fontsize=10)
    plt.grid(True, linestyle=':', linewidth=1)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'thrust_comparison.png'))
    plt.close()
    print(f"Thrust comparison plot saved in {save_dir}/thrust_comparison.png")


# Function to plot residuals (thrust, ion velocity, discharge current)
def plot_residuals_comparison(datasets, labels, save_dir="plots_comparison"):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    
    for idx, data in enumerate(datasets):
        observed_data = data['observed_data']
        optimized_data = data['optimized_data']
        initial_guess_data = data['initial_guess_data']

        # Calculate residuals (Observed - Optimized) for thrust, ion velocity, and discharge current
        thrust_residual = observed_data['thrust'][-1] - optimized_data['thrust'][-1]
        ion_velocity_residual = np.mean(np.abs(np.array(observed_data['ion_velocity'][0]) - np.array(optimized_data['ion_velocity'][0])))
        discharge_current_residual = observed_data['discharge_current'][-1] - optimized_data['discharge_current'][-1]

        # Plot residuals for each weight
        plt.bar(idx - 0.2, thrust_residual, width=0.2, label=f'{labels[idx]} Thrust Residual', color='blue', alpha=0.7)
        plt.bar(idx, ion_velocity_residual, width=0.2, label=f'{labels[idx]} Ion Velocity Residual', color='green', alpha=0.7)
        plt.bar(idx + 0.2, discharge_current_residual, width=0.2, label=f'{labels[idx]} Discharge Current Residual', color='orange', alpha=0.7)

    plt.title('Residuals Comparison (Thrust, Ion Velocity, Discharge Current)', fontsize=14)
    plt.xlabel('Ion Velocity Weight', fontsize=12)
    plt.ylabel('Residuals', fontsize=12)
    plt.xticks(range(len(labels)), labels, fontsize=10)
    plt.grid(True, linestyle=':', linewidth=1)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'residuals_comparison.png'))
    plt.close()
    print(f"Residuals comparison plot saved in {save_dir}/residuals_comparison.png")


# Function to plot convergence (loss over iterations)
def plot_convergence(loss_files, labels, save_dir="plots_comparison"):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    
    for idx, loss_file in enumerate(loss_files):
        # Load the loss values for each weight
        loss_values = load_json(loss_file)

        # Plot the loss over iterations for each weight
        plt.plot(range(1, len(loss_values) + 1), loss_values, marker='o', label=f'{labels[idx]} Loss', linewidth=2)

    plt.title('Convergence (Loss over Iterations)', fontsize=14)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle=':', linewidth=1)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'convergence_loss.png'))
    plt.close()
    print(f"Convergence plot saved in {save_dir}/convergence_loss.png")


def main():
    # Load data for ion_velocity_weight = 0.1
    data_w01 = {
        'observed_data': load_json('w_01_observed_data_map.json'),
        'optimized_data': load_json('w_01_optimized_twozonebohm_result.json'),
        'initial_guess_data': load_json('w_01_best_initial_guess_result.json')
    }

    # Load data for ion_velocity_weight = 1.0
    data_w1 = {
        'observed_data': load_json('w_01_observed_data_map.json'),  # Same observed data
        'optimized_data': load_json('w_1_optimized_twozonebohm_result.json'),
        'initial_guess_data': load_json('w_1_best_initial_guess_result.json')
    }
    
    # Load data for ion_velocity_weight = 2.0
    data_w2 = {
        'observed_data': load_json('w_01_observed_data_map.json'),  # Same observed data
        'optimized_data': load_json('w_2_optimized_twozonebohm_result.json'),
        'initial_guess_data': load_json('w_2_best_initial_guess_result.json')
    }
    
    # Load data for ion_velocity_weight = 3.0
    data_w3 = {
        'observed_data': load_json('w_01_observed_data_map.json'),  # Same observed data
        'optimized_data': load_json('w_3_optimized_twozonebohm_result.json'),
        'initial_guess_data': load_json('w_3_best_initial_guess_result.json')
    }

    datasets = [data_w01, data_w1, data_w2, data_w3]
    labels = ['Weight 0.1', 'Weight 1.0', 'Weight 2.0', 'Weight 3.0']

    # Plot ion velocity comparison
    plot_ion_velocity_comparison(datasets, labels, offset=True)

    # Plot thrust comparison
    plot_thrust_comparison(datasets, labels)

    # Plot residuals comparison
    plot_residuals_comparison(datasets, labels)

    # Plot convergence (loss) comparison
    loss_files = ['loss_values_w0.1.json', 'loss_values_w1_0.json', 'loss_values_w2_0.json', 'loss_values_w3_0.json']
    plot_convergence(loss_files, labels)


if __name__ == "__main__":
    main()
