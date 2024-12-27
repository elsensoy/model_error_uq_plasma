import json
import os
import matplotlib.pyplot as plt
import numpy as np
#from utils.save_data import load_json_data, subsample_data, save_results_to_json
from config.simulation import postprocess
# Helper function to load JSON data
def load_json(filename):
    file_path = os.path.join("..", "results-map", filename)  # Load from the results directory
    with open(file_path, 'r') as f:
        return json.load(f)

# Function to plot ion velocity comparison for multiple ion velocity weights
def plot_ion_velocity_comparison(datasets, labels, save_dir="plots", offset=True):
    os.makedirs(save_dir, exist_ok=True)
    ground_truth_postprocess = postprocess.copy()
    ground_truth_postprocess["output_file"] = "/mnt/c/Users/MRover/elsensoy/model_error_uq_plasma/hall_opt/map_/results-map/ground_truth.json"
    plt.figure(figsize=(12, 8))  # Larger figure size for better spacing
    
    for idx, data in enumerate(datasets):
        # Load the various data types for each weight
        observed_data = data['observed_data']
        optimized_data = data['optimized_data']
        initial_guess_data = data['initial_guess_data']
        
        # Convert ion velocity lists to numpy arrays
        observed_ion_velocity = np.array(observed_data["ui"][0])
        optimized_ion_velocity = np.array(optimized_data["ui"][0])
        initial_guess_ion_velocity = np.array(initial_guess_data["ui"][0])
        
        # Define grids based on ion velocity arrays
        z_grid_observed = np.linspace(0, 1, len(observed_ion_velocity))
        z_grid_optimized = np.linspace(0, 1, len(optimized_ion_velocity))
        z_grid_initial = np.linspace(0, 1, len(initial_guess_ion_velocity))

        # Apply offset to each dataset for better visual distinction (optional)
        offset_value = 1000 * idx if offset else 0

        # Plot the observed data (ground truth) once for all comparisons
        if idx == 0:
            plt.plot(z_grid_observed, observed_ion_velocity + offset_value, 
                     linestyle='-', color='red', label='Observed (Ground Truth for MAP)', linewidth=3, alpha=0.7)

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
def plot_thrust_comparison(datasets, labels, save_dir="plots"):
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


def main():

    # Load data for ion_velocity_weight = 2.0
    data_w2 = {
        'observed_data': load_json('observed_data_map.json'),  # Same observed data
        'optimized_data': load_json('initial_map_log.json'),
        'initial_guess_data': load_json('pre_mcmc_initial.json')
    }
    

    
    # Plot comparison for ion_velocity_weight = 0.1, 1.0, 2.0, and 3.0
    plot_ion_velocity_comparison(
        datasets=[data_w2],
        labels=[ 'Weight 2.0'],
        offset=True  # Set to True to apply vertical offset for visual separation
    )
    
    # Plot thrust comparison for the same datasets
    plot_thrust_comparison(
        datasets=[ data_w2],
        labels=[ 'Weight 2.0']
    )


if __name__ == "__main__":
    main()