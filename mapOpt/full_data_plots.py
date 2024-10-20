import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Helper function to load JSON data
def load_json(filename):
    file_path = os.path.join("..", "results-combined", filename)  # Load from the results directory
    with open(file_path, 'r') as f:
        return json.load(f)

# Function to plot ion velocity comparison for multiple ion velocity weights using full data
def plot_ion_velocity_comparison_full(datasets, labels, save_dir="plots_comparison_full", offset=True):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(12, 8))  # Larger figure size for better spacing
    
    for idx, data in enumerate(datasets):
        # Load the various data types for each weight (using full data)
        observed_data = data['observed_data']
        optimized_data = data['optimized_data']
        initial_guess_data = data['initial_guess_data']

        # Convert ion velocity lists to numpy arrays (use the full data here)
        observed_ion_velocity = np.array(observed_data['ion_velocity'][0])  # Full data for observed
        optimized_ion_velocity = np.array(optimized_data['ion_velocity'][0])  # Full data for optimized
        initial_guess_ion_velocity = np.array(initial_guess_data['ion_velocity'][0])  # Full data for initial guess

        # Directly use the full z_normalized values
        z_grid_observed = np.array(observed_data['z_normalized'])  # Full z grid
        z_grid_optimized = np.array(optimized_data['z_normalized'])  # Full z grid
        z_grid_initial = np.array(initial_guess_data['z_normalized'])  # Full z grid

        # Apply offset to each dataset for better visual distinction (optional)
        offset_value = np.zeros_like(z_grid_optimized)  # No offset at z=0
        offset_value[1:] = 700 * idx  # Apply partial offset to avoid overlap

        # Plot the observed data (ground truth) once for all comparisons
        if idx == 0:
            plt.plot(z_grid_observed, observed_ion_velocity + offset_value, 
                     linestyle='-', color='red', label='Observed (Ground Truth for MAP)', linewidth=3, alpha=0.7)

        # Plot the initial guess ion velocity for each dataset
        plt.plot(z_grid_initial, initial_guess_ion_velocity + offset_value, 
                 linestyle='--', label=f'{labels[idx]} Initial Guess', linewidth=1.5, marker='o', markersize=6, alpha=0.7)

        # Plot the optimized ion velocity for each dataset
        plt.plot(z_grid_optimized, optimized_ion_velocity + offset_value, 
                 linestyle='-.', label=f'{labels[idx]} Optimized', linewidth=1.5, alpha=0.7)

    # Finalize the plot with title, labels, and legend
    plt.title('Ion Velocity Comparison (Observed, Initial Guess, Optimized)', fontsize=14)
    plt.xlabel('Normalized z', fontsize=12)
    plt.ylabel('Ion Velocity', fontsize=12)
    plt.grid(True, linestyle=':', linewidth=1)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()  # Adjust layout to prevent overlapping elements
    plt.savefig(os.path.join(save_dir, 'ion_velocity_comparison_full.png'))
    plt.close()
    print(f"Ion velocity comparison plot saved in {save_dir}/ion_velocity_comparison_full.png")


def main():
    # Load data for ion_velocity_weight = 0.1
    data_w01 = {
        'observed_data': load_json('w_0.1_observed_data_map.json'),
        'optimized_data': load_json('w_0.1_optimized_twozonebohm_result.json'),
        'initial_guess_data': load_json('w_0.1_best_initial_guess_result.json')
    }

    # Load data for ion_velocity_weight = 1.0
    data_w1 = {
        'observed_data': load_json('w_1.0_observed_data_map.json'),  # Same observed data
        'optimized_data': load_json('w_1.0_optimized_twozonebohm_result.json'),
        'initial_guess_data': load_json('w_1.0_best_initial_guess_result.json')
    }
    
    # Load data for ion_velocity_weight = 2.0
    data_w2 = {
        'observed_data': load_json('w_2.0_observed_data_map.json'),  # Same observed data
        'optimized_data': load_json('w_2.0_optimized_twozonebohm_result.json'),
        'initial_guess_data': load_json('w_2.0_best_initial_guess_result.json')
    }
    
    # Load data for ion_velocity_weight = 3.0
    data_w3 = {
        'observed_data': load_json('w_3.0_observed_data_map.json'),  # Same observed data
        'optimized_data': load_json('w_3.0_optimized_twozonebohm_result.json'),
        'initial_guess_data': load_json('w_3.0_best_initial_guess_result.json')
    }

    # Load data for ion_velocity_weight = 5.0
    data_w5 = {
        'observed_data': load_json('w_5.0_observed_data_map.json'),
        'optimized_data': load_json('w_5.0_optimized_twozonebohm_result.json'),
        'initial_guess_data': load_json('w_5.0_best_initial_guess_result.json')
    }

    # Load data for ion_velocity_weight = 10.0
    data_w10 = {
        'observed_data': load_json('w_10.0_observed_data_map.json'),
        'optimized_data': load_json('w_10.0_optimized_twozonebohm_result.json'),
        'initial_guess_data': load_json('w_10.0_best_initial_guess_result.json')
    }
    
    # Plot comparison for ion_velocity_weight = 0.1, 1.0, 2.0, 3.0, 5.0, and 10.0 using full data
    plot_ion_velocity_comparison_full(
        datasets=[data_w01, data_w1, data_w2, data_w3, data_w5, data_w10],
        labels=['Weight 0.1', 'Weight 1.0', 'Weight 2.0', 'Weight 3.0', 'Weight 5.0', 'Weight 10.0'],
        offset=True  # Set to True to apply vertical offset for visual separation
    )


if __name__ == "__main__":
    main()
