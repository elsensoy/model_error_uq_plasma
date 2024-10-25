import json 
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Hilfsfunktion zum Laden von JSON-Daten
def load_json(filename):
    file_path = os.path.join("..", "results-Nelder-Mead", filename)  # Load from the results directory
    with open(file_path, 'r') as f:
        return json.load(f)
def plot_combined_weights_2_10_debug(datasets, labels, save_dir="nm_plots_comparison"):
    os.makedirs(save_dir, exist_ok=True)

    # Filter only for weights 2.0 and 10.0
    filtered_datasets = [datasets[idx] for idx, label in enumerate(labels) if '2.0' in label or '10.0' in label]
    filtered_labels = [label for label in labels if '2.0' in label or '10.0' in label]

    for idx, data in enumerate(filtered_datasets):
        plt.figure(figsize=(12, 8))  # Standard portrait size
        
        initial_guess_data = data['initial_guess_data']
        optimized_data = data['optimized_data']

        initial_guess_ion_velocity = np.array(initial_guess_data['ion_velocity'][0])
        optimized_ion_velocity = np.array(optimized_data['ion_velocity'][0])

        # Print values for debugging
        print(f"Initial Guess for {filtered_labels[idx]}: {initial_guess_ion_velocity}")
        print(f"Optimized Values for {filtered_labels[idx]}: {optimized_ion_velocity}")

        z_grid = np.linspace(0, 1, len(initial_guess_ion_velocity))

        # Assign a distinct color for each dataset
        color = plt.cm.get_cmap('tab10')(idx)

        # Plot the initial guess with distinct marker
        plt.plot(z_grid, initial_guess_ion_velocity, linestyle='--', linewidth=2, color=color, label=f'{filtered_labels[idx]} Initial Guess', alpha=0.8)
        plt.scatter(z_grid, initial_guess_ion_velocity, color=color, s=80, marker='o', alpha=0.8)

        # Plot the optimized values with 'x' marker and solid line
        plt.plot(z_grid, optimized_ion_velocity, linestyle='-', linewidth=3, color=color, label=f'{filtered_labels[idx]} Optimized', alpha=1)
        plt.scatter(z_grid, optimized_ion_velocity, color=color, s=100, marker='x')

        # Add labels for optimized values
        for i in range(len(z_grid)):
            plt.text(z_grid[i], optimized_ion_velocity[i], f'{optimized_ion_velocity[i]:.1f}', 
                     fontsize=9, color=color, ha='left', va='bottom')

        plt.title(f'Comparison for {filtered_labels[idx]}: Initial Guess and Optimized', fontsize=14)
        plt.xlabel('Normalized z', fontsize=12)
        plt.ylabel('Ion Velocity', fontsize=12)
        plt.grid(True, linestyle=':', linewidth=1)

        # Add a legend
        plt.legend(loc='best', fontsize=10)

        # Save the plot for each weight (2.0 and 10.0)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'combined_weights_2_10_{filtered_labels[idx].replace(" ", "_")}.png'))
        plt.close()
        print(f"Combined plot saved for {filtered_labels[idx]} in {save_dir}/combined_weights_2_10_{filtered_labels[idx].replace(' ', '_')}.png")

def main():


    data_w2 = {
        'observed_data': load_json('nm_w_2.0_observed_data_map.json'),  # Same observed data
        'optimized_data': load_json('nm_w_2.0_optimized_twozonebohm_result.json'),
        'initial_guess_data': load_json('nm_w_2.0_best_initial_guess_result.json')
    }

    #  data for ion_velocity_weight = 10.0
    data_w10 = {
        'observed_data': load_json('nm_w_10.0_observed_data_map.json'),
        'optimized_data': load_json('nm_w_10.0_optimized_twozonebohm_result.json'),
        'initial_guess_data': load_json('nm_w_10.0_best_initial_guess_result.json')
    }

    # Create a list of datasets and labels including the new weight 1e-10
    datasets = [data_w2, data_w10]
    labels = ['Weight 2.0', 'Weight 10.0']
    # Debugging: Print the first values for initial guess and optimized data for each dataset
    for idx, data in enumerate(datasets):
        initial_guess_data = data['initial_guess_data']
        optimized_data = data['optimized_data']

        # Print the first value for initial guess and optimized ion velocities
        print(f"First value for {labels[idx]} Initial Guess: {initial_guess_data['ion_velocity'][0][0]}")
        print(f"First value for {labels[idx]} Optimized: {optimized_data['ion_velocity'][0][0]}")
       
    plot_combined_weights_2_10_debug(datasets, labels)


if __name__ == "__main__":
    main()
