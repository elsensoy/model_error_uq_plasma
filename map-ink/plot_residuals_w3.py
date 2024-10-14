import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Helper function to load JSON data from the results directory
def load_json(filename):
    file_path = os.path.join("..", "results-combined", filename)  # Load from the results directory
    with open(file_path, 'r') as f:
        return json.load(f)

# Function to plot combined residuals with logarithmic scaling
def plot_combined_residuals_log_scale(datasets, labels, save_dir=os.path.join("..", "plots", "map_ink")):
    os.makedirs(save_dir, exist_ok=True)  # Create the plots directory if it doesn't exist

    plt.figure(figsize=(10, 6))
    
    for idx, data in enumerate(datasets):
        observed_data = data['observed_data']
        optimized_data = data['optimized_data']
        
        # Compute residuals
        thrust_residual = abs(observed_data['thrust'][-1] - optimized_data['thrust'][-1])
        ion_velocity_residual = np.mean(np.abs(np.array(observed_data['ion_velocity'][0]) - np.array(optimized_data['ion_velocity'][0])))
        discharge_current_residual = abs(observed_data['discharge_current'][-1] - optimized_data['discharge_current'][-1])

        # Plot residuals using logarithmic scaling
        plt.bar(idx - 0.2, thrust_residual, width=0.2, label=f'{labels[idx]} Thrust Residual', color='blue')
        plt.bar(idx, ion_velocity_residual, width=0.2, label=f'{labels[idx]} Ion Velocity Residual', color='green')
        plt.bar(idx + 0.2, discharge_current_residual, width=0.2, label=f'{labels[idx]} Discharge Current Residual', color='orange')

    plt.yscale('log')  # Apply logarithmic scale to y-axis
    plt.title('Residuals Comparison with Logarithmic Scale', fontsize=14)
    plt.xlabel('Ion Velocity Weight', fontsize=12)
    plt.ylabel('Residuals', fontsize=12)
    plt.xticks(range(len(labels)), labels, fontsize=10)
    plt.grid(True, linestyle=':', linewidth=1)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'combined_residuals_log_scale.png'))  # Save plot in the correct folder
    plt.close()
    print(f"Combined residuals plot with logarithmic scaling saved in {save_dir}/combined_residuals_log_scale.png")


# Function to plot separate residuals for thrust, ion velocity, and discharge current
def plot_separate_residuals(datasets, labels, save_dir=os.path.join("..", "plots", "map_ink")):
    os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

    # Plot Thrust Residuals
    plt.figure(figsize=(10, 6))
    for idx, data in enumerate(datasets):
        observed_data = data['observed_data']
        optimized_data = data['optimized_data']

        thrust_residual = abs(observed_data['thrust'][-1] - optimized_data['thrust'][-1])
        plt.bar(idx, thrust_residual, width=0.2, label=f'{labels[idx]} Thrust Residual', color='blue')

    plt.title('Thrust Residuals Comparison', fontsize=14)
    plt.xlabel('Ion Velocity Weight', fontsize=12)
    plt.ylabel('Thrust Residuals', fontsize=12)
    plt.xticks(range(len(labels)), labels, fontsize=10)
    plt.grid(True, linestyle=':', linewidth=1)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'thrust_residuals.png'))
    plt.close()
    print(f"Thrust residuals plot saved in {save_dir}/thrust_residuals.png")

    # Plot Ion Velocity Residuals
    plt.figure(figsize=(10, 6))
    for idx, data in enumerate(datasets):
        observed_data = data['observed_data']
        optimized_data = data['optimized_data']

        ion_velocity_residual = np.mean(np.abs(np.array(observed_data['ion_velocity'][0]) - np.array(optimized_data['ion_velocity'][0])))

        plt.bar(idx, ion_velocity_residual, width=0.2, label=f'{labels[idx]} Ion Velocity Residual', color='green')

    plt.title('Ion Velocity Residuals Comparison', fontsize=14)
    plt.xlabel('Ion Velocity Weight', fontsize=12)
    plt.ylabel('Ion Velocity Residuals', fontsize=12)
    plt.xticks(range(len(labels)), labels, fontsize=10)
    plt.grid(True, linestyle=':', linewidth=1)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ion_velocity_residuals.png'))
    plt.close()
    print(f"Ion velocity residuals plot saved in {save_dir}/ion_velocity_residuals.png")

    # Plot Discharge Current Residuals
    plt.figure(figsize=(10, 6))
    for idx, data in enumerate(datasets):
        observed_data = data['observed_data']
        optimized_data = data['optimized_data']

        discharge_current_residual = abs(observed_data['discharge_current'][-1] - optimized_data['discharge_current'][-1])
        plt.bar(idx, discharge_current_residual, width=0.2, label=f'{labels[idx]} Discharge Current Residual', color='orange')

    plt.title('Discharge Current Residuals Comparison', fontsize=14)
    plt.xlabel('Ion Velocity Weight', fontsize=12)
    plt.ylabel('Discharge Current Residuals', fontsize=12)
    plt.xticks(range(len(labels)), labels, fontsize=10)
    plt.grid(True, linestyle=':', linewidth=1)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'discharge_current_residuals.png'))
    plt.close()
    print(f"Discharge current residuals plot saved in {save_dir}/discharge_current_residuals.png")


def main():
    # Load data for ion_velocity_weight = 0.1
    data_w01 = {
        'observed_data': load_json('w_0.1_observed_data_map.json'),
        'optimized_data': load_json('w_0.1_optimized_twozonebohm_result.json')
    }

    # Load data for ion_velocity_weight = 1.0
    data_w1 = {
        'observed_data': load_json('w_1.0_observed_data_map.json'),  # Same observed data
        'optimized_data': load_json('w_1.0_optimized_twozonebohm_result.json')
    }
    
    # Load data for ion_velocity_weight = 2.0
    data_w2 = {
        'observed_data': load_json('w_2.0_observed_data_map.json'),  # Same observed data
        'optimized_data': load_json('w_2.0_optimized_twozonebohm_result.json')
    }
    
    # Load data for ion_velocity_weight = 3.0
    data_w3 = {
        'observed_data': load_json('w_3.0_observed_data_map.json'),  # Same observed data
        'optimized_data': load_json('w_3.0_optimized_twozonebohm_result.json')
    }
    
    # Prepare the datasets and labels
    datasets = [data_w01, data_w1, data_w2, data_w3]
    labels = ['Weight 0.1', 'Weight 1.0', 'Weight 2.0', 'Weight 3.0']

    # Plot combined residuals with logarithmic scaling
    plot_combined_residuals_log_scale(datasets, labels)

    # Plot separate residuals for thrust, ion velocity, and discharge current
    plot_separate_residuals(datasets, labels)

if __name__ == "__main__":
    main()
