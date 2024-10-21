import json 
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# Hilfsfunktion zum Laden von JSON-Daten
def load_json(filename):
    file_path = os.path.join("..", "results-LBFGSB", filename)  # Load from the results directory
    with open(file_path, 'r') as f:
        return json.load(f)


def plot_ion_velocity_comparison(datasets, labels, save_dir="plots_comparison", offset=True):
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))  
    
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

        # skip offset (-delete it-)
        offset_value = 0 * idx if offset else 0

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


def plot_thrust_comparison(datasets, labels, save_dir="plots_comparison"):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    
    for idx, data in enumerate(datasets):
        observed_data = data['observed_data']
        optimized_data = data['optimized_data']
        initial_guess_data = data['initial_guess_data']

        # Extract thrust values
        observed_thrust = observed_data['thrust'][-1]  #  observed thrust
        optimized_thrust = optimized_data['thrust'][-1]  #  optimized thrust
        initial_guess_thrust = initial_guess_data['thrust'][-1]  # initial guess thrust

        #  thrust for observed, initial guess, and optimized results
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

def plot_thrust_comparison_bar(datasets, labels, save_dir="plots_comparison"):
    os.makedirs(save_dir, exist_ok=True)

    observed_thrust = []
    optimized_thrust = []
    initial_guess_thrust = []

    #  thrust data for bar plot
    for data in datasets:
        observed_data = data['observed_data']
        optimized_data = data['optimized_data']
        initial_guess_data = data['initial_guess_data']
        
        observed_thrust.append(observed_data['thrust'][-1])  
        optimized_thrust.append(optimized_data['thrust'][-1])
        initial_guess_thrust.append(initial_guess_data['thrust'][-1])

    # Create bar plot
    plt.figure(figsize=(10, 6))
    width = 0.2  
    x = np.arange(len(labels)) 

    plt.bar(x - width, observed_thrust, width, label='Observed', color='red')
    plt.bar(x, initial_guess_thrust, width, label='Initial Guess', color='blue')
    plt.bar(x + width, optimized_thrust, width, label='Optimized', color='green')
    plt.xlabel('Ion Velocity Weight')
    plt.ylabel('Thrust')
    plt.title('Thrust Comparison (Observed, Initial Guess, Optimized)')
    plt.xticks(x, labels)
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'thrust_comparison_bar.png'))
    plt.close()
    print(f"Bar plot saved in {save_dir}/thrust_comparison_bar.png")

def plot_thrust_delta_comparison(datasets, labels, save_dir="plots_comparison"):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 6)) 
    for idx, data in enumerate(datasets):
        observed_data = data['observed_data']
        optimized_data = data['optimized_data']
        initial_guess_data = data['initial_guess_data']

        # Extract thrust values
        observed_thrust = np.array(observed_data['thrust'][-1])  
        optimized_thrust = np.array(optimized_data['thrust'][-1]) 
        initial_guess_thrust = np.array(initial_guess_data['thrust'][-1]) 

        # Calculate delta (difference from observed thrust)
        optimized_thrust_delta = optimized_thrust - observed_thrust
        initial_guess_thrust_delta = initial_guess_thrust - observed_thrust

        # Plot the delta values
        plt.bar(idx - 0.2, initial_guess_thrust_delta, width=0.2, label=f'{labels[idx]} Initial Guess (Delta)', color='blue')
        plt.bar(idx + 0.2, optimized_thrust_delta, width=0.2, label=f'{labels[idx]} Optimized (Delta)', color='green')

    plt.title('Thrust Delta Plot: Difference from Observed Thrust', fontsize=14)
    plt.xlabel('Ion Velocity Weight', fontsize=12)
    plt.ylabel('Difference from Observed Thrust', fontsize=12)
    plt.xticks(range(len(labels)), labels, fontsize=10)
    plt.grid(True, linestyle=':', linewidth=1)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'thrust_delta_comparison.png'))
    plt.close()
    print(f"Thrust delta plot saved in {save_dir}/thrust_delta_comparison.png")


def plot_discharge_current_delta_comparison(datasets, labels, save_dir="plots_comparison"):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))  

    for idx, data in enumerate(datasets):
        observed_data = data['observed_data']
        optimized_data = data['optimized_data']
        initial_guess_data = data['initial_guess_data']

        # Extract discharge current values
        observed_discharge = np.array(observed_data['discharge_current'][-1]) 
        optimized_discharge = np.array(optimized_data['discharge_current'][-1]) 
        initial_guess_discharge = np.array(initial_guess_data['discharge_current'][-1]) 
        # Calculate delta (difference from observed discharge current)
        optimized_discharge_delta = optimized_discharge - observed_discharge
        initial_guess_discharge_delta = initial_guess_discharge - observed_discharge

        # Plot the delta values
        plt.bar(idx - 0.2, initial_guess_discharge_delta, width=0.2, label=f'{labels[idx]} Initial Guess (Delta)', color='blue')
        plt.bar(idx + 0.2, optimized_discharge_delta, width=0.2, label=f'{labels[idx]} Optimized (Delta)', color='green')

    plt.title('Discharge Current Delta Plot: Difference from Observed', fontsize=14)
    plt.xlabel('Ion Velocity Weight', fontsize=12)
    plt.ylabel('Difference from Observed Discharge Current', fontsize=12)
    plt.xticks(range(len(labels)), labels, fontsize=10)
    plt.grid(True, linestyle=':', linewidth=1)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'discharge_current_delta_comparison.png'))
    plt.close()
    print(f"Discharge current delta plot saved in {save_dir}/discharge_current_delta_comparison.png")


def plot_difference_comparison(datasets, labels, save_dir="plots_comparison"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(12, 8))  

    for idx, data in enumerate(datasets):
        observed_data = data['observed_data']
        optimized_data = data['optimized_data']
        initial_guess_data = data['initial_guess_data']

        # Convert ion velocity lists to numpy arrays
        observed_ion_velocity = np.array(observed_data['ion_velocity'][0])
        optimized_ion_velocity = np.array(optimized_data['ion_velocity'][0])
        initial_guess_ion_velocity = np.array(initial_guess_data['ion_velocity'][0])

        # Use the provided z_normalized values
        z_grid = np.array(observed_data['z_normalized'])

        # Calculate the difference from the observed data (delta)
        optimized_delta = optimized_ion_velocity - observed_ion_velocity
        initial_delta = initial_guess_ion_velocity - observed_ion_velocity

        # Plot the delta values
        plt.plot(z_grid, initial_delta, linestyle='--', label=f'{labels[idx]} Initial Guess (Delta)', linewidth=1.5, alpha=0.7)
        plt.plot(z_grid, optimized_delta, linestyle='-.', label=f'{labels[idx]} Optimized (Delta)', linewidth=1.5, alpha=0.7)

    plt.title('Delta Plot: Difference from Observed Ion Velocity', fontsize=14)
    plt.xlabel('Normalized z', fontsize=12)
    plt.ylabel('Difference from Observed', fontsize=12)
    plt.grid(True, linestyle=':', linewidth=1)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'delta_plot_comparison.png'))
    plt.close()
    print(f"Difference comparison plot saved in {save_dir}/delta_plot_comparison.png")

def main():
    #  data for ion_velocity_weight = 0.1
    data_w01 = {
        'observed_data': load_json('w_0.1_observed_data_map.json'),
        'optimized_data': load_json('w_0.1_optimized_twozonebohm_result.json'),
        'initial_guess_data': load_json('w_0.1_best_initial_guess_result.json')
    }

    #  data for ion_velocity_weight = 1.0
    data_w1 = {
        'observed_data': load_json('w_1.0_observed_data_map.json'),  # Same observed data
        'optimized_data': load_json('w_1.0_optimized_twozonebohm_result.json'),
        'initial_guess_data': load_json('w_1.0_best_initial_guess_result.json')
    }

    #  data for ion_velocity_weight = 2.0
    data_w2 = {
        'observed_data': load_json('w_2.0_observed_data_map.json'),  # Same observed data
        'optimized_data': load_json('w_2.0_optimized_twozonebohm_result.json'),
        'initial_guess_data': load_json('w_2.0_best_initial_guess_result.json')
    }

    #  data for ion_velocity_weight = 3.0
    data_w3 = {
        'observed_data': load_json('w_3.0_observed_data_map.json'),  
        'optimized_data': load_json('w_3.0_optimized_twozonebohm_result.json'),
        'initial_guess_data': load_json('w_3.0_best_initial_guess_result.json')
    }

    #  data for ion_velocity_weight = 5.0
    data_w5 = {
        'observed_data': load_json('w_5.0_observed_data_map.json'),
        'optimized_data': load_json('w_5.0_optimized_twozonebohm_result.json'),
        'initial_guess_data': load_json('w_5.0_best_initial_guess_result.json')
    }

    #  data for ion_velocity_weight = 10.0
    data_w10 = {
        'observed_data': load_json('w_10.0_observed_data_map.json'),
        'optimized_data': load_json('w_10.0_optimized_twozonebohm_result.json'),
        'initial_guess_data': load_json('w_10.0_best_initial_guess_result.json')
    }

    # Load data for ion_velocity_weight = 1e-10
    data_w1e10 = {
        'observed_data': load_json('w_1e-10_observed_data_map.json'),
        'optimized_data': load_json('w_1e-10_optimized_twozonebohm_result.json'),
        'initial_guess_data': load_json('w_1e-10_best_initial_guess_result.json')
    }

    # Create a list of datasets and labels including the new weight 1e-10
    datasets = [data_w01, data_w1, data_w2, data_w3, data_w5, data_w10, data_w1e10]
    labels = ['Weight 0.1', 'Weight 1.0', 'Weight 2.0', 'Weight 3.0', 'Weight 5.0', 'Weight 10.0', 'Weight 1e-10']

    #  comparison for ion_velocity_weight = 0.1, 1.0, 2.0, 3.0, 5.0, 10.0, and 1e-10
    plot_ion_velocity_comparison(
        datasets=datasets,
        labels=labels,
        offset=False
    )

    #  thrust comparison bar
    plot_thrust_comparison_bar(
        datasets=datasets,
        labels=labels
    )

    #  thrust comparison
    plot_thrust_comparison(
        datasets=datasets,
        labels=labels
    )

    # Plot delta comparison (difference from observed) for ion velocity
    plot_difference_comparison(
        datasets=datasets,
        labels=labels
    )

    # Plot delta comparison (difference from observed) for thrust
    plot_thrust_delta_comparison(
        datasets=datasets,
        labels=labels
    )

    plot_discharge_current_delta_comparison(
        datasets=datasets,
        labels=labels
    )

if __name__ == "__main__":
    main()
