import json 
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# Hilfsfunktion zum Laden von JSON-Daten
def load_json(filename):
    file_path = os.path.join("..", "results-LBFGSB", filename)  # Load from the results directory
    with open(file_path, 'r') as f:
        return json.load(f)
import matplotlib.pyplot as plt
import matplotlib
def plot_ion_velocity_comparison(datasets, labels, save_dir="plots_comparison"):
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))  # Standard portrait size
    
    # Use a larger color palette like 'tab20' for more distinct colors
    color_map = matplotlib.colormaps.get_cmap('tab10')  # Get colormap once, outside the loop

    # Define different markers and line styles for each dataset
    optimized_marker = 'x'  # X marker for optimized
    initial_guess_marker = 'o'  # Circle marker for initial guesses
    optimized_line_style = '-'  # Solid line for optimized
    initial_guess_line_style = '--'  # Dashed line for initial guesses

    for idx, data in enumerate(datasets):
        observed_data = data['observed_data']
        optimized_data = data['optimized_data']
        initial_guess_data = data['initial_guess_data']

        observed_ion_velocity = np.array(observed_data['ion_velocity'][0])
        optimized_ion_velocity = np.array(optimized_data['ion_velocity'][0])
        initial_guess_ion_velocity = np.array(initial_guess_data['ion_velocity'][0])

        z_grid = np.linspace(0, 1, len(observed_ion_velocity))

        # Assign distinct color for each dataset based on the index
        color = color_map(idx)

        # Plot observed data (ground truth) only once for all comparisons
        if idx == 0:
            plt.plot(z_grid, observed_ion_velocity, 
                     linestyle='-', color='purple', linewidth=3, alpha=0.7, label='Observed (Ground Truth)')
            plt.scatter(z_grid, observed_ion_velocity, color='purple', s=50)

        # Plot the initial guess with distinct marker and line style (transparent)
        plt.plot(z_grid, initial_guess_ion_velocity, 
                 linestyle=initial_guess_line_style, linewidth=2, color=color, label=f'{labels[idx]} Initial Guess', alpha=0.9)
        plt.scatter(z_grid, initial_guess_ion_velocity, color=color, s=80, marker=initial_guess_marker, alpha=0.6)

        # Plot the optimized values with 'x' marker and solid line
        plt.plot(z_grid, optimized_ion_velocity, 
                 linestyle=optimized_line_style, linewidth=3, color=color, label=f'{labels[idx]} Optimized', alpha=1)
        plt.scatter(z_grid, optimized_ion_velocity, color=color, s=100, marker=optimized_marker)

    # Adjustments for plot aesthetics
    plt.title('Ion Velocity Comparison (Observed, Initial Guess, Optimized)', fontsize=14)
    plt.xlabel('Normalized z', fontsize=12)
    plt.ylabel('Ion Velocity', fontsize=12)
    plt.grid(True, linestyle=':', linewidth=1)  # Dotted grid lines for reduced distraction
    
    # Add a legend with distinct colors for clarity
    plt.legend(loc='best', fontsize=10)
    
    plt.tight_layout()  # Adjust layout to prevent overlapping elements
    
    # Save the plot
    plt.savefig(os.path.join(save_dir, 'ion_velocity_comparison_enhanced.png'))
    plt.close()
    print(f"Ion velocity comparison plot saved in {save_dir}/ion_velocity_comparison_enhanced.png")

def plot_each_combined_weight(datasets, labels, save_dir="plots_comparison"):
    os.makedirs(save_dir, exist_ok=True)

    # Define markers and line styles
    optimized_marker = 'x'  # X marker for optimized
    initial_guess_marker = 'o'  # Circle marker for initial guesses
    optimized_line_style = '-'  # Solid line for optimized
    initial_guess_line_style = '--'  # Dashed line for initial guesses

    for idx, data in enumerate(datasets):
        plt.figure(figsize=(12, 8))  # Standard portrait size
        
        initial_guess_data = data['initial_guess_data']
        optimized_data = data['optimized_data']
        initial_guess_ion_velocity = np.array(initial_guess_data['ion_velocity'][0])
        optimized_ion_velocity = np.array(optimized_data['ion_velocity'][0])
        z_grid = np.linspace(0, 1, len(initial_guess_ion_velocity))

        # Use distinct colors for each dataset
        color = matplotlib.colormaps.get_cmap('tab10')(idx)

        # Plot the initial guess with distinct marker and dashed line
        plt.plot(z_grid, initial_guess_ion_velocity, linestyle=initial_guess_line_style, linewidth=2, color=color, label=f'{labels[idx]} Initial Guess', alpha=0.8)
        plt.scatter(z_grid, initial_guess_ion_velocity, color=color, s=80, marker=initial_guess_marker, alpha=0.6)

        # Plot the optimized values with distinct marker and solid line
        plt.plot(z_grid, optimized_ion_velocity, linestyle=optimized_line_style, linewidth=3, color=color, label=f'{labels[idx]} Optimized', alpha=1)
        plt.scatter(z_grid, optimized_ion_velocity, color=color, s=100, marker=optimized_marker)

        # Plot adjustments
        plt.title(f'Comparison for {labels[idx]}: Initial Guess and Optimized', fontsize=14)
        plt.xlabel('Normalized z', fontsize=12)
        plt.ylabel('Ion Velocity', fontsize=12)
        plt.grid(True, linestyle=':', linewidth=1)
        plt.legend(loc='best', fontsize=10)

        # Save the plot for each combined weight
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'combined_initial_optimized_{labels[idx].replace(" ", "_")}.png'))
        plt.close()
        print(f"Combined initial guess and optimized plot saved for {labels[idx]} in {save_dir}/combined_initial_optimized_{labels[idx].replace(' ', '_')}.png")

# Separate Plot for Each Initial Guess
def plot_each_initial_guess(datasets, labels, save_dir="plots_comparison"):
    os.makedirs(save_dir, exist_ok=True)

    for idx, data in enumerate(datasets):
        plt.figure(figsize=(12, 8))  # Standard portrait size
        
        initial_guess_data = data['initial_guess_data']
        initial_guess_ion_velocity = np.array(initial_guess_data['ion_velocity'][0])
        z_grid = np.linspace(0, 1, len(initial_guess_ion_velocity))

        # Assign a distinct color for each dataset
        color = plt.cm.get_cmap('tab10')(idx)

        # Plot the initial guess with distinct marker
        plt.plot(z_grid, initial_guess_ion_velocity, linestyle='--', linewidth=2, color=color, label=f'{labels[idx]} Initial Guess', alpha=0.8)
        plt.scatter(z_grid, initial_guess_ion_velocity, color=color, s=80, marker='o', alpha=0.8)

        plt.title(f'Initial Guess Ion Velocity for {labels[idx]}', fontsize=14)
        plt.xlabel('Normalized z', fontsize=12)
        plt.ylabel('Ion Velocity', fontsize=12)
        plt.grid(True, linestyle=':', linewidth=1)

        # Add a legend
        plt.legend(loc='best', fontsize=10)

        # Save the plot for each initial guess
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'initial_guess_{labels[idx].replace(" ", "_")}.png'))
        plt.close()
        print(f"Initial guess plot saved for {labels[idx]} in {save_dir}/initial_guess_{labels[idx].replace(' ', '_')}.png")
# Separate Plot for Optimized Values Only
# Separate Plot for Optimized Values Only
def plot_each_optimized_value(datasets, labels, save_dir="plots_comparison"):
    os.makedirs(save_dir, exist_ok=True)

    for idx, data in enumerate(datasets):
        plt.figure(figsize=(12, 8))  # Standard portrait size
        
        optimized_data = data['optimized_data']
        optimized_ion_velocity = np.array(optimized_data['ion_velocity'][0])
        z_grid = np.linspace(0, 1, len(optimized_ion_velocity))

        # Use a larger color palette for distinct colors
        color = matplotlib.colormaps.get_cmap('tab10')(idx)

        # Plot the optimized values with distinct marker and solid line
        plt.plot(z_grid, optimized_ion_velocity, linestyle='-', linewidth=2, color=color, label=f'{labels[idx]} Optimized', alpha=0.8)
        plt.scatter(z_grid, optimized_ion_velocity, color=color, s=80, marker='x', alpha=0.8)

        # Add text labels for each optimized value
        for i, (z, val_opt) in enumerate(zip(z_grid, optimized_ion_velocity)):
            plt.text(z, val_opt, f'{val_opt:.1f}', fontsize=8, color=color, ha='right', va='bottom')

        plt.title(f'Optimized Ion Velocity for {labels[idx]}', fontsize=14)
        plt.xlabel('Normalized z', fontsize=12)
        plt.ylabel('Ion Velocity', fontsize=12)
        plt.grid(True, linestyle=':', linewidth=1)

        # Add a legend
        plt.legend(loc='best', fontsize=10)

        # Save the plot for each optimized value
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'optimized_value_{labels[idx].replace(" ", "_")}.png'))
        plt.close()

        # Correct print statement
        print(f"Optimized value plot saved for {labels[idx]} in {save_dir}/optimized_value_{labels[idx].replace(' ', '_')}.png")



def plot_initial_guess_only(datasets, labels, save_dir="plots_comparison"):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(12, 8))  # Standard portrait size
    
    # Use the new way to get colormap: matplotlib.colormaps.get_cmap
    color_map = matplotlib.colormaps.get_cmap('tab10')  # Removed the len(datasets) argument

    for idx, data in enumerate(datasets):
        # Load the initial guess data only
        initial_guess_data = data['initial_guess_data']

        # Convert ion velocity lists to numpy arrays
        initial_guess_ion_velocity = np.array(initial_guess_data['ion_velocity'][0])

        # Print the initial guess data for debugging
        print(f"Initial Guess for {labels[idx]}: {initial_guess_ion_velocity}")

        # Define grids based on ion velocity arrays
        z_grid = np.linspace(0, 1, len(initial_guess_ion_velocity))

        # Assign distinct color for each dataset based on the index
        color = color_map(idx)

        # Plot the initial guess ion velocity for each dataset
        plt.plot(z_grid, initial_guess_ion_velocity, 
                 linestyle='--', linewidth=2, color=color, label=f'{labels[idx]} Initial Guess')

        # Scatter plot for the initial guess grid points
        plt.scatter(z_grid, initial_guess_ion_velocity, color=color, s=50)

    # Adjustments for plot aesthetics
    plt.title('Initial Guess Ion Velocity Comparison', fontsize=14)
    plt.xlabel('Normalized z', fontsize=12)
    plt.ylabel('Ion Velocity', fontsize=12)
    plt.grid(True, linestyle=':', linewidth=1)
    
    # Add a legend with distinct colors for clarity
    plt.legend(loc='best', fontsize=10)
    
    plt.tight_layout()  # Adjust layout to prevent overlapping elements
    
    # Save the plot
    plt.savefig(os.path.join(save_dir, 'initial_guess_comparison_debug.png'))
    plt.close()
    print(f"Initial guess comparison plot saved in {save_dir}/initial_guess_comparison_debug.png")


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

   # Matching labels to datasets (ensure both have the same length)
    datasets = [ data_w01, data_w2, data_w10, data_w1e10]
    labels = ['Weight 0.1', 'Weight 2.0', 'Weight 10.0', 'Weight 1e-10']

    #  comparison for ion_velocity_weight = 0.1, 1.0, 2.0, 3.0, 5.0, 10.0, and 1e-10
    plot_ion_velocity_comparison(
        datasets=datasets,
        labels=labels,
    )
    plot_each_initial_guess(datasets=datasets, labels=labels, save_dir="plots_comparison")


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

    
    plot_each_combined_weight(datasets=datasets, labels=labels)
    plot_each_optimized_value(datasets=datasets, labels=labels)
    plot_each_initial_guess(datasets=datasets, labels=labels)

    plot_initial_guess_only(
        datasets=datasets,
        labels=labels)

if __name__ == "__main__":
    main()
