import os
import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

def load_json(filename):
    file_path = os.path.join("..", "results-Nelder-Mead", filename)  # Load from the results directory
    with open(file_path, 'r') as f:
        return json.load(f)

def plot_ion_velocity_comparison(datasets, labels, save_dir="n_plots_comparison"):

    os.makedirs(save_dir, exist_ok=True)
    
    z_grid = datasets[0]['observed_data']['z_normalized']  

  
    initial_guess = datasets[0]['initial_guess_data']['ion_velocity'][0] if all(
        dataset['initial_guess_data']['ion_velocity'][0] == datasets[0]['initial_guess_data']['ion_velocity'][0] 
        for dataset in datasets
    ) else None

    plt.figure(figsize=(12, 6))

    # Plot initial condition if it's a single, consistent line, with scatter points
    if initial_guess is not None:
        plt.plot(z_grid, initial_guess, color='gray', linestyle=':', label='Initial Condition', linewidth=2)
        plt.scatter(z_grid, initial_guess, color='gray', s=50)
    common_observed = datasets[0]['observed_data']['ion_velocity'][0] if all(
        dataset['observed_data']['ion_velocity'][0] == datasets[0]['observed_data']['ion_velocity'][0]
        for dataset in datasets
    ) else None

   
    if common_observed is not None:
        plt.plot(z_grid, common_observed, color='purple', linestyle='-', label='Observed Data', linewidth=2)
        plt.scatter(z_grid, common_observed, color='purple', s=50)
    else:
      
        for dataset, label in zip(datasets, labels):
            observed = dataset['observed_data']['ion_velocity'][0]
            plt.plot(z_grid, observed, linestyle='--', color='purple', linewidth=2)
            plt.scatter(z_grid, observed, color='purple', s=50)


    for idx, (dataset, label) in enumerate(zip(datasets, labels)):
        optimized = dataset['optimized_data']['ion_velocity'][0]
        plt.plot(z_grid, optimized, linestyle='-', marker='s', markersize=4, color=plt.cm.tab10(idx),
                 label=f'Weight {label}')
        plt.scatter(z_grid, optimized, color=plt.cm.tab10(idx), s=50, marker='s')

    plt.xlabel('z_normalized')
    plt.ylabel('Ion Velocity')
    plt.title('Ion Velocity Comparison Across Weights - Nelder-Mead')
    plt.grid(True, linestyle=':', linewidth=0.5)  # Adding light, dotted grid lines for precision
    plt.legend(loc='best', fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/ion_velocity_comparison_full.png", format="png")
    plt.close()

    print(f"Ion velocity comparison plot saved in {save_dir}/ion_velocity_comparison_full.png")


def plot_difference_comparison(datasets, labels, save_dir="n_plots_comparison"):
    """
    Plot the delta (difference) from the observed ion velocity data for initial guess and optimized results.
    Simplified legend for clarity.
    """
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

        # Calculate the delta (difference) from the observed data
        optimized_delta = optimized_ion_velocity - observed_ion_velocity
        initial_delta = initial_guess_ion_velocity - observed_ion_velocity

        # Plot the initial condition delta (only label once for clarity)
        if idx == 0:
            plt.plot(z_grid, initial_delta, linestyle='--', color='gray', linewidth=1.5, label='Initial Condition(Delta)')
            plt.scatter(z_grid, initial_delta, color='gray', s=50)

        # Plot the optimized delta with simplified labeling
        plt.plot(z_grid, optimized_delta, linestyle='-', marker='s', markersize=4, linewidth=1.5,
                 color=plt.cm.tab10(idx), label=f'Delta for {labels[idx]}')
        plt.scatter(z_grid, optimized_delta, color=plt.cm.tab10(idx), s=50, marker='s')

    # Add a single baseline label for observed data
    plt.axhline(0, color='purple', linestyle='-', linewidth=2, label='Observed Baseline')

    # Set plot labels, title, and grid for the delta plot
    plt.title('Delta Plot: Difference from Observed Ion Velocity Nelder-Mead', fontsize=14)
    plt.xlabel('Normalized z', fontsize=12)
    plt.ylabel('Difference from Observed', fontsize=12)
    plt.grid(True, linestyle=':', linewidth=1)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'delta_plot_comparison.png'))
    plt.close()
    print(f"Difference comparison plot saved in {save_dir}/delta_plot_comparison.png")

# def plot_each_optimized_value(datasets, labels):
#     optimized_values = [dataset['optimized_data']['ion_velocity'] for dataset in datasets]
    
#     plt.figure(figsize=(12, 6))
#     for i, optimized in enumerate(optimized_values):
#         plt.plot(optimized, label=labels[i])
    
#     plt.xlabel('Index')
#     plt.ylabel('Optimized Ion Velocity')
#     plt.title('Optimized Ion Velocity per Weight')
#     plt.legend()
#     plt.savefig("optimized_velocity_per_weight.png", format="png")  # Save as PNG
#     plt.close()


def main():

    data_w01 = {
        'observed_data': load_json('nm_w_0.1_observed_data_map.json'),
        'optimized_data': load_json('nm_w_0.1_optimized_twozonebohm_result.json'),
        'initial_guess_data': load_json('nm_w_0.1_best_initial_guess_result.json')
    }

    #  data for ion_velocity_weight = 1.0
    # data_w1 = {
    #     'observed_data': load_json('nm_w_1.0_observed_data_map.json'),  # Same observed data
    #     'optimized_data': load_json('nm_w_1.0_optimized_twozonebohm_result.json'),
    #     'initial_guess_data': load_json('nm_w_1.0_best_initial_guess_result.json')
    # }

    #  data for ion_velocity_weight = 2.0
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

    # Load data for ion_velocity_weight = 1e-10
    data_w1e10 = {
        'observed_data': load_json('nm_w_1e-10_observed_data_map.json'),
        'optimized_data': load_json('nm_w_1e-10_optimized_twozonebohm_result.json'),
        'initial_guess_data': load_json('nm_w_1e-10_best_initial_guess_result.json')
    }



   # Matching labels to datasets (ensure both have the same length)
    datasets = [ data_w1e10, data_w01, data_w2]
    labels = ['Weight 1e-10','Weight 0.1', 'Weight 2.0']
    plot_difference_comparison(datasets=datasets, labels=labels)
    #  comparison for ion_velocity_weight = 0.1, 1.0, 2.0, 3.0, 5.0, 10.0, and 1e-10
    plot_ion_velocity_comparison(
        datasets=datasets,
        labels=labels,
    )
    
    # plot_each_optimized_value(datasets=datasets, labels=labels)
	


if __name__ == "__main__":
    main()
