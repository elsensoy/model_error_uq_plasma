import json
import os
import matplotlib.pyplot as plt
import numpy as np

def load_json(filename):
    file_path = os.path.join("..", "results-LBFGSB", filename)  # Load from the results directory
    with open(file_path, 'r') as f:
        return json.load(f)

# Plotting loss values for multiple ion velocity weights on the same plot
def plot_loss_comparison(loss_files, labels, save_dir="plots_comparison"):
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))

    # Iterate over each loss file and plot the loss values
    for loss_file, label in zip(loss_files, labels):
        loss_values = load_json(loss_file)
        iterations = list(range(1, len(loss_values) + 1))

        # Plotten der Verlustwerte für dieses Gewicht
        plt.plot(iterations, loss_values, marker='o', linestyle='-', label=f'{label}', alpha=0.8)

    plt.title('Loss Comparison for Different Ion Velocity Weights')
    plt.xlabel('Iteration')
    plt.ylabel('Loss Value')
    plt.grid(True)
    plt.legend()

    plt.savefig(os.path.join(save_dir, 'loss_comparison.png'))
    plt.close()
    print(f"Loss comparison plot saved in {save_dir}/loss_comparison.png")

def main():
    # List of loss files and corresponding labels for different ion velocity weights
    loss_files = [
        'loss_values_w_0_1.json',
        'loss_values_w_1_0.json',
        'loss_values_w_2_0.json',
        'loss_values_w_3_0.json',
        'loss_values_w_5_0.json',
        'loss_values_w_10_0.json',
        'loss_values_w_1e-10.json'
    ]
    labels = [
        'Weight 0.1',
        'Weight 1.0',
        'Weight 2.0',
        'Weight 3.0',
        'Weight 5.0',
        'Weight 10.0',
        'Weight 1e-10'
    ]

    # plotten des Verlustvergleichs für alle Gewichte
    plot_loss_comparison(loss_files, labels, save_dir="plots_comparison")

if __name__ == "__main__":
    main()
