import os
import json
import matplotlib.pyplot as plt
from pathlib import Path

from ..config.verifier import verify_all_yaml

def load_results(filename):
    """Load JSON data from a file."""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        print(f"Data loaded successfully from {filename}")
        return data
    except FileNotFoundError:
        print(f"ERROR: File not found: {filename}")
        return None
    except json.JSONDecodeError as e:
        print(f"ERROR: Error decoding JSON: {e}")
        return None

def plot_ion_velocity(results, output_dir):
    """Plot ion velocity vs. z_normalized and save the figure."""
    
    if "ion_velocity" not in results or "z_normalized" not in results:
        print("ERROR: Missing required keys in results.")
        return

    # Extract Data
    ion_velocity = results["ion_velocity"]
    z_normalized = results["z_normalized"]

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    else:
        print(f" Output directory already exists: {output_dir}")

    # Debugging: Print final plot save location
    print(f" DEBUG: Plot will be saved to {output_dir}")
    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(z_normalized, ion_velocity, marker='o', linestyle='-', label="Ion Velocity")
    plt.xlabel("Z-Normalized Position")
    plt.ylabel("Ion Velocity (m/s)")
    plt.title("Ion Velocity vs. Z-Normalized")
    plt.legend()
    plt.grid(True)

    # Save Plot
    plot_filename = os.path.join(output_dir, "ground_truth_subsampled_plot.png")
    plt.savefig(plot_filename)
    plt.close()
    print(f"Plot saved to {plot_filename}")

