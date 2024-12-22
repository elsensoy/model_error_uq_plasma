import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pandas as pd
from matplotlib.colors import LogNorm

# Import the helper functions and paths
from common_setup import load_data, get_common_paths  

def plot_2d_kde(samples, plots_dir, map_v1, map_alpha):
    """
    Generate and save a 2D KDE plot of posterior samples with MAP estimate.

    Parameters:
        samples (DataFrame): MCMC samples with log_v1 and log_alpha.
        plots_dir (str): Directory to save the plot.
        map_v1 (float): log10(v1) for MAP estimate.
        map_alpha (float): log10(alpha) for MAP estimate.
    """
    # Extract log10 parameters for plotting
    log_v1_samples = samples["log_v1"].values
    log_alpha_samples = samples["log_alpha"].values

    # Create 2D KDE
    kde = gaussian_kde(np.vstack([log_v1_samples, log_alpha_samples]))

    # Create grid for evaluation
    x = np.linspace(log_v1_samples.min(), log_v1_samples.max(), 100)
    y = np.linspace(log_alpha_samples.min(), log_alpha_samples.max(), 100)
    X, Y = np.meshgrid(x, y)
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = kde(positions).reshape(X.shape)

    # Plot the KDE
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, Z, levels=50, cmap="viridis", norm=LogNorm())
    plt.colorbar(contour, label="Density")

    # Overlay credible regions
    plt.contour(X, Y, Z, levels=[0.01, 0.05, 0.1, 0.5, 0.9], colors="white", linewidths=0.8, linestyles='--')

    # Add the MAP estimate as a red point
    plt.scatter(map_v1, map_alpha, color='red', s=100, label="MAP Estimate")

    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)

    # Labels and title
    plt.xlabel("log10(v1)")
    plt.ylabel("log10(alpha)")
    plt.title("2D KDE of Posterior Distribution with MAP Estimate")

    # Legend
    plt.legend()

    # Save the plot
    plot_path = os.path.join(plots_dir, "2d_kde_with_map.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"2D KDE with MAP plot saved to {plot_path}")


def main():
    # Load data using helper functions
    samples, truth_data, pre_mcmc_data, initial_params = load_data()
    paths = get_common_paths()
    plots_dir = paths["plots_dir"]

    # Compute MAP estimate
    v1_map = 2.0124301397620146
    v2_map = 2.0028894367014445
    log_v1_map = np.log10(v1_map)
    log_alpha_map = np.log10(v2_map / v1_map)

    # Generate and save the 2D KDE plot with MAP estimate
    plot_2d_kde(samples, plots_dir, log_v1_map, log_alpha_map)

if __name__ == "__main__":
    main()
