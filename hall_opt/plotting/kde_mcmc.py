import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import gaussian_kde
from hall_opt.plotting.common_setup import get_common_paths
from hall_opt.utils.data_loader import load_data
from hall_opt.config.dict import Settings

def plot_mcmc_kde(settings: Settings):
    """
    Generate and save a 2D KDE plot of posterior samples with MAP estimate.
    """
    # 1. Get file paths
    paths = get_common_paths(settings, analysis_type="mcmc")
    plots_dir = paths["plots_dir"]

    # 2. Load MCMC data (DataFrame with log_c1 and log_alpha columns)
    samples = load_data(settings, analysis_type="mcmc")

    # Extract sample arrays
    log_c1_samples = samples["log_c1"].values
    log_alpha_samples = samples["log_alpha"].values

    # 3. Create 2D KDE
    kde = gaussian_kde(np.vstack([log_c1_samples, log_alpha_samples]))

    # Create grid
    x = np.linspace(log_c1_samples.min(), log_c1_samples.max(), 100)
    y = np.linspace(log_alpha_samples.min(), log_alpha_samples.max(), 100)
    X, Y = np.meshgrid(x, y)
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = kde(positions).reshape(X.shape)

    # 4. Plot the KDE
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, Z, levels=50, cmap="viridis", norm=LogNorm())
    plt.colorbar(contour, label="Density")

    # Overlay credible regions
    plt.contour(X, Y, Z, levels=[0.01, 0.05, 0.1, 0.5, 0.9], colors="white", linewidths=0.8, linestyles='--')

    # Get MAP estimate (max density point)
    idx_max = np.argmax(Z)
    c1_log_map = X.ravel()[idx_max]
    alpha_log_map = Y.ravel()[idx_max]

    plt.scatter(c1_log_map, alpha_log_map, color='red', s=100, label="MCMC")

    # Labels and title
    plt.xlabel("log(c1)")
    plt.ylabel("log(alpha)")
    plt.title("2D KDE of Posterior Distribution with MCMC")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 5. Save plot
    plot_path = os.path.join(plots_dir, "2d_kde_with_mcmc.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] 2D KDE with MAP plot saved to {plot_path}")
