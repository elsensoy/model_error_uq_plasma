import matplotlib.pyplot as plt 
import seaborn as sns
import arviz as az
import pandas as pd
import json
from typing import Optional
from hall_opt.plotting.common_setup import get_common_paths
from hall_opt.utils.data_loader import load_data
from hall_opt.config.dict import Settings
from .common_setup import interactive_plot_prompt
from .kde_mcmc import plot_mcmc_kde

def plot_autocorrelation(samples, output_dir):
    """Generate autocorrelation plots."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    az.plot_autocorr(samples["log_c1"].values, ax=axes[0])
    axes[0].set_title("Autocorrelation for log(c1)")
    az.plot_autocorr(samples["log_alpha"].values, ax=axes[1])
    axes[1].set_title("Autocorrelation for log(alpha)")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/autocorrelation_plots.png")
    plt.close()
    print(f"Saved autocorrelation plots to {output_dir}")

def plot_trace(samples, output_dir):
    """Generate trace plots."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes[0].plot(samples["log_c1"], color='blue')
    axes[0].set_title("Trace Plot for log(c1)")
    axes[1].plot(samples["log_alpha"], color='green')
    axes[1].set_title("Trace Plot for log(alpha)")

    plt.xlabel("Iteration")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/trace_plots.png")
    plt.close()
    print(f"Saved trace plots to {output_dir}")

def plot_posterior(samples, output_dir):
    """Generate posterior distributions."""
    posterior = az.from_dict(posterior={"log_c1": samples["log_c1"], "log_alpha": samples["log_alpha"]})
    az.plot_posterior(posterior)
    plt.savefig(f"{output_dir}/posterior_marginals.png")
    plt.close()
    print(f"Saved posterior plots to {output_dir}")

def plot_pair(samples, output_dir):
    """Generate pair plots for joint distributions."""
    sns.pairplot(samples[["log_c1", "log_alpha"]], kind="scatter", diag_kind="kde", corner=True)
    plt.savefig(f"{output_dir}/pair_plot.png")
    plt.close()
    print(f"Saved pair plot to {output_dir}")


def plot_mcmc_kde(samples, plots_dir):
    """
    Generate 2D KDE plot for posterior samples.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import gaussian_kde

    # Extract sample arrays
    log_c1_samples = samples["log_c1"].values
    log_alpha_samples = samples["log_alpha"].values

    kde = gaussian_kde(np.vstack([log_c1_samples, log_alpha_samples]))

    x = np.linspace(log_c1_samples.min(), log_c1_samples.max(), 100)
    y = np.linspace(log_alpha_samples.min(), log_alpha_samples.max(), 100)
    X, Y = np.meshgrid(x, y)
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = kde(positions).reshape(X.shape)

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, Z, levels=50, cmap="viridis")
    plt.colorbar(contour, label="Density")

    plt.xlabel("log(c1)")
    plt.ylabel("log(alpha)")
    plt.title("2D KDE of Posterior Samples")
    plt.tight_layout()

    kde_plot_path = plots_dir / "kde_2d_log_c1_log_alpha.png"
    plt.savefig(kde_plot_path, dpi=300)
    plt.close()

    print(f"[INFO] Saved 2D KDE plot to: {kde_plot_path}")

def generate_plots(settings: Settings, analysis_type: Optional[str] = None):
    """
    Main function to generate and save plots dynamically for both MAP and MCMC.
    """
    print("Generating plots...")

    # Interactive CLI input
    analysis_type, results_dir = interactive_plot_prompt(settings)

    # Get correct latest directory
    paths = get_common_paths(settings, analysis_type)
    latest_dir = paths["latest_results_dir"]
    plots_dir = latest_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # --- Select proper file based on analysis type ---
    try:
        if analysis_type == "map":
            log_file = latest_dir / "map_iteration_log.json"
            if not log_file.is_file():
                raise FileNotFoundError(f"[ERROR] MAP log file not found: {log_file}")
            with open(log_file, "r") as f:
                all_iters = json.load(f)
            samples = pd.DataFrame(all_iters)[["c1_log", "alpha_log"]]
            samples.rename(columns={"c1_log": "log_c1", "alpha_log": "log_alpha"}, inplace=True)

        elif analysis_type == "mcmc":
            csv_file = latest_dir / "final_samples_log.csv"
            if not csv_file.is_file():
                raise FileNotFoundError(f"[ERROR] MCMC sample file not found: {csv_file}")
            samples = pd.read_csv(csv_file, header=None, names=["log_c1", "log_alpha"])

        else:
            raise ValueError("[ERROR] Unknown analysis type passed to generate_plots()")

    except Exception as e:
        print(f"[ERROR] Failed to load samples for {analysis_type.upper()}: {e}")
        return

    # --- Generate standard posterior plots ---
    plot_autocorrelation(samples, plots_dir)
    plot_trace(samples, plots_dir)
    plot_posterior(samples, plots_dir)
    plot_pair(samples, plots_dir)
    if analysis_type == "mcmc":
        plot_mcmc_kde(samples, plots_dir)  # <-- passing plots_dir and DataFrame directly
    else:
        print(f"KDE plots skipped for analysis type '{analysis_type}'")

    print(f"[INFO] All {analysis_type.upper()} plots saved to {plots_dir}")
