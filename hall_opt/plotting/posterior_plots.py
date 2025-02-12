import matplotlib.pyplot as plt 
import seaborn as sns
import arviz as az
from ..plotting.common_setup import get_common_paths
from ..utils.data_loader import load_data
from ..config.dict import Settings

def plot_autocorrelation(samples, output_dir):
    """Generate autocorrelation plots."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    az.plot_autocorr(samples["log_v1"].values, ax=axes[0])
    axes[0].set_title("Autocorrelation for log(v1)")
    az.plot_autocorr(samples["log_alpha"].values, ax=axes[1])
    axes[1].set_title("Autocorrelation for log(alpha)")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/autocorrelation_plots.png")
    plt.close()
    print(f"Saved autocorrelation plots to {output_dir}")

def plot_trace(samples, output_dir):
    """Generate trace plots."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes[0].plot(samples["log_v1"], color='blue')
    axes[0].set_title("Trace Plot for log(v1)")
    axes[1].plot(samples["log_alpha"], color='green')
    axes[1].set_title("Trace Plot for log(alpha)")

    plt.xlabel("Iteration")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/trace_plots.png")
    plt.close()
    print(f"Saved trace plots to {output_dir}")

def plot_posterior(samples, output_dir):
    """Generate posterior distributions."""
    posterior = az.from_dict(posterior={"log_v1": samples["log_v1"], "log_alpha": samples["log_alpha"]})
    az.plot_posterior(posterior)
    plt.savefig(f"{output_dir}/posterior_marginals.png")
    plt.close()
    print(f"Saved posterior plots to {output_dir}")

def plot_pair(samples, output_dir):
    """Generate pair plots for joint distributions."""
    sns.pairplot(samples[["log_v1", "log_alpha"]], kind="scatter", diag_kind="kde", corner=True)
    plt.savefig(f"{output_dir}/pair_plot.png")
    plt.close()
    print(f"Saved pair plot to {output_dir}")

def generate_plots(settings: Settings):
    """
    Main function to generate and save plots dynamically for both MAP and MCMC.
    """

    # Determine analysis type dynamically
    if settings.general.run_map:
        analysis_type = "map"
    elif settings.general.run_mcmc:
        analysis_type = "mcmc"
    else:
        print("ERROR: Neither MAP nor MCMC is enabled! No plots will be generated.")
        return
    
    print(f"Running plotting for {analysis_type.upper()}.")

    # Get correct directories dynamically
    paths = get_common_paths(settings, analysis_type)
    output_dir = paths["plots_dir"]

    # Load data dynamically
    try:
        samples = load_data(settings, analysis_type)
        print(f"Data loaded successfully for {analysis_type.upper()}.")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    # Generate all plots
    plot_autocorrelation(samples, output_dir)
    plot_trace(samples, output_dir)
    plot_posterior(samples, output_dir)
    plot_pair(samples, output_dir)

    print(f"All plots for {analysis_type.upper()} have been saved to {output_dir}!")
