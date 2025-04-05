import matplotlib.pyplot as plt 
import seaborn as sns
import arviz as az
import json
import pandas as pd
from hall_opt.plotting.common_setup import get_common_paths,prompt_analysis_type
from hall_opt.utils.data_loader import load_data
from hall_opt.config.dict import Settings
from pathlib import Path

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

def generate_plots(settings: Settings):
    """
    Main function to generate and save plots dynamically for both MAP and MCMC.
    """

    # Prompt the user for which type to plot
    analysis_type = prompt_analysis_type()
    print(f"Running plotting for {analysis_type.upper()}.")
    
    # --- Get plot directory ---
    try:
        paths = get_common_paths(settings, analysis_type)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    output_dir = paths["plots_dir"]
    results_dir = Path(settings.map.base_dir)

    # --- Load data depending on mode ---
    if analysis_type == "map":
        iteration_file = results_dir / "map_iteration_log.json"
        if not iteration_file.exists():
            print(f"ERROR: MAP iteration file not found at: {iteration_file}")
            return

        with open(iteration_file, "r") as f:
            iter_data = json.load(f)

        samples = pd.DataFrame(iter_data)

        if samples.empty:
            print("ERROR: Loaded MAP iteration data is empty.")
            return

        # Rename for plotting consistency
        samples.rename(columns={
            "c1_log": "log_c1",
            "alpha_log": "log_alpha"
        }, inplace=True)
    else:
        # Load MCMC samples (same logic as before)
        samples = load_data(settings, analysis_type)


        print(f"Data loaded successfully for {analysis_type.upper()}.")

    # --- Generate Plots ---
    plot_autocorrelation(samples, output_dir)
    plot_trace(samples, output_dir)
    plot_posterior(samples, output_dir)
    plot_pair(samples, output_dir)

    print(f"All plots for {analysis_type.upper()} have been saved to {output_dir}!")