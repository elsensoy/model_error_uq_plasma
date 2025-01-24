import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import argparse
from hall_opt.plotting.common_setup import load_data, get_common_paths

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
    print("Saved autocorrelation plots.")

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
    print("Saved trace plots.")

def plot_posterior(samples, output_dir):
    """Generate posterior distributions."""
    posterior = az.from_dict(posterior={"log_v1": samples["log_v1"], "log_alpha": samples["log_alpha"]})
    az.plot_posterior(posterior)
    plt.savefig(f"{output_dir}/posterior_marginals.png")
    plt.close()
    print("Saved posterior plots.")

def plot_pair(samples, output_dir):
    """Generate pair plots for joint distributions."""
    sns.pairplot(samples[["log_v1", "log_alpha"]], kind="scatter", diag_kind="kde", corner=True)
    plt.savefig(f"{output_dir}/pair_plot.png")
    plt.close()
    print("Saved pair plot.")
