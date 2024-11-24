import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Directory to save plots
PLOTS_DIR = "mcmc-plots"
os.makedirs(PLOTS_DIR, exist_ok=True) 
RESULTS_DIR = os.path.join("..", "mcmc-results-11-23-24")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Paths to the MAP and MCMC results
MAP_RESULTS_PATH = os.path.join("..", "results-Nelder-Mead", "best_initial_guess_w_2_0.json")
MCMC_SAMPLES_PATH = os.path.join("..", "mcmc-results-11-23-24", "final_mcmc_samples_w_2.0_2.csv")

# Load MAP parameters
def load_map_results(filepath):
    import json
    with open(filepath, 'r') as f:
        map_results = json.load(f)
    return {"v1": map_results["v1"], "v2": map_results["v2"]}

# Load MCMC samples
def load_mcmc_samples(filepath):
    samples = pd.read_csv(filepath, header=None)
    samples.columns = ["log_v1", "log_alpha"]
    samples["v1"] = 10 ** samples["log_v1"]
    samples["alpha"] = 10 ** samples["log_alpha"]
    samples["v2"] = samples["v1"] * samples["alpha"]
    return samples

# Comparison function with MAP values in the legend
def compare_map_vs_mcmc(map_results, mcmc_samples):
    # Compute posterior statistics
    mcmc_summary = {
        "v1_mean": mcmc_samples["v1"].mean(),
        "v1_std": mcmc_samples["v1"].std(),
        "v2_mean": mcmc_samples["v2"].mean(),
        "v2_std": mcmc_samples["v2"].std(),
    }
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # v1 Comparison
    sns.kdeplot(mcmc_samples["v1"], ax=axes[0], fill=True, label="MCMC Posterior (v1)")
    axes[0].axvline(map_results["v1"], color="red", linestyle="--", 
                    label=f"MAP Estimate (v1): {map_results['v1']:.3f}")
    axes[0].set_title("Comparison of v1: MAP vs MCMC")
    axes[0].set_xlabel("v1")
    axes[0].legend()

    # v2 Comparison
    sns.kdeplot(mcmc_samples["v2"], ax=axes[1], fill=True, label="MCMC Posterior (v2)")
    axes[1].axvline(map_results["v2"], color="red", linestyle="--", 
                    label=f"MAP Estimate (v2): {map_results['v2']:.3f}")
    axes[1].set_title("Comparison of v2: MAP vs MCMC")
    axes[1].set_xlabel("v2")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("map_vs_mcmc_comparison_with_map_values.png")
    plt.show()

    # Print Summary Table
    comparison_table = pd.DataFrame({
        "Metric": ["MAP Estimate (v1)", "Posterior Mean (v1)", "Posterior Std (v1)",
                   "MAP Estimate (v2)", "Posterior Mean (v2)", "Posterior Std (v2)"],
        "Value": [map_results["v1"], mcmc_summary["v1_mean"], mcmc_summary["v1_std"],
                  map_results["v2"], mcmc_summary["v2_mean"], mcmc_summary["v2_std"]]
    })

    print(comparison_table)

# Main
def main():
    # Load data
    map_results = load_map_results(MAP_RESULTS_PATH)
    mcmc_samples = load_mcmc_samples(MCMC_SAMPLES_PATH)

    # Compare and visualize
    compare_map_vs_mcmc(map_results, mcmc_samples)

if __name__ == "__main__":
    main()
