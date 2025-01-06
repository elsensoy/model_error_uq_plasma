import numpy as np
import json
import os
import matplotlib.pyplot as plt
from config.simulation import simulation, config_spt_100, postprocess, update_twozonebohm_config, run_simulation_with_config
from utils.statistics import log_likelihood, prior_logpdf
from utils.save_data import load_json_data
# Add HallThruster Python API to sys.path
hallthruster_path = "/root/.julia/packages/HallThruster/J4Grt/python"
if hallthruster_path not in sys.path:
    sys.path.append(hallthruster_path)

print("Updated sys.path:", sys.path)

import hallthruster as het

observed_data_path = "/mnt/c/Users/MRover/elsensoy/model_error_uq_plasma/hall_opt/mcmc/results/mcmc_observed_data_map.json"
with open(observed_data_path, "r") as f:
    observed_data = json.load(f)

# Define fixed parameter points (v1_log, alpha_log)
fixed_points = [
    {"v1_log": np.log10(1 / 160), "alpha_log": 0.5},
    {"v1_log": np.log10(1 / 200), "alpha_log": 1.0},
    {"v1_log": np.log10(1 / 100), "alpha_log": 2.0},
]

# Directory setup for saving results and plots
results_dir = "results"
plots_dir = "plots"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# Initialize a list to store results
results = []

# Evaluate log-posterior for fixed points
for point in fixed_points:
    v1_log, alpha_log = point["v1_log"], point["alpha_log"]
    v1 = 10 ** v1_log
    alpha = 10 ** alpha_log
    v2 = alpha * v1

    # Compute log prior
    log_prior_value = prior_logpdf(v1_log, alpha_log)

    # Update configuration for simulation
    updated_config = update_twozonebohm_config(config_spt_100, v1, v2)
    simulation_result = run_simulation_with_config(
        updated_config, simulation, postprocess, config_type="TwoZoneBohm"
    )

    if simulation_result:
        # Extract simulation metrics
        metrics = simulation_result["output"]["average"]
        simulated_data = {
            "thrust": metrics.get("thrust", 0),
            "discharge_current": metrics.get("discharge_current", 0),
            "ui": metrics.get("ui", []),
            "z_coords": metrics.get("z_coords", []),
        }

        # Compute log likelihood
        log_likelihood_value = log_likelihood(simulated_data, observed_data, postprocess)

        # Compute log posterior
        log_posterior_value = log_prior_value + log_likelihood_value

        # Append results
        results.append({
            "v1_log": v1_log,
            "alpha_log": alpha_log,
            "log_prior": log_prior_value,
            "log_likelihood": log_likelihood_value,
            "log_posterior": log_posterior_value,
            "metrics": metrics,
        })

        print(f"Fixed Point: v1_log={v1_log:.4f}, alpha_log={alpha_log:.4f}")
        print(f"Log Prior: {log_prior_value:.4f}, Log Likelihood: {log_likelihood_value:.4f}")
        print(f"Log Posterior: {log_posterior_value:.4f}")
    else:
        print(f"Simulation failed for Fixed Point: v1_log={v1_log:.4f}, alpha_log={alpha_log:.4f}")
        results.append({
            "v1_log": v1_log,
            "alpha_log": alpha_log,
            "log_prior": log_prior_value,
            "log_likelihood": None,
            "log_posterior": None,
            "metrics": None,
        })

# Save results to JSON
results_file = os.path.join(results_dir, "fixed_points_results.json")
with open(results_file, "w") as f:
    json.dump(results, f, indent=4)
print(f"Results saved to {results_file}")

# Plotting
# Extract data for plotting
v1_values = [10**point["v1_log"] for point in results if point["log_posterior"] is not None]
log_posteriors = [point["log_posterior"] for point in results if point["log_posterior"] is not None]
thrust_values = [point["metrics"]["thrust"] for point in results if point["metrics"]]

# Plot Log Posterior vs v1
plt.figure(figsize=(8, 6))
plt.plot(v1_values, log_posteriors, marker="o", linestyle="-")
plt.xlabel("$v_1$")
plt.ylabel("Log Posterior")
plt.title("Log Posterior vs $v_1$")
plt.xscale("log")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
log_posterior_plot_file = os.path.join(plots_dir, "log_posterior_vs_v1.png")
plt.tight_layout()
plt.savefig(log_posterior_plot_file)
plt.close()
print(f"Log Posterior plot saved to {log_posterior_plot_file}")

# Plot Thrust vs v1
plt.figure(figsize=(8, 6))
plt.plot(v1_values, thrust_values, marker="o", linestyle="-")
plt.xlabel("$v_1$")
plt.ylabel("Thrust")
plt.title("Thrust vs $v_1$")
plt.xscale("log")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
thrust_plot_file = os.path.join(plots_dir, "thrust_vs_v1.png")
plt.tight_layout()
plt.savefig(thrust_plot_file)
plt.close()
print(f"Thrust plot saved to {thrust_plot_file}")
