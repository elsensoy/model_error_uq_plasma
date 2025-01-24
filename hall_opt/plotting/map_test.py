import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Path to the MAP iterations JSON file
map_iterations_path = "/home/elida/Public/users/elsensoy/model_error_uq_plasma/hall_opt/results/map_results/map_iterations.json"

# Output directory for plots
output_plot_dir = "/home/elida/Public/users/elsensoy/model_error_uq_plasma/hall_opt/results/map_results"
os.makedirs(output_plot_dir, exist_ok=True)

# Load the MAP iteration data
def load_map_iterations(file_path):
    """Load MAP optimization iteration data from JSON."""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None

# Extract MAP iteration data
map_iterations = load_map_iterations(map_iterations_path)

if map_iterations is None:
    exit()

# Extract iteration numbers and parameter values
iterations = []
c1_log_values = []
alpha_log_values = []

for entry in map_iterations:
    iterations.append(entry["iteration"])
    c1_log_values.append(entry["params"]["c1_log"])
    alpha_log_values.append(entry["params"]["alpha_log"])

# Convert to NumPy arrays
iterations = np.array(iterations)
c1_log_values = np.array(c1_log_values)
alpha_log_values = np.array(alpha_log_values)

# Create trace plots
plt.figure(figsize=(10, 6))

# Plot for c1_log
plt.subplot(2, 1, 1)
plt.plot(iterations, c1_log_values, label="log(c1)", color='b')
plt.xlabel("Iteration")
plt.ylabel("log(c1)")
plt.title("Trace Plot for log(c1)")
plt.grid()
plt.legend()

# Plot for alpha_log
plt.subplot(2, 1, 2)
plt.plot(iterations, alpha_log_values, label="log(alpha)", color='g')
plt.xlabel("Iteration")
plt.ylabel("log(alpha)")
plt.title("Trace Plot for log(alpha)")
plt.grid()
plt.legend()

# Save and display the plot
plot_path = os.path.join(output_plot_dir, "map_traceplot.png")
plt.tight_layout()
plt.savefig(plot_path, dpi=300)
plt.show()

print(f"Trace plot saved to: {plot_path}")
