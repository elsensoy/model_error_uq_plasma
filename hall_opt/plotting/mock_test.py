# import os
# import json
# import yaml
# import numpy as np
# import matplotlib.pyplot as plt

# # Load settings from YAML file
# def load_settings(yaml_path):
#     """Load YAML settings file."""
#     try:
#         with open(yaml_path, "r") as f:
#             settings = yaml.safe_load(f)
#         return settings
#     except FileNotFoundError:
#         print(f"Error: YAML settings file not found at {yaml_path}")
#         return None
#     except yaml.YAMLError as e:
#         print(f"Error parsing YAML: {e}")
#         return None

# # Path to settings.yaml
# settings_path = "/home/elida/Public/users/elsensoy/model_error_uq_plasma/hall_opt/settings.yaml"

# # Load YAML settings
# settings = load_settings(settings_path)

# if settings is None:
#     exit("Failed to load settings file.")

# # Extract the MAP iteration file path from YAML
# map_iterations_path = settings["optimization_params"]["map_params"]["iteration_log_file"]
# output_plot_dir = os.path.dirname(map_iterations_path)

# print(f"Loading MAP iteration data from: {map_iterations_path}")
# print(f"Saving plots to: {output_plot_dir}")

# # Function to load MAP iteration data from JSON
# def load_map_iterations(file_path):
#     """Load MAP optimization iteration data from JSON file."""
#     try:
#         with open(file_path, "r") as f:
#             data = json.load(f)
#         return data
#     except FileNotFoundError:
#         print(f"Error: File not found at {file_path}")
#         return None
#     except json.JSONDecodeError as e:
#         print(f"Error decoding JSON: {e}")
#         return None

# # Load the iteration data
# map_iterations = load_map_iterations(map_iterations_path)

# if map_iterations is None:
#     exit("Failed to load MAP iteration data.")

# # Extract iteration numbers and parameter values
# iterations = []
# c1_log_values = []
# alpha_log_values = []

# for entry in map_iterations:
#     iterations.append(entry["iteration"])
#     c1_log_values.append(entry["params"]["c1_log"])
#     alpha_log_values.append(entry["params"]["alpha_log"])

# # Convert to NumPy arrays
# iterations = np.array(iterations)
# c1_log_values = np.array(c1_log_values)
# alpha_log_values = np.array(alpha_log_values)

# # Create trace plots
# plt.figure(figsize=(10, 6))

# # Plot for log(c1)
# plt.subplot(2, 1, 1)
# plt.plot(iterations, c1_log_values, label="log(c1)", color='b')
# plt.xlabel("Iteration")
# plt.ylabel("log(c1)")
# plt.title("Trace Plot for log(c1)")
# plt.grid()
# plt.legend()

# # Plot for log(alpha)
# plt.subplot(2, 1, 2)
# plt.plot(iterations, alpha_log_values, label="log(alpha)", color='g')
# plt.xlabel("Iteration")
# plt.ylabel("log(alpha)")
# plt.title("Trace Plot for log(alpha)")
# plt.grid()
# plt.legend()

# # Save the plot
# plot_path = os.path.join(output_plot_dir, "map_traceplot.png")
# plt.tight_layout()
# plt.savefig(plot_path, dpi=300)
# plt.show()

# print(f"Trace plot saved to: {plot_path}")
