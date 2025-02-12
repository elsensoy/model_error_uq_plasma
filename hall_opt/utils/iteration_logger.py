import json
import numpy as np
from pathlib import Path

def iteration_callback(c_log, iteration_counter, iteration_logs, iteration_log_path, checkpoint_path):
    
    iteration_counter[0] += 1  # Increment iteration count
    c1_log, alpha_log = c_log
    c1, alpha = np.exp(c1_log), np.exp(alpha_log)  # Convert log-space to actual values

    # Store iteration data
    iteration_data = {
        "iteration": iteration_counter[0],
        "c1_log": c1_log,
        "alpha_log": alpha_log,
        "c1": c1,
        "alpha": alpha,
    }
    iteration_logs.append(iteration_data)

    # Save logs to JSON file after each iteration
    try:
        with open(iteration_log_path, "w") as log_file:
            json.dump(iteration_logs, log_file, indent=4)
        print(f"Saved iteration {iteration_counter[0]} to {iteration_log_path}")
    except Exception as e:
        print(f"WARNING: Failed to save iteration log: {e}")

    # Save checkpoint every 10th iteration
    if iteration_counter[0] % 10 == 0:
        try:
            checkpoint_data = iteration_data
            with open(checkpoint_path, "a") as checkpoint_file:  # Append mode
                json.dump(checkpoint_data, checkpoint_file)
                checkpoint_file.write("\n")  # Ensure new line for each entry
            print(f"Checkpoint saved at iteration {iteration_counter[0]} in {checkpoint_path}")
        except Exception as e:
            print(f"WARNING: Failed to save checkpoint: {e}")

    # Print iteration progress
    print(f"Iteration {iteration_counter[0]}: c1 = {c1:.4f} (log: {c1_log:.4f}), "
          f"alpha = {alpha:.4f} (log: {alpha_log:.4f})")
