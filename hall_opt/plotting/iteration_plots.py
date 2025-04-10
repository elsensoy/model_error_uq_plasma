import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List
from hall_opt.config.dict import Settings

def generate_iteration_metric_plots(settings: Settings, iter_metrics_dir: Path, save_dir: Path):
    """
    Generates evolution plots for metrics like thrust and discharge_current over iterations.

    Args:
        iter_metrics_dir: Path to the folder containing metrics_*.json files.
        save_dir: Path to save the output plots.
    """

    save_dir.mkdir(parents=True, exist_ok=True)
    # Load ground truth if available
    gt_file = Path(settings.output_dir) / "ground_truth" / "ground_truth_metrics.json"
    observed_thrust = None
    observed_current = None

    if gt_file.is_file():
        with open(gt_file, "r") as f:
            gt_data = json.load(f)
            observed_thrust = gt_data.get("thrust")
            observed_current = gt_data.get("discharge_current")

    # --- Load all JSON files in order ---
    metric_files = sorted(
        [f for f in iter_metrics_dir.glob("metrics_*.json")],
        key=lambda f: int(f.stem.split("_")[-1])
    )

    if not metric_files:
        print(f"[ERROR] No metric files found in {iter_metrics_dir}")
        return

    thrust_vals = []
    current_vals = []

    for file in metric_files:
        with open(file, "r") as f:
            data = json.load(f)
            thrust_vals.append(data.get("thrust"))
            current_vals.append(data.get("discharge_current"))

    iterations = list(range(1, len(thrust_vals) + 1))


    # --- Plot Thrust Evolution ---
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, thrust_vals, marker="o", label="Thrust [N]")
    plt.title("Thrust over MAP Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Thrust (N)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    thrust_plot_path = save_dir / "thrust_evolution.png"
    plt.savefig(thrust_plot_path)
    print(f"[INFO] Saved: {thrust_plot_path}")
    plt.close()

    # --- Plot Discharge Current Evolution ---
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, current_vals, marker="o", color="purple", label="Discharge Current [A]")
    plt.title("Discharge Current over MAP Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Discharge Current (A)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    current_plot_path = save_dir / "discharge_current_evolution.png"
    plt.savefig(current_plot_path)
    print(f"[INFO] Saved: {current_plot_path}")
    plt.close()
    
    plot_ion_velocity_iterations(metric_files=metric_files, save_dir=save_dir)

def plot_ion_velocity_iterations(metric_files: List[Path], save_dir: Path):
    """
    Simple plot of ion velocity across iterations over z_normalized.
    
    Args:
        metric_files: Sorted list of iteration metric JSON files.
        save_dir: Path where the figure will be saved.
    """
    import matplotlib.pyplot as plt
    import json

    save_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    for i, file in enumerate(metric_files):
        with open(file, "r") as f:
            data = json.load(f)
            z = data["z_normalized"]
            v = data["ion_velocity"]

            label = f"Iter {i+1}" if i == 0 or i == len(metric_files) - 1 else None
            plt.plot(z, v, alpha=0.3, linewidth=1.0, label=label)

    plt.xlabel("Normalized Axial Position (z)")
    plt.ylabel("Ion Velocity (m/s)")
    plt.title("Ion Velocity Evolution Over Iterations")
    plt.grid(True, linestyle="--", alpha=0.5)
    if plt.gca().get_legend_handles_labels()[1]:  # only show legend if any label was added
        plt.legend()
    plt.tight_layout()

    output_path = save_dir / "ion_velocity_iterations.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved simple ion velocity evolution plot: {output_path}")
