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

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, thrust_vals, marker="o", label="MAP Thrust")
    plt.axvline(iterations[-1], color="gray", linestyle="--", alpha=0.3)
    plt.scatter(iterations[-1], thrust_vals[-1], color="red", zorder=5)
    plt.text(iterations[-1], thrust_vals[-1] + 0.01, f"{thrust_vals[-1]:.3f}", fontsize=9, color="red")

    if observed_thrust:
        plt.axhline(observed_thrust, color="green", linestyle="--", label="Observed Thrust")
        plt.text(iterations[0], observed_thrust + 0.005, f"Obs: {observed_thrust:.3f}", fontsize=9, color="green")

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


    plt.figure(figsize=(10, 6))
    plt.plot(iterations, current_vals, marker="o", color="purple", label="MAP Discharge Current")
    plt.scatter(iterations[-1], current_vals[-1], color="red", zorder=5)
    plt.text(iterations[-1], current_vals[-1] + 0.5, f"{current_vals[-1]:.2f}", fontsize=9, color="red")

    if observed_current:
        plt.axhline(observed_current, color="green", linestyle="--", label="Observed Current")
        plt.text(iterations[0], observed_current + 0.5, f"Obs: {observed_current:.2f}", fontsize=9, color="green")

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
    
    plot_ion_velocity_iterations(settings, metric_files=metric_files, save_dir=save_dir)

def plot_ion_velocity_iterations(settings: Settings, metric_files: List[Path], save_dir: Path):
    """
    Enhanced ion velocity evolution plot:
    - Adds scatter markers to each iteration
    - Plots final iteration in bold
    - Overlays ground truth with label and annotation
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))

    final_z, final_v = None, None

    # --- Plot each iteration ---
    for i, file in enumerate(metric_files):
        with open(file, "r") as f:
            data = json.load(f)
            z = data["z_normalized"]
            v = data["ion_velocity"]

            is_first_or_last = i in (0, len(metric_files) - 1)
            label = f"Iter {i+1}" if is_first_or_last else None

            plt.plot(z, v, alpha=0.3, linewidth=1.0, label=label)
            plt.scatter(z, v, alpha=0.3, s=10, color="gray")

            if i == len(metric_files) - 1:
                final_z, final_v = z, v

    # --- Final iteration bold ---
    if final_z and final_v:
        plt.plot(final_z, final_v, color="blue", linewidth=2.5, label="Final Iteration")
        plt.scatter(final_z, final_v, color="blue", s=25)
        plt.text(final_z[-1], final_v[-1], f"{final_v[-1]:.1f}", fontsize=9, color="blue")

    # --- Ground truth overlay ---
# --- Plot Ground Truth if Available ---
    gt_file = Path(settings.output_dir) / "ground_truth" / "ground_truth_metrics.json"
    if gt_file.is_file():
        with open(gt_file, "r") as f:
            gt_data = json.load(f)

        z_gt = gt_data.get("z_normalized")
        v_gt = gt_data.get("ion_velocity") or gt_data.get("ui")

        if z_gt and v_gt and len(z_gt) == len(v_gt):
            plt.plot(z_gt, v_gt, color="green", linestyle="--", linewidth=2, label=f"Observed (Final: {v_gt[-1]:.1f} m/s)")
            plt.scatter(z_gt, v_gt, color="green", s=25)
        if isinstance(v_gt, list) and isinstance(v_gt[0], list):
            print("[DEBUG] Flattening nested ion velocity list from ground truth")
            v_gt = v_gt[0]

            # Annotate the final ground truth point
            plt.text(z_gt[-1], v_gt[-1], f"{v_gt[-1]:.1f}", fontsize=9, color="green")

    # --- Finalize layout ---
    plt.xlabel("Normalized Axial Position (z)")
    plt.ylabel("Ion Velocity (m/s)")
    plt.title("Ion Velocity Evolution Over Iterations")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    output_path = save_dir / "ion_velocity_iterations.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved ion velocity evolution plot: {output_path}")
