import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
from hall_opt.config.dict import Settings


def visualize_final_simplex(json_filepath_str: str):
    """
    Loads optimization results from a JSON file and plots the final simplex
    (specifically for Nelder-Mead results).

    Args:
        json_filepath_str: Path to the 'optimization_result.json' file.
    """
    settings: Settings
    json_filepath = Path(json_filepath_str)

    if not json_filepath.is_file():
        print(f"Error: JSON file not found at {json_filepath}")
        return

    try:
        with open(json_filepath, 'r') as f:
            result_data = json.load(f)
    except Exception as e:
        print(f"Error loading or parsing JSON file {json_filepath}: {e}")
        return

    # Check if final_simplex data exists (specific to Nelder-Mead)
    if 'final_simplex' not in result_data or not isinstance(result_data['final_simplex'], dict):
        print("Error: 'final_simplex' data not found or not in expected format in JSON.")
    try:
        vertices = np.array(result_data['final_simplex']['vertices']) # N+1 points in N dims
        values = np.array(result_data['final_simplex']['values'])     # Function value at each vertex
        best_point = np.array(result_data['x_log'])                      # Overall best point found
        fun_at_best = result_data['fun']                             # Objective value at best_point

        # Ensure we have 2D data for plotting (3 vertices for a 2D simplex)
        if vertices.shape[1] != 2 or vertices.shape[0] != 3:
             print(f"Error: Expected 3 vertices with 2 dimensions each, but got shape {vertices.shape}")
             return

    except KeyError as e:
        print(f"Error: Missing key in JSON data ('{e}'). Cannot extract simplex information.")
        return
    except Exception as e:
        print(f"Error processing simplex data from JSON: {e}")
        return

    # --- Create the Plot ---
    fig, ax = plt.subplots(figsize=(10, 8))

    # 1. Plot the simplex as a polygon outline
    #  the polygon must be 'closed' (close it by appending the first vertex to the end for plotting lines)
    closed_vertices = np.vstack([vertices, vertices[0]]) # Shape becomes (4, 2)
    ax.plot(closed_vertices[:, 0], closed_vertices[:, 1], 'b-', label='Final Simplex Edges', linewidth=1.5, alpha=0.7)

    # 2. Plot the vertices themselves
    scatter = ax.scatter(vertices[:, 0], vertices[:, 1], c=values, cmap='viridis', s=60, zorder=4, label='Simplex Vertices (Color by Func Value)')
    plt.colorbar(scatter, label='Objective Function Value (fun)')

    # 3. Highlight the best point found ('x') - should be one of the vertices
    ax.plot(best_point[0], best_point[1], 'r*', markersize=15, label=f'Best Point Found (x)\nfun={fun_at_best:.4f}', zorder=6)

    # Annotations and Labels 
    ax.set_xlabel("Parameter 1 (c1_log)")
    ax.set_ylabel("Parameter 2 ( alpha_log)")
    ax.set_title("Final Nelder-Mead Simplex at Termination")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_aspect('equal', adjustable='box') #  aspect ratio is equal if scales are similar

    # optional: Annotate vertices with their values (can get cluttered)1
    
    # for i, txt in enumerate(values):
    #    ax.annotate(f"{txt:.3f}", (vertices[i, 0], vertices[i, 1]), textcoords="offset points", xytext=(5,5), ha='center', fontsize=8)


    # Show the plot
    plt.tight_layout()
    plt.show()
    # Ensure the plot subdir exists inside the run folder
    plot_dir = json_filepath.parent / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    save_path = plot_dir / "final_simplex_visualization.png"

    try:
        fig.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    except Exception as e:
        print(f"Failed to save plot: {e}")