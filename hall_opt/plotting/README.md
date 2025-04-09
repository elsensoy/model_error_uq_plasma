# Goal 1: Visualize the final_simplex data obtained from a Nelder-Mead optimization result,
  which have been saved in a JSON file. The simplex exists in the 2D parameter space defined by c1_log and alpha_log.

Identify the Data: The relevant data is within the final_simplex dictionary in the JSON:

JSON

    "final_simplex": {
        "vertices": [
            [-0.270..., 0.425...],  // Vertex 1 (x, y) = (c1_log, alpha_log)
            [-0.230..., 0.462...],  // Vertex 2
            [-0.230..., 0.487...]   // Vertex 3
        ],
        "values": [
            2.200...,  // Function value at Vertex 1
            2.239...,  // Function value at Vertex 2
            2.239...   // Function value at Vertex 3
        ]
    }
This represents three points (vertices) in a 2D space (c1_log vs. alpha_log).


Import Libraries: Imports json, matplotlib.pyplot, numpy, and pathlib.
Load Data: Opens and reads the specified JSON file. Includes error handling.
Extract Data: Pulls out the final_simplex dictionary and the best point x. It specifically checks if final_simplex exists, as this visualization only makes sense for Nelder-Mead results. It converts the lists from the JSON into NumPy arrays. Includes validation for data shapes.
Create Plot: Sets up a Matplotlib figure and axes.
Plot Simplex Edges: Uses ax.plot on the vertex coordinates (appending the first vertex to the end to close the triangle) to draw the outline of the simplex.
Plot Vertices: Uses ax.scatter to draw points at each vertex. The c=values argument colors each point based on the objective function value stored in values, using the viridis colormap. A color bar is added for reference.
Highlight Best Point: Uses ax.plot with a red star marker ('r*') to clearly show the location of the overall best parameters found (x).
Labels & Formatting: Adds axis labels, a title, a legend, and grid lines. It also sets axis limits slightly larger than the simplex bounds for better visibility.
Show Plot: Displays the generated plot using plt.show().
Save Plot (Optional): Includes commented-out code to save the plot as a PNG file in the same directory as the JSON file.


NELDERMEAD:
https://alexdowad.github.io/visualizing-nelder-mead/
"The Nelder-Mead algorithm doesn’t always reach a minimum point, either; or at least, not in a reasonable number of iterations. Sometimes it gets close to a minimum point... and then moves very, very slowly towards it.

For that reason, when implementing Nelder-Mead, you need to limit the number of iterations so it doesn’t run for too long. Rather than running Nelder-Mead for a huge number of iterations, you will probably get better results by restarting it several times, with different starting points, and then picking the best overall solution found.[2]"