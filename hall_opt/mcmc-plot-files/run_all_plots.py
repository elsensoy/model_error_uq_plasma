import subprocess

# List of all plot scripts
plot_scripts = [
    "visualization.py",
    "delta_plots.py",
    "plot_mcmc_across_200_plots.py",
    # Add all other plot scripts
]

# Run each plot script
for script in plot_scripts:
    print(f"Running {script}...")
    subprocess.run(["python", script], check=True)
