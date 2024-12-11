import subprocess


plot_scripts = [
    "visualization.py",
    "delta_plots.py",
    "mcmc_iter_plots.py"
]


for script in plot_scripts:
    print(f"Running {script}...")
    subprocess.run(["python", script], check=True)
