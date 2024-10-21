# hall-opt

## Project Overview
This project uses the HallThruster.jl Julia package, integrated with Python through Juliacall, to simulate and optimize the behavior of a Hall Thruster.

# Key Features:
- MultiLogBohm Simulation: Simulates Hall Thruster behavior using the MultiLogBohm model.
- TwoZoneBohm Simulation: Uses MAP (Maximum A Posteriori) parameter optimization to simulate Hall Thruster behavior.
- MAP Estimation: Optimizes key model parameters using MAP.
- Result Comparison: Visualize and compare simulation results against observed data.

## Step 1: Install PDM
If PDM is not installed, install it via pip:
    #pip install pdm

## Step 2: Install Project Dependencies
Once PDM is installed, clone the repository and install the dependencies:

# Clone the repository
    git clone https://github.com/elsensoy/HallOpt.git
    cd hall-opt

# Install Python dependencies with PDM
    pdm install
PDM will create a virtual environment and install all necessary dependencies, including matplotlib, scipy, and juliacall.

    # Activate the PDM virtual environment:
    pdm venv activate

<!-- Julia Installation -->
Ensure Julia is installed. Also the following Julia packages:
    import Pkg
    Pkg.add("HallThruster")
    Pkg.add("JSON3")

    #Install HallThrusterPEM manually if needed
     git clone https://github.com/JANUS-Institute/HallThrusterPEM.git
     cd HallThrusterPEM
     pdm install


## Step 3: Running the Project
Once the dependencies are installed, you can run the main simulation script.

    cd mapOpt
    pdm run python map_ink.py

# The script include the following tasks:
Simulates the Hall Thruster using MultiLogBohm.
Optimizes parameters using MAP for TwoZoneBohm.
Saves results for different ion velocity weights.

    # The default ion velocity weights used in the project are:
    ion_velocity_weights = [1e-10, 0.1, 1.0, 2.0, 3.0, 5.0, 10.0]

    #Ion velocity weight is set in main but also in log_likelihood and run_map_multiple_initial_guesses functions. Function definitions might need a slight adjustment for the weight chosen.
    ion_velocity_weights = []

# Expected Outputs
Results are saved in the results-LBFGSB directory in JSON format. Key outputs include:
Ground Truth Data: Results from the initial MultiLogBohm simulation.
Optimized Parameters: Best-fit parameters from MAP estimation.
Performance Metrics: Metrics like thrust, ion velocity, and discharge current.

## Step 4: Visualizing Results
To visualize the results, you can use the plotting scripts. For example, you can generate comparison plots by running:
    pdm run python ion_velocity_weight_plots.py
This script will generate comparison plots between observed and optimized results, saving the plots in the plots_comparison directory.

# Dependencies
The Python dependencies are listed in the pyproject.toml. The main dependencies include:

    matplotlib
    juliacall
    scipy

# Project Structure

hall-project/
├── HallThrusterPEM
├── README.md
├── __pycache__
├── bfield_spt100.csv
├── extract_metrics.jl
├── mapOpt
    ├── ion_velocity_weight_plots.py
    ├── loss_comparison_plots.py
    ├── map_ink.py
    ├── plots_comparison
    └── residuals_plots_map.py
├── nelder-mead #still work in progress. dublicate files might exist. 
├── output               #these are some saved terminal outputs for debugging map_ink.py 
├── pdm.lock
├── pyproject.toml
├── requirements.txt #dependencies
├── results-LBFGSB      #lbfgsb results directory.
├── results-Nelder-Mead #nelder mead results directory.(still work in progress.)
├── src
└── tests #some unit tests, etc.

11 directories, 6 files
Contact
Feel free to contact me at elsensoy@umich.edu!
