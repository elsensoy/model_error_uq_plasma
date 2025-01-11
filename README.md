## Project Overview
This project aims to optimize the behavior of a Hall Thruster using the HallThruster.jl Julia package, integrated with Python through Juliacall. The project involves parameter estimation using MCMC sampling and MAP optimization, providing tools for comparison between simulated and observed data. It employs Bayesian inference techniques to refine the parameters of the **TwoZoneBohm** and **MultiLogBohm** models.

Please refer to:
- [HallThruster.jl (v0.18)](https://um-pepl.github.io/HallThruster.jl/dev/)
- [Simulation Tutorial](https://um-pepl.github.io/HallThruster.jl/dev/tutorials/simulation/)
- [Running Simulations from JSON](https://um-pepl.github.io/HallThruster.jl/dev/howto/json/)
- [Using HallThruster with Python](https://um-pepl.github.io/HallThruster.jl/dev/howto/python/)


## Key Features

- **MultiLogBohm Simulation**: Models Hall Thruster behavior using the MultiLogBohm framework.
- **TwoZoneBohm Simulation**: Uses MAP optimization and MCMC sampling to refine Hall Thruster parameters.
- **MAP Estimation**: Provides optimized initial guesses for key parameters using the Nelder-Mead optimization method.
- **MCMC Sampling**:
  - **Proposal Mechanism**: Gaussian random walk with adaptive scaling.
  - **DRAM Features**:
    - *Delayed Rejection*: Generates smaller-step secondary proposals upon rejection.
    - *Adaptive Covariance*: Updates proposal covariance for efficiency.
- **Comparison with Observed Data**: Simulates ion velocity, thrust, and discharge current and compares them with ground truth data.
- **Result Visualization**: Includes trace plots, posterior distributions, autocorrelation plots, and parameter-pair plots.

---

## Installation

### Step 1: Install PDM
If you don't already have PDM installed, you can install it using `pip`:
```
    pip install pdm
```

### Step 2: Clone the Repository
```
    git clone https://github.com/gorodetsky-umich/model_error_uq_plasma.git
    cd model_error_uq_plasma
```

### Step 3: Install Dependencies
1. Install the necessary dependencies with PDM:
```
    pdm install
```
2. Activate the environment:
```
pdm venv activate
```
Alternative: Use Python's venv
1. Alternatively, use Python's built-in venv module:
```
    python3 -m venv .venv
    source .venv/bin/activate
```
Install Dependencies: Within the virtual environment, use pip to install the dependencies and verify:
```
pip install -r requirements.txt
python -m pip list

```
### Step 4: Install MCMCIterators
Clone and install the **MCMCIterators** package for advanced MCMC sampling:
```
    git clone https://github.com/goroda/MCMCIterators.git
    cd MCMCIterators
    python setup.py install
```

### Step 5: Configure HallThruster.jl
Make sure HallThruster.jl is installed. It can be installed from Julia's package manager (`Pkg`):
```julia
    using Pkg
    Pkg.add("HallThruster")
```

---

## Running the Project

### Step 1: Pre-MCMC MAP Optimization
Run the MAP optimization script to generate the initial parameter guesses:
```
    pdm run python map.py
```

### Step 2: Run MCMC Sampling
Start the MCMC sampling process using the optimized parameters from MAP:
```
    pdm run python mcmc.py
```

This will:
- Estimate parameters (\(c_1\), \(c_2\)) using Bayesian inference.
- Save intermediate and final results in the `results/` directory.

---

## Expected Outputs

   - final_samples_log.csv: Raw MCMC output in log space.
   - final_samples_linear.csv: Transformed output in linear space for simulation and analysis.
   - failing_samples: Logs the failed model parameters during sampling.
   - mcmc_metadata: Post-processing information for each run: timestamp, initial covariance scaling, number of iterations, final acceptance rate, results directories, and additional model configuration information.
   - iteration_metrics : Metrics like thrust, ion velocity, and discharge current for each sample.

## Visualizing Results

You can visualize MCMC sampling results using the `run_all_plots.py` script:
```
    pdm run python run_all_plots.py
```

This will generate:
- Trace plots of parameters over iterations.
- Posterior distribution plots.
- Pair plots showing parameter correlations.

Plots are saved in the `plots/` subdirectory of the respective MCMC results folder.

# Project Structure 
```
.
├── Manifest.toml
├── Project.toml
├── __init__.py
hall_opt/
├── config/
│   ├── __init__.py
│   ├── simulation.py
│   └── settings.yaml
├── config_loader.py  # NEW: Contains the Config class
├── main.py
├── map.py
├── mcmc.py
└── utils/
    ├── __init__.py
    └── save_data.py

```
 **Additional Information**

- MCMCIterators: This package provides the tools for implementing Delayed Rejection Adaptive Metropolis (DRAM) sampling. [Learn more about MCMCIterators](https://github.com/goroda/MCMCIterators).
- **HallThruster.jl**: A Julia package for Hall Thruster simulations, providing core functionality for the MultiLogBohm and TwoZoneBohm models. See [HallThruster.jl documentation](https://um-pepl.github.io/HallThruster.jl/dev/).
