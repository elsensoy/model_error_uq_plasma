## Project Overview
This project aims to optimize the behavior of a Hall Thruster using the HallThruster.jl Julia package, integrated with Python through Juliacall. The project involves parameter estimation using MCMC sampling and MAP optimization, providing tools for comparison between simulated and observed data. It employs Bayesian inference techniques to refine the parameters of the **TwoZoneBohm** and **MultiLogBohm** models.

Please refer to:
- [HallThruster.jl (v0.18.1)](https://um-pepl.github.io/HallThruster.jl/dev/)
- [Simulation Tutorial](https://um-pepl.github.io/HallThruster.jl/dev/tutorials/simulation/)
- [Running Simulations from JSON](https://um-pepl.github.io/HallThruster.jl/dev/howto/json/)
- [Using HallThruster with Python](https://um-pepl.github.io/HallThruster.jl/dev/howto/python/)


## **Table of Contents**
1. [Features](#features)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Workflow Overview](#workflow-overview)
5. [Configuration Files](#configuration-files)
6. [Usage](#usage)
    - [Running MAP Optimization](#running-map-optimization)
    - [Running MCMC Sampling](#running-mcmc-sampling)
7. [Visualization](#visualization)
8. [Contact](#contact)

---

## **Features**

- **Simulation**: Uses HallThruster's simulation capabilities to run time-domain plasma simulations for a given thruster configuration.
- **MAP Estimation**: Implements Maximum A Posteriori (MAP) optimization to estimate parameters such as \( c_1 \) and \( \alpha \) for different anomalous transport models (e.g., TwoZoneBohm, MultiLogBohm).
- **MCMC Sampling**: Employs the Delayed Rejection Adaptive Metropolis (DRAM) sampler for Bayesian parameter estimation and uncertainty quantification.
- **Plots**: Automatically generates plots to visualize parameter convergence during optimization and sampling.

---

## **Requirements**

To run this project, you need the following:

### **Python**
- Python 3.10 or later
- Required Python packages (specified in `requirements.txt`):
  - `numpy`
  - `scipy`
  - `matplotlib`
  - `pydantic`
  - `pyyaml`
  - `MCMCIterators`

### **Julia**
- Julia 1.9 or later
- Make sure the **HallThruster** package installed in your Julia environment. See the references above. 
---

## **Installation**

1. Clone the repository:
    ```bash
    git clone ]((https://github.com/gorodetsky-umich/model_error_uq_plasma.git)
    cd model_error_uq_plasma.git
    ```

2. Create a Python virtual environment and activate it:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3. Install the Python dependencies:
    ```bash
    pdm install
    pip install -r requirements.txt
    ```

4. Install the **HallThruster** package in Julia:
    ```julia
    using Pkg
    Pkg.add("HallThruster")
    ```

5. Add the HallThruster Python path to your script:
    ```python
    import sys
    sys.path.append("<path-to-your-HallThruster-python>")
    import hallthruster as het
    ```

---

## **Workflow Overview**

The workflow consists of the following steps:
1. **Configuration**: Use the YAML configuration file (`settings.yaml`) to define simulation, optimization, and postprocessing parameters.
2. **MAP Optimization**: Run MAP estimation to find optimal values for parameters \( c_1 \) and \( alpha \) using the `TwoZoneBohm` or `MultiLogBohm` models.
3. **MCMC Sampling**: Perform `Delayed Rejection and Adaptive Metropolis` to generate samples.
4. **Visualization**: Generate plots for MAP and MCMC results to analyze parameter convergence and distribution.

---

## **Configuration Files**

The main configuration file is `settings.yaml`. The key sections as follows:

- **General Settings**:
  ```yaml
  general_settings:
    results_dir: "/hall_opt/results"
    gen_data: true
    run_map: true
    run_mcmc: false
    ion_velocity_weight: 2.0
    plotting: true
    iterations: 20
  ```

- **Optimization Parameters**:
  ```yaml
  optimization_params:
    map_params:
      method: "Nelder-Mead"
      map_initial_guess_path: "map-results/map_initial_guess.json"
      maxfev: 5000
      fatol: 1e-3
      xatol: 1e-3
      iteration_log_file: "/map-results/map_iterations.json"
      final_map_params: "/map_results/final_map_params.json"
    mcmc_params:
      save_interval: 10
      checkpoint_interval: 10
      save_metadata: true
      final_samples_file_log: "/mcmc-results/final_samples_log.csv"
      final_samples_file_linear: "/mcmc-results/final_samples_linear.csv"
      checkpoint_file: "/mcmc-results/checkpoint.json"
      metadata_file: "/mcmc-results/mcmc_metadata.json"
      initial_cov:
        - [0.1, -0.5]
        - [-0.5, 0.1]
  ```

- **Simulation Configuration**:
  ```yaml
  simulation_config:
    name: "SPT-100"
    geometry:
      channel_length: 0.025
      inner_radius: 0.0345
      outer_radius: 0.05
    magnetic_field:
      file: "config/bfield_spt100.csv"
    discharge_voltage: 300
    anode_mass_flow_rate: 0.0001
    domain: [0, 0.08]
    propellant: "Xenon"
    ncharge: 3
    anom_model:
      TwoZoneBohm:
        c1: -2.0
        c2: 0.5
      MultiLogBohm:
        zs: [0.0, 0.01, 0.02, 0.03]
        cs: [0.02, 0.024, 0.028, 0.033]
  ```

---

## **Usage**

### **Running MAP Optimization**
To perform MAP estimation:
```bash
python -m hall_opt.main --settings hall_opt/config/settings.yaml
```

### **Running MCMC Sampling**
To perform MCMC sampling:
1. Set `run_mcmc: true` in `settings.yaml`.
2. Run the command:
   ```bash
   python -m hall_opt.main --settings hall_opt/config/settings.yaml
   ```
---
## **Visualization**

- To generate the plots of MAP results:
1. Ensure that `/map_iterations.json` and `final_map_params.json` are saved in their respected folder after the MAP run. Set plotting flag 'true' in the general settings object in the settings.yaml file.
2. Use the provided plotting script:
   ```bash
   python -m hall_opt.main --settings hall_opt/plotting/plotting.yaml
   ```

- Similarly, to demonstrate MCMC sampling:
1. Load the samples from `final_samples_log.csv` and `mcmc_iterations.json` as well as `plotting.yaml`.
2. Ensure that the plotting flag 'true'.
3. Use Python's `matplotlib` library to generate plots.

---


## **Contact**

For questions, issues, feel free to contact:

- **Name**: Elida Sensoy 
- **Email**: elsensoy@umich.edu

