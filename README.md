
# **Project Overview**

This project aims to optimize the behavior of a Hall Thruster using the **HallThruster.jl** Julia package, integrated with Python through Juliacall. The project involves parameter estimation using MCMC sampling and MAP optimization, providing tools for comparison between simulated and observed data. It employs Bayesian inference techniques to refine the parameters of the **TwoZoneBohm** and **MultiLogBohm** models.

Please refer to:

- [HallThruster.jl (v0.18.1)](https://um-pepl.github.io/HallThruster.jl/dev/)
- [Simulation Tutorial](https://um-pepl.github.io/HallThruster.jl/dev/tutorials/simulation/)
- [Running Simulations from JSON](https://um-pepl.github.io/HallThruster.jl/dev/howto/json/)
- [Using HallThruster with Python](https://um-pepl.github.io/HallThruster.jl/dev/howto/python/)

---

## **Table of Contents**
1. [Features](#features)
2. [Requirements](#requirements)
3. [Installation](#installation)
   - [Using Python Virtual Environment](#using-python-virtual-environment)
   - [Using PDM for Dependency Management](#using-pdm-for-dependency-management)
   - [Windows Installation](#windows-installation)
   - [Ubuntu Installation](#ubuntu-installation)

   - [HallThruster.jl Installation](#hallthrusterjl-installation)
4. [Workflow Overview](#workflow-overview)
5. [Configuration Files](#configuration-files)
6. [Usage](#usage)
7. [Visualization](#visualization)
8. [Contact](#contact)

---

## **Requirements**

To run this project, ensure the following dependencies are installed:

### **Python**
- Python 3.10 or later
- Required Python packages (from `requirements.txt`):
  - `numpy`
  - `scipy`
  - `matplotlib`
  - `pydantic`
  - `pyyaml`
  - `MCMCIterators`
  - `arviz` *(for MCMC analysis)*

### **Julia**
- Julia 1.10 or later
- Install the **HallThruster** package using Julia's package manager.

---

## **Installation**

### **Using Python Virtual Environment**

1. **Clone the repository:**
    ```bash
    git clone https://github.com/gorodetsky-umich/model_error_uq_plasma.git
    cd model_error_uq_plasma
    ```

2. **Create a virtual environment and activate it:**
    ```bash
    python -m venv .venv
    # On Windows
    .venv\Scripts\activate

    # On Ubuntu/MacOS
    source .venv/bin/activate
    ```

3. **Install dependencies:**
    ```bash
    python -m pip install -r requirements.txt
    ```
---

### **Using PDM for Dependency Management**

[PDM (Python Dependency Manager)](https://pdm-project.org) offers a more streamlined way to manage dependencies and virtual environments.

#### **Step 1: Install PDM**

If you don't have PDM installed, install it via:

```bash
python -m pip install pdm
```

#### **Step 2: Clone the repository**

```bash
git clone https://github.com/gorodetsky-umich/model_error_uq_plasma.git
cd model_error_uq_plasma
```

#### **Step 3: Install dependencies with PDM**

Instead of using a virtual environment manually, PDM will handle it:

```bash
python pdm install
```

Verify the required packages are available in your environment.

```bash
pdm list
```

> **Note:** If you encounter an error such as `pdm not >found`, try run `python -m pdm install` instead of `pdm install`. 
>
> To check if any installed Python packages, including PDM, > need an update:
> ```
    > python -m pip list --outdated
>    
 
>If pdm appears in the list, update it using:
> ```
    > python -m pdm self update
    > python -m pip install --upgrade pip

    

#### **Step 4: Activate the PDM virtual environment**

```bash
python -m pdm venv activate
```

---
### **Windows Installation**

Follow the instructions under [Using Python Virtual Environment](#using-python-virtual-environment) or [Using PDM for Dependency Management](#using-pdm-for-dependency-management).

---

### **Ubuntu Installation**

1. **Install required dependencies:**
    ```bash
    sudo apt update && sudo apt install -y python3 python3-venv python3-pip git julia
    ```

2. **Clone the repository and set up the environment:**
    ```bash
    git clone https://github.com/gorodetsky-umich/model_error_uq_plasma.git
    cd model_error_uq_plasma
    python3 -m venv .venv
    source .venv/bin/activate
    ```
---

> **Note:** **Troubleshooting Dependency Conflicts**  
> If you encounter dependency resolution issues (such as conflicts with package versions, e.g., pandas), try the following commands to resolve them:  
>  
> ```bash
> pdm update --unconstrained
> pdm lock --refresh
> ```  
>  
> These commands will relax dependency constraints and attempt to find compatible versions automatically.

---
### **HallThruster.jl Installation**

1. **Install Julia (1.10 or later)** from [Julia Official Website](https://julialang.org/downloads/)

2. **Activate a project-specific environment:**

    ```bash
    mkdir hallthruster_project && cd hallthruster_project
    julia
    ```

### **HallThruster Installation and Python Integration**

To ensure HallThruster works correctly with Python, follow these steps to install and integrate it:

---

#### **1. Install HallThruster.jl in Julia**

1. Open Julia and activate your project environment:

   ```julia
   julia
   ```

2. Activate the project and install the package:

   ```julia
   (@v1.10) pkg> activate .
   (@v1.10) pkg> add HallThruster
   ```

---

#### **2. Locate HallThruster Python Path**

After installation, find the Python script path for HallThruster by running the following command in Julia:

```julia
using HallThruster
println(pathof(HallThruster))
```

The output will contain the package installation path. Typically, it looks like:

```
C:\Users\yourname\.julia\packages\HallThruster\yxE62\python
```

---

#### **3. Set the Python Path (PYTHONPATH) Permanently**

To make HallThruster available to Python, you can set the `PYTHONPATH` environment variable permanently. Here are the steps:

1. Open **PowerShell as Administrator**.
2. Run the following command to add the HallThruster path permanently to the user environment variables:

   ```powershell
   [System.Environment]::SetEnvironmentVariable("PYTHONPATH", "C:\path", [System.EnvironmentVariableTarget]::User)
   ```

3. Restart PowerShell or your system for changes to take effect.

---

#### **4. Verify the PYTHONPATH Variable**

To verify that the path was added correctly, open PowerShell and run:

```powershell
echo $env:PYTHONPATH
```

You should see the HallThruster path listed in the output.

---

#### **5. Import HallThruster in Python**

Once the environment variable is set, you can test the integration by running the following Python script:

```python
import hallthruster as het

# Check if the module is loaded correctly
print("HallThruster successfully imported!")
```
---
#### **6. Manually Add the Path (If Needed)**

If you prefer not to set the path permanently, you can add it manually in your Python scripts before importing HallThruster:

```python
import sys

hallthruster_path = "C:\\Users\\elsensoy\\.julia\\packages\\HallThruster\\yxE62\\python"
if hallthruster_path not in sys.path:
    sys.path.append(hallthruster_path)

import hallthruster as het
print("HallThruster imported successfully!")
```

---

## **Workflow Overview**

1. **Configuration:**  
   - Define simulation parameters and model settings in the `settings.yaml` configuration file.  
   - Specify options for simulation, optimization, and visualization in their respective .yaml configuration files.

2. **Data Generation:**  
   - Generate synthetic or experimental ground truth data based on the defined configuration.  

3. **MAP (Maximum A Posteriori) Optimization:**  
   - Estimate the optimal parameter values using the chosen anomalous transport models (e.g., `TwoZoneBohm`, `MultiLogBohm`).  
   - Optimizate to minimize the error between simulated and observed data.

4. **MCMC (Markov Chain Monte Carlo) Sampling:**  
   - Conduct Bayesian inference to obtain parameter distributions.  
   - Use MCMC techniques (e.g., Delayed Rejection Adaptive Metropolis) to quantify uncertainty.

5. **Visualization and Analysis:**  
   - Generate plots and statistical summaries to analyze parameter convergence and posterior distributions.  
   - Compare simulation results against observed data for model validation.
---
### **Run Project**

*Run:*
```bash
run.bat
```
## **Optional Command Line Arguments

- **`--gen_data`** – For data generation  
- **`--map`** – For MAP estimation  
- **`--mcmc`** – For MCMC sampling  
- **`--plotting`** – For visualization  


| Process     |   Flag to Enable      |   Command Line Argument
|-------------|--------------------  -|--------------------------|
| **Generate Data**  `gen_data: true` |   run.bat --gen_data
| **MAP Estimation** `run_map: true`  |    run.bat --map
| **MCMC Sampling** `run_mcmc: true`  |    run.bat --mcmc
| **Visualization** `plotting: true`  |    run.bat --plotting


To configure and run the workflow with yaml settings, modify the respective `settings.yaml` files listed below:  

Command-line arguments (--mcmc, --map, --gen_data) override the corresponding YAML flags (run_map, run_mcmc, gen_data). This way, users can either rely on the YAML configuration or override it via CLI arguments.

# Behavior:
Uses YAML settings: If no CLI arguments are passed, default YAML values are used.
CLI overrides YAML	If --mcmc, --map, or --gen_data is passed, it takes priority over the YAML file.
Works with run.bat	Automatically passes arguments from the batch file.
---

### Results directory:
```
    model_error_uq_plasma
    ├run.bat   
    ├hall_opt
    │   ├── config
    │   │   ├── bfield_spt100.csv
    │   │   ├── dict.py
    │   │   ├── run_model.py
    │   │   ├── settings.yaml
    │   │   └── verifier.py
    │   ├── main.py
    │   ├── plotting
    │   │   ├── __init__.py
    │   │   ├── __pycache__
    │   │   │   ├── __init__.cpython-311.pyc
    │   │   │   ├── common_setup.cpython-311.pyc
    │   │   │   ├── plot_ground_truth.cpython-311.pyc
    │   │   │   └── posterior_plots.cpython-311.pyc
    │   │   ├── common_setup.py
    │   │   ├── iteration_plots.py
    │   │   ├── plot_ground_truth.py
    │   │   ├── posterior_plots.py
    │   │   └── tests
    │   │       ├── 2d_plot_kde.py
    │   │       ├── delta_plots.py
    │   │       ├── generate_truth_data.py
    │   │       ├── map_test.py
    │   │       └── mock_test.py
    │   ├── posterior
    │   │   ├── __pycache__
    │   │   │   ├── log_likelihood.cpython-311.pyc
    │   │   │   └── statistics.cpython-311.pyc
    │   │   ├── log_likelihood.py
    │   │   └── statistics.py    
        /results/
        ├── map/
        │   ├── map-results-1/  Created dynamically per MAP run
        │   │   ├── iter_metrics/
        │   │   │   ├── metrics_1.json  Iteration metrics
        │   │   │   ├── metrics_2.json
        │   │   ├── map_iteration_log.json  Stores all iterations. Loaded in `load_data()`
        │   │   ├── final_map_params.json  Final MAP sample 
        │   │   ├── plots/  Automatically created using `get_common_paths()`
        │   ├── map-results-2/
        │   │   ├── iter_metrics/
        │   │   ├── map_iteration_log.json   
        │   │   ├── final_map_params.json
        │   │   ├── plots/
        ├── mcmc/
        │   ├── mcmc-results-1/
        │   │   ├── iter_metrics/
        │   │   │   ├── metrics_1.json
        │   │   │   ├── metrics_2.json
        │   │   ├── checkpoint.json  Checkpoint saving dynamically
        │   │   ├── final_samples_log.csv  Final sample logs, Loaded in `load_data()`
        │   │   ├── mcmc_metadata.json  Metadata for MCMC
        │   │   ├── plots/
        │   ├── mcmc-results-2/
        │   │   ├── iter_metrics/
        │   │   ├── checkpoint.json
        │   │   ├── final_samples_log.csv
        │   │   ├── mcmc_metadata.json
        │   │   ├── plots/

```
---

## **Contact**

For any questions or issues, contact:

- **Name:** Elida Sensoy  
- **Email:** elsensoy@umich.edu  

