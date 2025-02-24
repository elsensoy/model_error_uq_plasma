
# **Project Overview**

This project aims to optimize the behavior of a Hall Thruster using the **HallThruster.jl** Julia package, integrated with Python through Juliacall. The project involves parameter estimation using MCMC sampling and MAP optimization, providing tools for comparison between simulated and observed data. It employs Bayesian inference techniques to refine the parameters of the **TwoZoneBohm** and **MultiLogBohm** models.

Please refer to:

- [HallThruster.jl (v0.18.1)](https://um-pepl.github.io/HallThruster.jl/dev/)
- [Simulation Tutorial](https://um-pepl.github.io/HallThruster.jl/dev/tutorials/simulation/)
- [Running Simulations from JSON](https://um-pepl.github.io/HallThruster.jl/dev/howto/json/)
- [Using HallThruster with Python](https://um-pepl.github.io/HallThruster.jl/dev/howto/python/)

---

---

## **Table of Contents**
1. [Methods](#methods)
2. [Requirements](#requirements)
3. [Installation](#installation)
   - [Using Python Virtual Environment](#using-python-virtual-environment)
   - [Using PDM for Dependency Management](#using-pdm-for-dependency-management)
   - [HallThruster.jl Installation](#hallthrusterjl-installation)
4. [Run Project](#run-project)
6. [Usage](#usage)
8. [Contact](#contact)

---

## **Methods**

1. **Configuration:**  
   - Define simulation parameters and model settings in a configuration file. 
   - Specify options for simulation, optimization, and visualization to create a customized thruster model.

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
    ```
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

 ```
 python -m pdm self update
 python -m pip install --upgrade pip
 ```
    

#### **Step 4: Activate the PDM virtual environment**

```
python pdm venv activate
```

---
**Windows Installation**

Follow the instructions under [Using Python Virtual Environment](#using-python-virtual-environment) or [Using PDM for Dependency Management](#using-pdm-for-dependency-management).

---

**Ubuntu Installation**

1. **Install required dependencies:**
    ```
    sudo apt update && sudo apt install -y python3 python3-venv python3-pip git julia
    ```

2. **Clone the repository and set up the environment:**
    ```
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
### HallThruster Installation and Python Integration
To ensure HallThruster works correctly with Python, follow these steps to install and integrate it:

1. **Install Julia (1.10 or later)** from [Julia Official Website](https://julialang.org/downloads/)

Ensure Julia is installed and run:
   ```
    where julia  # Windows
    which julia  # macOS/Linux
   ```
---

#### 1. Install HallThruster.jl in Julia


    ```
    mkdir hallthruster_project && cd hallthruster_project
    ```

1. Open Julia and activate your project environment:

   ```julia
   julia
   ```
If it’s not found, add it to PATH or reinstall Julia.
 Activate the project and install the package:

   ```julia
   (@v1.10) pkg> activate .
   (@v1.10) pkg> add HallThruster
   ```

---

#### 1:Automatic Setup via `run.py` 

The script automatically finds Julia (whether installed via juliaup or standalone).
The correct HallThruster.PYTHON_PATH is dynamically retrieved. So after confirming
If you experience import errors, run the following in Julia:

    julia```
    using Pkg
    Pkg.add("HallThruster")

Then, try running run.py again.

After installation, find the Python script path by running the following command in Julia:

```julia
using HallThruster
println(pathof(HallThruster))
```
The output will contain the package installation path. 

#### 2: Verify the PYTHONPATH Variable
If you want to manually check that `HallThruster` was added to `PYTHONPATH`, run:

    ```powershell
    echo $env:PYTHONPATH
    ```
    ```sh
    echo $PYTHONPATH
    ```
You should see the `HallThruster` path included in the output.

---
---
#### 3: Manually Add the Path in `main.py`

If hallthruster is still not found, `main.py` you can add the snippet below at the top of main.py, 
This will check if `HallThruster` is available. If not, it will add the path:

    ```python
    import sys

    # Ensure HallThruster module is in the Python path
    hallthruster_path = "C:/Users/your-name/.julia/packages/HallThruster/yxE62/python"
    if hallthruster_path not in sys.path:
        sys.path.append(hallthruster_path)

    import hallthruster as het
    print("HallThruster imported successfully!")

    ```
#### 4 : Import HallThruster in Python
Once the environment is set (via `run.py`), you can test if `HallThruster` is available in Python:

    ```python
    import hallthruster as het

    # Check if the module is loaded correctly
    print("hallThruster successfully imported!")
    ```

    If `HallThruster` is available, the script will proceed normally.  
    If not available, `main.py` will manually attempt to add it.

    ---
---


### **Run Project**

**Command Line Arguments**

- **`gen_data.yaml`** – For data generation  
- **`map.yaml`** – For MAP estimation  
- **`mcmc.yaml`** – For MCMC sampling  
- **`plotting.yaml`** – For visualization  

*Run:*
```bash
python run.py `my-method`.yaml
```

This will:
- Automatically find the project root
- Locate the correct Python and Julia paths
- Ensure HallThruster is correctly imported
- Run the simulation with the specified YAML configuration.
---

### Results directory:
```
    model_error_uq_plasma/
    │── hall_opt/
    │   ├── main.py             # Main execution script
    │   ├── results/            # Stores all output files
    │   ├── results_test/       # Stores test results
    │   ├── config/             # Configuration files
    │   ├── scripts/            # Method scripts
    │   ├── utils/              # Helper scripts
    │   ├── posterior/          # Posterior calculations
    │── run.py                  # Entry point
    │── README.md               # Documentation
    │── pyproject.toml          # Python dependencies (PDM)
    │── .venv/                  # Virtual environment
```
## **Contact**

For any questions or issues, contact:

- **Name:** Elida Sensoy  
- **Email:** elsensoy@umich.edu  

