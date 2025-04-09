
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
   - [Dependency Management](#using-pdm-for-dependency-management)
   - [HallThruster.jl Installation](#hallthrusterjl-installation)
4. [Run Project](#run-project)
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

#### **Python**
- Python 3.10 or later
- Required Python packages (from `requirements.txt`):
  - `numpy`
  - `scipy`
  - `matplotlib`
  - `pydantic`
  - `pyyaml`
  - `MCMCIterators`
  - `arviz` *(for MCMC analysis)*

#### **Julia**
- Julia 1.10 or later
- Install the **HallThruster** package using Julia's package manager.

---

## **Installation**


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
    python3 -m venv .venv   
    source .venv/bin/activate
    ```

## **Dependency Management**

[PDM (Python Dependency Manager)](https://pdm-project.org) offers a more streamlined way to manage dependencies and virtual environments.


#### **Step 2: Install PDM**

If you don't have PDM installed, install it via:

```bash
python pip install pdm
```

#### **Step 3: Install dependencies with PDM**

Install dependencies:

```bash
pdm install
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
pdm venv activate
```
### Auto-Activation in ~/.bashrc, ~/.zshrc (optional)
    '''
    function cd {
        builtin cd "$@"
        # bash regex pdm-test / conditionally activate the venv in the current shell 
        if [[ "$PWD" =~ pdm-test ]]
        then
            eval $(pdm venv activate)
        fi
    }
    '''
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


#### Ensure Julia is installed and run:
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
placeholder half replace the hallthruster jls.e

---
If you experience import errors, run the following in Julia:

    julia```
    using Pkg
    Pkg.add("HallThruster")

Check if the package installation was successful:

    julia```
        using Pkg
        Pkg.status()
    ```

After installation, find the Python script path by running the following command in Julia:

```julia
using HallThruster; 
HallThruster.PYTHON_PATH
```

Once the environment is set, you can test if `HallThruster` is available in Python:

    ```python
    import hallthruster as het

    # Check if the module is loaded correctly
    print("hallThruster successfully imported!")
    ```

    If `HallThruster` is available, the script will proceed normally.  
    If not available, `main.py` will manually attempt to add it.

    ---

### 3: TROUBLESHOOT: Manually Add the Path in `main.py`

If hallthruster is still not found, in `main.py`,  edit the snippet below (located at the top of main.py) and replace the placeholder with your hallthruster python path, 
This will check if `HallThruster` is accessed. If not, it will add the path:

    ```python
    import sys

    # Ensure HallThruster module is in the Python path
    hallthruster_path = "/Users/your-name/.julia/packages/HallThruster/abc/python"
    if hallthruster_path not in sys.path:
        sys.path.append(hallthruster_path)

    import hallthruster as het
    print("HallThruster imported successfully in main!")

    ```


---
---
## 2: Automatic Setup via `run.py`

#### After hallthruster installation is completed, you are ready to run the project. See the guide for running the project below.

---

### **Run Project**

This project uses [PDM](https://pdm-project.org/) for managing dependencies and executing tasks. The `run.py` script serves as the main entry point, handling essential setup before launching the core logic.

**1. Install Dependencies (If not already done):**

```
    pdm install
```
**2. Execute the Analysis:**

Use the PDM script command defined in pyproject.toml (see the manually added entry `main = "python run.py"`  in the toml file) This allows define the console script entry points within the package. Now, we can easily run:

```
    pdm run main my_project.yaml
```

### Example:
```
    pdm run main configuration.yaml
```

#### Note: 

This command can be run from any directory within the project. PDM ensures correct execution from the project root. See link here for more details: https://pdm-project.org/latest/usage/scripts/ 

---
    

##               YAML VALIDATION & DEFAULTS WORKFLOW

Here is a breakdown of the internal steps typically taken after the project is launched via pdm run main 'my_project.yaml'. 

#### 1. START PROCESS
   ------------------------------------------------------------
- Main script execution begins in `main.py`.
- The YAML configuration file is read from command-line args.

#### 2. PARSE YAML CONFIGURATION
   ------------------------------------------------------------
- Read the YAML file using `load_yaml()`.
- Convert YAML content into a Python dictionary.

#### 3. VERIFY YAML CONTENT & CREATE DEFAULTS (Pydantic Validation)
   ------------------------------------------------------------
- `verify_all_yaml(yaml_data)` validates the config.
- If fields are missing, Pydantic assigns default values:
- `gen_data = False`
- `thruster = "SPT-100"`
- `anom_model = MultiLogBohm (default values)`
- If verification fails, exit process.

   ------------------------------------------------------------

#### 4. RESOLVE FILE PATHS

- Calls `resolve_yaml_paths(settings)`.
- Converts relative paths to absolute paths.

------------------------------------------------------------
#### 5. OVERRIDE DEFAULT SETTINGS WITH USER-SPECIFIED VALUES

- Flags from YAML (`gen_data`, `run_map`, etc.) override defaults.
- `setattr(settings.general, flag, value)` applies overrides.
------------------------------------------------------------

#### 6. CREATE RESULTS DIRECTORY

- Ensures `results_test/` directory exists.
- Creates subdirectories dynamically.
------------------------------------------------------------

#### 7. VALIDATE `gen_data` FLAG & RUN DATA GENERATION

- If `gen_data=True`:
- Calls `generate_ground_truth(settings)`.
- Runs `run_model()` to create synthetic data.
- Saves results in `results_test/postprocess/output_multilogbohm.json`.
- If `gen_data=False`, loads existing ground truth data.


#### 8. ERROR HANDLING & EXCEPTION MANAGEMENT
------------------------------------------------------------
- If missing config keys (`KeyError`), print error and exit.
- If `thruster` is None, assign a default model.
- If ground truth data is missing, stop execution.
------------------------------------------------------------

#### 9. CONTINUE EXECUTION OR EXIT

- If YAML is valid, execution continues to MAP/MCMC steps.
- If errors occur, exit process.
------------------------------------------------------------
#### 10. END PROCESS

- All settings are finalized.
- Ready to proceed with the next computational steps.

### Results directory:
```
    model_error_uq_plasma/
    │── hall_opt/
    │   ├── main.py             # Main execution script
    │   ├── results/            # Your output directory
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
