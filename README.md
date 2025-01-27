
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
  - Replace any occurrence of `sklearn` with `scikit-learn` to avoid installation errors.

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
python -m pdm install
```

> **Note:** If you encounter an error such as `pdm not >found`, try run `python -m pdm install` instead of `pdm >install`. 
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
2. **Install HallThruster.jl in Julia REPL:**

    **Inside Julia:**

    -> Load the Pkg module: 

    ```julia
    import Pkg
    ```
    
    -> Activate the project environment
    ```julia
    Pkg.activate(".")
    ```

    -> Add the HallThruster package
     ```julia
    Pkg.add("HallThruster")
     ```
    -> Update all installed packages to their latest versions
    
     ```julia
      Pkg.update()
     ```
    -> Check the installed packages and versions
    
    ```julia
    Pkg.status()
    ```
4. **Integrate HallThruster with Python**

   -> Find the Python script path:

   ```julia
   using HallThruster
   HallThruster.PYTHON_PATH
   ```

   -> Add the path to Python:

   ```python
   import sys
   sys.path.append("/path/to/HallThruster/python")

   import hallthruster as het
   ```

---

## **Workflow Overview**

1. **Configuration:** Define parameters in `settings.yaml`.
2. **MAP Optimization:** Find optimal values using the specified models.
3. **MCMC Sampling:** Generate samples via Bayesian inference.
4. **Visualization:** Analyze results with plots.

---

## **Usage**

To perform MAP estimation:

```bash
python -m hall_opt.main --settings hall_opt/config/settings.yaml
```

To perform MCMC sampling:

```bash
python -m hall_opt.main --settings hall_opt/config/settings.yaml
```

---

## **Visualization**

To generate plots:

```bash
python -m hall_opt.plotting.plotting --settings hall_opt/plotting/plotting.yaml
```
---

## **Contact**

For any questions or issues, contact:

- **Name:** Elida Sensoy  
- **Email:** elsensoy@umich.edu  

