
# README_MCMC_Steps

# Overview
This script performs MCMC sampling to estimate parameters for the TwoZoneBohm model using initial guesses from a prior optimization (using the Nelder-Mead algorithm) and observed data. The main steps involve setting up configurations, defining prior and likelihood functions, running MCMC sampling with checkpointing, and saving results along with metadata.

### Step-by-Step Explanation

## 1. Import Necessary Libraries
   - Imports modules for file handling, JSON parsing, scientific calculations, and MCMC sampling.
   - src.map_nelder_mead is imported to access functions and configurations used in the Nelder-Mead optimization step, which serves as the initial guess for MCMC.

## 2. Define Paths to Initial Guess and Observed Data
   - Sets up RESULTS_DIR, the directory where results and initial guesses from previous optimization runs are saved.
   - initial_guess_path: Path to the initial guess for MCMC based on the optimized parameters from Nelder-Mead.
   - observed_data_path: Path to observed data from the MultiLogBohm simulation, used as ground truth.

## 3. Set Up TwoZoneBohm Configuration
   - config_spt_100 is created as a copy of the MultiLogBohm configuration (`config_multilogbohm`) and modified to represent the TwoZoneBohm model.
   - This configuration will be used in the MCMC process to simulate the model with different parameter values.

## 4. Helper Functions for Saving/Loading Results
   - load_optimized_params: Loads the optimized parameters (v1 and v2) from a JSON file.
   - load_json_data: General-purpose function to load JSON data with error handling.
   - save_metadata: Saves metadata to a JSON file, helping document parameter values, acceptance rate, and other relevant details from the MCMC run.
   - create_specific_config: Extracts relevant configuration details and formats them for documentation, capturing essential information for reproducibility.

## 5. Define Prior and Likelihood Functions
   - prior_logpdf: Sets up Gaussian priors on the logarithmic scale for the parameters `v1` and the scaling factor `alpha`.
   - log_likelihood: Computes the likelihood of observing the data given the simulated data, with weights applied to certain metrics (e.g., ion velocity). The function compares simulated and observed values for `thrust`, `discharge_current`, and `ion_velocity`.
   - log_posterior: Combines the likelihood and prior functions to calculate the log-posterior. This is the main function used in the MCMC sampling to evaluate each parameter set.

## 6. MCMC Sampling with Checkpoints
   - mcmc_inference: Runs MCMC sampling using the Delayed Rejection Adaptive Metropolis (DRAM) algorithm. 
   - Checkpointing: Saves intermediate samples and acceptance rate at specified intervals (`save_interval`), allowing for resuming if the process is interrupted. Checkpoints are saved to both `.csv` and `.json` formats, providing robustness in case of failure.
   - Final samples are saved at the end of the process.

## 7. Run MCMC with Initial Optimized Parameters
   - run_mcmc_with_optimized_params: Initializes the MCMC sampling with the optimized parameters from Nelder-Mead with ion velocity weight 2.0 as the starting point.
   - Sets up initial parameters (v_log_initial), runs mcmc_inference with these, and saves both final samples and metadata.
   - Metadata includes initial guesses, acceptance rate, configuration details, and file paths for saved data, ensuring full documentation for reproducibility.

## 8. Main Function
   - Loads initial parameters and observed data.
   - Calls run_mcmc_with_optimized_params with loaded data and configuration, specifying 'ion_velocity_weight' and 'iterations' as key parameters.
   - Executing the MCMC process with the specified configuration and data.

## 9. Execution
   - To run the script, execute the main function. 

   pdm venv activate
   pdm run python mcmc.py

   - After running, the MCMC results, metadata, and checkpoint files will be saved in the `results-Nelder-Mead` directory.
