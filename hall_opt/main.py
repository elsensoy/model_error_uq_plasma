import argparse
import logging
import pathlib
import yaml
from typing import Any
from pydantic import BaseModel, Field
from scipy.stats import norm
from datetime import datetime
from MCMCIterators.samplers import DelayedRejectionAdaptiveMetropolis 
from map_.run_map import run_map
from mcmc.mcmc import run_mcmc_with_optimized_params
from config.simulation import simulation, config_multilogbohm, postprocess, run_simulation_with_config
from utils.plotting import generate_all_plots

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)  # Ensures hall_opt is at the top of the search path

# Add HallThruster Python API to sys.path
hallthruster_path = "/root/.julia/packages/HallThruster/J4Grt/python"
if hallthruster_path not in sys.path:
    sys.path.append(hallthruster_path)

print("Updated sys.path:", sys.path)

import hallthruster as het

#python main.py config.yaml
# Define the configuration model using Pydantic
class Config(BaseModel):
    data_dir: str = Field(..., description="Directory to save all results.")
    gen_data: bool = Field(..., description="Flag to generate data.")
    run_map: bool = Field(False, description="Flag to run MAP estimation.")
    run_mcmc: bool = Field(False, description="Flag to run MCMC sampling.")
    ion_velocity_weight: float = Field(2.0, description="Weight for ion velocity in MAP and MCMC.")
    plotting: bool = Field(True, description="Enable or disable plotting.")
    initial_guess_path: str = Field(..., description="Path to initial guess parameters for MCMC.")
    iterations: int = Field(1000, description="Number of MCMC iterations.")
    initial_cov: list = Field(..., description="Initial covariance matrix for MCMC.")


# Helper function to load YAML configuration
def load_yml_config(path: pathlib.Path, logger: logging.Logger) -> Any:
    """Load and return the YAML configuration."""
    try:
        with path.open("r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError as error:
        message = "Error: YAML config file not found."
        logger.exception(message)
        raise FileNotFoundError(error, message) from error
    except yaml.YAMLError as error:
        message = "Error: Invalid YAML format."
        logger.exception(message)
        raise ValueError(error, message) from error
    
def main():
    # Setup logger
    logger = logging.getLogger("main")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run MCMC and MAP estimation workflow.")
    parser.add_argument("config", type=str, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    # Load the YAML configuration
    logger.info("Loading configuration...")
    #YAML file provided as a command-line argument (config.yaml).
    yml_dict = load_yml_config(pathlib.Path(args.config), logger)
    
    #unpack the dictionary into keyword arguments
    config = Config(**yml_dict)

    # Ensure the data directory exists
    data_dir = pathlib.Path(config.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Data directory set to: {data_dir}")

    # Step 1: Generate ground truth data if gen_data is True
    if config.gen_data:
        logger.info("Generating ground truth data...")
        ground_truth_postprocess = postprocess.copy()
        ground_truth_postprocess["output_file"] = str(data_dir / "ground_truth.json")

        ground_truth_solution = run_simulation_with_config(
            config_multilogbohm, simulation, ground_truth_postprocess, config_type="MultiLogBohm"
        )

        if not ground_truth_solution:
            logger.error("Ground truth simulation failed. Exiting.")
            return

        # Extract observed data
        averaged_metrics = ground_truth_solution["output"]["average"]
        observed_data = {
            "thrust": averaged_metrics["thrust"],
            "discharge_current": averaged_metrics["discharge_current"],
            "ion_velocity": averaged_metrics["ui"][0],
            "z_normalized": averaged_metrics["z"],
        }
        logger.info("Observed data extracted.")
    else:
        observed_data = None

    # Step 2: Run MAP estimation if enabled
    if config.run_map:
        logger.info("Running MAP estimation...")
        v1_opt, alpha_opt = run_map(
            observed_data=observed_data,
            config=config_spt_100, 
            simulation=simulation,
            postprocess=postprocess,
            results_dir=str(data_dir),
            final_params_file="final_parameters.json",
            ion_velocity_weight=config.ion_velocity_weight,
        )

        if v1_opt is not None and alpha_opt is not None:
            logger.info(f"MAP optimization completed: v1={v1_opt}, alpha={alpha_opt}")
        else:
            logger.error("MAP optimization failed.")

    # Step 3: Run MCMC sampling if enabled
     #need to double check this 
    if config.run_mcmc:
        try:
            print("Starting MCMC sampling...")
            run_mcmc_with_optimized_params(
                json_path=initial_guess_path,
                observed_data=observed_data,
                config=config_spt_100,
                ion_velocity_weight=config.ion_velocity_weight,
                iterations=iterations,
                initial_cov=initial_cov,
                results_dir=results_dir
            )
            print("MCMC sampling completed successfully.")
        except Exception as e:
            print(f"Error during MCMC sampling: {e}")

    # Step 4: Generate plots if enabled
    if config.plotting:
        logger.info("Generating plots...")
        plots_dir = data_dir / "plots"
        generate_all_plots(input_dir=str(data_dir), output_dir=str(plots_dir))
        logger.info(f"All plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
