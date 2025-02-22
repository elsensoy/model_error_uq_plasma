import sys
import os
import json
import yaml
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from hall_opt.config.dict import Settings

def load_data(settings: Settings, analysis_type: str) -> pd.DataFrame:
    if analysis_type == "ground_truth":
        data_file = Path("results/postprocess/output_multilogbohm.json").resolve()
        print("Ground truth is loaded successfully")
    elif analysis_type == "map":
        data_file = os.path.join(settings.map.base_dir, "final_map_params.json")  #  Dynamically find path
    elif analysis_type == "mcmc":
        data_file = os.path.join(settings.mcmc.base_dir, "final_samples_log.csv")  #  Dynamically find path
    else:
        raise ValueError(" Invalid analysis type. Choose 'map' or 'mcmc'.")

    if not os.path.exists(data_file):
        raise FileNotFoundError(f" Error: Data file not found at {data_file}")

    if analysis_type == "map":
        with open(data_file, "r") as f:
            data = json.load(f)
        samples = pd.DataFrame([data])  #  Convert single MAP estimate to DataFrame
    else:  # MCMC
        samples = pd.read_csv(data_file)

    return samples

def extract_anom_model(settings: Settings, model_type: str) -> Dict[str, Any]:
    """Extracts the anomalous model configuration for the given model type."""
    try:
        anom_model_config = settings.config_settings.anom_model

        # Ensure user-specified `model_type` is valid
        if model_type not in ["TwoZoneBohm", "MultiLogBohm"]:
            raise KeyError(f"❌ ERROR: Unknown anomalous model type '{model_type}'.")

        # Ensure `type` matches what the user specified
        if anom_model_config.type != model_type:
            print(f"⚠️ WARNING: Overriding `type` from {anom_model_config.type} to {model_type}.")
            anom_model_config.type = model_type

        # ✅ Ensure that `model` is FLATTENED, not nested
        return {
            "type": model_type,
            **anom_model_config.model.model_dump()  # ✅ Spread model parameters directly inside `anom_model`
        }

    except KeyError as e:
        print(f"❌ ERROR: {e}")
        return {}


def load_config(config_path):
    """Load the YAML configuration file with debug info."""
    
    config_path = Path(config_path).resolve()  
    print(f"DEBUG: Attempting to load YAML config from {config_path}")

    if not config_path.exists():
        print(f"ERROR: Configuration file does not exist at {config_path}")
        return None

    try:
        with open(config_path, 'r') as file:
            config_data = yaml.safe_load(file)
            print("DEBUG: YAML configuration loaded successfully!")
            return config_data
    except yaml.YAMLError as e:
        print(f"ERROR: Failed to parse YAML file: {e}")
        return None