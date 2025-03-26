import os
import json
from pathlib import Path
from hall_opt.config.verifier import Settings
from hall_opt.config.run_model import run_model
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from hall_opt.config.dict import Settings
from hall_opt.config.verifier import Settings
from hall_opt.utils.data_loader import load_data



def generate_ground_truth(settings: Settings):
    """Run ground truth model and return result dict (no file I/O)."""
    output_file = Path(settings.postprocess.output_file["MultiLogBohm"])
    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        result = run_model(
            config_settings=settings.config_settings,
            settings=settings,
            simulation=settings.simulation.model_dump(),
            postprocess=settings.postprocess,
            model_type="MultiLogBohm",
        )

        if not result:
            print("ERROR: Simulation failed.")
            return None

        # Only return data — no saving   
        return result

    except Exception as e:
        print(f"[ERROR] during ground truth generation: {e}")
        return None


def get_ground_truth_data(settings: Settings) -> Optional[Tuple[dict, dict]]:
    """
    Returns a tuple of (raw_simulation_output, extracted_metrics_dict).
    If loading from CSV/JSON, returns (dataframe, None).
    """
    observed_data = None
    metrics = None

    primary_path = Path(settings.reference_data).resolve()
    fallback_path = Path(settings.postprocess.output_file["MultiLogBohm"])

    #   Option 1: Generate
    if settings.gen_data:
        try:
            print("[INFO] `gen_data=True` → Generating ground truth...")
            observed_data = generate_ground_truth(settings)

            if observed_data is None:
                return None, None

            #  metrics here but no saving (always outside of the library)
            metrics = observed_data.get("output", {}).get("average", {})
            print("[INFO] Ground truth successfully generated.")
            return observed_data, metrics
        except Exception as e:
            print(f"[ERROR] Ground truth generation failed: {e}")
            return None, None
    # Option 2: Load from reference_data if file exists
    print(f"[INFO] Attempting to load external ground truth from: {primary_path}")
    if primary_path.exists():
        try:
            if primary_path.suffix == ".csv":
                observed_data = pd.read_csv(primary_path)
                print("[INFO] Loaded ground truth from CSV.")
                return observed_data, None
            elif primary_path.suffix == ".json":
                with open(primary_path, "r") as f:
                    observed_data = json.load(f)
                print("[INFO] Loaded ground truth from JSON.")
                return observed_data, None
            else:
                print(f"[WARNING] Unsupported file extension for reference_data: {primary_path.suffix}")
        except Exception as e:
            print(f"[WARNING] Failed to load reference_data from {primary_path}: {e}")
            print(f"[WARNING] Skipping to fallback because reference_data could not be parsed.")


    # Option 3: Fallback to postprocess output
    print(f"[INFO] Trying fallback ground truth at: {fallback_path}")
    if fallback_path.exists():
        try:
            observed_data = load_data(settings, "ground_truth")
            print("[INFO] Ground truth loaded from fallback.")
            return observed_data, None
        except Exception as e:
            print(f"[WARNING] Failed to load fallback: {e}")

    # No data found
    print("[FATAL] No ground truth data could be found.")
    print("[SUGGESTION] Try setting `gen_data: true` in your input YAML or ensure `reference_data` path is valid.")
    return None, None


'''EXP:
------------------------------------------------------------
Step                                  | Behavior
gen_data: true                        | Generates new data and extracts metrics
reference_data provided               |Attempts to load that first if gen_data is false
reference_data fails or isn’t usable  |Falls back to postprocess output
postprocess.output_file also fails    |Logs a fatal error, suggests using gen_data: true

------------------------------------------------------------

'''