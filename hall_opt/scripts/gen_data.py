import os
import json
from pathlib import Path
from hall_opt.config.verifier import Settings
from hall_opt.config.run_model import run_model
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union 
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
    Returns a tuple of (observed_data, metrics).
    observed_data can be a dict (generated/JSON) or DataFrame (CSV).
    metrics is a dict or None.
    Prioritizes: Generate -> Load reference -> Load specific fallback -> Search fallback.
    """
    observed_data: Union[pd.DataFrame, Dict, None] = None # Type hint for clarity
    metrics: Optional[Dict] = None

    # --- Setup Paths ---
    # Resolve primary reference path if provided, else None
    primary_path = Path(settings.reference_data).resolve() if settings.reference_data else None
    fallback_path = Path(settings.postprocess.output_file["MultiLogBohm"])
    fallback_filename = fallback_path.name # Get filename for potential search


    # --- Option 1: Generate ---
    if settings.gen_data:
        try:
            print("[INFO] `gen_data=True` -> Generating ground truth...")
            # Assuming generate_ground_truth returns dict or None
            observed_data_dict = generate_ground_truth(settings)
            if observed_data_dict is None:
                print("[ERROR] Ground truth generation returned None.")
                return None, None # Stop if generation failed

            # Assign potentially generated data
            observed_data = observed_data_dict
            metrics = observed_data.get("output", {}).get("average", {})
            print("[INFO] Ground truth successfully generated.")
            return observed_data, metrics # <<< EXIT POINT IF GENERATED
        except Exception as e:
            print(f"[ERROR] Ground truth generation failed: {e}")
            return None, None # Stop if generation was requested but failed


    # --- Option 2: Load from Primary Reference Path ---
    # This runs only if gen_data is False
    if primary_path:
        print(f"[INFO] Attempting to load ground truth from primary reference: {primary_path}")
        if primary_path.exists():
            try:
                loaded_successfully = False
                if primary_path.suffix == ".csv":
                    observed_data = pd.read_csv(primary_path)
                    print("[INFO] Loaded ground truth from primary CSV.")
                    loaded_successfully = True
                elif primary_path.suffix == ".json":
                    with open(primary_path, "r") as f:
                        observed_data = json.load(f)
                    print("[INFO] Loaded ground truth from primary JSON.")
                    loaded_successfully = True
                else:
                    print(f"[WARNING] Unsupported file extension for reference_data: {primary_path.suffix}. Skipping.")
                    # observed_data remains None

                if loaded_successfully:
                    return observed_data, None # <<< EXIT POINT IF PRIMARY LOAD SUCCEEDS

            except Exception as e:
                print(f"[WARNING] Failed to load reference_data from {primary_path}: {e}")
                print("[INFO] Proceeding to fallback options as primary reference failed.")
                observed_data = None # Ensure observed_data is None so fallback proceeds
        else:
             print(f"[INFO] Primary reference file not found at: {primary_path}")
             # observed_data remains None
    else:
        print("[INFO] No primary reference_data path specified. Proceeding to fallback options.")
        # observed_data remains None


    # --- Option 3: Fallback (Load Specific -> Search) ---
    # This section runs ONLY if observed_data is still None after Options 1 & 2
    if observed_data is None:
        print("[INFO] Proceeding to fallback options...")

        # --- Attempt 3a: Try loading from the specific fallback path using load_data ---
        print(f"[INFO] Trying fallback ground truth via load_data (expected path: {fallback_path})")
        try:
            #  load_data handles FileNotFoundError or returns None if missing/invalid
            observed_data_fallback = load_data(settings, "ground_truth")

            if observed_data_fallback is not None:
                print("[INFO] Ground truth loaded successfully from specific fallback via load_data.")
                # If successful, assign to the main variable and return
                observed_data = observed_data_fallback
                return observed_data, None # <<< EXIT POINT IF SPECIFIC FALLBACK LOAD SUCCEEDS
            else:
                print("[INFO] load_data completed but returned no data (file might be missing/empty?).")
                # observed_data remains None

        except FileNotFoundError:
            print(f"[INFO] Specific fallback file not found by load_data at expected location.")
            # observed_data remains None
        except Exception as e:
            print(f"[WARNING] Failed to load specific fallback via load_data: {e}")
            # observed_data remains None


        # --- Attempt 3b: Search for the file if specific load failed ---
        # This code runs ONLY if observed_data is still None after Attempt 3a
        if observed_data is None:
            print(f"[INFO] Specific fallback failed. Searching for '{fallback_filename}'...")

            # Keep import local as per original structure
            try:
                from hall_opt.utils.parse import find_file_anywhere
            except ImportError:
                 print("[ERROR] Cannot search: Failed to import find_file_anywhere.")
                 # Fallback search not possible
            else:
                # Call the ORIGINAL find_file_anywhere (no exclusion added)
                # Passing only the filename derived from the fallback_path
                alt_fallback = find_file_anywhere(fallback_filename)

                if alt_fallback and alt_fallback.exists():
                    print(f"[INFO] Found fallback file via search at: {alt_fallback}")
                    try:
                        # Load the found file (
                        with open(alt_fallback, "r") as f:
                            observed_data = json.load(f) # Load into main variable
                        print(f"[INFO] Successfully loaded fallback ground truth from searched file.")
                        return observed_data, None # <<< EXIT POINT IF SEARCH/LOAD SUCCEEDS
                    
                    except Exception as e:
                        print(f"[WARNING] Failed to load fallback ground truth from found file {alt_fallback}: {e}")
                        # If loading the found file fails, observed_data remains None
                else:
                    # find_file_anywhere already prints a warning if file not found
                    print(f"[INFO] Search for '{fallback_filename}' did not find a valid file.")
                    # observed_data remains None


    # Final Outcome 
    # This point is reached only if ALL methods failed or were skipped
    if observed_data is None:
         print("[ERROR] CRITICAL: Failed to obtain ground truth data via all methods (generation, reference, specific fallback, search).")

    # Return whatever observed_data contains (could be None) and metrics (None unless generated)
    return observed_data, metrics

'''EXP:
------------------------------------------------------------
Step                                  | Behavior
gen_data: true                        | Generates new data and extracts metrics
reference_data provided               |Attempts to load that first if gen_data is false
reference_data fails or isn’t usable  |Falls back to postprocess output
postprocess.output_file also fails    |Logs a fatal error, suggests using gen_data: true

------------------------------------------------------------

'''