import json
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np

def restore_mcmc_checkpoint(checkpoint_path: Path, checkpoint_number: int) -> Optional[np.ndarray]:
    """
    Restores a specific checkpoint sample from a JSON file with multiple checkpoints.
    """
    if not checkpoint_path.is_file():
        print(f"[INFO] No checkpoint found at: {checkpoint_path}")
        return None

    try:
        with open(checkpoint_path, "r") as f:
            data = json.load(f)

        checkpoint_key = str(checkpoint_number)
        checkpoint_entry = data.get(checkpoint_key)

        if not checkpoint_entry or "sample" not in checkpoint_entry:
            print(f"[WARNING] No valid sample found for checkpoint #{checkpoint_number}.")
            return None

        last_sample = checkpoint_entry["sample"]
        print(f"[INFO] Restored sample from checkpoint #{checkpoint_number}: {last_sample}")
        return np.array(last_sample, dtype=np.float64)

    except Exception as e:
        print(f"[ERROR] Failed to restore checkpoint: {e}")
        return None
