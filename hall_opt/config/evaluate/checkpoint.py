import json
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np

def restore_mcmc_checkpoint(checkpoint_path: Path) -> Optional[np.ndarray]:
    """
    Restores the last saved sample from a JSON MCMC checkpoint.
    """
    if not checkpoint_path.is_file():
        print(f"[INFO] No checkpoint found at: {checkpoint_path}")
        return None

    try:
        with open(checkpoint_path, "r") as f:
            data = json.load(f)

        samples = data.get("checkpoint_samples")
        if not samples or not isinstance(samples, list):
            print(f"[WARNING] No valid samples found in checkpoint.")
            return None

        last_sample = samples[-1]
        print(f"[INFO] Restored last sample from checkpoint: {last_sample}")
        return np.array(last_sample, dtype=np.float64)

    except Exception as e:
        print(f"[ERROR] Failed to restore checkpoint: {e}")
        return None