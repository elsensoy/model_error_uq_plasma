import json
from pathlib import Path
from typing import Optional, Tuple, List

def restore_mcmc_checkpoint(checkpoint_path: Path) -> Optional[Tuple[int, List[float]]]:
    """
    Restores the latest checkpoint from a given JSON file path.

    Returns:
        A tuple of (iteration_number, last_sample) or None if not found or invalid.
    """
    if not checkpoint_path.is_file():
        print(f"[INFO] No checkpoint found at: {checkpoint_path}")
        return None

    try:
        with open(checkpoint_path, "r") as f:
            data = json.load(f)

        iteration = data.get("iteration")
        samples = data.get("checkpoint_samples")

        if not isinstance(samples, list) or not samples:
            print(f"[WARNING] No samples found in checkpoint file.")
            return None

        last_sample = samples[-1]
        print(f"[INFO] Restored checkpoint at iteration {iteration}.")
        return iteration, last_sample

    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint: {e}")
        return None
