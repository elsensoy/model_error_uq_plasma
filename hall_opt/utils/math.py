# Example: hall_opt/math_utils.py

from typing import Callable, Optional, Tuple, Union, List # Added List
import numpy as np
import sys
import statsmodels.tools.numdiff as smnd
_has_statsmodels = True
# Define type alias for the objective function type
ObjectiveFunc = Callable[[np.ndarray], float]


def calculate_hessian_at_point(
    objective_func: ObjectiveFunc,
    point: np.ndarray,
    use_statsmodels: bool = True # Optional flag if you want to disable it sometimes
) -> Tuple[Optional[Union[np.ndarray, str]], Optional[Union[np.ndarray, str]]]:
    """
    Calculates the Hessian and its inverse for a given objective function
    at a specific point using numerical differentiation (statsmodels if available).

    Args:
        objective_func: The function to differentiate (e.g., neg_log_posterior).
                        It should accept a NumPy array and return a float.
        point: The NumPy array point (parameters) at which to calculate the Hessian.
        use_statsmodels: Whether to attempt using the statsmodels library.

    Returns:
        A tuple containing:
        - Hessian matrix (np.ndarray), None (if skipped/failed), or error string.
        - Inverse Hessian matrix (np.ndarray), None (if skipped/failed/singular), or error string.
    """
    hessian_matrix: Optional[Union[np.ndarray, str]] = None
    inv_hessian_matrix: Optional[Union[np.ndarray, str]] = None
    can_calculate = use_statsmodels and _has_statsmodels # Check flag and import status

    if not point.size > 0: # Basic check on the input point
         print("[ERROR] calculate_hessian: Input point is empty.")
         return "Input point empty", None

    if not can_calculate:
        print("[INFO] Hessian calculation skipped (statsmodels not available or disabled).")
        return None, None # Return None for both Hessian and its inverse

    print(f"[INFO] Calculating Hessian of objective function at point {point}...")
    try:
        # approx_hess typically uses central differences by default
        hessian_matrix = smnd.approx_hess(point, objective_func)
        print("[INFO] Hessian calculated numerically.")

        # Print the Hessian matrix to the command line
        print("[DEBUG] Hessian matrix:")
        print(hessian_matrix)
        # --- Calculate Inverse Hessian ---
        try:
            # Ensure Hessian is finite before inverting
            if not np.all(np.isfinite(hessian_matrix)):
                 raise ValueError("Hessian contains non-finite values.")
            inv_hessian_matrix = np.linalg.inv(hessian_matrix)
            print("[INFO] Inverse Hessian calculated.")
        except np.linalg.LinAlgError:
            print("[WARNING] Hessian matrix is singular or near-singular. Cannot compute inverse.")
            # Keep the calculated hessian_matrix, but set inverse to indicator string
            inv_hessian_matrix = "Singular Hessian"
        except ValueError as ve:
             print(f"[WARNING] Cannot compute inverse Hessian: {ve}")
             inv_hessian_matrix = f"Error calculating inverse: {ve}"
        except Exception as inv_e:
             # Catch other potential errors during inversion
             print(f"[WARNING] Failed to compute inverse Hessian: {inv_e}")
             inv_hessian_matrix = f"Error calculating inverse: {inv_e}"

    except Exception as hess_e:
        # Catch errors during the Hessian calculation itself
        print(f"[WARNING] Failed to calculate Hessian matrix: {hess_e}")
        hessian_matrix = f"Error calculating Hessian: {hess_e}" # Store error string
        inv_hessian_matrix = None # Ensure inverse isn't attempted if Hessian fails

    return hessian_matrix, inv_hessian_matrix