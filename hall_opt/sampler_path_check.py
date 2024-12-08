import sys
import os
#source ~/.venvs/pdm/bin/activate
# Add the absolute path to `MCMCIterators`
module_path = "/mnt/c/Users/MRover/elsensoy/model_error_uq_plasma"
if module_path not in sys.path:
    sys.path.append(module_path)

print("Updated PYTHONPATH:", sys.path)  # For debugging
from MCMCIterators.samplers import DelayedRejectionAdaptiveMetropolis
