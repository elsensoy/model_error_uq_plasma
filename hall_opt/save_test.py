import os
import json
import numpy as np
import pathlib
from scipy.stats import norm
from hall_opt.utils.iter_methods import get_next_filename
from hall_opt.config.dict import Settings
from hall_opt.utils.save_data import save_results_to_json

test_metrics = {
    "thrust": 0.046,
    "discharge_current": 6.2,
    "ion_velocity": [1250, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000],
    "z_normalized": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
}

save_results_to_json(
    settings=Settings,
    result_dict=test_metrics,
    filename="metrics_test.json",
    results_dir="hall_opt/results/test",
    save_every_n_grid_points=2  # Subsampling every 2 points
)
