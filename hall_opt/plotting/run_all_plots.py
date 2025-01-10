import sys
from pathlib import Path
import hall_opt.plotting.posterior_plots as posterior_plots
import delta_plots
import mcmc_iter_plots

def run_all_plots(input_dir, output_dir):
    """
    Call individual plotting scripts with specified input/output directories.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running posterior plots...")
    posterior_plots.main(input_dir, output_dir)

    print(f"Running delta plots...")
    delta_plots.main(input_dir, output_dir)

    print(f"Running MCMC iteration plots...")
    mcmc_iter_plots.main(input_dir, output_dir)

if __name__ == "__main__":
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    run_all_plots(input_dir, output_dir)
