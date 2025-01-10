import pandas as pd
import matplotlib.pyplot as plt

# Define the MCMC parameters and values
mcmc_parameters = {
    "Parameter": [
        "Initial Sample",
        "Iterations",
        "Lower Bound",
        "Upper Bound",
        "Save Interval",
        "Initial Covariance",
        "Adapt Start",
        "Epsilon",
        "Proposal Scaling (sd)",
        "Level Scale"
    ],
    "Value": [
        "[2.0124301397620146, 2.0028894367014445]",  # Example initial sample
        10000,        # Number of iterations
        -5,           # Lower bound for parameters
        3,            # Upper bound for parameters
        10,           # Save interval
        "0.8 * Identity Matrix",  # Covariance
        10,           # Adaptation start
        1e-6,         # Epsilon for stabilization
        "2.4^2 / len(initial_sample)",  # Proposal scaling
        0.1           # Level scale for DRAM
    ]
}

# Convert to a DataFrame
mcmc_df = pd.DataFrame(mcmc_parameters)

# Display the table using matplotlib
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('tight')
ax.axis('off')
table = ax.table(
    cellText=mcmc_df.values,
    colLabels=mcmc_df.columns,
    cellLoc='center',
    loc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(len(mcmc_df.columns))))

# Save or display the table
plt.savefig("mcmc_parameters_table.png", bbox_inches="tight")
plt.show()
