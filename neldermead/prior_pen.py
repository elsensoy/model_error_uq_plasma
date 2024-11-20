import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Define prior and penalty functions
def prior_logpdf(v1_log, alpha_log):
    prior1 = norm.logpdf(v1_log, loc=np.log10(1/160), scale=np.sqrt(2))
    prior2 = norm.logpdf(alpha_log, loc=np.log10(1/16), scale=np.sqrt(2))
    return prior1 + prior2

def bounds_penalty(v_log):
    penalty = 0
    if not (-5 <= v_log[0] <= 0):
        penalty += (v_log[0] - max(-5, min(v_log[0], 0))) ** 2
    if not (0 <= v_log[1] <= 3):
        penalty += (v_log[1] - max(0, min(v_log[1], 3))) ** 2
    return penalty

# Create a grid for plotting
v1_log_vals = np.linspace(-5, 0, 200)
alpha_log_vals = np.linspace(0, 3, 200)
V1_LOG, ALPHA_LOG = np.meshgrid(v1_log_vals, alpha_log_vals)

# Compute prior and penalty
prior_values = np.array([prior_logpdf(v1, alpha) for v1, alpha in zip(V1_LOG.ravel(), ALPHA_LOG.ravel())])
prior_values = prior_values.reshape(V1_LOG.shape)

penalty_values = np.array([bounds_penalty([v1, alpha]) for v1, alpha in zip(V1_LOG.ravel(), ALPHA_LOG.ravel())])
penalty_values = penalty_values.reshape(V1_LOG.shape)

# Combine prior and penalty for visualization
combined_values = prior_values - penalty_values

# # Plotting
# fig, ax = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

# # Prior plot
# cont1 = ax[0].contourf(V1_LOG, ALPHA_LOG, prior_values, levels=50, cmap='viridis')
# fig.colorbar(cont1, ax=ax[0])
# ax[0].set_title("Prior Distribution")
# ax[0].set_xlabel("log10(v1)")
# ax[0].set_ylabel("log10(alpha)")

# # Penalty plot
# cont2 = ax[1].contourf(V1_LOG, ALPHA_LOG, penalty_values, levels=50, cmap='coolwarm')
# fig.colorbar(cont2, ax=ax[1])
# ax[1].set_title("Penalty Function")
# ax[1].set_xlabel("log10(v1)")
# ax[1].set_ylabel("log10(alpha)")

# # Combined plot
# cont3 = ax[2].contourf(V1_LOG, ALPHA_LOG, combined_values, levels=50, cmap='plasma')
# fig.colorbar(cont3, ax=ax[2])
# ax[2].set_title("Prior - Penalty (Adjusted)")
# ax[2].set_xlabel("log10(v1)")
# ax[2].set_ylabel("log10(alpha)")

# # Save plot to a PNG file
# plt.savefig("priors_with_penalty.png")
# plt.show()

fig, ax = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

# Prior plot
cont1 = ax[0].contour(V1_LOG, ALPHA_LOG, prior_values, levels=10, cmap='viridis')
ax[0].clabel(cont1, inline=True, fontsize=8)
ax[0].set_title("Prior Distribution")
ax[0].set_xlabel("log10(v1)")
ax[0].set_ylabel("log10(alpha)")

# Penalty plot
cont2 = ax[1].contour(V1_LOG, ALPHA_LOG, penalty_values, levels=10, cmap='coolwarm')
ax[1].clabel(cont2, inline=True, fontsize=8)
ax[1].set_title("Penalty Function")
ax[1].set_xlabel("log10(v1)")
ax[1].set_ylabel("log10(alpha)")

# Combined plot
cont3 = ax[2].contour(V1_LOG, ALPHA_LOG, combined_values, levels=10, cmap='plasma')
ax[2].clabel(cont3, inline=True, fontsize=8)
ax[2].set_title("Prior - Penalty (Adjusted)")
ax[2].set_xlabel("log10(v1)")
ax[2].set_ylabel("log10(alpha)")

# Save plot to a PNG file
output_path = "mcmc-plots/priors_with_penalty_lines.png"
plt.savefig(output_path)
plt.show()
