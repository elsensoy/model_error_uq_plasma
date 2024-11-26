# Redefine the MAP estimates for clarity
v1_log_map = np.log10(2.0124301397620146)  # MAP estimate for v1 from earlier
alpha_log_map = np.log10(2.0028894367014445 / 2.0124301397620146)  # MAP estimate for alpha

# Plot the distributions again with updated MAP estimates
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot for log10(c1)
axes[0].plot(x_c1, pdf_c1, label='Gaussian Prior for log10(c1)', color='blue')
axes[0].axvline(mean_c1, color='red', linestyle='--', label=f'Mean = {mean_c1:.2f}')
axes[0].axvline(min(x_c1), color='green', linestyle=':', label=f'Min = {min(x_c1):.2f}')
axes[0].axvline(max(x_c1), color='purple', linestyle=':', label=f'Max = {max(x_c1):.2f}')
axes[0].scatter(v1_log_map, norm.pdf(v1_log_map, loc=mean_c1, scale=std_c1), 
                color='black', label=f'MAP Estimate = {v1_log_map:.5f}')
axes[0].set_title('Prior Distribution for log10(c1)')
axes[0].set_xlabel('log10(c1)')
axes[0].set_ylabel('Probability Density')
axes[0].legend()
axes[0].grid()

# Plot for log10(alpha)
axes[1].plot(x_alpha, pdf_alpha, label='Log-Uniform Prior for log10(alpha)', color='green')
axes[1].axvline(alpha_min, color='red', linestyle='--', label=f'Min = {alpha_min:.2f}')
axes[1].axvline(alpha_max, color='purple', linestyle='--', label=f'Max = {alpha_max:.2f}')
axes[1].scatter(alpha_log_map, uniform.pdf(alpha_log_map, loc=alpha_min, scale=alpha_max - alpha_min), 
                color='black', label=f'MAP Estimate = {alpha_log_map:.5f}')
axes[1].set_title('Prior Distribution for log10(alpha)')
axes[1].set_xlabel('log10(alpha)')
axes[1].set_ylabel('Probability Density')
axes[1].legend()
axes[1].grid()

# Adjust layout to avoid cutoffs
plt.tight_layout()

# Save the revised plot
output_path = "/mnt/data/revised_priors_with_map_estimates_corrected.png"
plt.savefig(output_path)
plt.show()

output_path
