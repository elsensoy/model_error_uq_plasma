import numpy as np
from scipy.stats import norm
from ..config.dict import Settings
from .log_likelihood import log_likelihood
# -----------------------------
# 1. Prior
# -----------------------------
def prior_logpdf(c1_log: float, alpha_log: float) -> float:
    """Computes the log prior probability of the parameters."""
    prior1 = norm.logpdf(c1_log, loc=np.log10(1 / 160), scale=np.sqrt(2))
    prior2 = 0 if 0 < alpha_log <= 2 else -np.inf  # Uniform prior
    return prior1 + prior2

# -----------------------------
# 3. Posterior (Only Save Posterior Value)
# -----------------------------
def log_posterior(c_log: list[float], observed_data: dict, settings: Settings, yaml_file: str) -> float:

    # Compute Prior
    log_prior_value = prior_logpdf(*c_log)
    if not np.isfinite(log_prior_value):
        return -np.inf  
    
    log_likelihood_value = log_likelihood(c_log, observed_data, settings, yaml_file)

    if not np.isfinite(log_likelihood_value):
        return -np.inf  
    # Compute Log Posterior
    log_posterior_value = log_prior_value + log_likelihood_value
    return log_posterior_value