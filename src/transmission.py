from __future__ import annotations

import numpy as np
from scipy.stats import gamma


def gamma_discrete_weights(mean: float, sd: float, max_lag: int) -> np.ndarray:
    """Discretise a Gamma generation-interval distribution on integer days."""
    shape = (mean / sd) ** 2
    scale = sd**2 / mean
    lags = np.arange(1, max_lag + 1)
    weights = gamma.cdf(lags + 0.5, a=shape, scale=scale) - gamma.cdf(
        lags - 0.5, a=shape, scale=scale
    )
    weights = np.maximum(weights, 0.0)
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("Generation-interval weights sum to zero.")
    return weights / total


def infectiousness_series(incidence: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Renewal infectiousness Lambda_t = sum_s I_{t-s} w_s."""
    incidence = np.asarray(incidence, dtype=float)
    weights = np.asarray(weights, dtype=float)
    lam = np.zeros_like(incidence, dtype=float)
    max_lag = len(weights)
    for t in range(len(incidence)):
        m = min(t, max_lag)
        if m > 0:
            lam[t] = np.dot(incidence[t - np.arange(1, m + 1)], weights[:m])
    return lam
