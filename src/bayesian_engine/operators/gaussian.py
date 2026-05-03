"""Gaussian operator: exp(-(x-μ)² / 2σ²)."""

from typing import Callable

import numpy as np


def gaussian(mu: float, sigma: float) -> Callable[..., float]:
    """Return a weight function evaluating a Gaussian at the first parent value.

    Args:
        mu: Mean.
        sigma: Standard deviation (> 0).

    Returns:
        A callable ``f(x, *rest) -> float``.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    two_sigma_sq = 2.0 * sigma * sigma

    def f(*parent_values) -> float:
        x = float(parent_values[0])
        return float(np.exp(-((x - mu) ** 2) / two_sigma_sq))

    f.__operator_name__ = "gaussian"
    f.__operator_params__ = {"mu": mu, "sigma": sigma}
    return f
