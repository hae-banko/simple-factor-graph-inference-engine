"""Sigmoid operator — soft threshold with Gaussian compatibility scoring."""

from typing import Callable

import numpy as np


def sigmoid(w: np.ndarray, sigma: float = 0.1) -> Callable[..., float]:
    """Return a weight function that scores compatibility with a sigmoid prediction.

    The returned function receives all factor variable values (inputs then output).
    It computes ``predicted = σ(w · inputs)`` and returns a Gaussian compatibility
    score ``exp(-(output - predicted)² / 2σ²)``.

    Args:
        w: Weight vector. Must match the number of input variables.
        sigma: Compatibility bandwidth (default 0.1).

    Returns:
        A callable ``f(*inputs, output) -> float``.
    """
    w = np.asarray(w, dtype=float)
    two_sigma_sq = 2.0 * sigma * sigma

    def f(*all_values) -> float:
        *inputs, output = all_values
        x = np.asarray(inputs, dtype=float)
        predicted = float(1.0 / (1.0 + np.exp(-np.dot(w, x))))
        diff = float(output) - predicted
        return float(np.exp(-(diff**2) / two_sigma_sq))

    f.__operator_name__ = "sigmoid"
    f.__operator_params__ = {"w": w, "sigma": sigma}
    return f
