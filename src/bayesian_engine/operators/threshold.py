"""Threshold operator: indicator [x ≥ bound]."""

from typing import Callable


def threshold(bound: float) -> Callable[..., float]:
    """Return a weight function that returns 1.0 when the first value ≥ bound.

    Args:
        bound: The threshold value.

    Returns:
        A callable ``f(x, *rest) -> float``.
    """

    def f(*parent_values) -> float:
        return 1.0 if float(parent_values[0]) >= bound else 0.0

    f.__operator_name__ = "threshold"
    f.__operator_params__ = {"bound": bound}
    return f
