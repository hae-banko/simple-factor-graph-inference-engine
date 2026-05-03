"""Combinator operators: product, sum, max over parent values."""

import builtins
from typing import Callable


def product() -> Callable[..., float]:
    """Return a weight function that multiplies all values: ∏ w_i."""

    def f(*parent_values) -> float:
        result = 1.0
        for v in parent_values:
            result *= float(v)
        return result

    f.__operator_name__ = "product"
    f.__operator_params__ = {}
    return f


def sum() -> Callable[..., float]:
    """Return a weight function that sums all values: ∑ w_i."""

    def f(*parent_values) -> float:
        return float(builtins.sum(float(v) for v in parent_values))

    f.__operator_name__ = "sum"
    f.__operator_params__ = {}
    return f


def max() -> Callable[..., float]:
    """Return a weight function returning the maximum value: max(w_i)."""

    def f(*parent_values) -> float:
        if not parent_values:
            return 0.0
        return float(builtins.max(float(v) for v in parent_values))

    f.__operator_name__ = "max"
    f.__operator_params__ = {}
    return f
