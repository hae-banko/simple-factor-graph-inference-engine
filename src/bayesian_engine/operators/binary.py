"""Binary operator: indicator for exact match between parent values."""

from typing import Callable


def binary() -> Callable[..., float]:
    """Return a weight function that returns 1.0 when all parent values are equal.

    Returns:
        A callable ``f(*all_values) -> float``.
    """

    def f(*parent_values) -> float:
        if len(parent_values) < 2:
            return 1.0
        first = parent_values[0]
        return 1.0 if all(v == first for v in parent_values[1:]) else 0.0

    f.__operator_name__ = "binary"
    f.__operator_params__ = {}
    return f
