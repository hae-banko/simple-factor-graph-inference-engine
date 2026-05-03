"""Table operator: conditional probability table (CPT) lookup."""

from typing import Callable


def table(cpt: dict[tuple, float]) -> Callable[..., float]:
    """Return a weight function that looks up weights in a CPT.

    Args:
        cpt: A dictionary mapping ``(parent_val1, ..., output_val)`` to a weight.

    Returns:
        A callable ``f(*all_values) -> float``.
    """

    def f(*parent_values) -> float:
        return float(cpt.get(tuple(parent_values), 0.0))

    f.__operator_name__ = "table"
    f.__operator_params__ = {"cpt": cpt}
    return f
